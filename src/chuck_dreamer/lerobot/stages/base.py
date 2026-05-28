"""Core stage types shared across the stage families.

A :class:`Stage` declares ``produces`` (episode/metadata keys it fills),
``requires`` (names of stages that must run before it), its external
:class:`Requirement` list, and an :meth:`Stage.apply` transform. The
importer runs a resolved list of stages per episode; ``--doctor`` iterates
the same list and prints each stage's :meth:`Stage.requirements`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ...common import FK_MODEL_PATH


@dataclass(frozen=True)
class Requirement:
  """An external artifact a stage needs before it can run."""
  label: str
  path: Path
  remediation: str

  def satisfied(self) -> bool:
    return self.path.exists()


@dataclass
class StageContext:
  """Per-run shared state threaded through every stage call.

  Replaces the module-level ``_ensure_*`` caches that used to live in
  ``pipeline.py``: the FK evaluator, the object-localization runtime
  config, the per-source-repo estimator + smoother, and the per-episode
  mask cache (so a pose stage can read the masks a segmentation stage
  produced) all hang off one object with a lifetime of one import run.
  """
  source_repo: str
  config_paths: list[str] = field(default_factory=list)

  _fk: Any | None = None
  _ol_cfg: Any | None = None
  _estimator: Any | None = None
  _smoother: Any | None = None
  # per-episode, keyed by segmentation target name -> [mask | None]
  masks: dict[str, list[Any]] = field(default_factory=dict)

  def fk(self) -> Any:
    if self._fk is None:
      if not FK_MODEL_PATH.exists():
        raise FileNotFoundError(
          f"FK MuJoCo model not found at {FK_MODEL_PATH}. "
          "Restore assets/mujoco/so101_arm.xml from git.")
      from chuck_dreamer.real.fk_calibration.fk import FK
      self._fk = FK(FK_MODEL_PATH)
    return self._fk

  def ol_cfg(self) -> Any:
    if self._ol_cfg is None:
      from chuck_dreamer.config import load_config
      from chuck_dreamer.real.object_localization import (
        active, init_from_config,
      )
      init_from_config(load_config(self.config_paths))
      self._ol_cfg = active()
    return self._ol_cfg

  def estimator(self) -> Any:
    if self._estimator is None:
      ol_cfg = self.ol_cfg()
      cache_dir = Path(ol_cfg.calibration_cache)
      mesh_path = Path(ol_cfg.mesh_path)
      from chuck_dreamer.real.object_localization import (
        ObjectPoseEstimator, load_calibration,
      )
      cal = load_calibration(cache_dir, self.source_repo)
      if not mesh_path.exists():
        raise FileNotFoundError(
          f"mesh file not found at {mesh_path}; check "
          "object_localization.mesh_path in configs/default.yaml.")
      self._estimator = ObjectPoseEstimator(
        cal, mesh_path, device=ol_cfg.device,
        use_sam2=ol_cfg.use_sam2, scene_bg=None,
      )
    return self._estimator

  def smoother(self) -> Any:
    if self._smoother is None:
      ol_cfg = self.ol_cfg()
      from chuck_dreamer.real.object_localization.smoother import (
        SmoothedTrajectoryEstimator,
      )
      self._smoother = SmoothedTrajectoryEstimator(ol_cfg.raw.get("smoother") or {})
    return self._smoother


@runtime_checkable
class Stage(Protocol):
  name: str
  produces: tuple[str, ...]
  requires: tuple[str, ...]

  def requirements(self, ctx: StageContext) -> list[Requirement]: ...
  def apply(self, episode: dict[str, Any], metadata: dict[str, Any],
            ctx: StageContext) -> None: ...


def as_uint8_hwc(images: Any) -> np.ndarray:
  """Coerce a frame stack to ``(T, H, W, 3)`` uint8 for the estimator."""
  imgs = np.asarray(images)
  if imgs.dtype != np.uint8:
    imgs = (np.clip(imgs.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
  if imgs.ndim == 4 and imgs.shape[1] in (1, 3) and imgs.shape[-1] not in (1, 3):
    imgs = np.transpose(imgs, (0, 2, 3, 1))
  return imgs
