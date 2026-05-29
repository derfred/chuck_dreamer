"""Core stage types shared across the stage families.

A :class:`Stage` is bound to a :class:`RunContext` at construction and
declares ``produces`` (episode/metadata keys it fills), ``requires``
(names of stages that must run before it), its external
:class:`Requirement` list, and an :meth:`Stage.apply` transform. A
:class:`Pipeline` resolves the enabled stages per episode; the importer
calls ``stage.apply(episode, metadata)`` over them and ``--doctor``
iterates the same pipeline printing each stage's
:meth:`Stage.requirements`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ...common import FK_MODEL_PATH


@dataclass(frozen=True)
class Requirement:
  """An external artifact a stage needs before it can run.

  ``satisfied`` defaults to a plain ``path.exists()`` check; stages with
  finer-grained needs (e.g. a *per-episode* entry inside a shared sidecar
  file) pass a custom ``check`` callable.
  """
  label: str
  path: Path
  remediation: str
  check: Callable[[], bool] | None = None

  def satisfied(self) -> bool:
    if self.check is not None:
      return self.check()
    return self.path.exists()


@dataclass
class RunContext:
  """Run-level shared state, reused across every episode's pipeline.

  Replaces the module-level ``_ensure_*`` caches that used to live in
  ``pipeline.py``: the FK evaluator, the object-localization runtime
  config, the per-source-repo estimator + smoother, and the per-episode
  mask cache (so a pose stage can read the masks a segmentation stage
  produced) all hang off one object with a lifetime of one import run.

  Stages are bound to a context at construction and read these caches via
  the accessor methods below. The per-episode ``masks`` cache is reset by
  each :class:`Pipeline` on construction.
  """
  source_repo: str
  config_paths: list[str] = field(default_factory=list)
  # The episode the active pipeline runs on; set by Pipeline on construction
  # so stages can surface episode-specific requirements.
  episode_index: int = 0

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
      from chuck_dreamer.lerobot.object_localization import (
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
      from chuck_dreamer.lerobot.object_localization import (
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
      from chuck_dreamer.lerobot.object_localization.smoother import (
        SmoothedTrajectoryEstimator,
      )
      self._smoother = SmoothedTrajectoryEstimator(ol_cfg.raw.get("smoother") or {})
    return self._smoother

  def cache_dir(self) -> Path:
    """Root calibration-cache directory (``object_localization.calibration_cache``)."""
    return Path(self.ol_cfg().calibration_cache)

  def dataset_cache_dir(self) -> Path:
    """This source repo's per-dataset subdirectory under the cache root."""
    from chuck_dreamer.lerobot.object_localization.types import dataset_cache_dir
    return dataset_cache_dir(self.cache_dir(), self.source_repo)

  def dataset_slug(self) -> str:
    """Filesystem-safe slug for this source repo (used in cache/debug paths)."""
    from chuck_dreamer.lerobot.object_localization.types import dataset_slug
    return dataset_slug(self.source_repo)

  def keyframe_prompts(self, episode_index: int) -> dict[int, Any]:
    """Cached ``{frame_index: prompt}`` map for one episode of this source
    repo (empty if none cached). The segmentation and object-pose stages
    share this lookup so neither reaches into ``object_localization``."""
    from chuck_dreamer.lerobot.object_localization.prompts import (
      load_keyframe_prompts,
    )
    return load_keyframe_prompts(self.cache_dir(), self.source_repo, episode_index)


@runtime_checkable
class Stage(Protocol):
  name: str
  produces: tuple[str, ...]
  requires: tuple[str, ...]
  ctx: RunContext

  def requirements(self) -> list[Requirement]: ...
  def apply(self, episode: dict[str, Any], metadata: dict[str, Any]) -> None: ...


def as_uint8_hwc(images: Any) -> np.ndarray:
  """Coerce a frame stack to ``(T, H, W, 3)`` uint8 for the estimator."""
  imgs = np.asarray(images)
  if imgs.dtype != np.uint8:
    imgs = (np.clip(imgs.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
  if imgs.ndim == 4 and imgs.shape[1] in (1, 3) and imgs.shape[-1] not in (1, 3):
    imgs = np.transpose(imgs, (0, 2, 3, 1))
  return imgs
