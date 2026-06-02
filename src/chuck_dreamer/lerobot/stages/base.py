"""Core stage types shared across the stage families.

A :class:`Stage` is bound to a :class:`RunContext` at construction and
declares ``produces`` (episode/metadata keys it fills), ``requires``
(names of stages that must run before it), its external
:class:`Requirement` list, and an :meth:`Stage.apply` transform.
:class:`Run` resolves the enabled stages per episode; the importer
calls ``stage.apply(episode, metadata)`` over them and ``--doctor``
iterates the same stage list printing each stage's
:meth:`Stage.requirements`.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
  config, and the per-episode mask cache (so the object-pose stage can
  read the masks the segmentation stage produced) all hang off one object
  with a lifetime of one import run.

  Stages are bound to a context at construction and read these caches via
  the accessor methods below. The per-episode ``masks`` cache is reset by
  :class:`Run` each time it builds an episode's stage list.
  """
  source_repo: str
  config: Any = None

  _episode_index: int = 0
  _fk: Any | None = None
  _ol_cfg: Any | None = None
  # per-episode, keyed by segmentation target name -> [mask | None]
  masks: dict[str, list[Any]] = field(default_factory=dict)
  # this run's selected episode slices, keyed by episode_index; populated
  # by ``Run`` so stages can read an episode's video window / MP4 path
  # without re-opening ``LeRobotDatasetMetadata``.
  slices_by_index: dict[int, Any] = field(default_factory=dict)

  @property
  def episode_index(self) -> int:
    """The episode the current pipeline runs on. Set by ``Run.pipeline`` /
    :meth:`for_episode`; read by episode-specific stage requirements."""
    return self._episode_index

  @episode_index.setter
  def episode_index(self, value: int) -> None:
    self._episode_index = value

  @contextmanager
  def for_episode(self, episode_index: int) -> Iterator[None]:
    """Context manager for one episode's pipeline: sets the episode index and
    clears the per-episode mask cache."""
    self._episode_index = episode_index
    self.masks.clear()
    try:
      yield
    finally:
      self._episode_index = 0

  def fk(self) -> Any:
    if self._fk is None:
      if not FK_MODEL_PATH.exists():
        raise FileNotFoundError(
          f"FK MuJoCo model not found at {FK_MODEL_PATH}. "
          "Restore assets/mujoco/so101_arm.xml from git.")
      from chuck_dreamer.real.fk_calibration.fk import FK
      self._fk = FK(FK_MODEL_PATH)
    return self._fk

  @property
  def ol_cfg(self) -> Any:
    """Parsed ``object_localization`` view of this run's config, cached.

    Validates ``self.config`` once via ``init_from_config`` — no disk
    reload, the config is handed in at construction."""
    if self._ol_cfg is None:
      from chuck_dreamer.lerobot.object_localization import init_from_config
      self._ol_cfg = init_from_config(self.config)
    return self._ol_cfg

  def cache_dir(self) -> Path:
    """Root calibration-cache directory (``object_localization.calibration_cache``)."""
    return Path(self.ol_cfg.calibration_cache)

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

  def slice_for(self, episode_index: int) -> Any:
    """This run's :class:`EpisodeSlice` for one episode (its video window,
    MP4 path and chunk/file coordinates), populated by :class:`Run`.

    Raises ``KeyError`` if the episode isn't among the run's selected
    slices — a stage should only ever ask for the episode it's running on."""
    return self.slices_by_index[episode_index]


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
