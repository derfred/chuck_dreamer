"""Run-level context shared by every pipeline node.

Nodes are bound to a :class:`RunContext` at construction and read the run's
cached resources (FK evaluator, object-localization config view, artifact
store, episode slices) through it. Every user-supplied or cached artifact
resolves through the artifact store (``docs/trainer/artifact_store.md``) —
there is no legacy ``calibration_cache/`` fallback; a miss raises
:class:`~chuck_dreamer.store.MissingArtifact` naming the producing command.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chuck_dreamer.common import FK_MODEL_PATH


@dataclass
class RunContext:
  """Run-level shared state, reused across every episode's pipeline.

  The FK evaluator, the object-localization runtime config, and the artifact
  store hang off one object with a lifetime of one import run.

  Nodes are bound to a context at construction and read these caches via
  the accessor methods below. Inter-node *episode* state (e.g. the SAM2 masks
  handed from segmentation to object-pose) lives on the :class:`TrackSet` /
  :class:`Episode`, not here — this context is purely run-scoped, so
  concurrent producers in the parallel importer can each hold one without
  colliding.
  """
  source_repo: str
  config: Any = None

  _episode_index: int = 0
  _fk: Any | None = None
  _ol_cfg: Any | None = None
  _store: Any | None = None
  _store_resolved: bool = False
  # this run's selected episode slices, keyed by episode_index; populated
  # by ``Run`` so stages can read an episode's video window / MP4 path
  # without re-opening ``LeRobotDatasetMetadata``.
  slices_by_index: dict[int, Any] = field(default_factory=dict)

  @property
  def episode_index(self) -> int:
    """The episode the current pipeline runs on. Set by ``Run.pipeline`` /
    :meth:`for_episode`."""
    return self._episode_index

  @episode_index.setter
  def episode_index(self, value: int) -> None:
    self._episode_index = value

  @contextmanager
  def for_episode(self, episode_index: int) -> Iterator[None]:
    """Context manager for one episode's pipeline: sets the episode index for
    the duration and restores it afterward."""
    self._episode_index = episode_index
    try:
      yield
    finally:
      self._episode_index = 0

  def fk(self) -> Any:
    if self._fk is None:
      if not FK_MODEL_PATH.exists():
        raise FileNotFoundError(
          f"FK URDF model not found at {FK_MODEL_PATH}. "
          "Restore assets/urdf/SO101/so101_new_calib.urdf from git.")
      from chuck_dreamer.common.fk import FK
      self._fk = FK(FK_MODEL_PATH)
    return self._fk

  @property
  def ol_cfg(self) -> Any:
    """Parsed ``object_localization`` view of this run's config, cached.

    Validates ``self.config`` once via ``init_from_config`` — no disk
    reload, the config is handed in at construction."""
    if self._ol_cfg is None:
      from chuck_dreamer.perception.config import init_from_config
      self._ol_cfg = init_from_config(self.config)
    return self._ol_cfg

  @property
  def store(self) -> Any | None:
    """This run's :class:`~chuck_dreamer.store.ArtifactStore`, or ``None``
    when the config declares no ``store.root``."""
    if not self._store_resolved:
      self._store_resolved = True
      root = None
      if self.config is not None:
        from omegaconf import OmegaConf
        root = OmegaConf.select(self.config, "store.root")
      if root:
        from chuck_dreamer.store import ArtifactStore
        self._store = ArtifactStore(Path(root))
    return self._store

  def _require_store(self, what: str) -> Any:
    store = self.store
    if store is None:
      from chuck_dreamer.store import MissingArtifact
      raise MissingArtifact(
        f"{what} lives in the artifact store, but no store is configured; "
        "set store.root in the config")
    return store

  def dataset_slug(self) -> str:
    """Filesystem-safe slug for this source repo (used in debug paths)."""
    from chuck_dreamer.store import dataset_slug
    return dataset_slug(self.source_repo)

  def keyframe_prompts(self, episode_index: int) -> dict[int, Any]:
    """``{frame_index: prompt}`` map for one episode of this source repo
    (empty if not annotated). The segmentation and object-pose stages share
    this lookup so neither reaches into the annotation package."""
    return self.resolve_keyframe_prompts(episode_index)[0]

  # ---- store artifact resolution ---------------------------------------------
  # Each resolver returns ``(value, version)``: the artifact plus the content
  # identity the harness records in ``input_versions`` (the store record's
  # ``payload_hash``).

  def dataset_config(self) -> dict[str, Any] | None:
    """This dataset's ``dataset_config`` artifact, or ``None`` when the store
    is absent or holds none."""
    store = self.store
    if store is None:
      return None
    from chuck_dreamer.store import for_dataset
    if not store.has("dataset_config", for_dataset(self.source_repo)):
      return None
    payload, _ = store.get("dataset_config", for_dataset(self.source_repo))
    return dict(payload)

  def resolve_intrinsics(self) -> tuple[Any, str | None]:
    """Camera intrinsics: CAMERA-scoped store artifact, resolved through this
    dataset's ``dataset_config.camera_id``."""
    from chuck_dreamer.perception.types import Intrinsics
    from chuck_dreamer.store import MissingArtifact, for_camera

    store = self._require_store("intrinsics")
    cfg = self.dataset_config()
    camera_id = (cfg or {}).get("camera_id")
    if not camera_id:
      raise MissingArtifact(
        f"dataset_config.camera_id is not set for {self.source_repo}; run: "
        f"uv run python main.py import-lerobot annotate-mat {self.source_repo}")
    payload, record = store.get("intrinsics", for_camera(str(camera_id)))
    return Intrinsics.from_json(payload), record.payload_hash

  def resolve_extrinsics(self) -> tuple[Any, str | None]:
    """Camera extrinsics: DATASET-scoped store artifact."""
    from chuck_dreamer.perception.types import Extrinsics
    from chuck_dreamer.store import for_dataset

    store = self._require_store("extrinsics")
    payload, record = store.get("extrinsics", for_dataset(self.source_repo))
    return Extrinsics.from_json(payload), record.payload_hash

  def resolve_table_to_arm(self) -> tuple[dict[str, Any], str | None]:
    """The dataset's ``table_to_arm`` transform: DATASET-scoped store
    artifact, produced by ``extract_table_to_arm``. A miss names the
    producing command."""
    from chuck_dreamer.store import MissingArtifact, for_dataset

    store = self._require_store("table_to_arm")
    scope = for_dataset(self.source_repo)
    if not store.has("table_to_arm", scope):
      raise MissingArtifact(
        f"table_to_arm not in store for {self.source_repo}; run: "
        f"uv run python main.py import-lerobot extract-table-to-arm "
        f"{self.source_repo}")
    payload, record = store.get("table_to_arm", scope)
    return dict(payload), record.payload_hash

  def resolve_keyframe_prompts(self, episode_index: int
                               ) -> tuple[dict[int, Any], str | None]:
    """One episode's object prompts: EPISODE-scoped store artifact. Empty
    (with ``version=None``) when the episode has not been annotated."""
    from chuck_dreamer.store import for_episode

    store = self.store
    if store is None:
      return {}, None
    scope = for_episode(self.source_repo, episode_index)
    if not store.has("object_prompts", scope):
      return {}, None
    payload, record = store.get("object_prompts", scope)
    return ({int(fi): pr for fi, pr in payload.items()}, record.payload_hash)

  def slice_for(self, episode_index: int) -> Any:
    """This run's :class:`EpisodeSlice` for one episode (its video window,
    MP4 path and chunk/file coordinates), populated by :class:`Run`.

    Raises ``KeyError`` if the episode isn't among the run's selected
    slices — a node should only ever ask for the episode it's running on."""
    return self.slices_by_index[episode_index]
