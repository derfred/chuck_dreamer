"""The import-run object: pipeline nodes, run context, and dataset access.

A :class:`Run` ties an :class:`EpisodeSpec` to one set of importer flags. It
owns the shared, run-level :class:`RunContext` (so the FK evaluator and the
object-localization runtime config stay cached across episodes), resolves the
dataset's on-disk root + selected episode slices, and is the single factory
for a per-episode, dependency-ordered node list. Both the importer (which
runs the nodes through the harness) and ``import-lerobot doctor`` (which
prints each node's ``requirements()``) drive a :class:`Run`, so the
context/root/slice/node wiring lives in exactly one place.

A :class:`Run` also hides *all* LeRobot dataset access behind lazy accessors:
the :class:`LeRobotDataset` is constructed on first use, frames are grouped by
episode lazily, and :meth:`Run.episode_frames` hands back the stacked arrays
for one episode as an :class:`EpisodeFrames`. The importer never imports
``lerobot`` nor indexes a raw frame dict; it asks the run for slices, node
lists, and per-episode frame arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .context import RunContext

if TYPE_CHECKING:
  from chuck_dreamer.lerobot.episode_spec import EpisodeSlice, EpisodeSpec


def _frame_to_image(frame: Any) -> np.ndarray:
  """LeRobot returns video frames as CHW float32 in [0, 1]; the repo's
  episode format wants HWC uint8. Convert one frame."""
  arr = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
  arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
  return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class EpisodeFrames:
  """The stacked, decoded arrays for one episode, ready for the writer.

  Built by :meth:`Run.episode_frames` from the raw LeRobot frame dicts so
  the importer never touches a frame dict or a video key directly. ``T`` is
  the decoded frame count.
  """
  images: np.ndarray      # (T, H, W, 3) uint8
  action: np.ndarray      # (T, A) float32
  state: np.ndarray       # (T, J) float32
  timestamp: np.ndarray   # (T,) float32

  @property
  def length(self) -> int:
    return int(self.action.shape[0])

  @property
  def n_joints(self) -> int:
    return int(self.state.shape[1])


class Run:
  """One import run: an :class:`EpisodeSpec` plus the importer flags.

  Owns the shared :class:`RunContext`, is the sole factory for the per-episode
  node list, and is the single point of LeRobot dataset access. The importer
  and the doctor share the same context/root/slice resolution and flag→node
  wiring through it.
  """

  def __init__(self, config, spec: EpisodeSpec, params: dict[str, bool], video_key: str | None = None) -> None:
    self.spec                     = spec
    self.params                   = params
    self._video_key_pref          = video_key
    self.ctx                      = RunContext(config=config, source_repo=spec.dataset_id)
    self.local_root: Path | None  = (Path(spec.dataset_id) if Path(spec.dataset_id).is_dir() else None)
    self._nodes: list[Any] | None = None  # built lazily, reused per episode

  @property
  def dataset_id(self) -> str:
    return self.spec.dataset_id

  def set_object_localization_device(self, device: str) -> None:
    """Pin this run's SAM2 segmentation to ``device`` (e.g. ``"cuda:1"``).

    Used by the parallel importer to give each GPU producer its own device.
    Mutates *this run's* config in place; producers each hold their own
    ``Run`` (with a copied config — see the parallel orchestrator), so one
    producer's device never leaks into another's. The segmentation stage reads
    the value back through ``lookup_device(ctx.config,
    "object_localization.device")``."""
    from omegaconf import OmegaConf
    if self.ctx.config is None:
      return
    OmegaConf.update(
      self.ctx.config, "object_localization.device", device, force_add=True)

  # ---- slice / video-key resolution (offline ``meta/`` sidecars) ----------
  @cached_property
  def _resolved(self) -> tuple[list[EpisodeSlice], str]:
    """``(selected slices, resolved video key)``, read once from ``meta/``
    (no video touch) and cached for the run's lifetime."""
    return self.spec.read_episodes(video_key=self._video_key_pref, root=self.local_root)

  @property
  def slices(self) -> list[EpisodeSlice]:
    """This spec's selected episode slices."""
    return self._resolved[0]

  @property
  def video_key(self) -> str:
    """The resolved ``observation.images.*`` feature key the slices' video
    fields were populated from."""
    return self._resolved[1]

  def read_slices(self, video_key: str | None = None) -> tuple[list[EpisodeSlice], str]:
    """Resolve this spec's selected episode slices + the video key, reading
    only the offline ``meta/`` sidecars (no video touch).

    Kept for callers that want to override the run's video key; the no-arg
    form just returns the cached :attr:`slices` / :attr:`video_key`."""
    if video_key is None or video_key == self._video_key_pref:
      return self.slices, self.video_key
    return self.spec.read_episodes(video_key=video_key, root=self.local_root)

  # ---- lazy, per-episode LeRobot dataset access ---------------------------
  def episode_frames(self, episode_index: int) -> EpisodeFrames | None:
    """Decoded, stacked arrays for **one** episode, or ``None`` if no frames
    decoded. Hides the raw frame dicts and the video key behind the run.

    Builds a fresh ``LeRobotDataset`` scoped to just this episode, so only
    one episode's decoded frames are resident at a time — both the dataset
    and the frame dicts are released when this returns. Decoding *all*
    selected episodes up front (the previous approach) held tens to hundreds
    of GB for a multi-episode import and was OOM-killed.

    The ``lerobot`` import is deferred to here so merely constructing a
    :class:`Run` (the doctor's case) never pulls in the stack."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    ds = LeRobotDataset(
      self.dataset_id, root=self.local_root,
      episodes=[episode_index], download_videos=True)
    if len(ds) == 0:
      return None

    vkey = self.video_key
    images, action, state, timestamp = [], [], [], []
    for i in range(len(ds)):
      frame = ds[i]
      images.append(_frame_to_image(frame[vkey]))
      action.append(np.asarray(frame["action"], dtype=np.float32))
      state.append(np.asarray(frame["observation.state"], dtype=np.float32))
      timestamp.append(float(frame["timestamp"]))

    return EpisodeFrames(
      images=np.stack(images, axis=0),
      action=np.stack(action, axis=0),
      state=np.stack(state, axis=0),
      timestamp=np.asarray(timestamp, dtype=np.float32),
    )

  @cached_property
  def _dataset_meta(self) -> Any:
    """The dataset's ``LeRobotDatasetMetadata`` (full episode table), read
    once per run — :meth:`episode_joint_values` may be called for many
    touchpoint episodes in a row."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # type: ignore

    return LeRobotDatasetMetadata(self.dataset_id, root=self.local_root)

  def episode_joint_values(self, episode_index: int) -> np.ndarray:
    """``(T, J)`` float32 ``observation.state`` for one episode, read from
    the data parquet sidecars only — no video download or decode.

    Unlike :attr:`slices` this consults the dataset's *full* episode table,
    so it reaches episodes outside this run's selection: the touchpoint
    captures consumed by ``extract_table_to_arm`` are typically excluded
    from the imported episode range."""
    import pandas as pd  # type: ignore[import-untyped]

    meta = self._dataset_meta
    row  = next((r for r in meta.episodes if int(r["episode_index"]) == episode_index), None)
    if row is None:
      raise ValueError(
        f"{self.dataset_id}: episode {episode_index} not in the dataset")

    parquet = Path(meta.root) / meta.data_path.format(
      chunk_index=int(row["data/chunk_index"]),
      file_index=int(row["data/file_index"]))
    df     = pd.read_parquet(parquet, columns=["episode_index", "observation.state"])
    values = df.loc[df["episode_index"] == episode_index, "observation.state"]
    return np.stack(values.to_numpy()).astype(np.float32)

  # ---- per-episode node list ------------------------------------------------
  def pipeline(self, episode_index: int) -> list[Any]:
    """The dependency-ordered pipeline nodes for one episode, bound to this
    run's shared context.

    Sets the context's episode index, then builds the node list from the
    enabled flags: the arm-frame EE chain (``normalize_joints`` →
    ``ee_pos_arm`` → ``ee_rotation`` → ``ee_action_arm``) for
    ``with_ee_pos``, the object chain (``segmentation`` →
    ``obj_uv_extraction`` → ``object_pose``) for ``with_object_pose``, and
    the table-frame variants (``ee_pos_table`` / ``ee_quat_table`` /
    ``ee_action_table`` / ``obj_pos_arm``, all consuming the dataset's
    ``table_to_arm`` transform) for ``with_table_frame``. Node instances
    are built once and reused across episodes (they are episode-stateless;
    per-episode state flows through the
    :class:`~chuck_dreamer.lerobot.trackset.TrackSet`). The importer runs
    the list through
    :func:`~chuck_dreamer.lerobot.trackset.run_episode`."""
    from .nodes import (
      build_ee_nodes,
      build_ee_table_nodes,
      build_object_nodes,
      build_object_table_nodes,
    )

    self.ctx.episode_index = episode_index
    # Expose this run's slices (video window / MP4 path) to nodes so the
    # segmentation node can decode the video without re-reading metadata.
    if not self.ctx.slices_by_index:
      self.ctx.slices_by_index = {sl.episode_index: sl for sl in self.slices}

    if self._nodes is None:
      table_frame = self.params.get("with_table_frame", False)
      nodes: list[Any] = []
      if self.params.get("with_ee_pos", False):
        nodes.extend(build_ee_nodes(self.ctx))
        if table_frame:
          nodes.extend(build_ee_table_nodes(self.ctx))
      if self.params.get("with_object_pose", False):
        nodes.extend(build_object_nodes(self.ctx))
        if table_frame:
          nodes.extend(build_object_table_nodes(self.ctx))
      self._nodes = nodes
    return self._nodes

  def lane_pipeline(self, episode_index: int, lane: str) -> list[Any]:
    """The subset of :meth:`pipeline` whose nodes run on ``lane``, in the
    same dependency order.

    Used by the parallel importer to split each episode's nodes at the
    GPU/CPU boundary: the ``"producer"`` lane (the EE chain + SAM2
    segmentation) runs on the GPU producer, the ``"worker"`` lane (the
    object-pose fit) runs in the CPU pool. Because :meth:`pipeline` already
    sets the context's episode index and slice map, this just filters its
    result — call it once per lane for the same episode."""
    return [n for n in self.pipeline(episode_index) if getattr(n, "lane", "worker") == lane]
