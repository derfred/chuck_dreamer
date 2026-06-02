"""The import-run object: stages, run context, and dataset access in one place.

A :class:`Run` ties an :class:`EpisodeSpec` to one set of importer flags. It
owns the shared, run-level :class:`RunContext` (so the FK evaluator and the
object-localization runtime config stay cached across episodes), resolves the
dataset's on-disk root + selected episode slices, and
is the single factory for a per-episode, dependency-ordered stage list. Both
the importer (which calls ``stage.apply``) and ``import-lerobot --doctor``
(which calls ``stage.requirements``) drive a :class:`Run`, so the
context/root/slice/stage wiring lives in exactly one place.

A :class:`Run` also hides *all* LeRobot dataset access behind lazy accessors:
the :class:`LeRobotDataset` is constructed on first use, frames are grouped by
episode lazily, and :meth:`Run.episode_frames` hands back the stacked arrays
for one episode as an :class:`EpisodeFrames`. The importer never imports
``lerobot`` nor indexes a raw frame dict; it asks the run for slices, stage
lists, and per-episode frame arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .stages.base import RunContext, Stage
from .stages.registry import build_registry, resolve_stages

if TYPE_CHECKING:
  from chuck_dreamer.common.episode_spec import EpisodeSlice, EpisodeSpec


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
  stage list, and is the single point of LeRobot dataset access. The importer
  and the doctor share the same context/root/slice resolution and flag→stage
  wiring through it.
  """

  def __init__(self, config, spec: EpisodeSpec, params: dict[str, bool], video_key: str | None = None) -> None:
    self.spec             = spec
    self.params           = params
    self._video_key_pref  = video_key
    self.ctx              = RunContext(config=config, source_repo=spec.dataset_id)
    # A directory dataset_id is an on-disk LeRobot root; HF repo ids aren't.
    self.local_root: Path | None = (
      Path(spec.dataset_id) if Path(spec.dataset_id).is_dir() else None)
    self._registry = build_registry(self.ctx)

  @property
  def dataset_id(self) -> str:
    return self.spec.dataset_id

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

  # ---- lazy LeRobot dataset access ----------------------------------------
  @cached_property
  def dataset(self) -> Any:
    """The :class:`LeRobotDataset` for the selected episodes, built on first
    use (decodes video). The ``lerobot`` import is deferred to here so merely
    constructing a :class:`Run` (the doctor's case) never pulls in the stack."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    selected = [sl.episode_index for sl in self.slices]
    if not selected:
      raise RuntimeError(f"{self.dataset_id}: no episodes to import")
    return LeRobotDataset(
      self.dataset_id, root=self.local_root,
      episodes=selected, download_videos=True)

  @cached_property
  def _frames_by_episode(self) -> dict[int, list[dict]]:
    """The decoded frames grouped by ``episode_index``. LeRobotDataset
    flattens the selected episodes into one frame sequence; group consecutive
    frames back by their episode index."""
    ds = self.dataset
    grouped: dict[int, list[dict]] = {sl.episode_index: [] for sl in self.slices}
    for i in range(len(ds)):
      frame = ds[i]
      grouped[int(frame["episode_index"])].append(frame)
    return grouped

  def episode_frames(self, episode_index: int) -> EpisodeFrames | None:
    """Decoded, stacked arrays for one episode, or ``None`` if no frames
    decoded. Hides the raw frame dicts and the video key behind the run."""
    frames = self._frames_by_episode.get(episode_index, [])
    if not frames:
      return None
    vkey = self.video_key
    return EpisodeFrames(
      images=np.stack([_frame_to_image(f[vkey]) for f in frames], axis=0),
      action=np.stack([np.asarray(f["action"], dtype=np.float32) for f in frames], axis=0),
      state=np.stack([np.asarray(f["observation.state"], dtype=np.float32) for f in frames], axis=0),
      timestamp=np.asarray([float(f["timestamp"]) for f in frames], dtype=np.float32),
    )

  # ---- per-episode stage list ---------------------------------------------
  def pipeline(self, episode_index: int) -> list[Stage]:
    """The dependency-ordered stages for one episode, bound to this run's
    shared context and stage registry.

    Sets the context's episode index and clears the per-episode mask cache so
    a stage never sees a prior episode's masks, then resolves the enabled
    flags into a dependency-ordered list (auto-pulling each stage's
    ``requires``)."""
    self.ctx.episode_index = episode_index
    self.ctx.masks.clear()
    # Expose this run's slices (video window / MP4 path) to stages so a
    # segmentation stage can decode the video without re-reading metadata.
    if not self.ctx.slices_by_index:
      self.ctx.slices_by_index = {sl.episode_index: sl for sl in self.slices}

    enabled: set[str] = set()
    if self.params.get("with_ee_pos", False):
      enabled.add("ee_pos")
    if self.params.get("with_object_pose", False):
      enabled.add("object_pose")
    return resolve_stages(enabled, self.ctx, registry=self._registry)
