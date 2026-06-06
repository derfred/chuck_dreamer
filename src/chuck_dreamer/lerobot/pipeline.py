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

  # ---- per-episode stage list ---------------------------------------------
  def pipeline(self, episode_index: int) -> list[Stage]:
    """The dependency-ordered stages for one episode, bound to this run's
    shared context and stage registry.

    Sets the context's episode index, then resolves the enabled flags into a
    dependency-ordered list (auto-pulling each stage's ``requires``). Inter-stage
    mask state lives on the :class:`Episode` now, so there is no per-episode ctx
    cache to reset here."""
    self.ctx.episode_index = episode_index
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

  def lane_pipeline(self, episode_index: int, lane: str) -> list[Stage]:
    """The subset of :meth:`pipeline` whose stages run on ``lane``, in the
    same dependency order.

    Used by the parallel importer to split each episode's stages at the
    GPU/CPU boundary: the ``"producer"`` lane (decode-adjacent + SAM2
    segmentation) runs on the GPU producer, the ``"worker"`` lane (the
    object-pose fit) runs in the CPU pool. Because :meth:`pipeline` already
    sets the context's episode index and slice map, this just filters its
    result — call it once per lane for the same episode."""
    return [s for s in self.pipeline(episode_index) if getattr(s, "lane", "worker") == lane]
