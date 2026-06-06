"""SAM2 segmentation stage for the tracked object.

:class:`SegmentationStage` segments the object across an episode's video.
Masks are cached on the :class:`RunContext` so downstream stages (e.g.
object pose) can consume them; the ``segmentation_target`` / ``object_uv``
arrays are written onto the episode.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING, Any

from chuck_dreamer.config import lookup_device
import numpy as np

from .base import Requirement, RunContext

if TYPE_CHECKING:
  from chuck_dreamer.common.episode import Episode

SAM2_MODEL = "facebook/sam2-hiera-large"


def _masks_to_seg_and_uv(
    T: int, masks: list[np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray] | None:
  """Build the ``(T, H, W)`` uint8 ``segmentation_target`` array and the
  ``(T, 2)`` float32 ``object_uv`` centroid array from ``T`` per-frame SAM2
  masks, or ``None`` if no frame had a mask.

  Pixels carry label 1 inside the target mask, 0 elsewhere. Centroids are
  the mean pixel coordinate of each frame's mask, ``NaN`` for frames with
  no mask — so downstream tooling can tell drop-outs apart from "mask
  centred at origin".
  """
  if T == 0:
    return None
  ref = next((np.asarray(m) for m in masks if m is not None), None)
  if ref is None:
    return None
  H, W = ref.shape[:2]

  # Densify the ragged ``list[mask | None]`` into one (T, H, W) bool stack;
  # the per-frame work is just a slice assignment, no pixel loops.
  seg = np.zeros((T, H, W), dtype=bool)
  for t, m in enumerate(masks):
    if m is None:
      continue
    m_arr = np.asarray(m, dtype=bool)
    if m_arr.shape[:2] == (H, W):
      seg[t] = m_arr

  # Centroids for all frames at once: ``Σ coord·mask / Σ mask`` per frame.
  vs = np.arange(H, dtype=np.float32)
  us = np.arange(W, dtype=np.float32)
  counts = seg.sum(axis=(1, 2)).astype(np.float32)         # (T,)
  sum_u = (seg.sum(axis=1) * us).sum(axis=1)                # (T,)
  sum_v = (seg.sum(axis=2) * vs).sum(axis=1)                # (T,)
  uv = np.full((T, 2), np.nan, dtype=np.float32)
  nz = counts > 0
  uv[nz, 0] = sum_u[nz] / counts[nz]
  uv[nz, 1] = sum_v[nz] / counts[nz]

  return seg.astype(np.uint8), uv


def _largest_cc(mask: np.ndarray) -> np.ndarray | None:
  """Keep only the largest 8-connected component of a boolean mask.
  SAM2 can leak a few stray pixels far from the object"""
  import cv2

  m = mask.astype(np.uint8)
  if m.sum() == 0:
    return None
  _, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
  if labels.max() == 0:
    return None
  best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
  return np.asarray(labels == best)


class SegmentationStage:
  """Run SAM2 on the object, caching masks on ``ctx`` for downstream stages
  (e.g. pose fitting) and filling the ``segmentation_target`` / ``object_uv``
  arrays."""
  name = "segment:object"
  produces: tuple[str, ...] = ("segmentation_target", "object_uv")
  requires: tuple[str, ...] = ()
  # SAM2 runs on the GPU; this is the producer's GPU work.
  lane: str = "producer"

  def __init__(self, ctx: RunContext) -> None:
    self.ctx = ctx

  @contextmanager
  def _video_as_jpgs(self, metadata: dict[str, Any]):
    """Decode this episode's ``[from_ts, to_ts)`` video window to a temp
    directory of zero-padded ``%05d.jpg`` frames and yield its path.

    The MP4 path and decode window come straight off the episode's
    :class:`EpisodeSlice` (resolved once by :class:`Run` from the offline
    ``meta/`` sidecars), so the stage never re-opens the dataset metadata.
    SAM2's ``init_state`` consumes such a directory natively. The temp
    directory is removed on exit (success or error)."""
    import av
    import cv2

    episode_idx = int(metadata["config"]["episode_index"])
    sl          = self.ctx.slice_for(episode_idx)
    video_path  = sl.video_path
    from_ts     = float(sl.video_from_ts)
    to_ts       = float(sl.video_to_ts)

    jpg_dir = Path(tempfile.mkdtemp(prefix=f"seg_ep{episode_idx}_"))
    try:
      n = 0
      with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        tb = stream.time_base
        if tb is None:
          raise ValueError(f"{video_path}: video stream has no time_base")
        # av.seek with stream= takes an offset in the stream's time_base
        # (NOT microseconds); seek backward to the keyframe at/just before
        # from_ts, then drop frames until the window starts and stop at its
        # end.
        container.seek(int(from_ts / tb), stream=stream, any_frame=False, backward=True)
        for frame in container.decode(stream):
          if frame.pts is None:
            continue
          pts_s = float(frame.pts * tb)
          if pts_s < from_ts:
            continue
          if pts_s >= to_ts:
            break
          rgb = frame.to_ndarray(format="rgb24")
          cv2.imwrite(str(jpg_dir / f"{n:05d}.jpg"),
                      cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
          n += 1

      debug_dir = metadata.get("debug_dir")
      if debug_dir is not None:
        dest = Path(debug_dir) / "jpgs"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(jpg_dir, dest)
        print(f"[{self.name}] ep{episode_idx}: retained {n} JPEG frame(s) "
              f"at {dest}")

      yield jpg_dir
    finally:
      shutil.rmtree(jpg_dir, ignore_errors=True)

  def _segment_jpgs(self, jpg_dir: Path,
                    keyframes: dict[int, tuple[int, int]]) -> list[np.ndarray | None]:
    """Run SAM2's video predictor over a ``%05d.jpg`` directory, prompting
    at each ``{frame_idx: (u, v)}`` keyframe, and return one
    ``HxW bool`` mask (or ``None``) per frame.

    SAM2 ``init_state`` consumes the JPEG directory natively; per-frame
    logits are thresholded and reduced to their largest connected
    component."""
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    # Clamp prompts to the frames that were actually decoded: SAM2 indexes
    # its state by ``frame_idx``, so a keyframe past the decoded count (e.g.
    # if the window decoded short) would IndexError inside the predictor.
    n_jpgs = len(list(jpg_dir.glob("*.jpg")))
    in_range = {fi: pr for fi, pr in keyframes.items() if 0 <= fi < n_jpgs}
    if 0 not in in_range:
      raise RuntimeError(
        f"frame-0 prompt missing after clamping to {n_jpgs} decoded frame(s) "
        f"(keyframes were {sorted(keyframes)})")
    dropped = sorted(set(keyframes) - set(in_range))
    if dropped:
      print(f"[{self.name}] dropping out-of-range keyframe prompt(s) "
            f"{dropped} (only {n_jpgs} frames decoded)")

    device    = lookup_device(self.ctx.config, "object_localization.device", pytorch_only=True)
    predictor = SAM2VideoPredictor.from_pretrained(SAM2_MODEL, device=device)
    state     = predictor.init_state(str(jpg_dir), offload_video_to_cpu=True)

    for frame_idx, (u, v) in sorted(in_range.items()):
      predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=frame_idx,
        obj_id=0,
        points=np.array([[float(u), float(v)]], dtype=np.float32),
        labels=np.ones(1, dtype=np.int32),
      )

    per_frame: dict[int, np.ndarray | None] = {}
    for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
      mask = (mask_logits[0] > 0).cpu().numpy()
      if mask.ndim == 3:
        mask = mask[0]
      per_frame[int(frame_idx)] = _largest_cc(mask.astype(bool))

    if not per_frame:
      return []
    n_frames = max(per_frame) + 1
    return [per_frame.get(i) for i in range(n_frames)]

  def requirements(self) -> list[Requirement]:
    # SAM2 only needs the video and a frame-0 prompt. Camera calibration and
    # the mesh are the object-pose stage's concern (it declares them in its
    # own requirements), so they're intentionally not listed here.
    ds_dir = self.ctx.dataset_cache_dir()
    ep     = self.ctx.episode_index

    def frame0_prompt_present() -> bool:
      return 0 in self.ctx.keyframe_prompts(ep)

    return [
      Requirement(
        f"frame-0 object prompt (episode {ep})", ds_dir / "object_prompts.json",
        f"uv run python main.py prompt-episodes --dataset {self.ctx.source_repo}",
        check=frame0_prompt_present),
    ]

  def apply(self, episode: "Episode", metadata: dict[str, Any]) -> None:
    T = metadata.get("number_of_frames")
    if T is None or T == 0:
      return

    cfg           = metadata["config"]
    source_repo   = cfg["source_repo"]
    episode_index = int(cfg["episode_index"])

    keyframes = self.ctx.keyframe_prompts(episode_index)
    keyframes = {fi: pr for fi, pr in keyframes.items() if 0 <= fi < T}
    if 0 not in keyframes:
      raise FileNotFoundError(
        f"no cached frame-0 prompt for {source_repo} episode {episode_index}. "
        f"Run: `python main.py prompt-episodes --dataset {source_repo}`")
    print(f"[{self.name}] {source_repo} ep{episode_index}: "
          f"using {len(keyframes)} keyframe prompt(s) at frames "
          f"{sorted(keyframes.keys())}")

    with self._video_as_jpgs(metadata) as jpg_dir:
      masks = self._segment_jpgs(jpg_dir, keyframes)

    episode.set("object_masks", masks, persist=False)

    seg_and_uv = _masks_to_seg_and_uv(T, masks) if masks is not None else None
    if seg_and_uv is not None:
      episode["segmentation_target"] = seg_and_uv[0]
      episode["object_uv"]           = seg_and_uv[1]
