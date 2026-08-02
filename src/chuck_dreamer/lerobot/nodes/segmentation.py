"""Native segmentation nodes (spec §7.4 / §7.5).

:class:`SegmentationNode` segments the tracked object across an episode's
video with SAM2, seeded by the cached object prompts, producing the
``object_masks`` track (dense ``(T, H, W)`` bool + per-frame validity).
The track is declared ``Persist.CACHED``, so the *runner* caches it in the
artifact store keyed by the declared inputs' content versions — re-imports
with unchanged inputs skip SAM2 entirely without this node knowing.

:class:`ObjUvExtractionNode` derives the trajectory tracks from the masks:
``segmentation_target`` (the label image) and ``obj_uv`` (the mask
centroid per frame; ``NaN`` values on frames without a mask, with the
validity mask alongside).
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING, Any

import numpy as np

from chuck_dreamer.common.tracks import Frame, Persist, Track, TrackSpec, Unit
from chuck_dreamer.config import lookup_device

from ..context import RunContext
from ..harness import InputDecl

if TYPE_CHECKING:
  from ..trackset import NodeView


# ---------------------------------------------------------------------------
# Mask math (unchanged from the legacy stage — the numerics anchor)
# ---------------------------------------------------------------------------
def _masks_to_seg_and_uv(
    T: int, masks: list[np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray] | None:
  """Build the ``(T, H, W)`` uint8 ``segmentation_target`` array and the
  ``(T, 2)`` float32 ``obj_uv`` centroid array from ``T`` per-frame SAM2
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


def masks_to_dense(T: int, masks: list[np.ndarray | None]
                   ) -> tuple[np.ndarray, np.ndarray]:
  """``list[mask | None]`` → ``((T, H, W) bool, (T,) valid)``. Frames past
  the list's end, ``None`` entries, and shape-mismatched masks are invalid.
  With no mask anywhere the dense stack is a ``(T, 1, 1)`` placeholder."""
  ref = next((np.asarray(m) for m in masks if m is not None), None)
  H, W = (ref.shape[:2] if ref is not None else (1, 1))
  dense = np.zeros((T, H, W), dtype=bool)
  valid = np.zeros((T,), dtype=bool)
  for t, m in enumerate(masks[:T]):
    if m is None:
      continue
    m_arr = np.asarray(m, dtype=bool)
    if m_arr.shape[:2] == (H, W):
      dense[t] = m_arr
      valid[t] = True
  return dense, valid


def dense_to_masks(dense: np.ndarray, valid: np.ndarray
                   ) -> list[np.ndarray | None]:
  return [dense[t] if valid[t] else None for t in range(len(valid))]


# ---------------------------------------------------------------------------
# Segmentation node
# ---------------------------------------------------------------------------
class SegmentationNode:
  """Run SAM2 on the object; produce the ``object_masks`` track (CACHED —
  the runner stores/loads it keyed by the declared inputs' versions)."""
  name = "segmentation"
  scope_level = "episode"
  lane = "producer"          # SAM2 runs on the GPU producer
  inputs = (
    InputDecl("timestamp", unit=Unit.S),  # the episode's time axis
    InputDecl("object_prompts"),     # annotation artifact
    InputDecl("dataset.episodes"),   # this episode's video slice
    InputDecl("sam2_weights"),       # pinned model identity (assets/manifest.json)
  )
  outputs = (TrackSpec("object_masks", np.bool_, (), Frame.IMAGE,
                       Unit.UNITLESS, validity=True, persist=Persist.CACHED),)

  def __init__(self, ctx: RunContext) -> None:
    self.ctx = ctx

  # ---- video decode ----------------------------------------------------------
  @contextmanager
  def _video_as_jpgs(self, sl: Any, episode_index: int):
    """Decode this episode's ``[from_ts, to_ts)`` video window to a temp
    directory of zero-padded ``%05d.jpg`` frames and yield its path.

    SAM2's ``init_state`` consumes such a directory natively. The temp
    directory is removed on exit (success or error)."""
    import av
    import cv2

    video_path  = sl.video_path
    from_ts     = float(sl.video_from_ts)
    to_ts       = float(sl.video_to_ts)

    jpg_dir = Path(tempfile.mkdtemp(prefix=f"seg_ep{episode_index}_"))
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

      yield jpg_dir
    finally:
      shutil.rmtree(jpg_dir, ignore_errors=True)

  def _segment_jpgs(self, jpg_dir: Path,
                    keyframes: dict[int, tuple[int, int]],
                    hf_repo: str) -> list[np.ndarray | None]:
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
    predictor = SAM2VideoPredictor.from_pretrained(hf_repo, device=device)
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

  # ---- node body -------------------------------------------------------------
  def run(self, view: "NodeView") -> list[Track]:
    T = len(view.track("timestamp"))
    scope = view.scope

    keyframes = view.artifact("object_prompts")
    keyframes = {fi: pr for fi, pr in keyframes.items() if 0 <= fi < T}
    if 0 not in keyframes:
      raise FileNotFoundError(
        f"no cached frame-0 prompt for {scope.dataset_id} episode "
        f"{scope.episode_index}. Run: `python main.py import-lerobot "
        f"prompt-episodes {scope.dataset_id}`")
    print(f"[{self.name}] {scope.dataset_id} ep{scope.episode_index}: "
          f"using {len(keyframes)} keyframe prompt(s) at frames "
          f"{sorted(keyframes.keys())}")

    sl      = view.artifact("dataset.episodes")
    weights = view.artifact("sam2_weights")
    with self._video_as_jpgs(sl, scope.episode_index) as jpg_dir:
      masks = self._segment_jpgs(jpg_dir, keyframes, weights["hf_repo"])
    dense, valid = masks_to_dense(T, masks)

    spec = TrackSpec("object_masks", dense.dtype, tuple(dense.shape[1:]),
                     Frame.IMAGE, Unit.UNITLESS, validity=True,
                     persist=Persist.CACHED)
    return [Track(spec, dense, valid)]


# ---------------------------------------------------------------------------
# obj_uv extraction node
# ---------------------------------------------------------------------------
class ObjUvExtractionNode:
  """Spec §7.5: object image location per frame as the mask's center of
  mass. Inherits validity from the masks; the persisted ``obj_uv`` values
  keep the historical NaN-on-dropout convention."""
  name = "obj_uv_extraction"
  scope_level = "episode"
  lane = "producer"
  inputs = (InputDecl("object_masks", frame=Frame.IMAGE),
            InputDecl("image", frame=Frame.IMAGE))
  outputs = (
    TrackSpec("segmentation_target", np.uint8, (), Frame.IMAGE,
              Unit.UNITLESS, persist=Persist.TRAJECTORY),
    TrackSpec("obj_uv", np.float32, (2,), Frame.IMAGE,
              Unit.PX, validity=True, persist=Persist.TRAJECTORY),
  )

  def __init__(self, ctx: RunContext) -> None:
    self.ctx = ctx

  def run(self, view: "NodeView") -> list[Track]:
    masks = view.track("object_masks")
    T = len(masks)
    assert masks.valid is not None

    seg_and_uv = _masks_to_seg_and_uv(
      T, dense_to_masks(masks.values, masks.valid))
    if seg_and_uv is not None:
      seg, uv = seg_and_uv
    else:
      # No mask on any frame: empty label image at the video resolution
      # (legacy omitted the fields entirely; declared outputs are total).
      H, W = view.track("image").values.shape[1:3]
      seg = np.zeros((T, H, W), dtype=np.uint8)
      uv = np.full((T, 2), np.nan, dtype=np.float32)

    seg_spec = TrackSpec("segmentation_target", seg.dtype,
                         tuple(seg.shape[1:]), Frame.IMAGE, Unit.UNITLESS,
                         persist=Persist.TRAJECTORY)
    uv_spec = TrackSpec("obj_uv", np.float32, (2,), Frame.IMAGE, Unit.PX,
                        validity=True, persist=Persist.TRAJECTORY)
    return [Track(seg_spec, seg),
            Track(uv_spec, uv, masks.valid.copy())]
