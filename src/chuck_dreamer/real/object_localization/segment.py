"""Object segmentation backends used by ``ObjectPoseEstimator``.

The pipeline expects to be told *where* the object is on the first
frame (a 2D pixel prompt). SAM2's image and video predictors handle the
rest. There's a color-heuristic fallback for environments without SAM2;
it brightness-thresholds against the dark mat, which is correct
*enough* on the test rig but not what we recommend.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np

if TYPE_CHECKING:
  from .scene_bg import SceneBackground

logger = logging.getLogger(__name__)


@dataclass
class _SAM2Cached:
  image_predictor: object | None = None
  video_predictor: object | None = None


_SAM2_CACHE: dict[tuple[str, str], _SAM2Cached] = {}


def _get_sam2(checkpoint: str, device: str) -> _SAM2Cached:
  key = (checkpoint, device)
  if key in _SAM2_CACHE:
    return _SAM2_CACHE[key]
  entry = _SAM2Cached()
  try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
    entry.image_predictor = SAM2ImagePredictor.from_pretrained(checkpoint, device=device)
  except Exception as e:
    logger.warning("SAM2 image predictor unavailable (%s); will fall back.", e)
  try:
    from sam2.sam2_video_predictor import SAM2VideoPredictor  # type: ignore
    entry.video_predictor = SAM2VideoPredictor.from_pretrained(checkpoint, device=device)
  except Exception as e:
    logger.warning("SAM2 video predictor unavailable (%s); will fall back.", e)
  _SAM2_CACHE[key] = entry
  return entry


def segment_image(
  rgb: np.ndarray,
  prompt: tuple[int, int] | list[tuple[int, int]],
  use_sam2: bool, checkpoint: str, device: str,
  scene_bg: "SceneBackground | None" = None,
) -> np.ndarray | None:
  """Single-frame segmentation. Returns a ``HxW bool`` mask or ``None``.

  Backend selection (in order of preference):

    1. SAM2, if ``use_sam2`` is set and importable.
    2. Background subtraction, if a ``scene_bg`` reference is provided.
    3. Color floodfill from the click (the original fallback).
  """
  if use_sam2:
    sam = _get_sam2(checkpoint, device)
    pred = sam.image_predictor
    if pred is not None:
      try:
        pts = _prompt_to_array(prompt)
        labels = np.ones(pts.shape[0], dtype=np.int32)
        pred.set_image(rgb)
        masks, _, _ = pred.predict(point_coords=pts, point_labels=labels, multimask_output=False)
        return _largest_component(np.asarray(masks[0], dtype=bool))
      except Exception as e:
        logger.warning("SAM2 image predict failed (%s); using fallback.", e)
  if scene_bg is not None:
    mask = _bg_mask(rgb, prompt, scene_bg)
    if mask is not None:
      return mask
    logger.warning("background-subtraction segmenter returned no component "
                   "at the click; falling back to color floodfill.")
  return _fallback_mask(rgb, prompt)


def segment_video(
  frames: list[np.ndarray],
  first_frame_prompt: tuple[int, int] | list[tuple[int, int]],
  use_sam2: bool, checkpoint: str, device: str,
  scene_bg: "SceneBackground | None" = None,
) -> list[np.ndarray | None]:
  """Propagate a mask through an episode. Returns per-frame masks."""
  if use_sam2:
    sam = _get_sam2(checkpoint, device)
    pred = sam.video_predictor
    if pred is not None:
      try:
        return _sam2_video(pred, frames, first_frame_prompt)
      except Exception as e:
        logger.warning("SAM2 video predict failed (%s); falling back per-frame.", e)
  out: list[np.ndarray | None] = []
  for f in frames:
    mask = None
    if scene_bg is not None:
      mask = _bg_mask(f, first_frame_prompt, scene_bg)
    if mask is None:
      mask = _fallback_mask(f, first_frame_prompt)
    out.append(mask)
  return out


def _sam2_video(predictor, frames: list[np.ndarray],
                first_frame_prompt: tuple[int, int] | list[tuple[int, int]]
                ) -> list[np.ndarray | None]:
  pts = _prompt_to_array(first_frame_prompt)
  state = predictor.init_state(frames)
  predictor.add_new_points_or_box(
    inference_state=state, frame_idx=0, obj_id=0,
    points=pts, labels=np.ones(pts.shape[0], dtype=np.int32),
  )
  per_frame: dict[int, np.ndarray] = {}
  for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
    mask = (mask_logits[0] > 0).cpu().numpy()
    if mask.ndim == 3:
      mask = mask[0]
    per_frame[int(frame_idx)] = _largest_component(mask.astype(bool))
  return [per_frame.get(i) for i in range(len(frames))]


def _prompt_to_array(prompt) -> np.ndarray:
  if isinstance(prompt, tuple) and len(prompt) == 2 and isinstance(prompt[0], (int, float)):
    return np.array([[float(prompt[0]), float(prompt[1])]], dtype=np.float32)
  return np.array([[float(u), float(v)] for u, v in prompt], dtype=np.float32)


def _largest_component(mask: np.ndarray) -> np.ndarray | None:
  import cv2
  m = mask.astype(np.uint8)
  if m.sum() == 0:
    return None
  num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
  if num <= 1:
    return None
  areas = stats[1:, cv2.CC_STAT_AREA]
  best = int(np.argmax(areas)) + 1
  return labels == best


def _bg_mask(rgb: np.ndarray,
              prompt: tuple[int, int] | list[tuple[int, int]],
              scene_bg: "SceneBackground") -> np.ndarray | None:
  """Background-subtraction segmentation, anchored at the click pixel.

  Builds a global foreground mask via the scene model's per-pixel
  median/std, then returns the connected component that contains the
  click. Robust against shadows, the arm, and stray foreground blobs
  elsewhere in the frame — the click picks the one we want.
  """
  import cv2
  pts = _prompt_to_array(prompt)
  u0 = int(round(pts[0, 0]))
  v0 = int(round(pts[0, 1]))
  H, W = rgb.shape[:2]
  u0 = max(0, min(W - 1, u0))
  v0 = max(0, min(H - 1, v0))

  fg = scene_bg.foreground_mask(rgb)
  if not fg.any():
    return None

  # Connected components on the foreground mask; pick the one the click
  # lands in. If the click misses, fall back to the largest component
  # within a small radius — typical when the click clipped the edge.
  m = fg.astype(np.uint8)
  num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
  if num <= 1:
    return None

  chosen = int(labels[v0, u0])
  if chosen == 0:
    # Search a small window around the click for the nearest foreground.
    R = 30
    y0 = max(0, v0 - R); y1 = min(H, v0 + R + 1)
    x0 = max(0, u0 - R); x1 = min(W, u0 + R + 1)
    window = labels[y0:y1, x0:x1]
    cand = window[window != 0]
    if cand.size == 0:
      return None
    chosen = int(np.bincount(cand).argmax())
  if chosen == 0:
    return None
  return labels == chosen


def _fallback_mask(rgb: np.ndarray, prompt: tuple[int, int] | list[tuple[int, int]]
                   ) -> np.ndarray | None:
  """Color-similarity floodfill from the prompt click.

  Anchors the segmentation at the click pixel, then floodfills outward
  while pixel intensity stays within a tolerance of the click. This
  contains the segmentation to a single contiguous region of similar
  brightness, which is much more robust than a global threshold against
  scenes with multiple bright distractors (mat fiducials, walls).

  Caveat: still a heuristic. If the click is on a near-saturated patch
  the tolerance lets in too much; if it's on a shadowed edge it lets in
  too little. SAM2 is the right tool for production.
  """
  import cv2
  pts = _prompt_to_array(prompt)
  u0 = int(round(pts[0, 0]))
  v0 = int(round(pts[0, 1]))
  u0 = max(0, min(rgb.shape[1] - 1, u0))
  v0 = max(0, min(rgb.shape[0] - 1, v0))

  gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
  seed = int(gray[v0, u0])

  # Floodfill with a per-channel tolerance. cv2.floodFill mutates the
  # input mask in-place; ours is (H+2, W+2) per its API.
  H, W = gray.shape
  ff_mask = np.zeros((H + 2, W + 2), dtype=np.uint8)
  img = gray.copy()
  flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
  lo_diff = 30   # how much darker than the seed a pixel can be and still join
  hi_diff = 30   # how much brighter
  cv2.floodFill(img, ff_mask, (u0, v0), 255,
                loDiff=(lo_diff,), upDiff=(hi_diff,), flags=flags)
  mask = ff_mask[1:-1, 1:-1].astype(bool)

  if not mask.any():
    return None

  # Tiny morphological closing fills 1-px gaps from camera noise + JPEG.
  m = mask.astype(np.uint8)
  k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
  return m.astype(bool)
