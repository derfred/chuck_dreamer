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
from typing import Iterable, Optional

import numpy as np

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
) -> np.ndarray | None:
  """Single-frame segmentation. Returns a ``HxW bool`` mask or ``None``."""
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
  return _fallback_mask(rgb, prompt)


def segment_video(
  frames: list[np.ndarray],
  first_frame_prompt: tuple[int, int] | list[tuple[int, int]],
  use_sam2: bool, checkpoint: str, device: str,
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
    out.append(_fallback_mask(f, first_frame_prompt))
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


def _fallback_mask(rgb: np.ndarray, prompt: tuple[int, int] | list[tuple[int, int]]
                   ) -> np.ndarray | None:
  """Brightness-threshold fallback. Picks the connected component nearest
  the first prompt point. Works on the dark-mat / light-object rig but
  is intentionally crude — the spec calls for SAM2 in production.
  """
  import cv2
  pts = _prompt_to_array(prompt)
  hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
  v = hsv[..., 2]
  mask = v > max(80, int(np.percentile(v, 70)))
  m = mask.astype(np.uint8)
  num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
  if num <= 1:
    return None
  u0, v0 = int(round(pts[0, 0])), int(round(pts[0, 1]))
  u0 = max(0, min(rgb.shape[1] - 1, u0))
  v0 = max(0, min(rgb.shape[0] - 1, v0))
  chosen = int(labels[v0, u0])
  if chosen == 0:
    areas = stats[1:, cv2.CC_STAT_AREA]
    chosen = int(np.argmax(areas)) + 1
  return labels == chosen
