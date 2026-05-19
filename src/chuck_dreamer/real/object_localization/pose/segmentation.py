"""Per-frame object segmentation.

Uses SAM2 when available (best). Falls back to a contour-based heuristic
that masks bright/colored connected components inside a mat ROI. The
fallback is intended to keep the pipeline runnable end-to-end without
the heavy SAM2 install.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SegmentationPrompt:
  points: list[tuple[int, int]] | None = None
  bbox:   tuple[int, int, int, int] | None = None   # (x0, y0, x1, y1)
  mask:   np.ndarray | None = None


def _sam2_available() -> bool:
  try:
    import sam2  # type: ignore[import-not-found]  # noqa: F401
    return True
  except Exception:
    return False


class Sam2ImageSegmenter:
  """Thin wrapper over SAM2's image predictor."""

  def __init__(self, model_id: str = "facebook/sam2-hiera-large", device: str = "cuda"):
    from sam2.sam2_image_predictor import SAM2ImagePredictor   # type: ignore[import-not-found]
    self._predictor = SAM2ImagePredictor.from_pretrained(model_id)
    self._device    = device

  def predict(self, image: np.ndarray, prompt: SegmentationPrompt) -> np.ndarray:
    import torch
    self._predictor.set_image(image)
    point_coords = None
    point_labels = None
    box          = None
    mask_input   = None
    if prompt.points:
      point_coords = np.asarray(prompt.points, dtype=np.float32)
      point_labels = np.ones((len(prompt.points),), dtype=np.int32)
    if prompt.bbox is not None:
      box = np.asarray(prompt.bbox, dtype=np.float32)
    if prompt.mask is not None:
      mask_input = prompt.mask.astype(np.float32)[None]
    with torch.inference_mode():
      masks, scores, _ = self._predictor.predict(
        point_coords=point_coords, point_labels=point_labels,
        box=box, mask_input=mask_input, multimask_output=True,
      )
    best = int(np.argmax(scores))
    return np.asarray(masks[best].astype(bool))


class Sam2VideoSegmenter:
  """Wrapper over SAM2's video predictor for episode-wide propagation."""

  def __init__(self, model_id: str = "facebook/sam2-hiera-large", device: str = "cuda"):
    from sam2.sam2_video_predictor import SAM2VideoPredictor  # type: ignore[import-not-found]
    self._predictor = SAM2VideoPredictor.from_pretrained(model_id)
    self._device    = device

  def propagate(
    self,
    frames: list[np.ndarray],
    first_prompt: SegmentationPrompt,
  ) -> list[np.ndarray | None]:
    import torch
    with torch.inference_mode():
      state = self._predictor.init_state(video_frames=frames)
      if first_prompt.points:
        pts = np.asarray(first_prompt.points, dtype=np.float32)
        lbl = np.ones((len(first_prompt.points),), dtype=np.int32)
        self._predictor.add_new_points(state, frame_idx=0, obj_id=1,
                                       points=pts, labels=lbl)
      if first_prompt.bbox is not None:
        self._predictor.add_new_points_or_box(
          state, frame_idx=0, obj_id=1,
          box=np.asarray(first_prompt.bbox, dtype=np.float32))
      out: list[np.ndarray | None] = [None] * len(frames)
      for fid, _obj_ids, mask_logits in self._predictor.propagate_in_video(state):
        m = (mask_logits[0] > 0).cpu().numpy().astype(bool)
        if m.ndim == 3:
          m = m[0]
        out[fid] = m
    return out


def heuristic_mask(
  rgb: np.ndarray,
  mat_roi_mask: np.ndarray | None = None,
) -> np.ndarray | None:
  """Color-based object-mask heuristic.

  Picks the largest connected component inside ``mat_roi_mask`` whose
  mean color is far from the mat's median color. Returns ``None`` if no
  reasonable mask is found.
  """
  import cv2

  H, W = rgb.shape[:2]
  if mat_roi_mask is None:
    mat_roi_mask = np.ones((H, W), dtype=bool)

  hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
  mat_med = np.median(hsv[mat_roi_mask], axis=0).astype(np.float32) if mat_roi_mask.any() else np.array([0, 0, 128.0], dtype=np.float32)
  diff = np.linalg.norm(hsv.astype(np.float32) - mat_med[None, None, :], axis=-1)
  candidate = (diff > 40) & mat_roi_mask
  candidate = cv2.morphologyEx(candidate.astype(np.uint8) * 255,
                               cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
  num, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
  if num <= 1:
    return None
  # Skip the background (label 0); pick the largest remaining component
  areas = stats[1:, cv2.CC_STAT_AREA]
  if not areas.size:
    return None
  best = 1 + int(np.argmax(areas))
  mask = labels == best
  if mask.sum() < 200:
    return None
  return np.asarray(mask)


def make_segmenter(device: str = "cuda") -> "Sam2ImageSegmenter | None":
  """Try to construct a SAM2 image segmenter; return None if unavailable."""
  if not _sam2_available():
    logger.info("SAM2 not installed — segmentation falls back to color heuristic.")
    return None
  try:
    return Sam2ImageSegmenter(device=device)
  except Exception as e:
    logger.warning("SAM2 init failed (%s) — falling back to heuristic", e)
    return None
