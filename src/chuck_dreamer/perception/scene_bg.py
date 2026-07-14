"""Per-pixel scene background model — the pure image-space half.

:class:`SceneBackground` is the reference "scene without the object" used
for background-subtraction foreground detection, plus the mat-region ROI
helpers. Building the model from a LeRobot dataset and caching it live in
``chuck_dreamer.lerobot.annotation.scene_bg`` — this module never touches a
dataset.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SceneBackground:
  """Per-pixel reference scene built from the empty episode.

  ``mat_mask`` (optional) is a binary image-space ROI restricting where
  the foreground may live — typically the projection of the painted
  mat polygon. The white wall/table behind the mat differs from the
  reference too (lighting drift, etc.) but lies outside the mat ROI,
  so the foreground mask correctly excludes it. Compute it once from
  the camera extrinsics + mat geometry and attach it to the cached
  scene_bg; see ``build_mat_mask``.
  """
  median: np.ndarray   # (H, W, 3) uint8 — per-pixel median across samples
  std:    np.ndarray   # (H, W, 3) float32 — per-pixel std across samples
  n_frames: int
  mat_mask: np.ndarray | None = None   # (H, W) bool, optional ROI

  def foreground_mask(self, rgb: np.ndarray,
                       k: float = 3.0, floor: float = 15.0,
                       open_radius: int = 1) -> np.ndarray:
    """Binary mask where ``rgb`` differs from the reference (and lies
    inside ``mat_mask`` if one is attached).

    ``threshold = max(k*std, floor)`` per pixel + channel. A pixel is
    foreground iff the max-channel difference exceeds the threshold.
    A small morphological open removes 1-2 px speckle from sensor noise.
    """
    import cv2
    diff = np.abs(rgb.astype(np.float32) - self.median.astype(np.float32))
    thr  = np.maximum(self.std * k, floor)
    fg_per_channel = diff > thr
    fg = fg_per_channel.max(axis=-1)
    if open_radius > 0:
      m = fg.astype(np.uint8)
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (2 * open_radius + 1, 2 * open_radius + 1))
      m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
      m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
      fg = m.astype(bool)
    if self.mat_mask is not None:
      fg = fg & self.mat_mask
    return np.asarray(fg)


def build_mat_mask(
  image_size: tuple[int, int],
  calibration: "Any",
  extent_xmin_mm: float, extent_xmax_mm: float,
  extent_ymin_mm: float, extent_ymax_mm: float,
) -> np.ndarray:
  """Compute a binary image mask of the physical mat region.

  Projects the four corners of the world-frame mat rectangle into
  image space using ``calibration``'s intrinsics + extrinsics, then
  fills the convex hull. The world bbox should describe the physical
  mat (the entire dark region the object can travel on), not just
  the painted fiducial pattern.
  """
  import cv2

  W, H = image_size
  K    = np.asarray(calibration.intrinsics.K,    dtype=np.float64)
  dist = np.asarray(calibration.intrinsics.dist, dtype=np.float64)
  R    = np.asarray(calibration.extrinsics.R,    dtype=np.float64)
  t    = np.asarray(calibration.extrinsics.t,    dtype=np.float64).reshape(3)
  rvec, _ = cv2.Rodrigues(R)

  corners_world = np.array([
    [extent_xmin_mm, extent_ymin_mm, 0.0],
    [extent_xmin_mm, extent_ymax_mm, 0.0],
    [extent_xmax_mm, extent_ymax_mm, 0.0],
    [extent_xmax_mm, extent_ymin_mm, 0.0],
  ], dtype=np.float64)
  uv, _ = cv2.projectPoints(corners_world, rvec, t, K, dist)
  pts: np.ndarray = uv.reshape(-1, 2)
  # Clip projected corners to the image bounds. Wide-angle lens
  # distortion can push corner points to inf when they fall outside
  # the camera FOV; the convex hull then becomes invalid.
  pts = np.clip(pts, [-1e6, -1e6], [W + 1e6, H + 1e6])
  pts = np.nan_to_num(pts, nan=0.0, posinf=1e6, neginf=-1e6)
  pts = np.clip(pts, [-W, -H], [2 * W, 2 * H])
  hull = cv2.convexHull(pts.astype(np.float32))
  mask = np.zeros((H, W), dtype=np.uint8)
  cv2.fillConvexPoly(mask, hull.astype(np.int32), 1)
  return mask.astype(bool)


def detect_mat_mask(median_rgb: np.ndarray,
                     erode_px: int = 4, dilate_px: int = 4,
                     min_area_frac: float = 0.05) -> np.ndarray:
  """Detect the physical mat region (large dark area) from the empty-
  mat median frame.

  Steps:
    1. Convert median to grayscale.
    2. Otsu-threshold to split dark vs bright.
    3. Take the *darker* class as the mat candidate.
    4. Open + close to smooth, then keep only connected components
       larger than ``min_area_frac`` of the image.
    5. Fill the convex hull of those components.

  Falls back to an all-True mask (no ROI restriction) if no dark
  region passes the area threshold. Returns ``(H, W) bool``.
  """
  import cv2

  H, W = median_rgb.shape[:2]
  gray = cv2.cvtColor(median_rgb, cv2.COLOR_RGB2GRAY)
  # Otsu picks the split that maximises between-class variance.
  _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  # We want the darker class to be foreground (= mat). Otsu marks
  # bright pixels as 255 so invert.
  dark = (binary == 0).astype(np.uint8)

  # Smooth: close (fill small fiducial gaps inside the mat) then
  # open (drop carpet edges + thin dark strips).
  k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * dilate_px + 1, 2 * dilate_px + 1))
  k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * erode_px + 1, 2 * erode_px + 1))
  dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k_close)
  dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,  k_open)

  num, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
  if num <= 1:
    logger.warning("detect_mat_mask: no dark component found; falling back to "
                   "all-True mask.")
    return np.ones((H, W), dtype=bool)

  min_area = float(min_area_frac) * float(H * W)
  big = np.zeros((H, W), dtype=np.uint8)
  found = 0
  for i in range(1, num):
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
      big |= (labels == i).astype(np.uint8)
      found += 1
  if found == 0:
    logger.warning(
      "detect_mat_mask: largest dark component (%d px) below %d-px threshold "
      "(%.0f%% of image); falling back to all-True mask.",
      int(stats[1:, cv2.CC_STAT_AREA].max()), int(min_area), min_area_frac * 100)
    return np.ones((H, W), dtype=bool)

  # Convex hull of the union, to absorb small holes where fiducial
  # paint or the object's shadow sat during the empty-mat episode.
  ys, xs = np.where(big > 0)
  if xs.size == 0:
    return np.ones((H, W), dtype=bool)
  pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
  hull = cv2.convexHull(pts)
  mat = np.zeros((H, W), dtype=np.uint8)
  cv2.fillConvexPoly(mat, hull, 1)
  return mat.astype(bool)
