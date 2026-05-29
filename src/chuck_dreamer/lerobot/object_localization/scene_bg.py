"""Per-pixel background model built from the empty-mat episode.

The mat is static across episodes within one recording session. Episode 0
is the empty mat (no object), so the per-pixel median + std across a
sample of those frames is a tight reference for "what the scene looks
like without the object". Subtracting this from a frame from episodes
2/3/4 isolates the object far more reliably than any color heuristic.

Cached at ``<cache>/<slug>/scene_bg.npz`` so the ~10s sampling cost is
paid once per dataset.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import episode_bounds_from_meta, get_frame
from .types import dataset_cache_dir

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


def build_scene_bg(
  dataset_id: str, camera_key: str, episode_idx: int,
  n_samples: int = 16,
) -> SceneBackground:
  """Sample ``n_samples`` frames evenly across ``episode_idx`` and return
  the per-pixel median + std. Builds its own ``LeRobotDataset`` (slow on
  first call due to PyAV index).
  """
  from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

  t0 = time.perf_counter()
  fr, to = episode_bounds_from_meta(dataset_id, episode_idx)
  if to <= fr:
    raise ValueError(f"episode {episode_idx} of {dataset_id} is empty.")
  length = to - fr
  n = min(n_samples, length)
  indices = sorted({int(x) for x in np.linspace(fr, to - 1, num=n)})
  logger.info("[bg] %s: sampling %d frames from episode %d [%d, %d) ...",
              dataset_id, len(indices), episode_idx, fr, to)

  ds = LeRobotDataset(dataset_id)
  stack: list[np.ndarray] = []
  for i, idx in enumerate(indices):
    t = time.perf_counter()
    frame = get_frame(ds, idx, camera_key)
    stack.append(frame)
    logger.info("[bg]   frame %d/%d (idx=%d) in %.2fs",
                i + 1, len(indices), idx, time.perf_counter() - t)

  arr = np.stack(stack, axis=0)
  median = np.median(arr, axis=0).astype(np.uint8)
  std    = arr.astype(np.float32).std(axis=0)
  logger.info("[bg] %s: scene bg built in %.1fs (%d frames, shape=%s)",
              dataset_id, time.perf_counter() - t0, len(stack), median.shape)
  return SceneBackground(median=median, std=std, n_frames=len(stack))


def scene_bg_path(cache_dir: Path | str, dataset_id: str) -> Path:
  return dataset_cache_dir(cache_dir, dataset_id) / "scene_bg.npz"


def save_scene_bg(cache_dir: Path | str, dataset_id: str,
                  bg: SceneBackground) -> Path:
  p = scene_bg_path(cache_dir, dataset_id)
  p.parent.mkdir(parents=True, exist_ok=True)
  payload: dict[str, np.ndarray] = {
    "median":   bg.median,
    "std":      bg.std,
    "n_frames": np.asarray(bg.n_frames),
  }
  if bg.mat_mask is not None:
    payload["mat_mask"] = bg.mat_mask.astype(bool)
  # numpy stub models savez_compressed's **kwds poorly (collides with allow_pickle: bool).
  np.savez_compressed(p, **payload)  # type: ignore[arg-type]
  return p


def load_scene_bg(cache_dir: Path | str, dataset_id: str
                  ) -> SceneBackground | None:
  p = scene_bg_path(cache_dir, dataset_id)
  if not p.exists():
    return None
  with np.load(p) as f:
    mat_mask = np.asarray(f["mat_mask"], dtype=bool) if "mat_mask" in f else None
    return SceneBackground(
      median   = np.asarray(f["median"], dtype=np.uint8),
      std      = np.asarray(f["std"], dtype=np.float32),
      n_frames = int(f["n_frames"]),
      mat_mask = mat_mask,
    )


def ensure_scene_bg(
  cache_dir: Path | str, dataset_id: str, camera_key: str,
  episode_idx: int, n_samples: int = 16, rebuild: bool = False,
  calibration: "Any | None" = None,
  ol_cfg: "Any | None" = None,
  rebuild_mat_mask: bool = False,
) -> SceneBackground:
  """Load cached scene_bg, or build + save it. Idempotent.

  When ``calibration`` and ``ol_cfg`` are provided, attach a mat-region
  mask projected from the configured mat_extent_* bounds. Pass
  ``rebuild_mat_mask=True`` to force recomputation of an existing
  cached mat_mask (use this once after changing the extent values).
  """
  bg: SceneBackground | None = None
  if not rebuild:
    bg = load_scene_bg(cache_dir, dataset_id)
  if bg is None:
    bg = build_scene_bg(dataset_id, camera_key, episode_idx, n_samples)
  needs_mat = (bg.mat_mask is None) or rebuild_mat_mask
  if needs_mat and calibration is not None and ol_cfg is not None:
    try:
      mat_mask = build_mat_mask(
        image_size = bg.median.shape[1::-1],   # (W, H) from (H, W, 3)
        calibration = calibration,
        extent_xmin_mm = ol_cfg.mat_extent_xmin_mm,
        extent_xmax_mm = ol_cfg.mat_extent_xmax_mm,
        extent_ymin_mm = ol_cfg.mat_extent_ymin_mm,
        extent_ymax_mm = ol_cfg.mat_extent_ymax_mm,
      )
      bg.mat_mask = mat_mask
      logger.info("[bg] %s: attached mat_mask from world bbox "
                  "x=[%.0f, %.0f] y=[%.0f, %.0f] mm "
                  "(%d/%d pixels in ROI = %.1f%%)",
                  dataset_id,
                  ol_cfg.mat_extent_xmin_mm, ol_cfg.mat_extent_xmax_mm,
                  ol_cfg.mat_extent_ymin_mm, ol_cfg.mat_extent_ymax_mm,
                  int(mat_mask.sum()), mat_mask.size,
                  100.0 * mat_mask.sum() / mat_mask.size)
    except Exception as e:
      logger.warning("[bg] %s: could not build mat_mask (%s); proceeding "
                     "without ROI.", dataset_id, e)
  save_scene_bg(cache_dir, dataset_id, bg)
  return bg


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
