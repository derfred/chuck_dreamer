"""Per-dataset intrinsic calibration from the checkerboard episode.

Algorithm overview (full spec in ``object_localization_spec.md``,
Deliverable 1):

  1. Sample frames evenly across the checkerboard episode.
  2. Run ``findChessboardCornersSB`` + a sub-pixel refinement.
  3. Refuse to calibrate if spatial coverage is too thin.
  4. ``cv2.calibrateCamera`` on the surviving views.
  5. Write the camera matrix + per-view residuals to disk and render
     verification PNGs so the user can sanity-check.

The verification renders are the single most important artifact for the
user: ``coverage_mosaic.png`` shows whether the checkerboard explored
the whole field of view, which is what actually determines whether the
fit can be trusted away from the center.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..dataset import episode_bounds, get_frame, iter_episode_frame_indices
from ..runtime import ObjectLocalizationConfig
from ..types import Intrinsics, dataset_cache_dir, write_intrinsics

logger = logging.getLogger(__name__)


@dataclass
class DetectedFrame:
  frame_idx: int
  corners: np.ndarray   # (N, 1, 2) float32 image coords


@dataclass
class IntrinsicsResult:
  intrinsics: Intrinsics
  detections: list[DetectedFrame]
  per_view_rms: list[float]
  coverage_filled_cells: int
  coverage_total_cells: int


class IntrinsicsAbort(RuntimeError):
  """Raised when the calibration should not be persisted (coverage / count)."""


def calibrate_intrinsics(
  ds: Any, ol_cfg: ObjectLocalizationConfig,
  cache_dir: Path, dataset_id: str,
  force: bool = False,
) -> IntrinsicsResult:
  """Run the full intrinsics pipeline; raises ``IntrinsicsAbort`` if the
  spatial-coverage or min-frame thresholds aren't met.

  On success, writes ``intrinsics.json`` + verification artifacts under
  ``cache_dir/<slug>/``.
  """
  import cv2

  out_dir = dataset_cache_dir(cache_dir, dataset_id)
  out_dir.mkdir(parents=True, exist_ok=True)
  verify_dir = out_dir / "intrinsics_verify"

  t0 = time.perf_counter()
  logger.info("[%s] locating checkerboard episode (idx=%d) ...",
              dataset_id, ol_cfg.episode_checkerboard)
  ep_fr, ep_to = episode_bounds(ds, ol_cfg.episode_checkerboard)
  logger.info("[%s]   episode %d spans frames [%d, %d) (length=%d). "
              "Target: %d good detections (min %d).",
              dataset_id, ol_cfg.episode_checkerboard, ep_fr, ep_to, ep_to - ep_fr,
              ol_cfg.intrinsics_n_frames_target, ol_cfg.intrinsics_n_frames_min)

  initial, fallback = iter_episode_frame_indices(
    ds, ol_cfg.episode_checkerboard, ol_cfg.intrinsics_n_frames_target,
  )
  if not initial:
    raise IntrinsicsAbort(
      f"episode {ol_cfg.episode_checkerboard} of {dataset_id} is empty.")

  ck_size = tuple(int(x) for x in ol_cfg.checkerboard_size)
  ck_obj  = _checkerboard_object_points(ck_size, ol_cfg.checkerboard_square_mm)

  logger.info("[%s] starting corner detection on %d initial frames "
              "(fallback pool: %d). Each frame is loaded from the LeRobot "
              "video stream then run through findChessboardCornersSB.",
              dataset_id, len(initial), len(fallback))

  detections: list[DetectedFrame] = []
  obj_points:  list[np.ndarray]   = []
  img_points:  list[np.ndarray]   = []
  used_indices: list[int]         = []

  # Track frame W/H once we see the first frame.
  image_size: tuple[int, int] | None = None

  # Centroid-bucket dedupe: when the checkerboard is stationary across
  # many frames, near-duplicate views inflate the view count without
  # adding information (and bias the calibrate-camera fit). We bucket
  # the checkerboard centroid into a coverage-grid-sized image grid and
  # only keep the first HIT per bucket. The bucket grid is intentionally
  # tied to coverage_grid so "covered" and "deduped" use the same
  # spatial resolution.
  bucket_grid = ol_cfg.intrinsics_coverage_grid
  occupied_buckets: set[tuple[int, int]] = set()

  pending = list(initial)
  retry_pool = list(fallback)
  attempts = 0
  hits = 0
  misses = 0
  dupes = 0
  last_log = time.perf_counter()
  while pending and len(detections) < ol_cfg.intrinsics_n_frames_target:
    idx = pending.pop(0)
    attempts += 1
    t_load = time.perf_counter()
    frame = get_frame(ds, idx, ol_cfg.camera_key)
    t_loaded = time.perf_counter()
    if image_size is None:
      image_size = (frame.shape[1], frame.shape[0])
      logger.info("[%s]   first frame loaded in %.2fs; image_size=%dx%d",
                  dataset_id, t_loaded - t_load, image_size[0], image_size[1])
    corners = _detect_corners(frame, ck_size)
    t_det = time.perf_counter()
    if corners is None:
      misses += 1
      verdict = "MISS"
      if retry_pool:
        pending.append(retry_pool.pop(0))
    else:
      bucket = _centroid_bucket(corners, image_size, bucket_grid)
      if bucket in occupied_buckets:
        dupes += 1
        verdict = f"DUP@{bucket}"
        if retry_pool:
          pending.append(retry_pool.pop(0))
      else:
        hits += 1
        verdict = f"HIT@{bucket}"
        occupied_buckets.add(bucket)
        detections.append(DetectedFrame(frame_idx=idx, corners=corners))
        obj_points.append(ck_obj.copy())
        img_points.append(corners)
        used_indices.append(idx)

    now = time.perf_counter()
    if (now - last_log) > 5.0 or attempts <= 3 or attempts % 10 == 0:
      logger.info(
        "[%s]   frame %d: %s  load=%.2fs detect=%.2fs  "
        "hits=%d misses=%d dupes=%d  pending=%d retry_pool=%d  elapsed=%.1fs",
        dataset_id, idx, verdict,
        t_loaded - t_load, t_det - t_loaded,
        hits, misses, dupes, len(pending), len(retry_pool), now - t0,
      )
      last_log = now

  logger.info(
    "[%s] corner-detection loop done in %.1fs: %d hits / %d misses / %d dupes "
    "(%d frames attempted; %d in fallback pool unused; %d distinct buckets occupied).",
    dataset_id, time.perf_counter() - t0, hits, misses, dupes,
    attempts, len(retry_pool), len(occupied_buckets),
  )

  if image_size is None:
    raise IntrinsicsAbort(f"could not load any frame from {dataset_id}.")

  if len(detections) < ol_cfg.intrinsics_n_frames_min:
    _render_coverage(detections, ds, ol_cfg, verify_dir, image_size)
    raise IntrinsicsAbort(
      f"only {len(detections)} usable detections (need >= "
      f"{ol_cfg.intrinsics_n_frames_min}). Coverage written to {verify_dir}.")

  filled, total = _coverage_fill(detections, image_size, ol_cfg.intrinsics_coverage_grid)
  logger.info("[%s] spatial coverage: %d / %d cells (%.0f%%).",
              dataset_id, filled, total, 100.0 * filled / max(1, total))
  if filled / max(1, total) < 0.8:
    _render_coverage(detections, ds, ol_cfg, verify_dir, image_size)
    raise IntrinsicsAbort(
      f"checkerboard covers only {filled}/{total} grid cells (<80%). "
      f"Re-record episode {ol_cfg.episode_checkerboard} with the board "
      f"swept through the full field of view. Mosaic at {verify_dir}.")

  logger.info("[%s] running cv2.calibrateCamera on %d views ...",
              dataset_id, len(detections))
  t_cal = time.perf_counter()
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
  rms_px, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, image_size, None, None, criteria=criteria,
  )
  logger.info("[%s]   calibrateCamera done in %.1fs (rms=%.3fpx).",
              dataset_id, time.perf_counter() - t_cal, float(rms_px))
  K    = np.asarray(K, dtype=np.float64)
  dist = np.asarray(dist, dtype=np.float64).reshape(-1)
  if dist.size < 5:
    dist = np.concatenate([dist, np.zeros(5 - dist.size, dtype=np.float64)])
  else:
    dist = dist[:5]

  per_view_rms = _per_view_rms(obj_points, img_points, rvecs, tvecs, K, dist)

  intrinsics = Intrinsics(
    image_size   = image_size,
    K            = K,
    dist         = dist,
    rms_px       = float(rms_px),
    n_frames_used = len(detections),
  )

  write_intrinsics(cache_dir, dataset_id, intrinsics, extra={
    "frame_indices":   used_indices,
    "per_view_rms_px": [float(x) for x in per_view_rms],
    "rms_threshold_px": float(ol_cfg.intrinsics_rms_threshold_px),
  })

  logger.info("[%s] rendering verification artifacts to %s ...",
              dataset_id, verify_dir)
  t_render = time.perf_counter()
  _render_verification(
    detections, ds, ol_cfg, verify_dir, image_size,
    K=K, dist=dist, rvecs=rvecs, tvecs=tvecs,
    per_view_rms=per_view_rms, ck_obj=ck_obj,
  )
  logger.info("[%s]   verification render done in %.1fs (total elapsed %.1fs).",
              dataset_id, time.perf_counter() - t_render, time.perf_counter() - t0)

  return IntrinsicsResult(
    intrinsics            = intrinsics,
    detections            = detections,
    per_view_rms          = list(per_view_rms),
    coverage_filled_cells = filled,
    coverage_total_cells  = total,
  )


def _checkerboard_object_points(size: tuple[int, int], square_mm: float) -> np.ndarray:
  cols, rows = size
  objp = np.zeros((cols * rows, 3), dtype=np.float32)
  grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
  objp[:, :2] = grid * float(square_mm)
  return objp


def _detect_corners(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
  import cv2
  gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
  flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
  found, corners = cv2.findChessboardCornersSB(gray, size, flags=flags)
  if not found or corners is None:
    return None
  refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
  corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (11, 11), (-1, -1), refine_criteria)
  return corners


def _centroid_bucket(corners: np.ndarray, image_size: tuple[int, int],
                     grid: tuple[int, int]) -> tuple[int, int]:
  """Quantize the checkerboard centroid into a ``grid``-cell bucket."""
  pts = corners.reshape(-1, 2)
  cu = float(pts[:, 0].mean())
  cv = float(pts[:, 1].mean())
  W, H = image_size
  gx, gy = max(1, grid[0]), max(1, grid[1])
  bx = min(gx - 1, max(0, int(cu / (W / gx))))
  by = min(gy - 1, max(0, int(cv / (H / gy))))
  return (bx, by)


def _coverage_fill(detections: list[DetectedFrame], image_size: tuple[int, int],
                   grid: tuple[int, int]) -> tuple[int, int]:
  W, H = image_size
  gx, gy = max(1, grid[0]), max(1, grid[1])
  cell_w, cell_h = W / gx, H / gy
  filled = np.zeros((gy, gx), dtype=bool)
  for d in detections:
    pts = d.corners.reshape(-1, 2)
    cx = np.clip((pts[:, 0] / cell_w).astype(int), 0, gx - 1)
    cy = np.clip((pts[:, 1] / cell_h).astype(int), 0, gy - 1)
    filled[cy, cx] = True
  return int(filled.sum()), int(gx * gy)


def _per_view_rms(obj_points: list[np.ndarray], img_points: list[np.ndarray],
                  rvecs: list[np.ndarray], tvecs: list[np.ndarray],
                  K: np.ndarray, dist: np.ndarray) -> list[float]:
  import cv2
  out: list[float] = []
  for op, ip, rv, tv in zip(obj_points, img_points, rvecs, tvecs):
    proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
    diff = (proj.reshape(-1, 2) - ip.reshape(-1, 2))
    rms = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    out.append(rms)
  return out


def _representative_frame(ds: Any, ol_cfg: ObjectLocalizationConfig) -> np.ndarray | None:
  try:
    from ..dataset import episode_bounds
    fr, _ = episode_bounds(ds, ol_cfg.episode_empty)
    return get_frame(ds, fr, ol_cfg.camera_key)
  except Exception:
    return None


def _render_coverage(detections: list[DetectedFrame], ds: Any,
                     ol_cfg: ObjectLocalizationConfig, out_dir: Path,
                     image_size: tuple[int, int]) -> None:
  import cv2
  out_dir.mkdir(parents=True, exist_ok=True)
  W, H = image_size
  base = _representative_frame(ds, ol_cfg)
  if base is None:
    base = np.full((H, W, 3), 32, dtype=np.uint8)
  canvas = cv2.cvtColor(base.copy(), cv2.COLOR_RGB2BGR)

  gx, gy = ol_cfg.intrinsics_coverage_grid
  cell_w, cell_h = W / gx, H / gy
  filled = np.zeros((gy, gx), dtype=bool)
  for d in detections:
    pts = d.corners.reshape(-1, 2)
    cx = np.clip((pts[:, 0] / cell_w).astype(int), 0, gx - 1)
    cy = np.clip((pts[:, 1] / cell_h).astype(int), 0, gy - 1)
    filled[cy, cx] = True

  overlay = canvas.copy()
  for j in range(gy):
    for i in range(gx):
      x0, y0 = int(i * cell_w), int(j * cell_h)
      x1, y1 = int((i + 1) * cell_w), int((j + 1) * cell_h)
      if not filled[j, i]:
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 200), -1)
  canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)

  for j in range(gy + 1):
    y = int(j * cell_h)
    cv2.line(canvas, (0, y), (W, y), (140, 140, 140), 1)
  for i in range(gx + 1):
    x = int(i * cell_w)
    cv2.line(canvas, (x, 0), (x, H), (140, 140, 140), 1)

  for k, d in enumerate(detections):
    hue = int(180 * (k / max(1, len(detections))))
    color_bgr = tuple(int(v) for v in cv2.cvtColor(
      np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
    for u, v in d.corners.reshape(-1, 2):
      cv2.circle(canvas, (int(u), int(v)), 2, color_bgr, -1)

  cv2.imwrite(str(out_dir / "coverage_mosaic.png"), canvas)


def _render_verification(
  detections: list[DetectedFrame], ds: Any, ol_cfg: ObjectLocalizationConfig,
  out_dir: Path, image_size: tuple[int, int],
  K: np.ndarray, dist: np.ndarray, rvecs: list[np.ndarray], tvecs: list[np.ndarray],
  per_view_rms: list[float], ck_obj: np.ndarray,
) -> None:
  import cv2
  out_dir.mkdir(parents=True, exist_ok=True)
  _render_coverage(detections, ds, ol_cfg, out_dir, image_size)

  per_frame_dir = out_dir / "per_frame"
  per_frame_dir.mkdir(parents=True, exist_ok=True)

  if not detections:
    return

  n_render = min(10, len(detections))
  pick = np.linspace(0, len(detections) - 1, num=n_render, dtype=int)
  for i in pick:
    d = detections[int(i)]
    frame = get_frame(ds, d.frame_idx, ol_cfg.camera_key)
    canvas = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
    proj, _ = cv2.projectPoints(ck_obj, rvecs[int(i)], tvecs[int(i)], K, dist)
    obs = d.corners.reshape(-1, 2)
    rep = proj.reshape(-1, 2)
    for o in obs:
      cv2.circle(canvas, (int(o[0]), int(o[1])), 4, (0, 200, 0), -1)
    for r in rep:
      cv2.drawMarker(canvas, (int(r[0]), int(r[1])), (0, 0, 255),
                     cv2.MARKER_CROSS, 8, 1)
    text = f"frame={d.frame_idx}  rms={per_view_rms[int(i)]:.3f}px"
    cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(per_frame_dir / f"{d.frame_idx:05d}.png"), canvas)
