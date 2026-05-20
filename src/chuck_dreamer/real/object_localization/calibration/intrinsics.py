"""Checkerboard intrinsic calibration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .. import constants

logger = logging.getLogger(__name__)


@dataclass
class IntrinsicResult:
  K:                np.ndarray             # 3x3
  dist:             np.ndarray             # (5,)
  rms_px:           float
  image_size:       tuple[int, int]        # (W, H)
  used_corners:     list[np.ndarray]       # for debugging
  used_frame_idxs:  list[int]              # indices into the input list


def _world_grid() -> np.ndarray:
  """Object points for one board view, shape (N, 3) in millimeters."""
  cols, rows = constants.checkerboard_inner_corners()
  obj = np.zeros((rows * cols, 3), dtype=np.float32)
  obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
  obj *= constants.checkerboard_square_mm()
  return obj


def _detect_corners(gray: np.ndarray) -> np.ndarray | None:
  import cv2
  cols, rows = constants.checkerboard_inner_corners()
  ok, corners = cv2.findChessboardCornersSB(
    gray, (cols, rows),
    flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
  )
  if ok:
    return corners
  ok, corners = cv2.findChessboardCorners(
    gray, (cols, rows),
    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
  )
  if not ok:
    return None
  term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
  corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
  return corners


def calibrate_intrinsics(frames: list[np.ndarray]) -> IntrinsicResult:
  """Run :func:`cv2.calibrateCamera` over checkerboard detections.

  ``frames`` is a list of HxWx3 uint8 RGB images. The returned result
  carries the detected corners and the input indices that survived
  detection + filtering, so callers can save them for debugging.
  """
  import cv2

  if not frames:
    raise ValueError("calibrate_intrinsics: empty frame list")

  H, W = frames[0].shape[:2]
  obj_template = _world_grid()

  obj_pts:    list[np.ndarray] = []
  img_pts:    list[np.ndarray] = []
  used_idxs:  list[int] = []
  centers:    list[tuple[float, float]] = []

  for i, frame in enumerate(frames):
    if frame.shape[:2] != (H, W):
      logger.warning("frame %d: size mismatch (%s vs %s), skipping",
                     i, frame.shape[:2], (H, W))
      continue
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corners = _detect_corners(gray)
    if corners is None:
      continue
    pts = corners.reshape(-1, 2)
    span_x = (pts[:, 0].max() - pts[:, 0].min()) / W
    span_y = (pts[:, 1].max() - pts[:, 1].min()) / H
    if span_x < 0.15 or span_y < 0.15:
      logger.debug("frame %d: rejected (span %.2f x %.2f)", i, span_x, span_y)
      continue
    obj_pts.append(obj_template.copy())
    img_pts.append(corners.astype(np.float32))
    used_idxs.append(i)
    centers.append((float(pts[:, 0].mean()), float(pts[:, 1].mean())))

  if len(obj_pts) < 8:
    raise RuntimeError(
      f"only {len(obj_pts)} usable checkerboard detections (need >=8). "
      f"Either the checkerboard episode is too short or detection is failing.")

  if centers:
    cx = np.array([c[0] for c in centers]) / W
    cy = np.array([c[1] for c in centers]) / H
    span_x = float(cx.max() - cx.min())
    span_y = float(cy.max() - cy.min())
    if span_x < 0.5 or span_y < 0.5:
      logger.warning("checkerboard centers span only %.2f x %.2f of the image — "
                     "intrinsic estimate may be poorly conditioned", span_x, span_y)

  rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
    obj_pts, img_pts, (W, H), None, None,  # type: ignore[call-overload]
    flags=cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3,
  )
  rms = float(rms)
  if rms > 0.5:
    logger.warning("intrinsic RMS %.3f px > 0.5 (target < 0.3)", rms)
  else:
    logger.info("intrinsic RMS %.3f px from %d boards", rms, len(obj_pts))

  dist = np.asarray(dist, dtype=np.float64).reshape(-1)
  # Trim to the standard (k1, k2, p1, p2, k3) slot if cv2 returned more.
  if dist.size >= 5:
    dist = dist[:5]

  return IntrinsicResult(
    K=np.asarray(K, dtype=np.float64),
    dist=dist,
    rms_px=rms,
    image_size=(W, H),
    used_corners=[c.reshape(-1, 2) for c in img_pts],
    used_frame_idxs=used_idxs,
  )
