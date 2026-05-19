"""Extrinsic calibration: solvePnP from mat markings → known 3D points."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..geometry import circle_points_world, line_endpoints_world
from .mat_detection import MatDetection

logger = logging.getLogger(__name__)


@dataclass
class ExtrinsicResult:
  R:       np.ndarray   # 3x3, world -> camera
  t:       np.ndarray   # (3,)
  rms_px:  float
  obj_pts: np.ndarray   # (N, 3) used 3D points
  img_pts: np.ndarray   # (N, 2) used 2D points


def _associate_circle_samples(
  samples_uv: np.ndarray,
  K: np.ndarray, dist: np.ndarray,
  R: np.ndarray, t: np.ndarray,
  n_world: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Match each 2D ellipse sample to the closest projected 3D circle point.

  Returns ``(obj_pts, img_pts)`` arrays usable in PnP refinement.
  """
  import cv2

  world = circle_points_world(n_world)
  rvec, _ = cv2.Rodrigues(R)
  proj, _ = cv2.projectPoints(world.reshape(-1, 1, 3), rvec, t.reshape(3, 1), K, dist)
  proj = proj.reshape(-1, 2)

  # Greedy nearest matching: for each image sample take the closest projected
  # world point (with replacement). Good enough — both sets are well-spread.
  obj_out: list[np.ndarray] = []
  img_out: list[np.ndarray] = []
  for s in samples_uv:
    d = np.linalg.norm(proj - s, axis=1)
    j = int(np.argmin(d))
    obj_out.append(world[j])
    img_out.append(s)
  return np.asarray(obj_out, dtype=np.float64), np.asarray(img_out, dtype=np.float64)


def solve_extrinsics(
  detection:  MatDetection,
  K:          np.ndarray,
  dist:       np.ndarray,
) -> ExtrinsicResult:
  """Solve PnP for the world frame given a single mat detection.

  Pipeline:
    1. SQPNP on the 4 line endpoints + the ellipse center (5 correspondences).
    2. Associate the N circle samples to projected world circle points.
    3. Refine with LM over all correspondences (line endpoints + circle).
  """
  import cv2

  eps = line_endpoints_world()
  initial_obj = np.stack([
    eps["line1_near"],
    eps["line1_far"],
    eps["line2_near"],
    eps["line2_far"],
    np.array([0.0, 0.0, 0.0]),
  ], axis=0).astype(np.float64)
  initial_img = np.stack([
    detection.line1_near,
    detection.line1_far,
    detection.line2_near,
    detection.line2_far,
    detection.circle_center,
  ], axis=0).astype(np.float64)

  ok, rvec, tvec = cv2.solvePnP(
    initial_obj.reshape(-1, 1, 3),
    initial_img.reshape(-1, 1, 2),
    K, dist,
    flags=cv2.SOLVEPNP_SQPNP,
  )
  if not ok:
    raise RuntimeError("cv2.solvePnP (SQPNP) failed on mat correspondences")

  R, _ = cv2.Rodrigues(rvec)
  t = np.asarray(tvec, dtype=np.float64).reshape(3)

  circle_obj, circle_img = _associate_circle_samples(
    detection.circle_samples_uv, K, dist, R, t,
    n_world=max(detection.circle_samples_uv.shape[0], 32),
  )

  full_obj = np.concatenate([initial_obj, circle_obj], axis=0)
  full_img = np.concatenate([initial_img, circle_img], axis=0)

  rvec_ref, tvec_ref = cv2.solvePnPRefineLM(
    full_obj.reshape(-1, 1, 3),
    full_img.reshape(-1, 1, 2),
    K, dist, rvec, tvec,
  )
  R, _ = cv2.Rodrigues(rvec_ref)
  t = np.asarray(tvec_ref, dtype=np.float64).reshape(3)

  proj, _ = cv2.projectPoints(full_obj.reshape(-1, 1, 3),
                              rvec_ref, tvec_ref, K, dist)
  proj = proj.reshape(-1, 2)
  err = np.linalg.norm(proj - full_img, axis=1)
  rms = float(np.sqrt(np.mean(err ** 2)))
  if rms > 2.0:
    logger.warning("extrinsic RMS %.2f px > 2.0 (target < 1.0)", rms)
  else:
    logger.info("extrinsic RMS %.2f px on %d correspondences", rms, len(err))

  return ExtrinsicResult(
    R=R, t=t, rms_px=rms,
    obj_pts=full_obj,
    img_pts=full_img,
  )
