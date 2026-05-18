"""Tests for camera intrinsics estimation.

Builds a virtual checkerboard with a known pinhole camera, projects
corner points through several poses, and feeds the synthetic correspondences
into :func:`solve_intrinsics`. The recovered K should match the ground truth
to a tight tolerance.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from chuck_dreamer.real.scene_registration import (
  CheckerboardSpec,
  solve_intrinsics,
)
from chuck_dreamer.real.scene_types import CalibrationCapture


def _synthetic_captures(
  spec: CheckerboardSpec,
  K: np.ndarray,
  image_size: tuple[int, int],
  n_views: int = 12,
  rng_seed: int = 0,
) -> list[CalibrationCapture]:
  rng = np.random.default_rng(rng_seed)
  objp = spec.object_points()
  captures: list[CalibrationCapture] = []
  for i in range(n_views):
    # Random small rotations, board ~0.5m away with some lateral jitter.
    rvec = rng.uniform(-0.4, 0.4, size=3)
    tvec = np.array([
      rng.uniform(-0.1, 0.1),
      rng.uniform(-0.05, 0.05),
      rng.uniform(0.4, 0.7),
    ])
    img_pts, _ = cv2.projectPoints(objp, rvec, tvec, K, distCoeffs=np.zeros(5))
    img_pts = img_pts.reshape(-1, 2).astype(np.float32)
    captures.append(CalibrationCapture(
      image_path=Path(f"/tmp/synthetic-{i:03d}.png"),
      description="",
      corners_px=img_pts,
    ))
  return captures


def test_solve_intrinsics_recovers_known_K():
  spec = CheckerboardSpec(cols=9, rows=6, square_size_m=0.025)
  W, H = 640, 480
  K_true = np.array([
    [600.0, 0.0, 320.0],
    [0.0, 600.0, 240.0],
    [0.0, 0.0, 1.0],
  ])
  captures = _synthetic_captures(spec, K_true, (W, H), n_views=15)
  intr = solve_intrinsics(captures, spec, (W, H))

  # Pinhole synthetic data → cv2 should recover K within sub-pixel error.
  assert intr.reproj_error < 0.5, f"unexpected RMS reproj error {intr.reproj_error}"
  K_est = intr.K
  # Focal lengths within 1%, principal point within 1 px.
  assert abs(K_est[0, 0] - K_true[0, 0]) / K_true[0, 0] < 0.01
  assert abs(K_est[1, 1] - K_true[1, 1]) / K_true[1, 1] < 0.01
  assert abs(K_est[0, 2] - K_true[0, 2]) < 1.0
  assert abs(K_est[1, 2] - K_true[1, 2]) < 1.0
  assert intr.image_size == (W, H)
  assert len(intr.captures) == 15
