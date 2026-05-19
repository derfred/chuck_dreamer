"""Render the known mat geometry on top of a frame using a calibration."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..geometry import circle_points_world, line_endpoints_world
from .serialization import CameraCalibration

logger = logging.getLogger(__name__)


def render_overlay(rgb: np.ndarray, cal: CameraCalibration) -> np.ndarray:
  """Return a copy of ``rgb`` with the mat geometry projected on top."""
  import cv2

  rvec, _ = cv2.Rodrigues(cal.R)
  tvec = cal.t.reshape(3, 1)

  eps = line_endpoints_world()
  line1 = np.stack([eps["line1_near"], eps["line1_far"]])
  line2 = np.stack([eps["line2_near"], eps["line2_far"]])
  circle = circle_points_world(96)
  axes_world = np.array([
    [0, 0, 0], [50, 0, 0],
    [0, 0, 0], [0, 50, 0],
    [0, 0, 0], [0, 0, 50],
  ], dtype=np.float64)

  img = rgb.copy()
  if img.dtype != np.uint8:
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
  bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

  def project(pts):
    pts = pts.reshape(-1, 1, 3).astype(np.float64)
    pp, _ = cv2.projectPoints(pts, rvec, tvec, cal.K, cal.dist)
    return pp.reshape(-1, 2).astype(int)

  for line, color in ((line1, (0, 255, 0)), (line2, (255, 0, 0))):
    pp = project(line)
    cv2.line(bgr, tuple(pp[0]), tuple(pp[1]), color, 2)

  pp_circle = project(circle)
  for a, b in zip(pp_circle, np.roll(pp_circle, -1, axis=0)):
    cv2.line(bgr, tuple(a), tuple(b), (0, 165, 255), 2)

  pp_axes = project(axes_world)
  for i, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
    cv2.arrowedLine(bgr, tuple(pp_axes[2 * i]), tuple(pp_axes[2 * i + 1]),
                    color, 2, tipLength=0.15)

  return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_overlay(rgb: np.ndarray, cal: CameraCalibration, out_path: Path) -> None:
  import cv2
  out_path.parent.mkdir(parents=True, exist_ok=True)
  overlay = render_overlay(rgb, cal)
  cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
