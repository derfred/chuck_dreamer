"""Detect the mat's two parallel lines + circle in the empty-scene image.

Algorithm: **circle-anchored search**. Find circle candidates with
``cv2.HoughCircles`` (filtered to those sitting on a coarse mat ROI),
then for each candidate search a fan of (heading θ, near-end offset)
for two parallel lines extending leftward at distance ±D/2; score by
edge response along the predicted line pixels with bonus weight on the
better line so one occluded line is OK.

The other detectors (contour / hough_lines / ransac_template) lived
here previously but were removed — they performed poorly on the
real datasets we tested. See git history if you need them back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .. import constants

logger = logging.getLogger(__name__)


@dataclass
class MatDetection:
  # Endpoints in pixel coords, each shape (2,):
  line1_near: np.ndarray
  line1_far:  np.ndarray
  line2_near: np.ndarray
  line2_far:  np.ndarray
  circle_center: np.ndarray         # (2,)
  circle_axes:   tuple[float, float]   # (semi-major, semi-minor) in pixels
  circle_angle:  float              # degrees, cv2 convention
  circle_samples_uv: np.ndarray     # (N, 2) pixel samples around the ellipse


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fit_line(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """TLS-fit a line through ``pts``. Returns (direction, origin, ep_min, ep_max)."""
  import cv2
  vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32),
                               cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
  d = np.array([vx, vy], dtype=np.float64)
  origin = np.array([x0, y0], dtype=np.float64)
  t = (pts.astype(np.float64) - origin) @ d
  t_min, t_max = float(t.min()), float(t.max())
  return d, origin, origin + t_min * d, origin + t_max * d


def _ellipse_samples(center: np.ndarray, axes: tuple[float, float],
                     angle_deg: float, n: int = 64) -> np.ndarray:
  a, b = axes[0] / 2.0, axes[1] / 2.0
  theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
  base = np.stack([a * np.cos(theta), b * np.sin(theta)], axis=-1)
  c, s = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
  R = np.array([[c, -s], [s, c]])
  return np.asarray((base @ R.T) + center)


def _centrality_score(point: np.ndarray, image_size: tuple[int, int]) -> float:
  """1.0 at the image center, decays with normalized distance to the edge."""
  W, H = image_size
  cx, cy = W / 2.0, H / 2.0
  dx = (point[0] - cx) / (W / 2.0)
  dy = (point[1] - cy) / (H / 2.0)
  return float(np.exp(-(dx * dx + dy * dy)))


def _split_near_far(ep1: np.ndarray, ep2: np.ndarray,
                    circle_center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Return (near, far) endpoints — near is closer to ``circle_center``."""
  d1 = float(np.linalg.norm(ep1 - circle_center))
  d2 = float(np.linalg.norm(ep2 - circle_center))
  return (ep1, ep2) if d1 < d2 else (ep2, ep1)


def _assign_line_roles(
  l1_ep: tuple[np.ndarray, np.ndarray],
  l2_ep: tuple[np.ndarray, np.ndarray],
  circle_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Compute the (line1_near, line1_far, line2_near, line2_far) endpoints.

  Line 1 sits at +Y in the world frame; given the spec camera viewpoint
  this maps to the line with the larger v-pixel coordinate.
  """
  l1n, l1f = _split_near_far(l1_ep[0], l1_ep[1], circle_center)
  l2n, l2f = _split_near_far(l2_ep[0], l2_ep[1], circle_center)
  if l1n[1] < l2n[1]:
    l1n, l1f, l2n, l2f = l2n, l2f, l1n, l1f
  return l1n, l1f, l2n, l2f


def _build_detection(l1_ep, l2_ep, circle_center, circle_axes,
                     circle_angle) -> MatDetection:
  l1n, l1f, l2n, l2f = _assign_line_roles(l1_ep, l2_ep, circle_center)
  samples = _ellipse_samples(circle_center, circle_axes, circle_angle, n=64)
  return MatDetection(
    line1_near=l1n, line1_far=l1f,
    line2_near=l2n, line2_far=l2f,
    circle_center=circle_center,
    circle_axes=circle_axes,
    circle_angle=circle_angle,
    circle_samples_uv=samples,
  )


def _canny_edges(rgb: np.ndarray) -> np.ndarray:
  import cv2
  gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
  blur = cv2.GaussianBlur(gray, (5, 5), 0)
  return cv2.Canny(blur, 60, 180)


def _estimate_mat_roi(rgb: np.ndarray) -> np.ndarray:
  """Return a boolean mask (H, W) that approximates the mat region.

  Heuristic: the mat is darker than the surrounding carpet/wall, so an
  Otsu split on the *blurred* grayscale separates them. We keep the
  dark connected component closest to the image center (which the spec
  says is where the mat sits), then dilate slightly so lines that ride
  along the mat edge still count as "on the mat".
  """
  import cv2
  H, W = rgb.shape[:2]
  gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
  blur = cv2.GaussianBlur(gray, (9, 9), 0)
  _, dark = cv2.threshold(255 - blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
  num, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
  if num <= 1:
    return np.ones((H, W), dtype=bool)
  cx, cy = W / 2.0, H / 2.0
  best_idx = -1
  best_dist = np.inf
  for i in range(1, num):
    area = float(stats[i, cv2.CC_STAT_AREA])
    if area < 0.02 * H * W:
      continue
    cxi, cyi = centroids[i]
    d = (cxi - cx) ** 2 + (cyi - cy) ** 2
    if d < best_dist:
      best_dist = d
      best_idx = i
  if best_idx < 0:
    return np.ones((H, W), dtype=bool)
  mask = (labels == best_idx).astype(np.uint8) * 255
  mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
  return mask.astype(bool)


def _line_on_mat_fraction(
  mat_roi: np.ndarray, p1: np.ndarray, p2: np.ndarray, n_samples: int = 32,
) -> float:
  """Fraction of points along the segment p1->p2 that fall inside the mat ROI."""
  H, W = mat_roi.shape
  ts = np.linspace(0.0, 1.0, n_samples)
  pts = p1[None, :] * (1 - ts[:, None]) + p2[None, :] * ts[:, None]
  ii = np.clip(pts[:, 1].astype(int), 0, H - 1)
  jj = np.clip(pts[:, 0].astype(int), 0, W - 1)
  return float(mat_roi[ii, jj].mean())


def _sample_line_edge_response(
  edges: np.ndarray,
  p1: np.ndarray, p2: np.ndarray,
  n_samples: int,
) -> float:
  """Average edge response along ``n_samples`` points on the line p1->p2."""
  H, W = edges.shape
  if n_samples < 2:
    n_samples = 2
  ts = np.linspace(0.0, 1.0, n_samples)
  pts = p1[None, :] * (1 - ts[:, None]) + p2[None, :] * ts[:, None]
  ii = np.clip(pts[:, 1].astype(int), 0, H - 1)
  jj = np.clip(pts[:, 0].astype(int), 0, W - 1)
  return float((edges[ii, jj] > 0).mean())


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def detect_mat_markings(rgb: np.ndarray) -> MatDetection | None:
  """Detect 2 lines + 1 circle in the mat-only frame.

  Returns ``None`` if no satisfactory (line, line, circle) triple can be
  found. Callers should fall back to interactive annotation in that case.
  """
  import cv2
  cfg = constants.active()
  H, W = rgb.shape[:2]
  gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
  edges = _canny_edges(rgb)
  edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

  mat_roi = _estimate_mat_roi(rgb)
  mat_roi_fraction = float(mat_roi.mean())
  if mat_roi_fraction < 0.02 or mat_roi_fraction > 0.95:
    mat_roi = np.ones_like(mat_roi)
    logger.info("mat ROI estimate degenerate (fraction=%.2f) — disabling gate",
                mat_roi_fraction)

  circles_raw = cv2.HoughCircles(
    gray, cv2.HOUGH_GRADIENT,
    dp=1.0, minDist=30, param1=120, param2=18,
    minRadius=4, maxRadius=int(0.15 * min(W, H)),
  )
  if circles_raw is None:
    logger.info("no circles from HoughCircles — falling back")
    return None
  circles = circles_raw[0]   # (N, 3) (x, y, r)

  in_mat = np.array([
    bool(mat_roi[int(np.clip(c[1], 0, H - 1)), int(np.clip(c[0], 0, W - 1))])
    for c in circles
  ])
  circles = circles[in_mat]
  if len(circles) == 0:
    logger.info("no Hough circles inside the mat ROI — falling back")
    return None

  if len(circles) > 8:
    centralities = np.array([
      _centrality_score(np.array([c[0], c[1]]), (W, H)) for c in circles
    ])
    keep_idx = np.argsort(-centralities)[:8]
    circles = circles[keep_idx]

  D_over_r = cfg.mat_line_separation_mm / cfg.mat_circle_radius_mm
  L_over_r = cfg.mat_line_length_mm / cfg.mat_circle_radius_mm

  n_angles = 72   # 5° steps
  # The circle's midpoint should lie on the segment connecting the two lines'
  # near endpoints — i.e. the circle's offset along the line direction should
  # be ~0 (it sits next to the near ends, not behind them). We sweep a small
  # range around zero and bias the score toward small absolute offsets.
  thetas = np.linspace(0.0, np.pi, n_angles, endpoint=False)
  min_on_mat_fraction = 0.6
  best = None
  best_score = -np.inf
  for cx, cy, cr in circles:
    if cr < 3:
      continue
    center = np.array([cx, cy])
    central = _centrality_score(center, (W, H))
    if central < 0.05:
      continue
    D_px = cr * D_over_r
    L_px = cr * L_over_r
    half_D = D_px / 2.0
    n_line_samples = max(20, int(L_px / 4))
    # Offset sweep: ± half_D, but the scoring bias makes small offsets win.
    offset_range = 0.5 * half_D
    offsets = np.linspace(-offset_range, +offset_range, 5)
    for theta in thetas:
      direction = np.array([np.cos(theta), np.sin(theta)])
      normal = np.array([-direction[1], direction[0]])
      for offset in offsets:
        near = center - offset * direction
        l1_near = near + half_D * normal
        l2_near = near - half_D * normal
        l1_far  = l1_near - L_px * direction
        l2_far  = l2_near - L_px * direction
        on1 = _line_on_mat_fraction(mat_roi, l1_near, l1_far)
        on2 = _line_on_mat_fraction(mat_roi, l2_near, l2_far)
        if on1 < min_on_mat_fraction or on2 < min_on_mat_fraction:
          continue
        r1 = _sample_line_edge_response(edges_dilated, l1_near, l1_far, n_line_samples)
        r2 = _sample_line_edge_response(edges_dilated, l2_near, l2_far, n_line_samples)
        if max(r1, r2) < 0.15:
          continue
        line_score = 0.5 * (r1 + r2) + 0.5 * max(r1, r2)
        # Penalise offsets that pull the circle off the connecting line of
        # the two lines' near endpoints. A Gaussian with sigma = half_D/3
        # peaks at offset=0 and decays smoothly.
        on_line_bonus = float(np.exp(-(offset ** 2) / (2 * (half_D / 3.0) ** 2)))
        score = line_score * central * on_line_bonus
        if score > best_score:
          best_score = score
          best = (l1_near, l1_far, l2_near, l2_far, center, float(cr))

  if best is None or best_score < 0.1:
    logger.info("best score %.3f below threshold — falling back",
                best_score if best is not None else -1.0)
    return None
  l1n, l1f, l2n, l2f, center, cr = best
  logger.info("mat detection: best score=%.3f", best_score)

  # Snap each line to nearby edge pixels via TLS refit.
  ys, xs = np.where(edges > 0)
  ep_all = np.stack([xs, ys], axis=-1).astype(np.float64)
  refined = []
  for p1, p2 in ((l1n, l1f), (l2n, l2f)):
    length = float(np.linalg.norm(p2 - p1))
    if length < 1e-6:
      refined.append((p1, p2))
      continue
    d = (p2 - p1) / length
    n = np.array([-d[1], d[0]])
    perp  = (ep_all - p1) @ n
    along = (ep_all - p1) @ d
    keep = (np.abs(perp) < 6.0) & (along > -10.0) & (along < length + 10.0)
    if int(keep.sum()) >= 20:
      _, _, ep_min, ep_max = _fit_line(ep_all[keep])
      refined.append((ep_min, ep_max))
    else:
      refined.append((p1, p2))
  return _build_detection(
    refined[0], refined[1], center, (2 * cr, 2 * cr), 0.0,
  )
