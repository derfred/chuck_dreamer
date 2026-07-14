"""Sub-pixel refinement of the 4 line-endpoint clicks.

Two passes, in order:

  1. ``subpixel_snap_clicks`` runs ``cv2.cornerSubPix`` on each raw
     click within a small window. For a true corner-like feature
     (line tip on a contrasting background) this snaps to ~0.1 px.
     If the snap drifts more than ``_SNAP_REJECT_PX`` from the raw
     click we keep the raw click (the click likely wasn't near a
     real corner).

  2. ``refine_line_endpoints`` takes a pair of (NEAR, FAR) clicks
     describing one painted line, samples perpendicular cross-sections
     across the line at K positions, picks the brightness peak on
     each cross-section, refits a 1D line through the peaks with
     ``cv2.fitLine``, and walks along that refit line to find the
     endpoints by tracking where the line's brightness drops below
     a baseline + threshold. This is the standard sub-pixel
     edge-tracking trick for thin painted features.

Both helpers operate on the full-resolution RGB frame; the UI's
display-scaling is irrelevant here.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# cornerSubPix search window half-size (in pixels). The function
# searches a (2W+1)x(2W+1) window centred on each seed click for the
# local image-gradient saddle/extremum. 11 = 23x23 px window — big
# enough to find a real tip even if the click was a couple of px off,
# small enough to avoid picking up an unrelated feature.
_SUBPIX_WIN = 11

# If cornerSubPix moves a click by more than this many pixels the
# refinement is rejected and the raw click is kept.
_SNAP_REJECT_PX = 10.0

# Line-fit refinement parameters.
_LINE_CROSS_SAMPLES = 60     # K perpendicular sections sampled along the line
_LINE_CROSS_HALF_PX = 12     # +/- half-width of each cross-section
_LINE_SUBSAMPLE_PX  = 0.5    # sub-pixel sampling stride within a cross-section
_LINE_PEAK_MIN_CONTRAST = 30 # minimum brightness over baseline to accept a peak
_LINE_ENDPOINT_FALLOFF  = 0.5  # endpoint = where brightness drops below
                               # baseline + falloff * (peak - baseline)
_LINE_ENDPOINT_PAD_PX   = 30   # how far past the click to walk looking for
                               # the tip; clipped to image bounds


def subpixel_snap_clicks(rgb: np.ndarray, clicks: np.ndarray) -> np.ndarray:
  """Run ``cornerSubPix`` on each click; return refined ``(N, 2)`` array.

  Rejected snaps (those that moved more than ``_SNAP_REJECT_PX`` from
  the raw click) fall back to the raw click so the output is always
  at least as good as the input.
  """
  import cv2

  raw = np.asarray(clicks, dtype=np.float32).reshape(-1, 1, 2)
  gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
  try:
    refined = cv2.cornerSubPix(
      gray, raw.copy(), (_SUBPIX_WIN, _SUBPIX_WIN), (-1, -1), criteria,
    )
  except cv2.error as e:
    logger.warning("cornerSubPix failed: %s", e)
    return np.asarray(clicks, dtype=np.float64)

  out = raw.reshape(-1, 2).copy()
  for i in range(out.shape[0]):
    dx = float(refined[i, 0, 0] - raw[i, 0, 0])
    dy = float(refined[i, 0, 1] - raw[i, 0, 1])
    if abs(dx) + abs(dy) > _SNAP_REJECT_PX * 2:
      logger.info("subpixel snap rejected for click %d (moved %.1f, %.1f)",
                  i, dx, dy)
      continue
    out[i] = refined[i, 0]
    logger.debug("subpixel snap click %d: (%.2f, %.2f) -> (%.2f, %.2f)",
                  i, raw[i, 0, 0], raw[i, 0, 1], out[i, 0], out[i, 1])
  return out.astype(np.float64)


def refine_line_endpoints(rgb: np.ndarray,
                          near_click: np.ndarray,
                          far_click: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
  """Refine one line's NEAR/FAR endpoints by perpendicular edge tracking.

  The two input clicks define a rough segment along the painted line.
  We sample ``_LINE_CROSS_SAMPLES`` perpendicular cross-sections evenly
  spaced along the segment, find the brightness peak on each, refit a
  line through those peaks with ``cv2.fitLine``, and walk along the
  refit line outward from each endpoint until the brightness drops
  below a baseline + threshold.

  Returns ``(near_refined, far_refined)`` as ``(2,) float64`` arrays.
  If the refinement fails (insufficient peaks, geometry degenerate,
  etc.) the raw clicks are returned unchanged.
  """
  import cv2

  gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
  H, W = gray.shape

  near = np.asarray(near_click, dtype=np.float64).reshape(2)
  far  = np.asarray(far_click,  dtype=np.float64).reshape(2)
  d = far - near
  L = float(np.linalg.norm(d))
  if L < 8.0:
    logger.info("refine_line_endpoints: segment too short (%.1f px); skipped.", L)
    return near, far
  tangent = d / L
  # Perpendicular in image space (right-hand rotation by 90deg).
  normal = np.array([-tangent[1], tangent[0]])

  # Sample K perpendicular cross-sections across the line.
  ts = np.linspace(0.1, 0.9, _LINE_CROSS_SAMPLES)   # avoid the very tips
  peaks: list[np.ndarray] = []
  baselines: list[float] = []
  for t in ts:
    centre = near + t * d
    peak = _find_perp_peak(gray, centre, normal)
    if peak is None:
      continue
    pos, brightness, baseline = peak
    peaks.append(pos)
    baselines.append(baseline)

  if len(peaks) < 6:
    logger.info("refine_line_endpoints: only %d perp peaks; skipping line fit.",
                len(peaks))
    return near, far

  peaks_arr = np.array(peaks, dtype=np.float32).reshape(-1, 1, 2)
  # cv2.fitLine returns (vx, vy, x0, y0): unit direction + a point on the line.
  vx, vy, x0, y0 = cv2.fitLine(peaks_arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
  line_dir = np.array([float(vx), float(vy)])
  line_pt  = np.array([float(x0), float(y0)])

  # Re-align direction so it points from NEAR toward FAR.
  if np.dot(line_dir, tangent) < 0:
    line_dir = -line_dir

  # Project each peak onto the refit line; the median brightness on the
  # line is our baseline, the 90th-percentile is our peak reference.
  baseline = float(np.percentile(baselines, 50)) if baselines else 0.0
  # Median peak brightness along the line, for the falloff threshold.
  peak_brights: list[float] = []
  for p in peaks:
    px, py = int(round(p[0])), int(round(p[1]))
    if 0 <= px < W and 0 <= py < H:
      peak_brights.append(float(gray[py, px]))
  peak_b = float(np.median(peak_brights)) if peak_brights else baseline + 50.0
  threshold = baseline + _LINE_ENDPOINT_FALLOFF * (peak_b - baseline)

  # Walk outward from each click along the refit line, looking for the
  # tip (first sample where brightness drops below threshold).
  near_refined = _walk_to_tip(gray, line_pt, line_dir, near,
                               direction=-1, threshold=threshold)
  far_refined  = _walk_to_tip(gray, line_pt, line_dir, far,
                               direction=+1, threshold=threshold)
  return near_refined, far_refined


def _find_perp_peak(gray: np.ndarray, centre: np.ndarray, normal: np.ndarray
                    ) -> tuple[np.ndarray, float, float] | None:
  """Sample a perpendicular cross-section through ``centre`` along
  ``normal``; return ``(peak_pos, peak_brightness, baseline)`` or None
  if the cross-section is below contrast or out of bounds.
  """
  H, W = gray.shape
  half = _LINE_CROSS_HALF_PX
  stride = _LINE_SUBSAMPLE_PX
  n_samples = int(2 * half / stride) + 1
  offsets = np.linspace(-half, half, n_samples)
  pts = centre[None, :] + offsets[:, None] * normal[None, :]
  u = pts[:, 0]
  v = pts[:, 1]
  # Discard samples outside the image; bilinear-sample interior.
  valid = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
  if valid.sum() < 5:
    return None
  brights = _bilinear_sample(gray, u[valid], v[valid])

  baseline = float(np.percentile(brights, 10))
  peak_idx = int(np.argmax(brights))
  peak_val = float(brights[peak_idx])
  if peak_val - baseline < _LINE_PEAK_MIN_CONTRAST:
    return None
  # Convert peak_idx back into image-space (u, v).
  global_idx = np.where(valid)[0][peak_idx]
  return pts[global_idx], peak_val, baseline


def _bilinear_sample(gray: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
  u0 = np.floor(u).astype(int)
  v0 = np.floor(v).astype(int)
  u1 = u0 + 1
  v1 = v0 + 1
  du = u - u0
  dv = v - v0
  Ia = gray[v0, u0]
  Ib = gray[v0, u1]
  Ic = gray[v1, u0]
  Id = gray[v1, u1]
  return np.asarray((1 - du) * (1 - dv) * Ia + du * (1 - dv) * Ib
                    + (1 - du) * dv * Ic + du * dv * Id)


def _walk_to_tip(gray: np.ndarray, line_pt: np.ndarray, line_dir: np.ndarray,
                 seed_click: np.ndarray, direction: int, threshold: float
                 ) -> np.ndarray:
  """Walk along the line away from the centre in ``direction`` until
  brightness drops below ``threshold`` for two consecutive samples
  (catches noise).

  ``direction = +1`` walks toward FAR, ``-1`` toward NEAR (relative
  to the line_dir orientation, which points NEAR->FAR).

  The walk starts from the projection of ``seed_click`` onto the line
  and continues for at most ``_LINE_ENDPOINT_PAD_PX`` past the seed.
  """
  H, W = gray.shape
  seed_proj_t = float(np.dot(seed_click - line_pt, line_dir))
  step = 0.5  # px per step
  max_steps = int(_LINE_ENDPOINT_PAD_PX / step)
  last_above = seed_proj_t
  below_streak = 0
  for k in range(max_steps):
    t = seed_proj_t + direction * k * step
    p = line_pt + t * line_dir
    px, py = int(round(p[0])), int(round(p[1]))
    if not (0 <= px < W and 0 <= py < H):
      break
    bright = float(gray[py, px])
    if bright >= threshold:
      last_above = t
      below_streak = 0
    else:
      below_streak += 1
      if below_streak >= 2:
        break
  return (line_pt + last_above * line_dir).astype(np.float64)
