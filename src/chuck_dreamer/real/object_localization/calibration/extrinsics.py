"""Per-dataset extrinsics solve from a single mat annotation.

Inputs:
  - intrinsics (K, dist) already cached for the dataset.
  - one ``MatDetection`` (4 line endpoints + N circle samples in pixels).

Output: world->camera rotation/translation in millimeters, plus a
reprojection-RMS quality number that's surfaced to the user before
anything is persisted.

The non-trivial step is **phase alignment**: ``cv2.fitEllipse``'s
parametrization starts from the ellipse major axis with an arbitrary
sign, but our world-side samples start from world theta=0. We sweep
both phase offset and traversal direction with a 4-point seed PnP,
pick the lowest-error pair, then re-solve with all 4+N correspondences.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..geometry import circle_samples_world, line_world_points
from ..runtime import ObjectLocalizationConfig
from ..types import Extrinsics, MatDetection

logger = logging.getLogger(__name__)


@dataclass
class ExtrinsicsResult:
  extrinsics: Extrinsics
  phase_offset: int
  reverse: bool
  seed_rms_px: float
  refined_rms_px: float
  world_points: np.ndarray   # (4 + N, 3)
  image_points: np.ndarray   # (4 + N, 2)


def solve_extrinsics(
  detection: MatDetection,
  K: np.ndarray, dist: np.ndarray,
  ol_cfg: ObjectLocalizationConfig,
) -> ExtrinsicsResult:
  """Run the full extrinsics solve. Raises ``RuntimeError`` on failure."""
  import cv2

  line_img = np.stack([
    detection.line1_near,
    detection.line1_far,
    detection.line2_near,
    detection.line2_far,
  ], axis=0).astype(np.float64)
  line_world = line_world_points(ol_cfg.mat_line_length_mm,
                                 ol_cfg.mat_line_separation_mm)
  # Re-order line_world to match line_img: (line1_near, line1_far, line2_near, line2_far)
  # line_world_points returns them already in this order.

  seed_rvec, seed_tvec, seed_rms = _seed_pnp(line_world, line_img, K, dist)

  circle_uv = np.asarray(detection.circle_samples_uv, dtype=np.float64)
  N = circle_uv.shape[0]

  best = None
  for reverse in (False, True):
    for k in range(N):
      world = circle_samples_world(ol_cfg.mat_circle_radius_mm, N,
                                   phase_offset=k, reverse=reverse)
      proj, _ = cv2.projectPoints(world, seed_rvec, seed_tvec, K, dist)
      diff = proj.reshape(-1, 2) - circle_uv
      err = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
      if best is None or err < best[0]:
        best = (err, k, reverse)
  assert best is not None
  _, k_best, reverse_best = best
  circle_world = circle_samples_world(ol_cfg.mat_circle_radius_mm, N,
                                      phase_offset=k_best, reverse=reverse_best)

  world_pts = np.concatenate([line_world, circle_world], axis=0)
  img_pts   = np.concatenate([line_img, circle_uv], axis=0)

  ok, rvec, tvec = cv2.solvePnP(
    world_pts.astype(np.float64), img_pts.astype(np.float64),
    K.astype(np.float64), dist.astype(np.float64),
    flags=cv2.SOLVEPNP_SQPNP,
  )
  if not ok:
    raise RuntimeError("cv2.solvePnP (SQPNP) failed on the 4+N mat correspondences.")

  # Disambiguate SQPNP: ensure it converged to the same twin as the
  # IPPE seed (rather than flipping to the mirror image). The two twins
  # share reprojection RMS so SQPNP can pick either; we check by
  # comparing camera positions and, if they differ in sign, fall back
  # to ITERATIVE seeded with the IPPE solution to lock the basin.
  R_sq, _ = cv2.Rodrigues(rvec)
  cam_pos_sq = -R_sq.T @ tvec.reshape(3)
  R_seed, _  = cv2.Rodrigues(seed_rvec)
  cam_pos_seed = -R_seed.T @ seed_tvec.reshape(3)
  if cam_pos_sq[2] * cam_pos_seed[2] < 0:
    logger.warning("SQPNP picked the mirror twin of the IPPE seed "
                   "(SQPNP cam_z=%.1fmm, seed cam_z=%.1fmm); falling back "
                   "to ITERATIVE seeded by IPPE.",
                   float(cam_pos_sq[2]), float(cam_pos_seed[2]))
    ok2, rvec2, tvec2 = cv2.solvePnP(
      world_pts.astype(np.float64), img_pts.astype(np.float64),
      K.astype(np.float64), dist.astype(np.float64),
      rvec=seed_rvec.copy(), tvec=seed_tvec.copy(),
      useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if ok2:
      rvec, tvec = rvec2, tvec2

  refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8)
  rvec, tvec = cv2.solvePnPRefineLM(
    world_pts.astype(np.float64), img_pts.astype(np.float64),
    K.astype(np.float64), dist.astype(np.float64),
    rvec, tvec, criteria=refine_criteria,
  )
  R, _ = cv2.Rodrigues(rvec)

  # Final sanity log: report which side of the mat the camera ended
  # up on. Negative = the world frame's +Z is flipped relative to the
  # camera, which is fine for downstream code that detects + handles
  # it; positive = the +Z-up convention worked as designed.
  cam_pos_final = -R.T @ tvec.reshape(3)
  logger.info("extrinsics solved: camera_z_world=%+.1fmm (negative is OK; "
              "indicates flipped +Z convention).", float(cam_pos_final[2]))

  proj, _ = cv2.projectPoints(world_pts, rvec, tvec, K, dist)
  diff = proj.reshape(-1, 2) - img_pts
  refined_rms = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

  extrinsics = Extrinsics(
    R = R.astype(np.float64),
    t = tvec.astype(np.float64).reshape(3, 1),
    rms_px = refined_rms,
  )
  return ExtrinsicsResult(
    extrinsics    = extrinsics,
    phase_offset  = k_best,
    reverse       = reverse_best,
    seed_rms_px   = seed_rms,
    refined_rms_px = refined_rms,
    world_points  = world_pts,
    image_points  = img_pts,
  )


def _seed_pnp(world: np.ndarray, image: np.ndarray,
              K: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
  """4-point coplanar PnP via IPPE. Returns ``(rvec, tvec, rms_px)``.

  IPPE on coplanar points returns two solutions (the "twin" or "mirror"
  ambiguity inherent to planar-target PnP). Both solutions reproject
  the input points equally well, so picking the one with lower
  reprojection RMS is a numerical coin-flip — and a wrong choice
  produces a flipped world frame where the camera sits below the
  mat. We pick the physically valid solution (camera above mat,
  cam_pos_world.z > 0). If both have z > 0 we fall back to the lower
  RMS; if both have z < 0 the world-frame convention is upside down
  and we raise.
  """
  import cv2
  ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
    world.astype(np.float64), image.astype(np.float64),
    K.astype(np.float64), dist.astype(np.float64),
    flags=cv2.SOLVEPNP_IPPE,
  )
  if not ok or len(rvecs) == 0:
    raise RuntimeError("seed PnP (IPPE) failed on 4 line endpoints.")

  candidates: list[tuple[float, int, float]] = []   # (cam_z, index, rms)
  for i, (rvec_i, tvec_i, err_i) in enumerate(zip(rvecs, tvecs, errs)):
    R_i, _ = cv2.Rodrigues(rvec_i)
    cam_pos = -R_i.T @ tvec_i.reshape(3)
    candidates.append((float(cam_pos[2]), i, float(err_i)))

  above = [c for c in candidates if c[0] > 0.0]
  if above:
    # Pick the above-mat solution with the lower reprojection error.
    above.sort(key=lambda c: c[2])
    chosen_idx = above[0][1]
  else:
    # Both IPPE solutions have the camera "below" the mat plane. That
    # means the world-frame convention (origin at circle, +Z out of mat
    # toward where the camera is "supposed" to be) doesn't match how
    # this mat is physically mounted. Pick the lower-RMS solution and
    # let downstream object-pose code detect the flipped Z and flip
    # the resting-face hypothesis accordingly.
    candidates.sort(key=lambda c: c[2])
    chosen_idx = candidates[0][1]
    logger.warning(
      "PnP: both twin solutions have camera-below-mat (cam_z values: %s). "
      "Picking lower-RMS twin (rms=%.2fpx, cam_z=%.1fmm). World-frame +Z "
      "is inverted relative to the camera; object-pose code should detect "
      "this and flip its resting-face hypothesis. If you intended +Z to "
      "be 'up toward camera', re-annotate with NEAR/FAR clicks swapped on "
      "one line.",
      [round(c[0], 1) for c in candidates],
      candidates[0][2], candidates[0][0])

  rvec, tvec = rvecs[chosen_idx], tvecs[chosen_idx]
  proj, _ = cv2.projectPoints(world, rvec, tvec, K, dist)
  diff = proj.reshape(-1, 2) - image
  rms = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
  return rvec, tvec, rms


def render_extrinsics_overlay(
  frame_rgb: np.ndarray, extrinsics: Extrinsics, K: np.ndarray, dist: np.ndarray,
  ol_cfg: ObjectLocalizationConfig, detection: MatDetection | None,
  result: ExtrinsicsResult | None,
) -> np.ndarray:
  """Reprojected mat geometry on top of ``frame_rgb`` (BGR canvas out)."""
  import cv2

  rvec, _ = cv2.Rodrigues(extrinsics.R)
  tvec = extrinsics.t.reshape(3)
  canvas = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)

  line_world = line_world_points(ol_cfg.mat_line_length_mm,
                                 ol_cfg.mat_line_separation_mm)
  line_proj, _ = cv2.projectPoints(line_world, rvec, tvec, K, dist)
  line_proj = line_proj.reshape(-1, 2)
  cv2.line(canvas, _ipt(line_proj[0]), _ipt(line_proj[1]), (0, 255, 0), 2)
  cv2.line(canvas, _ipt(line_proj[2]), _ipt(line_proj[3]), (0, 0, 255), 2)

  circle_world = circle_samples_world(ol_cfg.mat_circle_radius_mm, 256)
  circle_proj, _ = cv2.projectPoints(circle_world, rvec, tvec, K, dist)
  circle_proj = circle_proj.reshape(-1, 2)
  pts = np.array([_ipt(p) for p in circle_proj], dtype=np.int32)
  cv2.polylines(canvas, [pts], isClosed=True, color=(0, 165, 255), thickness=2)

  # Mat border (configured world bbox) — green polyline. Each edge is
  # sampled at multiple points so wide-angle lens distortion shows as
  # the curved projection it actually is, rather than straight chords
  # between the 4 corner points.
  border_world = _mat_border_samples_world(ol_cfg, n_per_edge=32)
  border_proj, _ = cv2.projectPoints(border_world, rvec, tvec, K, dist)
  border_proj = border_proj.reshape(-1, 2)
  border_pts = np.array([_ipt(p) for p in border_proj], dtype=np.int32)
  cv2.polylines(canvas, [border_pts], isClosed=True, color=(80, 255, 80), thickness=2)

  if detection is not None:
    # Fitted ellipse from the annotation, drawn faintly.
    ax_w, ax_h = detection.circle_axes
    cv2.ellipse(canvas,
                (_ipt(detection.circle_center)),
                (int(round(ax_w / 2)), int(round(ax_h / 2))),
                float(detection.circle_angle), 0, 360,
                (180, 180, 0), 1, cv2.LINE_AA)
    for name, pt in (("L1n", detection.line1_near), ("L1f", detection.line1_far),
                     ("L2n", detection.line2_near), ("L2f", detection.line2_far)):
      cv2.drawMarker(canvas, _ipt(pt), (255, 255, 255), cv2.MARKER_CROSS, 10, 2)
      cv2.putText(canvas, name, (_ipt(pt)[0] + 6, _ipt(pt)[1] - 6),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

  if result is not None:
    for i, (img_pt, world_pt) in enumerate(zip(result.image_points, result.world_points)):
      proj_pt, _ = cv2.projectPoints(world_pt.reshape(1, 3), rvec, tvec, K, dist)
      proj_pt = proj_pt.reshape(2)
      err = float(np.linalg.norm(proj_pt - img_pt))
      if i < 4 or i % 8 == 0:
        cv2.putText(canvas, f"{err:.1f}", (_ipt(img_pt)[0] + 4, _ipt(img_pt)[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

  return canvas


def draw_extrinsics_overlay_bgr(
  canvas_bgr: np.ndarray, extrinsics: Extrinsics, K: np.ndarray, dist: np.ndarray,
  ol_cfg: ObjectLocalizationConfig,
) -> None:
  """Draw the mat geometry overlay directly onto a BGR canvas in-place.

  Same lines as :func:`render_extrinsics_overlay` but without the
  annotation-time decorations (per-click markers, per-point reprojection
  error labels): just the two painted lines, the world circle, and the
  mat border. Cheap enough to run every frame in the live runner.
  """
  import cv2

  rvec, _ = cv2.Rodrigues(extrinsics.R)
  tvec = extrinsics.t.reshape(3)

  line_world = line_world_points(ol_cfg.mat_line_length_mm,
                                 ol_cfg.mat_line_separation_mm)
  line_proj, _ = cv2.projectPoints(line_world, rvec, tvec, K, dist)
  line_proj = line_proj.reshape(-1, 2)
  cv2.line(canvas_bgr, _ipt(line_proj[0]), _ipt(line_proj[1]), (0, 255, 0), 2)
  cv2.line(canvas_bgr, _ipt(line_proj[2]), _ipt(line_proj[3]), (0, 0, 255), 2)

  circle_world = circle_samples_world(ol_cfg.mat_circle_radius_mm, 256)
  circle_proj, _ = cv2.projectPoints(circle_world, rvec, tvec, K, dist)
  circle_proj = circle_proj.reshape(-1, 2)
  pts = np.array([_ipt(p) for p in circle_proj], dtype=np.int32)
  cv2.polylines(canvas_bgr, [pts], isClosed=True, color=(0, 165, 255), thickness=2)

  border_world = _mat_border_samples_world(ol_cfg, n_per_edge=32)
  border_proj, _ = cv2.projectPoints(border_world, rvec, tvec, K, dist)
  border_proj = border_proj.reshape(-1, 2)
  border_pts = np.array([_ipt(p) for p in border_proj], dtype=np.int32)
  cv2.polylines(canvas_bgr, [border_pts], isClosed=True, color=(80, 255, 80), thickness=2)


def render_world_axes(
  frame_rgb: np.ndarray, extrinsics: Extrinsics, K: np.ndarray, dist: np.ndarray,
  axis_mm: float = 100.0,
) -> np.ndarray:
  """Draw a world-frame triad at the origin."""
  import cv2
  rvec, _ = cv2.Rodrigues(extrinsics.R)
  tvec = extrinsics.t.reshape(3)
  canvas = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
  pts3d = np.array([
    [0, 0, 0],
    [axis_mm, 0, 0],
    [0, axis_mm, 0],
    [0, 0, axis_mm],
  ], dtype=np.float64)
  proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
  proj = proj.reshape(-1, 2)
  origin = _ipt(proj[0])
  cv2.arrowedLine(canvas, origin, _ipt(proj[1]), (0, 0, 255), 2, tipLength=0.2)
  cv2.arrowedLine(canvas, origin, _ipt(proj[2]), (0, 255, 0), 2, tipLength=0.2)
  cv2.arrowedLine(canvas, origin, _ipt(proj[3]), (255, 0, 0), 2, tipLength=0.2)
  cv2.putText(canvas, "+X", _ipt(proj[1] + np.array([6, -6])),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
  cv2.putText(canvas, "+Y", _ipt(proj[2] + np.array([6, -6])),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
  cv2.putText(canvas, "+Z", _ipt(proj[3] + np.array([6, -6])),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
  return canvas


def _ipt(p: np.ndarray) -> tuple[int, int]:
  return int(round(float(p[0]))), int(round(float(p[1])))


def _mat_border_samples_world(ol_cfg: ObjectLocalizationConfig,
                               n_per_edge: int = 32) -> np.ndarray:
  """Sample the configured mat bbox as a closed polyline at z=0.

  Returns ``(4 * n_per_edge, 3)`` world points. Sampling along each
  edge means wide-angle lens distortion shows up as a curved
  projection instead of straight chords between the 4 corners.
  """
  x0 = ol_cfg.mat_extent_xmin_mm
  x1 = ol_cfg.mat_extent_xmax_mm
  y0 = ol_cfg.mat_extent_ymin_mm
  y1 = ol_cfg.mat_extent_ymax_mm
  ts = np.linspace(0.0, 1.0, n_per_edge, endpoint=False)
  pts: list[np.ndarray] = []
  # (x0, y0) -> (x1, y0)
  pts.append(np.stack([x0 + ts * (x1 - x0), np.full_like(ts, y0)], axis=1))
  # (x1, y0) -> (x1, y1)
  pts.append(np.stack([np.full_like(ts, x1), y0 + ts * (y1 - y0)], axis=1))
  # (x1, y1) -> (x0, y1)
  pts.append(np.stack([x1 + ts * (x0 - x1), np.full_like(ts, y1)], axis=1))
  # (x0, y1) -> (x0, y0)
  pts.append(np.stack([np.full_like(ts, x0), y1 + ts * (y0 - y1)], axis=1))
  arr = np.concatenate(pts, axis=0)
  return np.concatenate([arr, np.zeros((arr.shape[0], 1))], axis=1).astype(np.float64)
