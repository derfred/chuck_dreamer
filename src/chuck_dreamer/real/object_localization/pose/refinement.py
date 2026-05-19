"""Render-and-compare pose refinement.

Two stages, both via :func:`scipy.optimize.least_squares` over
``(x, y, yaw)`` with z and tilt fixed by the resting-face hypothesis:

  1. **Silhouette** stage — minimize a Chamfer-style residual between
     the observed mask boundary and the rendered silhouette boundary.
  2. **Edge** stage — minimize the perpendicular distance from each
     rendered visible edge to the nearest image edge along the rendered
     edge's normal direction (RAPID-style).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from .initial import RestingFaceHypothesis, yaw_rotation
from .mesh import ObjectMesh
from .rendering import render_silhouette_and_edges

logger = logging.getLogger(__name__)


@dataclass
class RefinedPose:
  R_wo: np.ndarray      # 3x3, object frame -> world frame
  t_wo: np.ndarray      # (3,) — world coords (mm)
  residual: float
  reprojection_error_px: float


def _boundary_points(mask: np.ndarray, n_samples: int | None = None) -> np.ndarray:
  """Return (N, 2) pixel coordinates on the silhouette boundary.

  If ``n_samples`` is given, points are uniformly resampled along the
  concatenated contour arc so callers can request a fixed-size output
  (required by Levenberg-Marquardt, which assumes a constant residual
  shape across iterations).
  """
  import cv2
  m = mask.astype(np.uint8)
  if not m.any():
    if n_samples is not None:
      return np.zeros((n_samples, 2), dtype=np.float32)
    return np.zeros((0, 2), dtype=np.float32)
  contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  if not contours:
    if n_samples is not None:
      return np.zeros((n_samples, 2), dtype=np.float32)
    return np.zeros((0, 2), dtype=np.float32)
  pts = np.concatenate([c.reshape(-1, 2) for c in contours], axis=0).astype(np.float32)
  if n_samples is None or pts.shape[0] == 0:
    return pts
  if pts.shape[0] >= n_samples:
    idx = np.linspace(0, pts.shape[0] - 1, n_samples).astype(int)
    return np.asarray(pts[idx])
  # Repeat to fill (very small contour)
  reps = (n_samples + pts.shape[0] - 1) // pts.shape[0]
  return np.asarray(np.tile(pts, (reps, 1))[:n_samples])


def _distance_transform(mask: np.ndarray) -> np.ndarray:
  """Distance to the boundary of ``mask`` (signed: positive outside).

  Returned as float32 with a tiny Gaussian blur — the blur smooths the
  step-function discontinuities at the boundary so finite-difference
  gradients used by LM are well-behaved.
  """
  import cv2
  m = mask.astype(np.uint8) * 255
  if m.any():
    inv = 255 - m
    d_out = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    d_in  = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    field = (d_out - d_in).astype(np.float32)
  else:
    H, W = mask.shape
    field = np.full((H, W), float(max(H, W)), dtype=np.float32)
  return cv2.GaussianBlur(field, (5, 5), 1.5)


_BOUNDARY_SAMPLES = 200


def _silhouette_residual(
  params: np.ndarray,
  *,
  mesh:         ObjectMesh,
  hypothesis:   RestingFaceHypothesis,
  cal_K:        np.ndarray,
  cal_dist:     np.ndarray,
  cal_R_cw:     np.ndarray,
  cal_t_cw:     np.ndarray,
  image_size:   tuple[int, int],
  obs_dt:       np.ndarray,           # distance transform of observed mask
  mesh_dt_pts:  np.ndarray,           # observed boundary points (_BOUNDARY_SAMPLES, 2)
) -> np.ndarray:
  x, y, yaw = params
  R_wo = yaw_rotation(yaw) @ hypothesis.R_face
  t_wo = np.array([float(x), float(y), float(hypothesis.z_center)])
  rendered = render_silhouette_and_edges(
    mesh, R_wo, t_wo, cal_R_cw, cal_t_cw, cal_K, cal_dist, image_size,
  )

  rendered_boundary = _boundary_points(rendered.silhouette, n_samples=_BOUNDARY_SAMPLES)
  if rendered_boundary.shape[0] == 0 or not rendered.silhouette.any():
    # Punish: large constant residual of the expected length.
    return np.full((2 * _BOUNDARY_SAMPLES,), 50.0, dtype=np.float32)

  # mesh boundary -> observed dt
  ri = np.clip(rendered_boundary[:, 1].astype(int), 0, obs_dt.shape[0] - 1)
  rj = np.clip(rendered_boundary[:, 0].astype(int), 0, obs_dt.shape[1] - 1)
  rendered_to_obs = np.abs(obs_dt[ri, rj])

  # observed boundary -> rendered dt
  rendered_dt = _distance_transform(rendered.silhouette)
  oi = np.clip(mesh_dt_pts[:, 1].astype(int), 0, rendered_dt.shape[0] - 1)
  oj = np.clip(mesh_dt_pts[:, 0].astype(int), 0, rendered_dt.shape[1] - 1)
  obs_to_rendered = np.abs(rendered_dt[oi, oj])

  return np.asarray(np.concatenate([rendered_to_obs, obs_to_rendered]).astype(np.float32))


def refine_silhouette(
  mesh:        ObjectMesh,
  hypothesis:  RestingFaceHypothesis,
  init_xy:     tuple[float, float],
  init_yaw:    float,
  observed_mask: np.ndarray,
  cal_K:       np.ndarray,
  cal_dist:    np.ndarray,
  cal_R_cw:    np.ndarray,
  cal_t_cw:    np.ndarray,
  image_size:  tuple[int, int],
) -> RefinedPose:
  """Levenberg-Marquardt over (x, y, yaw) using a Chamfer-style residual."""
  obs_dt      = _distance_transform(observed_mask)
  obs_boundary = _boundary_points(observed_mask, n_samples=_BOUNDARY_SAMPLES)
  x0 = np.array([init_xy[0], init_xy[1], init_yaw], dtype=np.float64)

  # Coarse-to-fine schedule: each pass uses a larger numerical step so LM
  # can escape local plateaus before fine-tuning. Step sizes are in the
  # native units of each parameter (mm, mm, rad).
  x_cur = np.array([init_xy[0], init_xy[1], init_yaw], dtype=np.float64)
  kwargs = {
    "mesh": mesh, "hypothesis": hypothesis,
    "cal_K": cal_K, "cal_dist": cal_dist,
    "cal_R_cw": cal_R_cw, "cal_t_cw": cal_t_cw,
    "image_size": image_size,
    "obs_dt": obs_dt, "mesh_dt_pts": obs_boundary,
  }
  res = None
  for diff_step in (3e-2, 5e-3):
    res = least_squares(
      _silhouette_residual, x_cur, method="lm",
      max_nfev=120, diff_step=diff_step, kwargs=kwargs,
    )
    x_cur = res.x
  assert res is not None
  x, y, yaw = res.x
  R_wo = yaw_rotation(float(yaw)) @ hypothesis.R_face
  t_wo = np.array([float(x), float(y), float(hypothesis.z_center)])
  return RefinedPose(
    R_wo=R_wo, t_wo=t_wo,
    residual=float(np.sqrt(np.mean(res.fun ** 2))),
    reprojection_error_px=float(np.median(np.abs(res.fun))),
  )


_EDGE_SAMPLES_PER_FIT = 600


def _sample_edges(edges: np.ndarray, n_total: int) -> np.ndarray:
  """Return ``(n_total, 2)`` points sampled along the given edges.

  Distribution: split the budget by edge length so longer edges get
  more samples. When ``edges`` is empty, returns a zero-filled buffer
  so the LM residual still has the expected length.
  """
  if edges.shape[0] == 0:
    return np.zeros((n_total, 2), dtype=np.float32)
  lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
  total = float(lengths.sum())
  if total < 1e-6:
    return np.tile(edges[0, 0][None, :], (n_total, 1)).astype(np.float32)
  per = np.maximum((lengths / total * n_total).astype(int), 1)
  samples: list[np.ndarray] = []
  for (s, e), k in zip(edges, per):
    ts = np.linspace(0.0, 1.0, int(k))[:, None]
    samples.append(s[None, :] * (1 - ts) + e[None, :] * ts)
  out = np.concatenate(samples, axis=0)
  if out.shape[0] >= n_total:
    idx = np.linspace(0, out.shape[0] - 1, n_total).astype(int)
    return np.asarray(out[idx].astype(np.float32))
  pad = np.tile(out[-1:], (n_total - out.shape[0], 1))
  return np.asarray(np.concatenate([out, pad], axis=0).astype(np.float32))


def _edge_residual(
  params:      np.ndarray,
  *,
  mesh:        ObjectMesh,
  hypothesis:  RestingFaceHypothesis,
  cal_K:       np.ndarray,
  cal_dist:    np.ndarray,
  cal_R_cw:    np.ndarray,
  cal_t_cw:    np.ndarray,
  image_size:  tuple[int, int],
  image_edge_dt: np.ndarray,
) -> np.ndarray:
  x, y, yaw = params
  R_wo = yaw_rotation(yaw) @ hypothesis.R_face
  t_wo = np.array([float(x), float(y), float(hypothesis.z_center)])
  rendered = render_silhouette_and_edges(
    mesh, R_wo, t_wo, cal_R_cw, cal_t_cw, cal_K, cal_dist, image_size,
  )
  pts = _sample_edges(rendered.visible_edges, _EDGE_SAMPLES_PER_FIT)
  ii = np.clip(pts[:, 1].astype(int), 0, image_edge_dt.shape[0] - 1)
  jj = np.clip(pts[:, 0].astype(int), 0, image_edge_dt.shape[1] - 1)
  return np.asarray(image_edge_dt[ii, jj].astype(np.float32))


def refine_edges(
  mesh:       ObjectMesh,
  hypothesis: RestingFaceHypothesis,
  prev:       RefinedPose,
  image:      np.ndarray,
  cal_K:      np.ndarray,
  cal_dist:   np.ndarray,
  cal_R_cw:   np.ndarray,
  cal_t_cw:   np.ndarray,
  image_size: tuple[int, int],
  canny_lo:   int = 60,
  canny_hi:   int = 180,
) -> RefinedPose:
  """Refine pose by minimizing the distance from rendered edges to image edges."""
  import cv2
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  edges = cv2.Canny(gray, canny_lo, canny_hi)
  inv = 255 - edges
  image_edge_dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

  yaw0 = float(np.arctan2(prev.R_wo[1, 0], prev.R_wo[0, 0]))
  x0 = np.array([prev.t_wo[0], prev.t_wo[1], yaw0], dtype=np.float64)

  res = least_squares(
    _edge_residual, x0, method="lm", max_nfev=80,
    kwargs={
      "mesh": mesh, "hypothesis": hypothesis,
      "cal_K": cal_K, "cal_dist": cal_dist,
      "cal_R_cw": cal_R_cw, "cal_t_cw": cal_t_cw,
      "image_size": image_size,
      "image_edge_dt": image_edge_dt,
    },
  )
  x, y, yaw = res.x
  R_wo = yaw_rotation(float(yaw)) @ hypothesis.R_face
  t_wo = np.array([float(x), float(y), float(hypothesis.z_center)])
  return RefinedPose(
    R_wo=R_wo, t_wo=t_wo,
    residual=float(np.sqrt(np.mean(res.fun ** 2))),
    reprojection_error_px=float(np.median(np.abs(res.fun))),
  )
