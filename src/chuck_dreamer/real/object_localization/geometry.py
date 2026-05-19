"""World-frame geometry: known 3D points on the mat, ray-plane intersection.

The world frame is defined in the spec:

* origin at the white circle's center
* +X points away from the lines
* +Z points up out of the mat plane
* +Y is fixed by right-hand rule (Y = Z × X)

Line 1 sits at ``y = +D/2`` (bottom of the image given the camera setup);
Line 2 sits at ``y = -D/2``. The lines extend from ``x = 0`` (near end,
beside the circle) to ``x = -L`` (far end).
"""

from __future__ import annotations

import numpy as np

from . import constants


def line_endpoints_world() -> dict[str, np.ndarray]:
  """Return the 4 line endpoints as a dict of (3,) arrays."""
  L = constants.mat_line_length_mm()
  D = constants.mat_line_separation_mm()
  return {
    "line1_near": np.array([ 0.0,  +D / 2, 0.0], dtype=np.float64),
    "line1_far":  np.array([-L,    +D / 2, 0.0], dtype=np.float64),
    "line2_near": np.array([ 0.0,  -D / 2, 0.0], dtype=np.float64),
    "line2_far":  np.array([-L,    -D / 2, 0.0], dtype=np.float64),
  }


def circle_points_world(n: int = 32) -> np.ndarray:
  """Return ``n`` evenly spaced points on the mat circle, shape (n, 3)."""
  r = constants.mat_circle_radius_mm()
  theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
  pts = np.zeros((n, 3), dtype=np.float64)
  pts[:, 0] = r * np.cos(theta)
  pts[:, 1] = r * np.sin(theta)
  return pts


def ray_plane_intersection_z0(
  K:    np.ndarray,
  R:    np.ndarray,    # world -> camera
  t:    np.ndarray,    # world -> camera, shape (3,) or (3, 1)
  uv:   np.ndarray,    # pixel coordinates, shape (..., 2)
  dist: np.ndarray | None = None,
) -> np.ndarray:
  """Back-project pixel(s) ``uv`` onto the world plane ``Z = 0``.

  Returns world-frame ``(x, y, z=0)`` points with the same leading
  shape as ``uv``. If ``dist`` is provided, the pixels are first
  undistorted with :func:`cv2.undistortPoints` so the back-projection
  matches the physical camera model.
  """
  import cv2

  uv_arr = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2).astype(np.float32)
  if dist is not None and np.any(dist):
    norm = cv2.undistortPoints(uv_arr, K, dist).reshape(-1, 2)
  else:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    norm = np.stack([(uv_arr.reshape(-1, 2)[:, 0] - cx) / fx,
                     (uv_arr.reshape(-1, 2)[:, 1] - cy) / fy], axis=-1)
  rays_cam = np.concatenate([norm, np.ones((norm.shape[0], 1), dtype=norm.dtype)], axis=-1)

  R = np.asarray(R, dtype=np.float64)
  t = np.asarray(t, dtype=np.float64).reshape(3)
  cam_center_world = -R.T @ t                            # (3,)
  rays_world       = (R.T @ rays_cam.T).T                # (N, 3)

  # Intersect each world ray with z = 0:  cam_center + s * ray = (x, y, 0)
  denom = rays_world[:, 2]
  safe  = np.where(np.abs(denom) < 1e-12, np.nan, denom)
  s     = -cam_center_world[2] / safe
  pts   = cam_center_world[None, :] + s[:, None] * rays_world
  pts[:, 2] = 0.0
  return np.asarray(pts.reshape(*np.shape(uv)[:-1], 3))


def camera_center_world(R: np.ndarray, t: np.ndarray) -> np.ndarray:
  """Return the camera center in world coordinates: ``-R.T @ t``."""
  R = np.asarray(R, dtype=np.float64)
  t = np.asarray(t, dtype=np.float64).reshape(3)
  return np.asarray(-R.T @ t)
