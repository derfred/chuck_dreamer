"""Lightweight CPU mesh projection used by the pose estimator.

We don't need a full off-screen GL pipeline for what the estimator does:

  * ``project_silhouette`` rasterizes only the front-facing triangles into
    a binary mask, then returns its outer contour. That's what the
    chamfer-distance silhouette loss in the spec needs.
  * ``project_visible_edges`` projects the edges shared by exactly two
    front-facing triangles (i.e. the visible mesh wireframe), which is
    what the RAPID-style normal-search refinement needs.

Both helpers do their own back-face culling against the world->camera
extrinsics; no GL state, no depth buffer, no shader. This trades fidelity
for portability — good enough for the rhombicuboctahedron geometry the
spec targets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
  K: np.ndarray
  dist: np.ndarray
  R_world_cam: np.ndarray
  t_world_cam: np.ndarray   # (3,) world->camera translation in mm
  image_size: tuple[int, int]   # (W, H)


def transform_object(vertices_mm: np.ndarray, R_world_obj: np.ndarray,
                     xyz_mm: np.ndarray) -> np.ndarray:
  """Apply object->world transform to the mesh vertices."""
  return np.asarray((R_world_obj @ vertices_mm.T).T + xyz_mm.reshape(1, 3))


def project_points(pts_world: np.ndarray, cam: Camera) -> np.ndarray:
  import cv2
  rvec, _ = cv2.Rodrigues(cam.R_world_cam)
  proj, _ = cv2.projectPoints(pts_world, rvec, cam.t_world_cam.reshape(3),
                              cam.K, cam.dist)
  return proj.reshape(-1, 2)


def visible_face_mask(vertices_world: np.ndarray, triangles: np.ndarray,
                      cam: Camera) -> np.ndarray:
  """``(T,) bool`` mask: True for triangles whose outward normal faces the camera."""
  v0 = vertices_world[triangles[:, 0]]
  v1 = vertices_world[triangles[:, 1]]
  v2 = vertices_world[triangles[:, 2]]
  normals = np.cross(v1 - v0, v2 - v0)
  centers = (v0 + v1 + v2) / 3.0
  cam_pos_world = -cam.R_world_cam.T @ cam.t_world_cam.reshape(3)
  to_cam = cam_pos_world.reshape(1, 3) - centers
  return np.asarray(np.sum(normals * to_cam, axis=1) > 0)


def project_silhouette(vertices_world: np.ndarray, triangles: np.ndarray,
                       cam: Camera) -> tuple[np.ndarray, np.ndarray | None]:
  """Render the front-facing-triangle silhouette.

  Returns ``(mask, contour)`` where ``mask`` is ``HxW uint8`` and
  ``contour`` is the longest outer contour (``Nx2`` float64) or ``None``.
  """
  import cv2
  W, H = cam.image_size
  uvs = project_points(vertices_world, cam)
  visible = visible_face_mask(vertices_world, triangles, cam)
  mask = np.zeros((H, W), dtype=np.uint8)
  for tri_idx in np.where(visible)[0]:
    pts = uvs[triangles[tri_idx]].astype(np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(mask, pts, 255)
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  if not contours:
    return mask, None
  contour = max(contours, key=len)
  return mask, contour.reshape(-1, 2).astype(np.float64)


def project_visible_edges(vertices_world: np.ndarray, triangles: np.ndarray,
                          cam: Camera) -> list[tuple[np.ndarray, np.ndarray]]:
  """Edges shared by visible (front-facing) triangles.

  Returns a list of ``(p0_uv, p1_uv)`` segments in image space, dedup'd
  so each edge appears at most once.
  """
  visible = visible_face_mask(vertices_world, triangles, cam)
  uvs = project_points(vertices_world, cam)
  seen: set[tuple[int, int]] = set()
  out: list[tuple[np.ndarray, np.ndarray]] = []
  for tri_idx in np.where(visible)[0]:
    a, b, c = (int(x) for x in triangles[tri_idx])
    for i, j in ((a, b), (b, c), (c, a)):
      key = (min(i, j), max(i, j))
      if key in seen:
        continue
      seen.add(key)
      out.append((uvs[i], uvs[j]))
  return out
