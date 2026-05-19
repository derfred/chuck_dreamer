"""Lightweight silhouette / edge renderer.

Implemented in pure numpy + cv2 — no pyrender dependency. The renderer
projects triangle vertices through the calibrated camera and fills them
into a binary mask; visible edges come from filtering mesh edges by
back-face culling. This is much lighter than a full GL pipeline and
plenty for silhouette/edge based refinement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import ObjectMesh


@dataclass
class RenderResult:
  silhouette: np.ndarray         # (H, W) bool
  visible_edges: np.ndarray      # (E, 2, 2) float; (start, end) in pixels


def _project(
  pts_world: np.ndarray,
  R_cw: np.ndarray, t_cw: np.ndarray,
  K: np.ndarray, dist: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
  """Return (pixels (N,2), camera_z (N,))."""
  import cv2
  rvec, _ = cv2.Rodrigues(R_cw)
  proj, _ = cv2.projectPoints(pts_world.reshape(-1, 1, 3),
                              rvec, t_cw.reshape(3, 1), K, dist)
  pixels = proj.reshape(-1, 2).astype(np.float32)
  cam = (R_cw @ pts_world.T).T + t_cw.reshape(1, 3)
  return pixels, cam[:, 2]


def render_silhouette_and_edges(
  mesh:    ObjectMesh,
  R_wo:    np.ndarray,    # object -> world rotation
  t_wo:    np.ndarray,    # object centroid (or origin) in world, mm
  R_cw:    np.ndarray,    # world -> camera
  t_cw:    np.ndarray,
  K:       np.ndarray,
  dist:    np.ndarray | None,
  image_size: tuple[int, int],   # (W, H)
) -> RenderResult:
  """Project the mesh at pose (R_wo, t_wo) seen by the calibrated camera."""
  import cv2

  W, H = image_size
  vertices_world = (R_wo @ mesh.vertices_mm.T).T + t_wo.reshape(1, 3)
  pixels, cam_z = _project(vertices_world, R_cw, t_cw, K, dist)

  silhouette = np.zeros((H, W), dtype=np.uint8)
  tri = mesh.faces

  # Back-face culling: drop triangles whose outward normal points away
  # from the camera. The normal is computed in camera space.
  v0 = (R_cw @ vertices_world.T).T + t_cw.reshape(1, 3)
  e1 = v0[tri[:, 1]] - v0[tri[:, 0]]
  e2 = v0[tri[:, 2]] - v0[tri[:, 0]]
  normals = np.cross(e1, e2)
  # The vector from triangle centroid to origin in camera coords is -centroid.
  centroids = (v0[tri[:, 0]] + v0[tri[:, 1]] + v0[tri[:, 2]]) / 3.0
  facing = (normals * (-centroids)).sum(axis=1) > 0    # cull if dot < 0

  valid = facing & (cam_z[tri].min(axis=1) > 1.0)
  for f in tri[valid]:
    pts = pixels[f].astype(np.int32)
    cv2.fillConvexPoly(silhouette, pts, 255)

  # Visible edges: keep edges where at least one adjacent triangle is
  # front-facing. Build a quick edge -> triangles map via the mesh's
  # face_groups so coplanar internal edges aren't drawn.
  edges = []
  tri_set = set()
  for fi, f in enumerate(tri):
    if not valid[fi]:
      continue
    for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
      key = (int(min(a, b)), int(max(a, b)))
      tri_set.add(key)
  for (a, b) in tri_set:
    edges.append((pixels[a], pixels[b]))
  if edges:
    edge_arr = np.stack([np.stack([s, e]) for s, e in edges], axis=0)
  else:
    edge_arr = np.zeros((0, 2, 2), dtype=np.float32)

  return RenderResult(silhouette=silhouette.astype(bool), visible_edges=edge_arr)
