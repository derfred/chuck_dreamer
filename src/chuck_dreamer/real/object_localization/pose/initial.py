"""Resting-face hypothesis + initial pose from a mask + calibration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..geometry import ray_plane_intersection_z0
from .mesh import ObjectMesh

logger = logging.getLogger(__name__)


@dataclass
class RestingFaceHypothesis:
  R_face: np.ndarray   # 3x3, rotates object frame so chosen face is at -Z
  z_center: float      # mm — z of the mesh centroid in object frame after rotation
  face_type: str       # "square" or "triangle"
  face_index: int      # representative face id within the symmetry class


def _face_normal(mesh: ObjectMesh, face_idx: int) -> np.ndarray:
  f = mesh.faces[face_idx]
  v0, v1, v2 = mesh.vertices_mm[f]
  return np.asarray(np.cross(v1 - v0, v2 - v0))


def _representative_face_indices(mesh: ObjectMesh) -> tuple[int, int]:
  """Return (square_face_idx, triangle_face_idx) — one face per resting class.

  We classify resting-face hypotheses by **outward normal direction**:
  picking one face whose normal points roughly along a coordinate axis
  (a "square" of the rhombicuboctahedron, normal ≈ ±x/±y/±z) and one
  whose normal points along the body diagonal (a "triangle", normal
  ≈ ±(1,1,1)/√3). This is more robust than relying on the mesh's
  facet grouping, which sometimes merges coplanar tris in surprising
  ways.
  """
  square_target   = np.array([0.0, 0.0, 1.0])
  triangle_target = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)

  best_sq_idx, best_sq_score = -1, -1.0
  best_tri_idx, best_tri_score = -1, -1.0
  for i in range(len(mesh.faces)):
    n = _face_normal(mesh, i)
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
      continue
    n = n / norm
    sq_score  = float(abs(n @ square_target))
    tri_score = float(abs(n @ triangle_target))
    if sq_score > best_sq_score:
      best_sq_score, best_sq_idx = sq_score, i
    if tri_score > best_tri_score:
      best_tri_score, best_tri_idx = tri_score, i

  if best_sq_idx < 0 or best_tri_idx < 0:
    raise RuntimeError("could not pick representative square/triangle faces")
  return best_sq_idx, best_tri_idx


def _rotation_to_align_normal_to_neg_z(normal: np.ndarray) -> np.ndarray:
  """Return R such that R @ normal == [0, 0, -1]."""
  n = normal / max(np.linalg.norm(normal), 1e-9)
  target = np.array([0.0, 0.0, -1.0])
  v = np.cross(n, target)
  s = float(np.linalg.norm(v))
  c = float(np.dot(n, target))
  if s < 1e-9:
    return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
  vx = np.array([[ 0,   -v[2],  v[1]],
                 [ v[2],   0,  -v[0]],
                 [-v[1], v[0],    0 ]])
  return np.asarray(np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s)))


def build_resting_hypotheses(mesh: ObjectMesh) -> list[RestingFaceHypothesis]:
  """Return one hypothesis per resting class (square, triangle).

  For each chosen representative face: compute the rotation that points
  its outward normal at -Z (so the face touches the mat plane in the
  object frame). The mesh centroid's z after rotation is what we put at
  z_center to keep the object sitting on Z=0 in world coords.
  """
  sq_idx, tri_idx = _representative_face_indices(mesh)
  hypotheses: list[RestingFaceHypothesis] = []
  for face_type, face_idx in (("square", sq_idx), ("triangle", tri_idx)):
    f = mesh.faces[face_idx]
    v0, v1, v2 = mesh.vertices_mm[f]
    normal = np.cross(v1 - v0, v2 - v0)
    R_face = _rotation_to_align_normal_to_neg_z(normal)
    rotated = (R_face @ mesh.vertices_mm.T).T
    face_z = float(rotated[f].mean(axis=0)[2])
    centroid = rotated.mean(axis=0)
    z_center = float(centroid[2] - face_z)  # raise the mesh so face sits at z=0
    hypotheses.append(RestingFaceHypothesis(
      R_face=R_face, z_center=z_center,
      face_type=face_type, face_index=face_idx,
    ))
  return hypotheses


def initial_translation_from_mask(
  mask: np.ndarray,
  K:    np.ndarray, dist: np.ndarray,
  R_cw: np.ndarray, t_cw: np.ndarray,
  z_center: float,
) -> np.ndarray:
  """Back-project the mask centroid onto the plane ``Z = z_center``.

  Implementation note: we back-project onto ``Z = 0`` and then translate
  along the ray to the desired height. That keeps a single code path
  (the existing ``ray_plane_intersection_z0``).
  """
  ys, xs = np.where(mask)
  if ys.size == 0:
    return np.array([0.0, 0.0, z_center])
  uv = np.array([float(xs.mean()), float(ys.mean())])
  xy0 = ray_plane_intersection_z0(K, R_cw, t_cw, uv[None, :], dist=dist).reshape(3)

  # Walk from camera center along the ray to z = z_center.
  cam_center = -R_cw.T @ t_cw.reshape(3)
  ray_dir = xy0 - cam_center
  ray_dir /= max(np.linalg.norm(ray_dir), 1e-9)
  if abs(ray_dir[2]) < 1e-6:
    return np.array([xy0[0], xy0[1], z_center])
  s = (z_center - cam_center[2]) / ray_dir[2]
  pt = cam_center + s * ray_dir
  return np.array([float(pt[0]), float(pt[1]), float(z_center)])


def yaw_rotation(yaw: float) -> np.ndarray:
  c, s = np.cos(yaw), np.sin(yaw)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
