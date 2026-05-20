"""Per-frame object pose estimation.

Pipeline per frame (see ``object_localization_spec.md`` Deliverable 3):

  1. Segment with SAM2 (or color-heuristic fallback).
  2. Back-project mask centroid to the mat plane Z=0 as the seed (x, y).
  3. First frame of an episode: search 2 rest classes x N yaw inits and
     keep the lowest silhouette residual.
  4. Silhouette refinement via bidirectional chamfer over (x, y, yaw).
  5. Edge-based refinement via RAPID-style 1D normal search + LM.
  6. Build the (x, y, yaw) -> world transform and return.

The expensive bit is the per-iteration mesh rasterization, which is done
on CPU here. For datasets with hundreds of frames per episode that is
slow but tractable; SAM2 dominates wall-clock anyway. If you need a
real-time pipeline replace ``render.project_silhouette`` with a GL
rasterizer.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import numpy as np

from .mesh import Mesh, load_mesh
from .render import (
  Camera,
  project_silhouette,
  project_visible_edges,
  transform_object,
)
from .segment import segment_image, segment_video
from .types import CameraCalibration

logger = logging.getLogger(__name__)


_RestingFaceClass = Literal["square", "triangle"]


@dataclass
class ObjectPose:
  xy_mm: np.ndarray              # (2,) world-frame (x, y) on the mat
  xyz_mm: np.ndarray             # (3,) world-frame (x, y, z); z reflects resting face
  R_world_obj: np.ndarray        # (3, 3) object -> world rotation
  resting_face_class: _RestingFaceClass
  confidence: float              # [0, 1]
  reprojection_error_px: float

  yaw_rad: float = 0.0
  tilt_rad: float = 0.0          # tilt angle baked into R_world_obj


@dataclass
class _RestHypothesis:
  cls: _RestingFaceClass
  z_mm: float                    # face-to-centroid distance for that face
  R_tilt: np.ndarray             # 3x3 rotation that orients the resting face down


@dataclass
class _EpisodeContext:
  resting: _RestHypothesis | None = None


class ObjectPoseEstimator:
  """Stateful per-frame estimator. Construct once per camera + dataset."""

  def __init__(
    self,
    calibration: CameraCalibration,
    mesh_path: Path,
    config: dict | None = None,
    device: str = "cuda",
    use_sam2: bool | None = None,
  ) -> None:
    self.calibration = calibration
    self.mesh        = load_mesh(Path(mesh_path))
    self.device      = device

    config = dict(config or {})
    self.sam2_checkpoint = str(config.get("sam2_checkpoint", "facebook/sam2-hiera-large"))
    self.use_sam2        = bool(use_sam2) if use_sam2 is not None else bool(config.get("use_sam2", True))
    self.n_yaw_inits     = int(config.get("n_yaw_inits", 4))
    self.chamfer_max_iter = int((config.get("silhouette") or {}).get("chamfer_max_iter", 100))
    edges_cfg = config.get("edges") or {}
    self.canny_low       = int(edges_cfg.get("canny_low", 50))
    self.canny_high      = int(edges_cfg.get("canny_high", 150))
    self.edges_search_px = int(edges_cfg.get("search_range_px", 25))
    self.edges_max_iter  = int(edges_cfg.get("max_iter", 50))

    self.camera = Camera(
      K           = calibration.intrinsics.K,
      dist        = calibration.intrinsics.dist,
      R_world_cam = calibration.extrinsics.R,
      t_world_cam = calibration.extrinsics.t.reshape(3),
      image_size  = calibration.intrinsics.image_size,
    )

    self._rest_classes = _build_rest_classes(self.mesh)
    self._episode = _EpisodeContext()

  # -------------------------------------------------------------------------
  # Public API
  # -------------------------------------------------------------------------
  def estimate(
    self,
    image: np.ndarray,
    prev_pose: ObjectPose | None = None,
    prompt: tuple[int, int] | list[tuple[int, int]] | None = None,
  ) -> ObjectPose | None:
    """Estimate the pose in a single frame."""
    if prompt is None and prev_pose is None:
      raise ValueError("estimate(): one of `prompt` or `prev_pose` is required.")

    if prev_pose is not None:
      effective_prompt = prompt if prompt is not None else _pose_to_prompt(prev_pose, self.camera)
    else:
      effective_prompt = prompt
    mask = segment_image(image, effective_prompt, self.use_sam2,
                         self.sam2_checkpoint, self.device)
    if mask is None:
      return None
    return self._fit_pose(image, mask, prev_pose)

  def estimate_episode(
    self,
    frames: Iterable[np.ndarray],
    first_frame_prompt: tuple[int, int] | list[tuple[int, int]] | None = None,
  ) -> list[ObjectPose | None]:
    """Estimate poses across a whole episode (sequence of frames)."""
    frame_list = list(frames)
    if not frame_list:
      return []
    if first_frame_prompt is None:
      raise ValueError("estimate_episode(): first_frame_prompt is required.")

    masks = segment_video(frame_list, first_frame_prompt, self.use_sam2,
                          self.sam2_checkpoint, self.device)

    out: list[ObjectPose | None] = []
    prev: ObjectPose | None = None
    self._episode = _EpisodeContext()
    for img, mask in zip(frame_list, masks):
      if mask is None:
        out.append(None)
        continue
      pose = self._fit_pose(img, mask, prev)
      out.append(pose)
      if pose is not None:
        prev = pose
    return out

  # -------------------------------------------------------------------------
  # Pose fitting
  # -------------------------------------------------------------------------
  def _fit_pose(self, image: np.ndarray, mask: np.ndarray,
                prev_pose: ObjectPose | None) -> ObjectPose | None:
    xy0 = _back_project_centroid(mask, self.camera)
    if xy0 is None:
      return None

    if prev_pose is not None and self._episode.resting is not None:
      rest = self._episode.resting
      yaw  = prev_pose.yaw_rad
      xy   = prev_pose.xy_mm.copy()
    else:
      rest, yaw, xy = self._search_rest_class(image, mask, xy0)
      if rest is None:
        return None
      self._episode.resting = rest

    xy_ref, yaw_ref, sil_rms = self._refine_silhouette(mask, rest, xy, yaw)
    xy_ed, yaw_ed, edge_rms = self._refine_edges(image, mask, rest, xy_ref, yaw_ref)

    return _make_pose(self.mesh, rest, xy_ed, yaw_ed,
                      residual_px=edge_rms if edge_rms is not None else sil_rms,
                      mask=mask, camera=self.camera)

  def _search_rest_class(self, image: np.ndarray, mask: np.ndarray,
                          xy_seed: np.ndarray) -> tuple[_RestHypothesis | None, float, np.ndarray]:
    best: tuple[float, _RestHypothesis, float, np.ndarray] | None = None
    for rest in self._rest_classes:
      for k in range(max(1, self.n_yaw_inits)):
        yaw0 = 2.0 * math.pi * k / max(1, self.n_yaw_inits)
        xy_ref, yaw_ref, residual = self._refine_silhouette(mask, rest, xy_seed, yaw0)
        if residual is None:
          continue
        if best is None or residual < best[0]:
          best = (residual, rest, yaw_ref, xy_ref)
    if best is None:
      return None, 0.0, xy_seed
    _, rest, yaw, xy = best
    return rest, yaw, xy

  def _refine_silhouette(self, mask: np.ndarray, rest: _RestHypothesis,
                          xy0: np.ndarray, yaw0: float
                          ) -> tuple[np.ndarray, float, float | None]:
    from scipy.optimize import least_squares

    obs_contour = _mask_contour(mask)
    if obs_contour is None or len(obs_contour) < 5:
      return xy0, yaw0, None
    obs_dt = _distance_transform(mask, self.camera.image_size)

    def residual(params: np.ndarray) -> np.ndarray:
      x, y, yaw = params
      ren_mask, ren_contour = self._render_silhouette(rest, np.array([x, y]), yaw)
      if ren_contour is None or len(ren_contour) < 5:
        return np.full(64, 1e3, dtype=np.float64)
      ren_dt = _distance_transform_from_contour(ren_contour, self.camera.image_size)
      d_obs_to_ren = ren_dt[obs_contour[:, 1].astype(int), obs_contour[:, 0].astype(int)]
      d_ren_to_obs = obs_dt[ren_contour[:, 1].astype(int), ren_contour[:, 0].astype(int)]
      return np.concatenate([d_obs_to_ren, d_ren_to_obs]).astype(np.float64)

    x0 = np.array([xy0[0], xy0[1], yaw0], dtype=np.float64)
    try:
      sol = least_squares(residual, x0, method="lm",
                          max_nfev=self.chamfer_max_iter, xtol=1e-6, ftol=1e-6)
    except Exception:
      return xy0, yaw0, None
    res_norm = float(np.sqrt(np.mean(sol.fun ** 2))) if sol.fun.size else None
    return np.asarray(sol.x[:2]), float(sol.x[2]), res_norm

  def _refine_edges(self, image: np.ndarray, mask: np.ndarray, rest: _RestHypothesis,
                     xy0: np.ndarray, yaw0: float
                     ) -> tuple[np.ndarray, float, float | None]:
    import cv2
    from scipy.optimize import least_squares

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, self.canny_low, self.canny_high)
    edge_dt = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)

    W, H = self.camera.image_size

    def residual(params: np.ndarray) -> np.ndarray:
      x, y, yaw = params
      edges_proj = self._render_edges(rest, np.array([x, y]), yaw)
      if not edges_proj:
        return np.full(64, 1e3, dtype=np.float64)
      sample_pts: list[np.ndarray] = []
      for p0, p1 in edges_proj:
        n = max(2, int(np.linalg.norm(p1 - p0) / 4.0))
        ts = np.linspace(0.0, 1.0, n)
        seg = p0[None, :] * (1 - ts[:, None]) + p1[None, :] * ts[:, None]
        sample_pts.append(seg)
      pts = np.concatenate(sample_pts, axis=0)
      u = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
      v = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
      return edge_dt[v, u].astype(np.float64)

    x0 = np.array([xy0[0], xy0[1], yaw0], dtype=np.float64)
    try:
      sol = least_squares(residual, x0, method="lm",
                          max_nfev=self.edges_max_iter, xtol=1e-6, ftol=1e-6)
    except Exception:
      return xy0, yaw0, None
    res_norm = float(np.sqrt(np.mean(sol.fun ** 2))) if sol.fun.size else None
    return np.asarray(sol.x[:2]), float(sol.x[2]), res_norm

  # -------------------------------------------------------------------------
  # Rendering helpers
  # -------------------------------------------------------------------------
  def _world_vertices(self, rest: _RestHypothesis, xy: np.ndarray, yaw: float) -> np.ndarray:
    R = _yaw_about_z(yaw) @ rest.R_tilt
    xyz = np.array([xy[0], xy[1], rest.z_mm])
    return transform_object(self.mesh.vertices_mm, R, xyz)

  def _render_silhouette(self, rest: _RestHypothesis, xy: np.ndarray, yaw: float
                          ) -> tuple[np.ndarray, np.ndarray | None]:
    verts = self._world_vertices(rest, xy, yaw)
    return project_silhouette(verts, self.mesh.triangles, self.camera)

  def _render_edges(self, rest: _RestHypothesis, xy: np.ndarray, yaw: float
                     ) -> list[tuple[np.ndarray, np.ndarray]]:
    verts = self._world_vertices(rest, xy, yaw)
    return project_visible_edges(verts, self.mesh.triangles, self.camera)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_rest_classes(mesh: Mesh) -> list[_RestHypothesis]:
  """Enumerate every distinct face-orientation as a rest hypothesis.

  The previous implementation picked a single representative per
  ``cls`` (square / triangle), which is wrong for the
  rhombicuboctahedron: there are 18 square faces split into two
  symmetry classes (6 axis-aligned squares with normals along
  ±X/±Y/±Z, and 12 diagonal squares whose normals point along
  face-diagonals like (±1, ±1, 0)/√2). The two classes have
  different face-to-centroid distances and very different rendered
  silhouettes, so picking only one means the optimizer can never
  recover the other. Same kind of trap exists for any non-uniform
  polyhedron.

  This version returns one hypothesis per coplanar face group. The
  outer pose search iterates them all; the ``cls`` label is kept for
  the public ObjectPose API but no longer collapses the search space.
  """
  centers = (mesh.vertices_mm[mesh.triangles[:, 0]]
             + mesh.vertices_mm[mesh.triangles[:, 1]]
             + mesh.vertices_mm[mesh.triangles[:, 2]]) / 3.0
  centroid = mesh.vertices_mm.mean(axis=0)

  groups = _group_coplanar_faces(mesh.face_normals, centers)
  hypotheses: list[_RestHypothesis] = []
  for g in groups:
    n = mesh.face_normals[g].mean(axis=0)
    n = n / (np.linalg.norm(n) + 1e-9)
    face_center = centers[g].mean(axis=0)
    z_mm = abs(float(np.dot(face_center - centroid, n)))
    R_tilt = _rotation_align(n, np.array([0.0, 0.0, -1.0]))
    cls: _RestingFaceClass = "square" if len(g) >= 2 else "triangle"
    hypotheses.append(_RestHypothesis(cls=cls, z_mm=z_mm, R_tilt=R_tilt))

  # Dedupe by silhouette signature: rotated vertex z-distribution.
  # Two rest hypotheses produce the same silhouette (modulo the yaw
  # rotation the optimizer already searches over) iff the set of vertex
  # heights above the mat plane is the same. Buckets at 0.5mm precision
  # to absorb float noise.
  unique: list[_RestHypothesis] = []
  for h in hypotheses:
    if not any(_same_silhouette_signature(h, u, mesh.vertices_mm)
               for u in unique):
      unique.append(h)
  return unique


def _same_silhouette_signature(a: _RestHypothesis, b: _RestHypothesis,
                                vertices_mm: np.ndarray) -> bool:
  """Return True iff a and b place the mesh in geometrically equivalent
  poses (up to in-plane rotation about world +Z).
  """
  if abs(a.z_mm - b.z_mm) > 0.5:
    return False
  za = np.sort(np.round((a.R_tilt @ vertices_mm.T).T[:, 2], 1))
  zb = np.sort(np.round((b.R_tilt @ vertices_mm.T).T[:, 2], 1))
  if za.shape != zb.shape:
    return False
  return bool(np.allclose(za, zb, atol=0.5))


def _group_coplanar_faces(normals: np.ndarray, centers: np.ndarray) -> list[list[int]]:
  """Group triangles by shared (normal, plane-offset). O(T^2); T is small."""
  T = len(normals)
  used = np.zeros(T, dtype=bool)
  groups: list[list[int]] = []
  for i in range(T):
    if used[i]:
      continue
    members = [i]
    used[i] = True
    n_i = normals[i]
    d_i = float(np.dot(n_i, centers[i]))
    for j in range(i + 1, T):
      if used[j]:
        continue
      if np.dot(n_i, normals[j]) < 0.999:
        continue
      if abs(float(np.dot(n_i, centers[j])) - d_i) > 0.5:
        continue
      members.append(j)
      used[j] = True
    groups.append(members)
  return groups


def _rotation_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
  """Rotation that maps unit ``src`` to unit ``dst``."""
  src = src / (np.linalg.norm(src) + 1e-9)
  dst = dst / (np.linalg.norm(dst) + 1e-9)
  v = np.cross(src, dst)
  c = float(np.dot(src, dst))
  if c > 1.0 - 1e-9:
    return np.eye(3)
  if c < -1.0 + 1e-9:
    helper = np.array([1.0, 0.0, 0.0]) if abs(src[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    axis = np.cross(src, helper)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    return _axis_angle(axis, np.pi)
  s = np.linalg.norm(v)
  vx = np.array([
    [0.0, -v[2], v[1]],
    [v[2], 0.0, -v[0]],
    [-v[1], v[0], 0.0],
  ])
  return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s + 1e-12))


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
  x, y, z = axis
  c, s = math.cos(angle), math.sin(angle)
  C = 1 - c
  return np.array([
    [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
    [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
    [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
  ])


def _yaw_about_z(yaw: float) -> np.ndarray:
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([
    [c, -s, 0.0],
    [s,  c, 0.0],
    [0.0, 0.0, 1.0],
  ])


def _back_project_centroid(mask: np.ndarray, cam: Camera) -> np.ndarray | None:
  import cv2
  ys, xs = np.where(mask)
  if xs.size == 0:
    return None
  u = float(xs.mean())
  v = float(ys.mean())
  undist = cv2.undistortPoints(
    np.array([[[u, v]]], dtype=np.float64),
    cam.K, cam.dist, P=None,
  ).reshape(2)
  ray_cam = np.array([undist[0], undist[1], 1.0])
  ray_world = cam.R_world_cam.T @ ray_cam
  origin_world = -cam.R_world_cam.T @ cam.t_world_cam.reshape(3)
  if abs(ray_world[2]) < 1e-9:
    return None
  s = -origin_world[2] / ray_world[2]
  hit = origin_world + s * ray_world
  return hit[:2]


def _mask_contour(mask: np.ndarray) -> np.ndarray | None:
  import cv2
  m = (mask.astype(np.uint8) * 255)
  contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  if not contours:
    return None
  c = max(contours, key=len)
  return c.reshape(-1, 2).astype(np.float64)


def _distance_transform(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
  import cv2
  W, H = image_size
  inv = (mask == 0).astype(np.uint8)
  return cv2.distanceTransform(inv, cv2.DIST_L2, 3)


def _distance_transform_from_contour(contour: np.ndarray,
                                     image_size: tuple[int, int]) -> np.ndarray:
  import cv2
  W, H = image_size
  bg = np.ones((H, W), dtype=np.uint8)
  pts = np.clip(np.round(contour).astype(int), [0, 0], [W - 1, H - 1])
  bg[pts[:, 1], pts[:, 0]] = 0
  return cv2.distanceTransform(bg, cv2.DIST_L2, 3)


def _make_pose(mesh: Mesh, rest: _RestHypothesis, xy: np.ndarray, yaw: float,
               residual_px: float | None, mask: np.ndarray, camera: Camera) -> ObjectPose:
  R = _yaw_about_z(yaw) @ rest.R_tilt
  xyz = np.array([xy[0], xy[1], rest.z_mm], dtype=np.float64)
  confidence = _score_confidence(mesh, rest, xy, yaw, residual_px, mask, camera)
  return ObjectPose(
    xy_mm                = xy.astype(np.float64),
    xyz_mm               = xyz,
    R_world_obj          = R,
    resting_face_class   = rest.cls,
    confidence           = confidence,
    reprojection_error_px = float(residual_px) if residual_px is not None else float("nan"),
    yaw_rad              = float(yaw),
    tilt_rad             = 0.0,
  )


def _score_confidence(mesh: Mesh, rest: _RestHypothesis, xy: np.ndarray, yaw: float,
                       residual_px: float | None, mask: np.ndarray,
                       camera: Camera) -> float:
  verts = transform_object(mesh.vertices_mm, _yaw_about_z(yaw) @ rest.R_tilt,
                           np.array([xy[0], xy[1], rest.z_mm]))
  ren_mask, _ = project_silhouette(verts, mesh.triangles, camera)
  observed_area = float(mask.sum())
  rendered_area = float((ren_mask > 0).sum())
  if observed_area == 0 or rendered_area == 0:
    return 0.0
  area_ratio = min(observed_area, rendered_area) / max(observed_area, rendered_area)
  if residual_px is None or not math.isfinite(residual_px):
    residual_score = 0.0
  else:
    residual_score = max(0.0, 1.0 - residual_px / 10.0)
  return float(max(0.0, min(1.0, 0.5 * area_ratio + 0.5 * residual_score)))


def _pose_to_prompt(prev: ObjectPose, cam: Camera) -> tuple[int, int]:
  """Project the previous frame's pose centroid down to an image pixel,
  so single-frame ``estimate`` calls can warm-start SAM2 without an
  explicit prompt.
  """
  pt = prev.xyz_mm.reshape(1, 3)
  import cv2
  rvec, _ = cv2.Rodrigues(cam.R_world_cam)
  uv, _ = cv2.projectPoints(pt, rvec, cam.t_world_cam.reshape(3), cam.K, cam.dist)
  u, v = uv.reshape(2)
  return int(round(u)), int(round(v))
