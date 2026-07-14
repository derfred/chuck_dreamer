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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal

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

if TYPE_CHECKING:
  from .scene_bg import SceneBackground

logger = logging.getLogger(__name__)


_RestingFaceClass = Literal["square", "triangle"]


# 6-DOF refinement parameter vector: [x, y, dz, droll, dpitch, dyaw]
#                                     mm mm mm   rad    rad   rad
#
# `x_scale` makes the LM step proportional to each scale, so a step of
# 1 unit in the rescaled space corresponds to ~1 mm or ~0.01 rad
# (~0.6°). Without this, LM rescales by the diagonal of J^T J and the
# angular DOFs end up with effectively-zero step (because rasterized
# residuals have no gradient under sub-pixel angular changes).
_PARAM_SCALE_6DOF = np.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01])

# Finite-difference step for the trf method. Multiplied by `x_scale`
# under the hood, so 0.05 here = 0.05 mm for positional DOFs and
# 0.0005 rad (~0.03°) for angular DOFs. Smaller than the silhouette
# pixel resolution but combined with the trust-region method it
# produces stable gradient estimates without overshooting into a
# different local minimum.
_DIFF_STEP_6DOF   = 0.05

# Per-call xy bounds: ±50 mm from the back-projected seed (centroid
# of the segmenter's mask). Keeps the optimizer from sliding the
# object across the image into a contour belonging to the arm or
# some other foreground blob. Position bounds for dz / dtilt are
# fixed because they're already in object-local units.
_XY_WANDER_MM   = 50.0
_DZ_BOUND_MM    = 20.0
_TILT_BOUND_RAD = 0.26    # ±15°
_YAW_BOUND_RAD  = 10.0    # >2π; finite is required by trf


def _build_bounds_6dof(xy_seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Return TRF-compatible (lower, upper) bounds anchored to the seed.

  xy is constrained to a ``±_XY_WANDER_MM`` box around the seed so
  the optimizer can't escape into an unrelated local minimum
  elsewhere in the image; the other DOFs use fixed bounds.
  """
  lo = np.array([xy_seed[0] - _XY_WANDER_MM, xy_seed[1] - _XY_WANDER_MM,
                 -_DZ_BOUND_MM, -_TILT_BOUND_RAD, -_TILT_BOUND_RAD, -_YAW_BOUND_RAD])
  hi = np.array([xy_seed[0] + _XY_WANDER_MM, xy_seed[1] + _XY_WANDER_MM,
                  _DZ_BOUND_MM,  _TILT_BOUND_RAD,  _TILT_BOUND_RAD,  _YAW_BOUND_RAD])
  return lo, hi


# Minimum number of points to resample the rendered contour / edges
# to. The residual vector must be constant-length across optimizer
# iterations, but the raw render returns variable-length output as
# the pose changes. Resampling to a fixed count >= max(observed,
# this) keeps the gradient signal sharp without losing detail.
_RESAMPLE_MIN = 128

# Edge-residual padding. The rhombicuboctahedron has at most ~30
# visible edges from any one camera angle, and we sample each at a
# fixed count. Total residual length is _EDGE_MAX_VISIBLE *
# _EDGE_SAMPLES_PER_EDGE; missing edges get sentinel-padded so the
# vector stays constant across optimizer iterations.
_EDGE_MAX_VISIBLE      = 40
_EDGE_SAMPLES_PER_EDGE = 16


def _resample_contour(contour: np.ndarray, n: int) -> np.ndarray:
  """Resample a polyline contour to exactly ``n`` evenly-spaced points
  by arc length.

  Returns ``(n, 2) float64`` (u, v) pixel coordinates. Handles the
  closed-contour case implicitly because cv2.findContours returns
  closed polylines and we don't append the wrap-around segment
  (visually negligible at typical contour lengths > 100).
  """
  pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
  if len(pts) <= 1:
    return np.repeat(pts[:1] if len(pts) else np.zeros((1, 2)), n, axis=0)
  segs = np.diff(pts, axis=0)
  seg_len = np.sqrt(np.einsum("ij,ij->i", segs, segs))
  cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])
  total = float(cumlen[-1])
  if total <= 0.0:
    return np.repeat(pts[:1], n, axis=0)
  targets = np.linspace(0.0, total, n, endpoint=False)
  u_resampled = np.interp(targets, cumlen, pts[:, 0])
  v_resampled = np.interp(targets, cumlen, pts[:, 1])
  return np.stack([u_resampled, v_resampled], axis=1)


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
    scene_bg: "SceneBackground | None" = None,
  ) -> None:
    self.calibration = calibration
    self.mesh        = load_mesh(Path(mesh_path))
    self.device      = device
    self.scene_bg    = scene_bg

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

    # Detect whether the extrinsics put the camera "above" or "below"
    # the mat plane. For datasets where the mat is mounted such that
    # the camera ends up on the -Z side of the world frame, we flip
    # the resting-face hypotheses so the mesh sits on the correct
    # (camera-visible) side of the mat. Either case is geometrically
    # valid; we just need the mesh and the camera to agree on which
    # side of z=0 the object lives on.
    cam_pos_world = -calibration.extrinsics.R.T @ calibration.extrinsics.t.reshape(3)
    self._z_sign = -1.0 if cam_pos_world[2] < 0 else +1.0
    if self._z_sign < 0:
      logger.info("ObjectPoseEstimator: camera is on -Z side of mat "
                  "(cam_z_world=%.1fmm); flipping rest-face hypotheses to "
                  "match.", float(cam_pos_world[2]))

    self._rest_classes = _build_rest_classes(self.mesh, z_sign=self._z_sign)
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

    effective_prompt: tuple[int, int] | list[tuple[int, int]] | None
    if prev_pose is not None:
      effective_prompt = prompt if prompt is not None else _pose_to_prompt(prev_pose, self.camera)
    else:
      effective_prompt = prompt
    # Guaranteed non-None: the guard above raises when both prompt and
    # prev_pose are None, and the prev_pose branch always sets a prompt.
    assert effective_prompt is not None
    mask = segment_image(image, effective_prompt, self.use_sam2,
                         self.sam2_checkpoint, self.device,
                         scene_bg=self.scene_bg)
    if mask is None:
      return None
    return self._fit_pose(image, mask, prev_pose)

  @staticmethod
  def _resolve_keyframes(
    first_frame_prompt: tuple[int, int] | list[tuple[int, int]] | None,
    keyframe_prompts: dict[int, "tuple[int, int] | list[tuple[int, int]]"] | None,
    *, caller: str,
  ) -> dict[int, Any]:
    kfs: dict[int, Any] = dict(keyframe_prompts or {})
    if 0 not in kfs and first_frame_prompt is not None:
      kfs[0] = first_frame_prompt
    if 0 not in kfs:
      raise ValueError(
        f"{caller}(): a frame-0 prompt is required (pass via "
        "first_frame_prompt or keyframe_prompts[0]).")
    return kfs

  def segment_episode(
    self,
    frames: Iterable[np.ndarray],
    first_frame_prompt: tuple[int, int] | list[tuple[int, int]] | None = None,
    keyframe_prompts: dict[int, "tuple[int, int] | list[tuple[int, int]]"] | None = None,
    target_name: str = "object",
  ) -> list[np.ndarray | None]:
    """Run SAM2 across an episode and return per-frame masks.

    Segmentation half of :meth:`estimate_episode`, separated so callers
    can obtain masks without pose fitting (and, in future, segment
    targets other than the pushed object). ``target_name`` selects which
    target is being masked; only ``"object"`` is wired today, so it does
    not yet thread into the SAM2 ``obj_id``.
    """
    frame_list = list(frames)
    if not frame_list:
      return []
    kfs = self._resolve_keyframes(
      first_frame_prompt, keyframe_prompts, caller="segment_episode")
    masks = segment_video(frame_list, kfs,
                          self.sam2_checkpoint, self.device)
    self.last_masks = masks
    return masks

  def fit_poses_from_masks(
    self,
    frames: Iterable[np.ndarray],
    masks: list[np.ndarray | None],
    first_frame_prompt: tuple[int, int] | list[tuple[int, int]] | None = None,
    keyframe_prompts: dict[int, "tuple[int, int] | list[tuple[int, int]]"] | None = None,
  ) -> list[ObjectPose | None]:
    """Fit per-frame poses given pre-computed masks.

    Pose-fitting half of :meth:`estimate_episode`. ``masks`` is the
    output of :meth:`segment_episode`; keyframe handling matches the
    original orchestration.
    """
    frame_list = list(frames)
    if not frame_list:
      return []
    kfs = self._resolve_keyframes(
      first_frame_prompt, keyframe_prompts, caller="fit_poses_from_masks")

    out: list[ObjectPose | None] = []
    prev: ObjectPose | None = None
    self._episode = _EpisodeContext()
    for t, (img, mask) in enumerate(zip(frame_list, masks)):
      if mask is None:
        out.append(None)
        continue
      is_keyframe = (t in kfs)
      if is_keyframe and t != 0:
        # Cold-start at this keyframe: bypass the warm-start path by
        # passing prev=None. The cached rest hypothesis stays so we
        # don't re-run the 8-fit rest-class search, but xy/dz/tilt/yaw
        # all get re-derived from the current frame's mask.
        pose = self._fit_pose(img, mask, prev_pose=None)
      else:
        pose = self._fit_pose(img, mask, prev)
      if pose is not None and is_keyframe:
        # Flag keyframe poses as maximally confident so the smoother's
        # observation-noise heuristic treats them as anchors and pulls
        # warm-started frames toward them.
        pose.confidence = 1.0
      out.append(pose)
      if pose is not None:
        prev = pose
    return out

  def estimate_episode(
    self,
    frames: Iterable[np.ndarray],
    first_frame_prompt: tuple[int, int] | list[tuple[int, int]] | None = None,
    keyframe_prompts: dict[int, "tuple[int, int] | list[tuple[int, int]]"] | None = None,
  ) -> list[ObjectPose | None]:
    """Estimate poses across a whole episode.

    ``first_frame_prompt`` is the click at frame 0 (required if no
    ``keyframe_prompts`` is given). ``keyframe_prompts`` is an
    ``{episode_relative_frame_idx: prompt}`` map; at each keyframe the
    estimator re-runs the cold-start path (rest hypothesis stays
    cached, but the (x, y) is re-anchored to the prompt's
    back-projection) so the warm-start chain can't drift forever.

    If both are given, ``keyframe_prompts[0]`` wins for frame 0.

    Thin wrapper over :meth:`segment_episode` + :meth:`fit_poses_from_masks`.
    """
    frame_list = list(frames)
    if not frame_list:
      return []
    masks = self.segment_episode(
      frame_list, first_frame_prompt, keyframe_prompts)
    return self.fit_poses_from_masks(
      frame_list, masks, first_frame_prompt, keyframe_prompts)

  # -------------------------------------------------------------------------
  # Pose fitting
  # -------------------------------------------------------------------------
  def _fit_pose(self, image: np.ndarray, mask: np.ndarray,
                prev_pose: ObjectPose | None) -> ObjectPose | None:
    """Fit the object pose to one frame.

    Two code paths, by far the dominant cost is the silhouette/edge
    rasterizer (CPU) inside the LM optimizer:

      * **First frame of an episode (cold start).** Run the 2×N rest-
        class search, then a 6-DOF silhouette refinement (recovers any
        small tilt/dz the rest hypothesis can't), then a 6-DOF edge
        refinement (sub-pixel precision on yaw). This costs many seconds
        but only happens once per episode.

      * **Warm-started frames.** The object can't move far between
        consecutive frames, the rest class is settled, and tilt/dz
        physically don't change. Run only a quick 3-DOF silhouette
        refinement on (x, y, yaw), seeded from the previous frame.
        ~5-10 LM evaluations, sub-second.

    The cached resting hypothesis (``self._episode.resting``) is what
    distinguishes the two paths. ``estimate_episode`` resets it; single-
    frame ``estimate`` calls re-search every time.
    """
    xy0 = _back_project_centroid(mask, self.camera)
    if xy0 is None:
      return None

    cold_start = (prev_pose is None or self._episode.resting is None)
    if cold_start:
      rest, yaw, xy = self._search_rest_class(image, mask, xy0)
      if rest is None:
        return None
      self._episode.resting = rest
      # Cold start: full 6-DOF silhouette + 6-DOF edge refinement.
      dz0  = 0.0
      rpy0 = (0.0, 0.0, yaw)
      xy_ref, dz_ref, rpy_ref, sil_rms = self._refine_silhouette_6dof(
        mask, rest, xy, dz0, rpy0)
      xy_ed, dz_ed, rpy_ed, edge_rms = self._refine_edges_6dof(
        image, mask, rest, xy_ref, dz_ref, rpy_ref)
      return _make_pose(
        self.mesh, rest, xy_ed, dz_ed, rpy_ed,
        residual_px=edge_rms if edge_rms is not None else sil_rms,
        mask=mask, camera=self.camera,
      )

    # Warm-started frame: pull rest hypothesis + tilt/dz from the
    # episode-locked cache (rest). Yaw warm-starts from the previous
    # frame (the object can't rotate far between frames). For (x, y)
    # we re-seed from the current frame's mask-centroid back-projected
    # to the mesh-centroid plane (z = prev_pose.xyz_mm[2]). The
    # polyhedron is nearly spherical in silhouette, so silhouette
    # gradient w.r.t. (x, y) is weak: if we seeded at the previous
    # frame's xy the optimizer would converge immediately on the
    # already-overlapping render and never actually track the object.
    # Not cold_start guarantees both are set (see the cold_start predicate).
    assert prev_pose is not None and self._episode.resting is not None
    rest = self._episode.resting
    yaw  = prev_pose.yaw_rad
    prev_z_world = float(prev_pose.xyz_mm[2])
    xy_seed = _back_project_centroid(mask, self.camera, z_plane=prev_z_world)
    if xy_seed is None:
      xy_seed = prev_pose.xy_mm.copy()
    prev_dz = float(prev_pose.xyz_mm[2] - rest.z_mm * self._z_sign)
    xy_ref, yaw_ref, sil_rms = self._refine_silhouette(mask, rest, xy_seed, yaw)
    return _make_pose(
      self.mesh, rest, xy_ref, prev_dz, (0.0, 0.0, yaw_ref),
      residual_px=sil_rms,
      mask=mask, camera=self.camera,
    )

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

  def _refine_silhouette_6dof(self, mask: np.ndarray, rest: _RestHypothesis,
                               xy0: np.ndarray, dz0: float,
                               rpy0: tuple[float, float, float]
                               ) -> tuple[np.ndarray, float,
                                          tuple[float, float, float],
                                          float | None]:
    """6-DOF silhouette refinement. Params: (x, y, dz, droll, dpitch, dyaw).

    `dz`, `droll`, `dpitch` are deltas from the rest hypothesis. Uses
    ``method='trf'`` with ``x_scale`` so the per-coordinate LM step
    size is comparable across mm (positional) and rad (angular) DOFs,
    and ``diff_step`` so finite-difference probes are large enough to
    actually move silhouette pixels (otherwise the rasterized residual
    has zero gradient with respect to sub-pixel parameter changes).
    """
    from scipy.optimize import least_squares

    obs_contour = _mask_contour(mask)
    if obs_contour is None or len(obs_contour) < 5:
      return xy0, dz0, rpy0, None
    obs_dt = _distance_transform(mask, self.camera.image_size)

    # least_squares requires a constant-length residual vector. The
    # rendered silhouette contour grows/shrinks with mesh pose, so we
    # resample it to a fixed number of points before computing the
    # distance transforms.
    n_ren_samples = max(_RESAMPLE_MIN, len(obs_contour))
    obs_idx_v = obs_contour[:, 1].astype(int)
    obs_idx_u = obs_contour[:, 0].astype(int)
    sentinel  = np.full(len(obs_contour) + n_ren_samples, 1e3, dtype=np.float64)

    def residual(params: np.ndarray) -> np.ndarray:
      x, y, dz, droll, dpitch, dyaw = params
      ren_mask, ren_contour = self._render_silhouette_6dof(
        rest, np.array([x, y]), dz, (droll, dpitch, dyaw))
      if ren_contour is None or len(ren_contour) < 5:
        return sentinel.copy()
      ren_dt = _distance_transform_from_contour(ren_contour, self.camera.image_size)
      ren_resampled = _resample_contour(ren_contour, n_ren_samples)
      d_obs_to_ren = ren_dt[obs_idx_v, obs_idx_u]
      d_ren_to_obs = obs_dt[ren_resampled[:, 1].astype(int),
                             ren_resampled[:, 0].astype(int)]
      return np.asarray(np.concatenate([d_obs_to_ren, d_ren_to_obs]), dtype=np.float64)

    x0 = np.array([xy0[0], xy0[1], dz0, rpy0[0], rpy0[1], rpy0[2]],
                  dtype=np.float64)
    bounds = _build_bounds_6dof(xy0)
    try:
      sol = least_squares(residual, x0, method="trf",
                          x_scale=_PARAM_SCALE_6DOF,
                          diff_step=_DIFF_STEP_6DOF,
                          bounds=bounds,
                          max_nfev=self.chamfer_max_iter * 2,
                          xtol=1e-6, ftol=1e-6)
    except Exception as e:
      logger.warning("silhouette 6dof refinement failed: %s", e)
      return xy0, dz0, rpy0, None
    logger.debug("silhouette 6dof: nfev=%d status=%d cost=%.3f x=%s",
                  sol.nfev, sol.status, float(sol.cost), sol.x)
    res_norm = float(np.sqrt(np.mean(sol.fun ** 2))) if sol.fun.size else None
    return (np.asarray(sol.x[:2]), float(sol.x[2]),
            (float(sol.x[3]), float(sol.x[4]), float(sol.x[5])),
            res_norm)

  def _refine_edges_6dof(self, image: np.ndarray, mask: np.ndarray,
                          rest: _RestHypothesis,
                          xy0: np.ndarray, dz0: float,
                          rpy0: tuple[float, float, float]
                          ) -> tuple[np.ndarray, float,
                                     tuple[float, float, float],
                                     float | None]:
    """6-DOF edge refinement; same shape as _refine_silhouette_6dof."""
    import cv2
    from scipy.optimize import least_squares

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, self.canny_low, self.canny_high)
    edge_dt = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)

    W, H = self.camera.image_size

    # Constant-length residual: every iteration must report
    # _EDGE_TOTAL_SAMPLES distances, regardless of how many visible
    # edges the current mesh pose has. Each edge contributes
    # `n_per_edge` samples; we pad with sentinels if the pose has
    # fewer edges than at first eval. The cap on edge count limits
    # the worst case so the optimizer's Jacobian stays small.
    n_edge_cap   = _EDGE_MAX_VISIBLE
    n_per_edge   = _EDGE_SAMPLES_PER_EDGE
    n_total      = n_edge_cap * n_per_edge
    sentinel_edges = np.full(n_total, 1e3, dtype=np.float64)
    ts_template  = np.linspace(0.0, 1.0, n_per_edge)

    def residual(params: np.ndarray) -> np.ndarray:
      x, y, dz, droll, dpitch, dyaw = params
      edges_proj = self._render_edges_6dof(
        rest, np.array([x, y]), dz, (droll, dpitch, dyaw))
      if not edges_proj:
        return sentinel_edges.copy()
      out = np.full(n_total, 1e3, dtype=np.float64)
      n_edges = min(n_edge_cap, len(edges_proj))
      for i in range(n_edges):
        p0, p1 = edges_proj[i]
        seg = p0[None, :] * (1 - ts_template[:, None]) + p1[None, :] * ts_template[:, None]
        u = np.clip(np.round(seg[:, 0]).astype(int), 0, W - 1)
        v = np.clip(np.round(seg[:, 1]).astype(int), 0, H - 1)
        out[i * n_per_edge:(i + 1) * n_per_edge] = edge_dt[v, u]
      return out

    x0 = np.array([xy0[0], xy0[1], dz0, rpy0[0], rpy0[1], rpy0[2]],
                  dtype=np.float64)
    bounds = _build_bounds_6dof(xy0)
    try:
      sol = least_squares(residual, x0, method="trf",
                          x_scale=_PARAM_SCALE_6DOF,
                          diff_step=_DIFF_STEP_6DOF,
                          bounds=bounds,
                          max_nfev=self.edges_max_iter * 2,
                          xtol=1e-6, ftol=1e-6)
    except Exception as e:
      logger.warning("edges 6dof refinement failed: %s", e)
      return xy0, dz0, rpy0, None
    logger.debug("edges 6dof: nfev=%d status=%d cost=%.3f x=%s",
                  sol.nfev, sol.status, float(sol.cost), sol.x)
    res_norm = float(np.sqrt(np.mean(sol.fun ** 2))) if sol.fun.size else None
    return (np.asarray(sol.x[:2]), float(sol.x[2]),
            (float(sol.x[3]), float(sol.x[4]), float(sol.x[5])),
            res_norm)

  def _refine_silhouette(self, mask: np.ndarray, rest: _RestHypothesis,
                          xy0: np.ndarray, yaw0: float
                          ) -> tuple[np.ndarray, float, float | None]:
    """3-DOF silhouette refinement used by the rest-class search.

    Constant-length residual (same trick as ``_refine_silhouette_6dof``)
    so the optimizer doesn't crash when the rendered contour length
    changes across iterations.
    """
    from scipy.optimize import least_squares

    obs_contour = _mask_contour(mask)
    if obs_contour is None or len(obs_contour) < 5:
      return xy0, yaw0, None
    obs_dt = _distance_transform(mask, self.camera.image_size)

    n_ren_samples = max(_RESAMPLE_MIN, len(obs_contour))
    obs_idx_v = obs_contour[:, 1].astype(int)
    obs_idx_u = obs_contour[:, 0].astype(int)
    sentinel  = np.full(len(obs_contour) + n_ren_samples, 1e3, dtype=np.float64)

    def residual(params: np.ndarray) -> np.ndarray:
      x, y, yaw = params
      ren_mask, ren_contour = self._render_silhouette(rest, np.array([x, y]), yaw)
      if ren_contour is None or len(ren_contour) < 5:
        return sentinel.copy()
      ren_dt = _distance_transform_from_contour(ren_contour, self.camera.image_size)
      ren_resampled = _resample_contour(ren_contour, n_ren_samples)
      d_obs_to_ren = ren_dt[obs_idx_v, obs_idx_u]
      d_ren_to_obs = obs_dt[ren_resampled[:, 1].astype(int),
                             ren_resampled[:, 0].astype(int)]
      return np.asarray(np.concatenate([d_obs_to_ren, d_ren_to_obs]), dtype=np.float64)

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
    """3-DOF edge refinement used by the rest-class search.

    Constant-length residual (same fixed-edge-count trick as
    ``_refine_edges_6dof``).
    """
    import cv2
    from scipy.optimize import least_squares

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, self.canny_low, self.canny_high)
    edge_dt = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)

    W, H = self.camera.image_size

    n_edge_cap = _EDGE_MAX_VISIBLE
    n_per_edge = _EDGE_SAMPLES_PER_EDGE
    n_total    = n_edge_cap * n_per_edge
    sentinel_edges = np.full(n_total, 1e3, dtype=np.float64)
    ts_template = np.linspace(0.0, 1.0, n_per_edge)

    def residual(params: np.ndarray) -> np.ndarray:
      x, y, yaw = params
      edges_proj = self._render_edges(rest, np.array([x, y]), yaw)
      if not edges_proj:
        return sentinel_edges.copy()
      out = np.full(n_total, 1e3, dtype=np.float64)
      n_edges = min(n_edge_cap, len(edges_proj))
      for i in range(n_edges):
        p0, p1 = edges_proj[i]
        seg = p0[None, :] * (1 - ts_template[:, None]) + p1[None, :] * ts_template[:, None]
        u = np.clip(np.round(seg[:, 0]).astype(int), 0, W - 1)
        v = np.clip(np.round(seg[:, 1]).astype(int), 0, H - 1)
        out[i * n_per_edge:(i + 1) * n_per_edge] = edge_dt[v, u]
      return out

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

  def _world_vertices_6dof(self, rest: _RestHypothesis, xy: np.ndarray,
                            dz: float, rpy: tuple[float, float, float]) -> np.ndarray:
    """6-DOF analogue: applies dyaw o tilt-delta o R_tilt, plus (xy, z+dz)."""
    droll, dpitch, dyaw = rpy
    R = (_yaw_about_z(dyaw)
         @ _small_tilt(droll, dpitch)
         @ rest.R_tilt)
    xyz = np.array([xy[0], xy[1], rest.z_mm + dz])
    return transform_object(self.mesh.vertices_mm, R, xyz)

  def _render_silhouette(self, rest: _RestHypothesis, xy: np.ndarray, yaw: float
                          ) -> tuple[np.ndarray, np.ndarray | None]:
    verts = self._world_vertices(rest, xy, yaw)
    return project_silhouette(verts, self.mesh.triangles, self.camera)

  def _render_silhouette_6dof(self, rest: _RestHypothesis, xy: np.ndarray,
                                dz: float, rpy: tuple[float, float, float]
                                ) -> tuple[np.ndarray, np.ndarray | None]:
    verts = self._world_vertices_6dof(rest, xy, dz, rpy)
    return project_silhouette(verts, self.mesh.triangles, self.camera)

  def _render_edges(self, rest: _RestHypothesis, xy: np.ndarray, yaw: float
                     ) -> list[tuple[np.ndarray, np.ndarray]]:
    verts = self._world_vertices(rest, xy, yaw)
    return project_visible_edges(verts, self.mesh.triangles, self.camera)

  def _render_edges_6dof(self, rest: _RestHypothesis, xy: np.ndarray,
                          dz: float, rpy: tuple[float, float, float]
                          ) -> list[tuple[np.ndarray, np.ndarray]]:
    verts = self._world_vertices_6dof(rest, xy, dz, rpy)
    return project_visible_edges(verts, self.mesh.triangles, self.camera)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_rest_classes(mesh: Mesh, z_sign: float = +1.0
                         ) -> list[_RestHypothesis]:
  """Enumerate every distinct face-orientation as a rest hypothesis.

  ``z_sign`` controls which world-frame half-space the mesh sits in.
  +1 = the usual convention ("mat at z=0, object above"); -1 = the
  mirrored convention used when PnP recovered a camera-below-mat
  extrinsics (the world frame's +Z is flipped relative to the
  camera's view, so the object needs to be on -Z to match the
  observed silhouette). For each rest hypothesis we rotate the
  face normal to ``(0, 0, -z_sign)`` so the face faces away from
  the camera, and we translate by ``z_mm * z_sign`` so the object
  sits on the visible side of the mat.

  The rhombicuboctahedron has 18 square faces split into two
  symmetry classes (6 axis-aligned with normals along ±X/±Y/±Z,
  and 12 diagonal squares whose normals point along face-diagonals
  like (±1, ±1, 0)/√2). Each class has a distinct rendered
  silhouette, so all candidates are needed for the search.
  """
  centers = (mesh.vertices_mm[mesh.triangles[:, 0]]
             + mesh.vertices_mm[mesh.triangles[:, 1]]
             + mesh.vertices_mm[mesh.triangles[:, 2]]) / 3.0
  centroid = mesh.vertices_mm.mean(axis=0)

  # In the standard +Z convention the resting face's outward normal
  # points to -Z (face down on the mat). In the flipped convention
  # the face's outward normal points to +Z (face "up" toward the
  # mat plane from below).
  target_normal = np.array([0.0, 0.0, -float(z_sign)])

  groups = _group_coplanar_faces(mesh.face_normals, centers)
  hypotheses: list[_RestHypothesis] = []
  for g in groups:
    n = mesh.face_normals[g].mean(axis=0)
    n = n / (np.linalg.norm(n) + 1e-9)
    face_center = centers[g].mean(axis=0)
    z_mm = abs(float(np.dot(face_center - centroid, n))) * float(z_sign)
    R_tilt = _rotation_align(n, target_normal)
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
  """Rotation that maps unit ``src`` to unit ``dst``.

  Uses Rodrigues' formula for the general case; falls back to a 180°
  flip about an arbitrary perpendicular axis when src and dst are
  antiparallel. The antipodal threshold is intentionally generous
  (``c < -0.999``) because the inputs come from unit normals that may
  carry a few ULPs of float drift — a tighter cutoff was missing the
  case ``src=(0,0,1) -> dst=(0,0,-1)`` and silently returning identity.
  """
  src = src / np.linalg.norm(src)
  dst = dst / np.linalg.norm(dst)
  c = float(np.dot(src, dst))
  if c > 0.999999:
    return np.eye(3)
  if c < -0.999999:
    helper = np.array([1.0, 0.0, 0.0]) if abs(src[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    axis = np.cross(src, helper)
    axis = axis / np.linalg.norm(axis)
    return _axis_angle(axis, np.pi)
  v = np.cross(src, dst)
  s = np.linalg.norm(v)
  vx = np.array([
    [0.0, -v[2], v[1]],
    [v[2], 0.0, -v[0]],
    [-v[1], v[0], 0.0],
  ])
  return np.asarray(np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s)))


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


def _small_tilt(roll: float, pitch: float) -> np.ndarray:
  """Tilt about world +X then +Y, composed left-to-right.

  Intended for small angles (a few degrees at most). The 6-DOF refinement
  uses this to let the rest face wobble slightly off horizontal without
  reaching for gimbal-lock-sensitive Euler-angle gymnastics.
  """
  cr, sr = math.cos(roll), math.sin(roll)
  cp, sp = math.cos(pitch), math.sin(pitch)
  Rx = np.array([
    [1.0, 0.0, 0.0],
    [0.0,  cr, -sr],
    [0.0,  sr,  cr],
  ])
  Ry = np.array([
    [cp, 0.0,  sp],
    [0.0, 1.0, 0.0],
    [-sp, 0.0,  cp],
  ])
  return np.asarray(Ry @ Rx)


def _back_project_centroid(mask: np.ndarray, cam: Camera,
                            z_plane: float = 0.0) -> np.ndarray | None:
  """Back-project the mask centroid onto the world plane at z=``z_plane``.

  Defaults to the mat plane (z=0). Pass ``z_plane = rest.z_mm`` to land
  at the height of the mesh centroid instead, which is what warm-started
  per-frame fits should seed from.
  """
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
  s = (z_plane - origin_world[2]) / ray_world[2]
  hit = origin_world + s * ray_world
  return np.asarray(hit[:2])


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
  return np.asarray(cv2.distanceTransform(inv, cv2.DIST_L2, 3))


def _distance_transform_from_contour(contour: np.ndarray,
                                     image_size: tuple[int, int]) -> np.ndarray:
  import cv2
  W, H = image_size
  bg = np.ones((H, W), dtype=np.uint8)
  pts = np.clip(np.round(contour).astype(int), [0, 0], [W - 1, H - 1])
  bg[pts[:, 1], pts[:, 0]] = 0
  return cv2.distanceTransform(bg, cv2.DIST_L2, 3)


def _make_pose(mesh: Mesh, rest: _RestHypothesis, xy: np.ndarray,
               dz: float, rpy: tuple[float, float, float],
               residual_px: float | None, mask: np.ndarray,
               camera: Camera) -> ObjectPose:
  """Build an ObjectPose from the 6-DOF refinement output.

  ``dz`` is the height delta from the rest hypothesis; ``rpy`` is
  ``(droll, dpitch, dyaw)`` where roll/pitch are small tilts off-axis
  and dyaw is the full yaw about world +Z.
  """
  droll, dpitch, dyaw = rpy
  R = (_yaw_about_z(dyaw)
       @ _small_tilt(droll, dpitch)
       @ rest.R_tilt)
  z  = float(rest.z_mm + dz)
  xyz = np.array([xy[0], xy[1], z], dtype=np.float64)
  # tilt magnitude (radians off vertical); zero when droll = dpitch = 0.
  tilt_rad = float(math.hypot(droll, dpitch))
  confidence = _score_confidence_6dof(
    mesh, rest, xy, dz, rpy, residual_px, mask, camera)
  return ObjectPose(
    xy_mm                = xy.astype(np.float64),
    xyz_mm               = xyz,
    R_world_obj          = R,
    resting_face_class   = rest.cls,
    confidence           = confidence,
    reprojection_error_px = float(residual_px) if residual_px is not None else float("nan"),
    yaw_rad              = float(dyaw),
    tilt_rad             = tilt_rad,
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


def _score_confidence_6dof(mesh: Mesh, rest: _RestHypothesis, xy: np.ndarray,
                            dz: float, rpy: tuple[float, float, float],
                            residual_px: float | None, mask: np.ndarray,
                            camera: Camera) -> float:
  """6-DOF analogue of _score_confidence."""
  droll, dpitch, dyaw = rpy
  R = (_yaw_about_z(dyaw)
       @ _small_tilt(droll, dpitch)
       @ rest.R_tilt)
  verts = transform_object(mesh.vertices_mm, R,
                           np.array([xy[0], xy[1], rest.z_mm + dz]))
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


def pose_to_prompt(prev: ObjectPose, cam: Camera) -> tuple[int, int]:
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


# Historical private name; external callers should use :func:`pose_to_prompt`.
_pose_to_prompt = pose_to_prompt
