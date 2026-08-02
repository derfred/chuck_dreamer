"""Object-pose node (spec §7.10): per-frame 6-DoF pose fit by
analysis-by-synthesis.

A direct port of ``notebooks/reconstruct_explained.ipynb``: for each frame
with an object mask, render the mesh silhouette under a candidate pose and
run Nelder-Mead to maximise silhouette IoU against the SAM2 mask. The fit
is raw per-frame (no temporal smoothing); it warm-starts from the previous
frame's fit — state *within* one episode, per the spec's granularity
constraint.

The node produces:

  * ``obj_pos_table`` — the fitted ``(x, y, z)`` per frame in the table frame
    (the mat/world frame the extrinsics are calibrated against;
    ``Frame.TABLE``), in millimetres, with a validity mask (frames with no
    fit stay ``0``). The writers persist the mask as the
    ``obj_pos_table.valid`` sibling — the historical ``no_pose_fit`` field
    was its inversion and is gone.
  * ``object_mesh_overlay`` — a ``(T, H, W)`` uint8 binary silhouette of the
    fitted mesh. The writer tints it into a translucent overlay and logs it
    as ``camera/mesh_overlay`` so the viewer composites it over the RGB
    video. (Kept binary, not RGBA, so a full-res episode doesn't hold an
    8 GB RGBA stack in memory.)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from chuck_dreamer.common.tracks import Frame, Persist, Track, TrackSpec, Unit

from ..context import RunContext
from ..harness import InputDecl
from .segmentation import dense_to_masks

if TYPE_CHECKING:
  from ..trackset import NodeView

# Pose-fit knobs (mirroring the notebook).
_INIT_DEPTH_M = 0.8       # seed depth along the camera ray, metres
_MAX_ITER = 1500          # Nelder-Mead function-eval budget per frame


# ---------------------------------------------------------------------------
# Analysis-by-synthesis pose fit (notebook port; everything in metres)
# ---------------------------------------------------------------------------
def _quat_to_R(q: np.ndarray) -> np.ndarray:
  """Quaternion ``[w, x, y, z]`` -> 3x3 rotation matrix."""
  w, x, y, z = q
  n = np.sqrt(w * w + x * x + y * y + z * z) + 1e-12
  w, x, y, z = w / n, x / n, y / n, z / n
  return np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ], dtype=np.float32)


def _project_silhouette(vertices, faces, K, R_cam, T_cam,
                        pose_xyz, pose_quat, image_wh) -> np.ndarray:
  """Render a filled binary silhouette of the mesh under ``(pose_xyz,
  pose_quat)``. Returns an ``(H, W)`` uint8 mask in ``{0, 1}``.

  ``vertices``, ``T_cam`` and ``pose_xyz`` are all in metres. Pinhole
  projection with no lens distortion, matching the notebook."""
  import cv2

  W, H = image_wh
  R_obj   = _quat_to_R(pose_quat)
  v_world = vertices @ R_obj.T + pose_xyz[None, :]      # object -> world
  v_cam   = v_world @ R_cam.T + T_cam[None, :]          # world  -> camera
  proj    = v_cam @ K.T                                 # camera -> pixels
  uv      = proj[:, :2] / np.clip(proj[:, 2:3], 1e-6, None)

  uv_i = np.round(uv).astype(np.int32)
  mask = np.zeros((H, W), dtype=np.uint8)
  for face in faces:
    cv2.drawContours(mask, [uv_i[face]], 0, color=1, thickness=-1)
  return mask


def _silhouette_iou(pred: np.ndarray, gt: np.ndarray) -> float:
  p = pred > 0
  g = gt > 0
  inter = float(np.count_nonzero(p & g))
  union = float(np.count_nonzero(p)) + float(np.count_nonzero(g)) - inter
  return inter / (union + 1e-6)


def _init_xyz_from_mask(gt_mask, K, R_cam, T_cam,
                        init_depth=_INIT_DEPTH_M) -> np.ndarray | None:
  """Back-project the mask centroid to a world-frame xyz at ``init_depth``
  (metres) along the camera ray. Returns ``None`` for an empty mask."""
  ys, xs = np.where(gt_mask > 0)
  if xs.size == 0:
    return None
  u_c, v_c = float(xs.mean()), float(ys.mean())
  fx, fy = K[0, 0], K[1, 1]
  cx, cy = K[0, 2], K[1, 2]
  ray_cam = np.array([(u_c - cx) / fx, (v_c - cy) / fy, 1.0], dtype=np.float32)
  p_cam   = ray_cam * init_depth
  xyz = R_cam.T @ (p_cam - T_cam)
  return np.asarray(xyz, dtype=np.float32)


def _fit_pose(vertices, faces, K, R_cam, T_cam, gt_mask, image_wh,
              init_xyz, init_quat, maxiter=_MAX_ITER):
  """Nelder-Mead fit of ``[xyz, quat]`` maximising silhouette IoU against
  ``gt_mask``. Returns ``(opt_xyz_m, opt_quat, final_iou)``."""
  from scipy.optimize import minimize

  def objective(params):
    xyz  = params[:3]
    quat = params[3:7]
    quat = quat / (np.linalg.norm(quat) + 1e-10)
    pred = _project_silhouette(vertices, faces, K, R_cam, T_cam,
                               xyz, quat, image_wh)
    return -_silhouette_iou(pred, gt_mask)

  x0 = np.concatenate([init_xyz, init_quat]).astype(np.float64)
  result = minimize(
    objective, x0, method="Nelder-Mead",
    options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-5, "adaptive": True},
  )
  opt_xyz  = result.x[:3].astype(np.float32)
  opt_quat = result.x[3:7]
  opt_quat = (opt_quat / np.linalg.norm(opt_quat)).astype(np.float32)
  return opt_xyz, opt_quat, float(-result.fun)


# ---------------------------------------------------------------------------
# Calibration / mesh, converted to metres (matching the notebook + mesh frame)
# ---------------------------------------------------------------------------
def _calibration_m(intrinsics, extrinsics):
  """``(K, R_cam, T_cam_m, image_wh)`` from the calibration artifacts, in
  metres. ``K`` and ``R_cam`` come straight from the calibration; the
  extrinsic translation is converted mm -> m to match the metre-frame
  mesh."""
  K     = np.asarray(intrinsics.K, dtype=np.float32)
  R_cam = np.asarray(extrinsics.R, dtype=np.float32)
  T_cam = np.asarray(extrinsics.t, dtype=np.float32).reshape(3) / 1000.0
  W, H  = intrinsics.image_size
  return K, R_cam, T_cam, (int(W), int(H))


def _mesh_m(mesh):
  """``(vertices_m, faces)`` for the tracked object, vertices in metres."""
  vertices = np.asarray(mesh.vertices_mm, dtype=np.float32) / 1000.0
  faces    = np.asarray(mesh.triangles, dtype=np.int32)
  return vertices, faces


class ObjectPoseNode:
  name = "object_pose"
  scope_level = "episode"
  # CPU-bound per-frame Nelder-Mead fit; runs in the worker pool.
  lane = "worker"
  inputs = (
    InputDecl("image", frame=Frame.IMAGE),
    InputDecl("object_masks", frame=Frame.IMAGE),
    InputDecl("intrinsics"),
    InputDecl("extrinsics"),
    InputDecl("object_mesh"),
  )
  outputs = (
    TrackSpec("obj_pos_table", np.float32, (3,), Frame.TABLE, Unit.MM,
              validity=True, persist=Persist.TRAJECTORY),
    TrackSpec("object_mesh_overlay", np.uint8, (), Frame.IMAGE,
              Unit.UNITLESS, persist=Persist.DIAGNOSTIC),
  )

  def __init__(self, ctx: RunContext) -> None:
    self.ctx = ctx

  def run(self, view: "NodeView") -> list[Track]:
    scope = view.scope
    images = view.track("image").values
    T, H, W = len(images), images.shape[1], images.shape[2]

    # Masks come from the segmentation node as a validity-carrying track;
    # in the parallel importer they cross the process boundary through the
    # episode container (`object_masks` + `object_masks.valid` scratch
    # fields), so this read is identical in the serial and worker paths.
    masks_track = view.track("object_masks")
    assert masks_track.valid is not None
    masks = dense_to_masks(masks_track.values, masks_track.valid)

    K, R_cam, T_cam, image_wh = _calibration_m(
      view.artifact("intrinsics"), view.artifact("extrinsics"))
    vertices, faces = _mesh_m(view.artifact("object_mesh"))
    init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    obj_pos = np.zeros((T, 3), dtype=np.float32)
    gap       = np.ones((T,), dtype=np.bool_)
    # Binary fitted-mesh silhouette per frame; the writer colours it into a
    # translucent overlay at log time. Kept as (T, H, W) uint8 (not RGBA) so
    # a full-res episode doesn't hold an 8 GB RGBA stack in memory.
    overlay   = np.zeros((T, H, W), dtype=np.uint8)

    from tqdm import tqdm

    n_ok = 0
    prev_xyz: np.ndarray | None = None
    prev_quat = init_quat
    for t in tqdm(range(T), desc=f"[{self.name}] ep{scope.episode_index} pose fit",
                  unit="frame", leave=False):
      mask = masks[t] if t < len(masks) else None
      if mask is None:
        continue
      gt = np.asarray(mask, dtype=np.uint8)
      # Warm-start from the previous fitted pose when we have one; otherwise
      # cold-start by back-projecting the mask centroid.
      init_xyz: np.ndarray | None
      if prev_xyz is not None:
        init_xyz, init_q = prev_xyz, prev_quat
      else:
        init_xyz = _init_xyz_from_mask(gt, K, R_cam, T_cam)
        init_q = init_quat
      if init_xyz is None:
        continue

      opt_xyz, opt_quat, _iou = _fit_pose(
        vertices, faces, K, R_cam, T_cam, gt, image_wh, init_xyz, init_q)

      obj_pos[t] = opt_xyz * 1000.0         # metres -> millimetres
      gap[t] = False

      sil = _project_silhouette(vertices, faces, K, R_cam, T_cam,
                                opt_xyz, opt_quat, image_wh)
      if sil.shape[:2] != (H, W):
        import cv2
        sil = cv2.resize(sil, (W, H), interpolation=cv2.INTER_NEAREST)
      overlay[t] = sil

      prev_xyz, prev_quat = opt_xyz, opt_quat
      n_ok += 1

    print(f"[object_pose] {scope.dataset_id} ep{scope.episode_index}: "
          f"{n_ok}/{T} frames fitted")

    pos_spec = TrackSpec("obj_pos_table", np.float32, (3,), Frame.TABLE,
                         Unit.MM, validity=True, persist=Persist.TRAJECTORY)
    overlay_spec = TrackSpec("object_mesh_overlay", overlay.dtype,
                             tuple(overlay.shape[1:]), Frame.IMAGE,
                             Unit.UNITLESS, persist=Persist.DIAGNOSTIC)
    return [Track(pos_spec, obj_pos, ~gap),
            Track(overlay_spec, overlay)]
