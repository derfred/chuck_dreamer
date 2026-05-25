"""Post-import stages for the LeRobot importer.

Two stages are applied to each freshly-imported episode dict, each guarded
by its own CLI flag in ``main.py import-lerobot``:

  * :func:`apply_ee_pos` — convert ``joint_qpos`` (degrees, as stored in
    LeRobot ``observation.state``) to radians for the 5 positioning
    joints, then run the MuJoCo FK in
    :mod:`chuck_dreamer.real.fk_calibration.fk` to fill ``ee_pos``,
    ``ee_quat`` and ``ee_action``.

  * :func:`apply_object_pose` — run the calibrated
    :mod:`chuck_dreamer.real.object_localization` pipeline on the
    episode's frames (SAM2 segmentation + per-frame pose fit + RTS
    smoothing) to fill ``object_xy`` and ``object_gap_too_long``.
    Requires a per-source-repo calibration cache entry and a cached
    frame-0 prompt; missing either is a hard error.

Module-level caches keyed by ``source_repo`` amortise FK-model and
object-localization runtime loads across episodes in one import run.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Stage: ee_pos / ee_quat / ee_action
# ---------------------------------------------------------------------------

# Path to the SO-101 MuJoCo model used by the FK module. Mirrors the
# default used by ``_build_arm_calibration`` in ``main.py``.
FK_MODEL_PATH = (
  Path(__file__).resolve().parents[3]
  / "assets" / "mujoco" / "so101_arm.xml"
)

# MuJoCo [w, x, y, z] for a 180° rotation about x — gripper z points down.
_EE_QUAT_DOWN: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)

# Number of positioning joints consumed by FK (rest of the state vector is
# the gripper, which doesn't affect EE position).
_N_POSITIONING_JOINTS = 5


_fk = None


def _ensure_fk():
  """Lazy-load the MuJoCo FK evaluator. Cached across episodes."""
  global _fk
  if _fk is None:
    if not FK_MODEL_PATH.exists():
      raise FileNotFoundError(
        f"FK MuJoCo model not found at {FK_MODEL_PATH}. "
        f"Restore assets/mujoco/so101_arm.xml from git.")
    from chuck_dreamer.real.fk_calibration.fk import FK
    _fk = FK(FK_MODEL_PATH)
  return _fk


def apply_ee_pos(episode: dict[str, Any], metadata: dict[str, Any]) -> None:
  """Convert joint_qpos to radians, run FK, and fill ee_* in place.

  ``joint_qpos`` comes in from LeRobot ``observation.state`` in
  **degrees** with 6 columns (5 positioning joints + gripper). We
  convert the positioning joints to radians (the gripper column is left
  untouched if present) and call the MuJoCo FK for each frame.
  """
  qpos = episode.get("joint_qpos")
  if qpos is None or qpos.size == 0:
    return

  qpos = np.asarray(qpos, dtype=np.float32)
  n = min(qpos.shape[1], _N_POSITIONING_JOINTS)
  rescaled = qpos.copy()
  rescaled[:, :n] = np.deg2rad(rescaled[:, :n])
  episode["joint_qpos"] = rescaled

  fk = _ensure_fk()
  T = rescaled.shape[0]
  ee_pos = np.empty((T, 3), dtype=np.float32)
  for t in range(T):
    ee_pos[t] = fk(rescaled[t, :_N_POSITIONING_JOINTS]).astype(np.float32)
  episode["ee_pos"] = ee_pos

  ee_quat = np.tile(np.asarray(_EE_QUAT_DOWN, dtype=np.float32), (T, 1))
  episode["ee_quat"] = ee_quat

  # action[t] is the target pose for step t+1; duplicate the last pose
  # so the action stream matches the recording length.
  ee_pose = np.concatenate([ee_pos, ee_quat], axis=-1)
  episode["ee_action"] = np.concatenate(
    [ee_pose[1:], ee_pose[-1:]], axis=0,
  ).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage: object pose (SAM2 segmentation + per-frame fit + RTS smoothing)
# ---------------------------------------------------------------------------

_estimator_cache: dict[str, Any] = {}
_smoother_cache:  dict[str, Any] = {}
_ol_cfg_initialized: bool = False


def _ensure_ol_runtime() -> Any:
  """Load chuck_dreamer's default config and install the OL runtime.

  Raises ``RuntimeError`` if the package or config is unavailable —
  object-pose estimation needs the runtime to be installable; we do
  not fall back to zero-fill.
  """
  global _ol_cfg_initialized
  from chuck_dreamer.config import load_config
  from chuck_dreamer.real.object_localization import active, init_from_config

  if not _ol_cfg_initialized:
    init_from_config(load_config())
    _ol_cfg_initialized = True
  return active()


def _ensure_estimator(source_repo: str) -> Any:
  """Build (or fetch) an ObjectPoseEstimator for ``source_repo``.

  Raises ``FileNotFoundError`` if the calibration cache entry or mesh
  is missing.
  """
  if source_repo in _estimator_cache:
    return _estimator_cache[source_repo]

  ol_cfg = _ensure_ol_runtime()
  cache_dir = Path(ol_cfg.calibration_cache)
  mesh_path = Path(ol_cfg.mesh_path)

  from chuck_dreamer.real.object_localization import (
    ObjectPoseEstimator,
    load_calibration,
  )

  cal = load_calibration(cache_dir, source_repo)  # may raise FileNotFoundError

  if not mesh_path.exists():
    raise FileNotFoundError(
      f"mesh file not found at {mesh_path}; check "
      f"object_localization.mesh_path in configs/default.yaml.")

  estimator = ObjectPoseEstimator(
    cal, mesh_path, device=ol_cfg.device, use_sam2=ol_cfg.use_sam2,
    scene_bg=None,
  )
  _estimator_cache[source_repo] = estimator
  return estimator


def _ensure_smoother(source_repo: str) -> Any:
  if source_repo in _smoother_cache:
    return _smoother_cache[source_repo]
  ol_cfg = _ensure_ol_runtime()
  from chuck_dreamer.real.object_localization.smoother import SmoothedTrajectoryEstimator
  smoother_cfg = ol_cfg.raw.get("smoother") or {}
  smoother = SmoothedTrajectoryEstimator(smoother_cfg)
  _smoother_cache[source_repo] = smoother
  return smoother


def apply_object_pose(episode: dict[str, Any], metadata: dict[str, Any]) -> None:
  """Run pose estimation + RTS smoothing in place.

  Writes ``object_xy`` and ``object_gap_too_long``. Requires a cached
  calibration entry for ``metadata['config']['source_repo']`` and a
  cached frame-0 prompt under ``calibration_cache/<slug>/object_prompts.json``.
  Missing either raises (no interactive fallback).
  """
  cfg = metadata.get("config")
  if not isinstance(cfg, dict):
    raise RuntimeError("apply_object_pose: metadata['config'] missing or invalid")
  source_repo = cfg.get("source_repo")
  if not isinstance(source_repo, str) or not source_repo:
    raise RuntimeError("apply_object_pose: metadata['config']['source_repo'] missing")
  images = episode.get("image")
  if images is None or len(images) == 0:
    return

  estimator = _ensure_estimator(source_repo)

  imgs = np.asarray(images)
  if imgs.dtype != np.uint8:
    imgs = (np.clip(imgs.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
  if imgs.ndim == 4 and imgs.shape[1] in (1, 3) and imgs.shape[-1] not in (1, 3):
    imgs = np.transpose(imgs, (0, 2, 3, 1))

  ol_cfg = _ensure_ol_runtime()
  cache_dir = Path(ol_cfg.calibration_cache)
  episode_index = int(cfg.get("episode_index", 0))

  from chuck_dreamer.real.object_localization.prompts import load_keyframe_prompts

  keyframes = load_keyframe_prompts(cache_dir, source_repo, episode_index)
  if 0 not in keyframes:
    raise FileNotFoundError(
      f"no cached frame-0 prompt for {source_repo} episode {episode_index}. "
      f"Run: `python main.py prompt-episodes --dataset {source_repo}`")

  T = len(imgs)
  keyframes = {fi: pr for fi, pr in keyframes.items() if 0 <= fi < T}
  print(f"[apply_object_pose] {source_repo} ep{episode_index}: "
        f"using {len(keyframes)} keyframe prompt(s) at frames "
        f"{sorted(keyframes.keys())}")

  poses = estimator.estimate_episode(
    list(imgs),
    first_frame_prompt=keyframes[0],
    keyframe_prompts=keyframes,
  )
  n_ok = sum(1 for p in poses if p is not None)
  print(f"[apply_object_pose] {source_repo} ep{episode_index}: "
        f"estimator returned {n_ok}/{len(poses)} non-None poses")

  masks = getattr(estimator, "last_masks", None)
  if masks is not None:
    _store_masks_and_uv(episode, list(imgs), masks)

  debug_dir = os.environ.get("IMPORT_LEROBOT_DEBUG_MASKS")
  if debug_dir:
    _dump_debug_masks(
      Path(debug_dir), source_repo, episode_index,
      list(imgs), getattr(estimator, "last_masks", None),
    )

  smoother = _ensure_smoother(source_repo)
  if all(p is None for p in poses) or poses[0] is None:
    out = np.zeros((T, 2), dtype=np.float32)
    for i, p in enumerate(poses):
      if p is not None:
        out[i] = p.xy_mm.astype(np.float32)
    episode["object_xy"] = out
    episode["object_gap_too_long"] = np.ones((T,), dtype=np.bool_)
    return

  frames = smoother.smooth_episode(poses)
  xy_arr = np.stack([f.xy_mm for f in frames], axis=0).astype(np.float32)
  episode["object_xy"] = xy_arr
  episode["object_gap_too_long"] = np.array(
    [f.gap_too_long for f in frames], dtype=np.bool_,
  )


def _store_masks_and_uv(episode: dict[str, Any],
                        imgs: list[np.ndarray],
                        masks: list[np.ndarray | None]) -> None:
  """Fill ``segmentation_target`` (T, H, W) uint8 and ``object_uv`` (T, 2)
  float32 on the episode dict from per-frame SAM2 masks.

  Pixels carry label 1 inside the target mask, 0 elsewhere. For frames
  where SAM2 returned no mask we emit a zero mask and ``NaN`` centroids
  so downstream tooling can tell drop-outs apart from "mask centred at
  origin".
  """
  T = len(imgs)
  if T == 0:
    return
  ref = next((np.asarray(m) for m in masks if m is not None), None)
  if ref is None:
    return
  H, W = ref.shape[:2]
  seg_arr = np.zeros((T, H, W), dtype=np.uint8)
  uv_arr  = np.full((T, 2), np.nan, dtype=np.float32)
  for t, m in enumerate(masks):
    if m is None:
      continue
    m_arr = np.asarray(m, dtype=bool)
    if m_arr.shape[:2] != (H, W):
      continue
    seg_arr[t][m_arr] = 1
    ys, xs = np.where(m_arr)
    if xs.size:
      uv_arr[t, 0] = float(xs.mean())
      uv_arr[t, 1] = float(ys.mean())
  episode["segmentation_target"] = seg_arr
  episode["object_uv"]           = uv_arr


def _dump_debug_masks(out_root: Path, source_repo: str, episode_index: int,
                      imgs: list[np.ndarray],
                      masks: list[np.ndarray | None] | None) -> None:
  """Dump per-frame mask overlays for visual inspection.

  Triggered by setting ``IMPORT_LEROBOT_DEBUG_MASKS=<dir>``. Writes one
  PNG per frame to ``<dir>/<dataset_slug>/ep<NNN>/frame_<NNNNN>.png``,
  with the SAM2 mask shown as a red 50% overlay on the source image.
  Frames where segmentation failed are still dumped (no overlay) so
  drop-outs are visible by scrubbing the folder.
  """
  if masks is None:
    print(f"[debug-masks] no masks available on estimator; skipping dump")
    return
  try:
    from PIL import Image
  except ImportError:
    print(f"[debug-masks] PIL not available; skipping dump")
    return

  from chuck_dreamer.real.object_localization.types import dataset_slug

  ep_dir = out_root / dataset_slug(source_repo) / f"ep{episode_index:03d}"
  ep_dir.mkdir(parents=True, exist_ok=True)
  n_with_mask = 0
  for t, (img, mask) in enumerate(zip(imgs, masks)):
    if img.dtype != np.uint8:
      img_u8 = (np.clip(img.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
      img_u8 = img
    if img_u8.ndim == 3 and img_u8.shape[-1] == 1:
      img_u8 = np.repeat(img_u8, 3, axis=-1)
    if img_u8.ndim == 2:
      img_u8 = np.stack([img_u8] * 3, axis=-1)
    out = img_u8.copy()
    if mask is not None:
      n_with_mask += 1
      m = np.asarray(mask, dtype=bool)
      if m.shape[:2] == out.shape[:2]:
        red = np.array([255, 0, 0], dtype=np.uint8)
        out[m] = (out[m].astype(np.uint16) * 1 // 2
                  + red.astype(np.uint16) * 1 // 2).astype(np.uint8)
    Image.fromarray(out).save(ep_dir / f"frame_{t:05d}.png")
  print(f"[debug-masks] wrote {len(imgs)} frames "
        f"({n_with_mask} with mask) to {ep_dir}")
