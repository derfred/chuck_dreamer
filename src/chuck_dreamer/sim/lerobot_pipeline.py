"""Post-import stages for the LeRobot importer.

Two stages are applied to each freshly-imported episode dict, each guarded
by its own CLI flag in ``main.py import-lerobot``:

  * :func:`apply_ee_pos` — rescale ``joint_qpos`` from servo units to
    radians, then run the MLX FK MLP (trained by
    ``scripts/train_fk_mlp.py``) to fill ``ee_pos``, ``ee_quat`` and
    ``ee_action``.

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

import math
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten


# ---------------------------------------------------------------------------
# Stage: ee_pos / ee_quat / ee_action
# ---------------------------------------------------------------------------

# (low, high) such that input -100 maps to ``low`` and +100 maps to ``high``.
# joint0/3/4/5 flip sign; joints 1/2 go monotonically in their stated direction.
_JOINT_RANGES: tuple[tuple[float, float], ...] = (
  (+math.pi / 2,  -math.pi / 2),  # joint 0
  ( 0.0,          -math.pi),      # joint 1
  (+math.pi,       0.0),          # joint 2
  (+math.pi / 2,  -math.pi / 2),  # joint 3
  (+math.pi / 2,  -math.pi / 2),  # joint 4
  (+math.pi / 2,  -math.pi / 2),  # joint 5
)

_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[3] / "ee_pos_model.safetensors"

# MuJoCo [w, x, y, z] for a 180° rotation about x — gripper z points down.
_EE_QUAT_DOWN: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)


class _MLP(nn.Module):
  def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int):
    super().__init__()
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
      layers.append(nn.Linear(prev, h))
      layers.append(nn.GELU())
      prev = h
    layers.append(nn.Linear(prev, out_dim))
    self.net = nn.Sequential(*layers)

  def __call__(self, x: mx.array) -> mx.array:
    return self.net(x)


def _rescale_joints(qpos: np.ndarray) -> np.ndarray:
  out = qpos.astype(np.float32, copy=True)
  n   = min(out.shape[1], len(_JOINT_RANGES))
  np.clip(out[:, :n], -100.0, 100.0, out=out[:, :n])
  t = (out[:, :n] + 100.0) / 200.0
  for i in range(n):
    lo, hi = _JOINT_RANGES[i]
    out[:, i] = lo + t[:, i] * (hi - lo)
  return out


def _hidden_dims_from_weights(weights: dict[str, mx.array]) -> tuple[int, ...]:
  idxs = sorted(
    int(k.split(".")[2]) for k in weights
    if k.startswith("net.layers.") and k.endswith(".weight")
  )
  return tuple(weights[f"net.layers.{i}.weight"].shape[0] for i in idxs[:-1])


def _build_model(weights_path: Path) -> tuple[_MLP, mx.array, mx.array]:
  weights = mx.load(str(weights_path))
  x_mean  = weights.pop("_x_mean")
  x_std   = weights.pop("_x_std")
  hidden  = _hidden_dims_from_weights(weights)
  in_dim  = weights["net.layers.0.weight"].shape[1]
  last    = max(int(k.split(".")[2]) for k in weights if k.endswith(".weight"))
  out_dim = weights[f"net.layers.{last}.weight"].shape[0]

  model = _MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
  model.update(tree_unflatten(list(weights.items())))
  mx.eval(model.parameters())
  return model, x_mean, x_std


_model: _MLP | None = None
_x_mean: mx.array | None = None
_x_std:  mx.array | None = None


def _ensure_model() -> tuple[_MLP, mx.array, mx.array]:
  global _model, _x_mean, _x_std
  if _model is None:
    weights_path = Path(os.environ.get("FK_MLP_WEIGHTS", str(_DEFAULT_WEIGHTS)))
    if not weights_path.exists():
      raise FileNotFoundError(
        f"FK weights not found at {weights_path}. Train them first: "
        f"`uv run python scripts/train_fk_mlp.py --save {weights_path}`")
    _model, _x_mean, _x_std = _build_model(weights_path)
  assert _x_mean is not None and _x_std is not None
  return _model, _x_mean, _x_std


def apply_ee_pos(episode: dict[str, Any], metadata: dict[str, Any]) -> None:
  """Rescale joint_qpos, run FK, and fill ee_pos/ee_quat/ee_action in place."""
  qpos = episode.get("joint_qpos")
  if qpos is None or qpos.size == 0:
    return

  rescaled = _rescale_joints(np.asarray(qpos))
  episode["joint_qpos"] = rescaled

  model, x_mean, x_std = _ensure_model()
  fk_input = rescaled[:, :6]
  x = (mx.array(fk_input) - x_mean) / x_std
  ee_pos = np.asarray(model(x), dtype=np.float32)
  episode["ee_pos"] = ee_pos

  T = ee_pos.shape[0]
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
