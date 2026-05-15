"""Convert raw sim-recorded episodes into the replay buffer's schema.

Episode readers (``episode_loader``) produce raw dicts of arrays straight
from the writer format. Processors here project them onto the modal
observation chosen by ``config.env.obs_mode`` and align with the buffer's
``(T+1 obs, T action)`` invariant.

The set of processors mirrors the env's ``policy_obs()`` projection on the
online side: ``StateVectorProcessor`` matches ``obs_mode="state"``,
``ImageProcessor`` matches ``"image"``, ``ImageProprioProcessor`` matches
``"image_proprio"``.
"""

from __future__ import annotations

from typing import Any, Protocol

import cv2  # type: ignore[import-not-found]
import numpy as np


RawEpisode = dict[str, Any]
Episode = dict[str, Any]


_STEP_INFO_KEYS = ("object_xy", "ee_pos", "ee_quat")


class EpisodeProcessor(Protocol):
  """Converts a raw sim episode into replay-buffer schema.

  A processor must return a dict with keys ``obs`` (T+1, *obs_shape),
  ``action`` (T, action_dim), ``reward`` (T,), ``done`` (T,), and
  ``step_info`` (a dict of T+1-length arrays mirroring StepInfo columns).
  """

  def __call__(self, raw: RawEpisode) -> Episode: ...


def _slice_step_info(raw: RawEpisode, n: int) -> dict[str, np.ndarray]:
  """Extract per-step info columns sliced to length ``n``.

  In the recorded format, ``step_info[t]`` describes the state AFTER
  action_t — i.e. it aligns with the buffer's ``reward[t]`` (reward
  earned by taking ``action_t``). Callers pass ``n = T = N - 1`` so the
  result aligns with the action/reward/done axis.
  """
  out: dict[str, np.ndarray] = {}
  for key in _STEP_INFO_KEYS:
    out[key] = np.asarray(raw[key], dtype=np.float32)[:n]
  if "timestamp" in raw:
    out["time"] = np.asarray(raw["timestamp"], dtype=np.float32)[:n]
  return out


def _resolve_action(raw: RawEpisode) -> np.ndarray:
  if "joint_action" in raw:
    return np.asarray(raw["joint_action"], dtype=np.float32)
  if "ee_action" in raw:
    return np.asarray(raw["ee_action"], dtype=np.float32)
  raise KeyError("raw episode missing an action field (joint_action / ee_action)")


def _drop_last_and_pack(obs: np.ndarray, raw: RawEpisode) -> Episode:
  """Align a per-step obs array with the buffer's (T+1 obs, T action) layout.

  Sim episodes record N aligned steps of (obs_t, action_t, reward_t).
  We treat the N recorded obs as ``obs[0..N-1]`` and drop the last
  action/reward, yielding N obs and N-1 actions — i.e. T = N-1 with
  T+1 = N obs. The final ``done`` is set to True. ``step_info`` is
  sliced to length N (matching obs).
  """
  N = obs.shape[0]
  if N < 2:
    raise ValueError(f"episode too short: {N} steps (need >= 2)")

  action = _resolve_action(raw)[: N - 1]
  reward = np.asarray(raw["reward"], dtype=np.float32)[: N - 1]
  done = np.zeros((N - 1,), dtype=bool)
  done[-1] = True

  goal_xy = None
  if "goal_xy" in raw:
    goal_xy = np.asarray(raw["goal_xy"], dtype=np.float32)

  ep: Episode = {
    "obs": obs.astype(obs.dtype, copy=False),
    "action": action,
    "reward": reward,
    "done": done,
    "step_info": _slice_step_info(raw, N - 1),
  }
  if goal_xy is not None:
    ep["goal_xy"] = goal_xy
  return ep


def _resize_image_stack(images: np.ndarray, target: int) -> np.ndarray:
  """Resize a (N, H, W, 3) uint8 stack to (N, target, target, 3) uint8."""
  if images.ndim != 4 or images.shape[-1] != 3:
    raise ValueError(f"expected (N, H, W, 3) image stack; got shape {images.shape}")
  N, H, W, _ = images.shape
  if H == target and W == target:
    return images.astype(np.uint8, copy=False)
  out = np.empty((N, target, target, 3), dtype=np.uint8)
  for i in range(N):
    out[i] = cv2.resize(images[i], (target, target), interpolation=cv2.INTER_AREA)
  return out


def _normalize_image_stack(images: np.ndarray) -> np.ndarray:
  """Map a (N, H, W, 3) uint8 stack to float32 in ``[-0.5, 0.5]``.

  Matches the CNN encoder's input contract so downstream code can feed
  the processor output straight to the encoder without re-normalizing.
  Float inputs are returned as float32 unchanged.
  """
  arr = np.asarray(images)
  if arr.dtype == np.uint8:
    return arr.astype(np.float32) / 255.0 - 0.5
  return arr.astype(np.float32, copy=False)


class StateVectorProcessor:
  """Default processor: concat ee_pos + ee_quat + object_xy + joint_qpos.

  Produces a flat float32 observation vector per step. Order is fixed
  so downstream consumers can unpack it consistently.
  """

  def __call__(self, raw: RawEpisode) -> Episode:
    ee_pos = np.asarray(raw["ee_pos"], dtype=np.float32)
    ee_quat = np.asarray(raw["ee_quat"], dtype=np.float32)
    object_xy = np.asarray(raw["object_xy"], dtype=np.float32)
    joint_qpos = np.asarray(raw["joint_qpos"], dtype=np.float32)
    obs = np.concatenate([ee_pos, ee_quat, object_xy, joint_qpos], axis=1)
    return _drop_last_and_pack(obs, raw)


class ImageProcessor:
  """Image-only obs: ``(N, image_size, image_size, 3)`` float32 per episode.

  Images are resized to ``image_size`` then normalized to ``[-0.5, 0.5]``
  to match the encoder's input contract, so the model can consume the
  obs straight from the buffer without re-normalizing.
  """

  def __init__(self, image_size: int):
    self.image_size = int(image_size)

  def __call__(self, raw: RawEpisode) -> Episode:
    images = np.asarray(raw["image"], dtype=np.uint8)
    obs = _normalize_image_stack(_resize_image_stack(images, self.image_size))
    return _drop_last_and_pack(obs, raw)


class ImageProprioProcessor:
  """Image + proprio obs as a dict ``{"image": ..., "proprio": ...}``.

  proprio = ee_pos (3) + ee_quat (4) + joint_qpos (n_joints). No object_xy:
  proprioception is body-internal. Images are resized to ``image_size``
  then normalized to float32 in ``[-0.5, 0.5]`` so they're ready for the
  encoder without further normalization.
  """

  def __init__(self, image_size: int):
    self.image_size = int(image_size)

  def __call__(self, raw: RawEpisode) -> Episode:
    images = np.asarray(raw["image"], dtype=np.uint8)
    images = _normalize_image_stack(_resize_image_stack(images, self.image_size))
    ee_pos = np.asarray(raw["ee_pos"], dtype=np.float32)
    ee_quat = np.asarray(raw["ee_quat"], dtype=np.float32)
    joint_qpos = np.asarray(raw["joint_qpos"], dtype=np.float32)
    proprio = np.concatenate([ee_pos, ee_quat, joint_qpos], axis=1)

    N = images.shape[0]
    if N < 2:
      raise ValueError(f"episode too short: {N} steps (need >= 2)")

    action = _resolve_action(raw)[: N - 1]
    reward = np.asarray(raw["reward"], dtype=np.float32)[: N - 1]
    done = np.zeros((N - 1,), dtype=bool)
    done[-1] = True

    ep: Episode = {
      "obs": {"image": images, "proprio": proprio},
      "action": action,
      "reward": reward,
      "done": done,
      "step_info": _slice_step_info(raw, N - 1),
    }
    if "goal_xy" in raw:
      ep["goal_xy"] = np.asarray(raw["goal_xy"], dtype=np.float32)
    return ep


def processor_for(config) -> EpisodeProcessor:
  """Build the processor matching ``config.env.obs_mode``.

  For image modes ``config.model.encoder.image_size`` (computed by the
  ``derive_image_size`` OmegaConf resolver from the encoder strides) is
  the side length the loader resizes captured frames to.
  """
  obs_mode = str(config.env.obs_mode)
  if obs_mode == "state":
    return StateVectorProcessor()
  if obs_mode in ("image", "image_proprio"):
    image_size = int(config.model.encoder.image_size)
    return ImageProcessor(image_size) if obs_mode == "image" else ImageProprioProcessor(image_size)
  raise ValueError(f"Unknown obs_mode: {obs_mode!r}")
