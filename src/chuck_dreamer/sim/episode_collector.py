"""Headless episode collection — env-agnostic of policy internals.

Used both by ``generate-scenes`` (offline collection) and by the
trainer's collect phase. The collector maps:

  reset() → SceneConfig
  run()   → (RawEpisode | None, outcome)

It does not poll ``policy.is_done`` or ``policy.state``: termination is
the env's job. ``outcome ∈ {"done", "terminated", "timeout", "crashed"}``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ..common.episode import Episode
from .observation_image import ObservationImage
from .scene_config import SceneConfig
from .step_info import StepInfo, stack_step_infos

if TYPE_CHECKING:
  from ..policy import Policy
  from .pushing_env import PushingEnv

logger = logging.getLogger(__name__)


RawEpisode = dict[str, Any]


_SHARED_KEYS = (
  "reward", "timestamp",
  "joint_qpos", "ee_pos", "ee_quat", "object_xy",
)


def mask_centroids_uv(masks: np.ndarray) -> np.ndarray:
  """Per-step pixel centroid ``(u, v)`` of a ``(T, H, W)`` bool mask stack.

  ``u`` is the column coordinate, ``v`` is the row coordinate (matching
  the image-coordinate convention used by OpenCV / Rerun). Steps whose
  mask is empty get ``NaN`` so downstream code can detect dropouts
  without silently snapping to 0. Returns ``(T, 2)`` float32.
  """
  if masks.ndim != 3:
    raise ValueError(f"mask_centroids_uv expects (T, H, W); got shape {masks.shape!r}")
  m = masks.astype(np.float32, copy=False)
  counts = m.sum(axis=(1, 2))                                    # (T,)
  H, W   = m.shape[1], m.shape[2]
  vs     = m.sum(axis=2) @ np.arange(H, dtype=np.float32)        # (T,) row weight
  us     = m.sum(axis=1) @ np.arange(W, dtype=np.float32)        # (T,) col weight
  with np.errstate(invalid="ignore", divide="ignore"):
    u = np.where(counts > 0, us / counts, np.float32("nan"))
    v = np.where(counts > 0, vs / counts, np.float32("nan"))
  return np.stack([u, v], axis=-1).astype(np.float32)


def _stack_image_obs(image_obs_list: list[ObservationImage]) -> dict[str, np.ndarray]:
  """Stack a per-step list of :class:`ObservationImage` into T-stacked arrays.

  Returns ``image`` ``(T, H, W, 3)`` and a ``segmentation_*`` field per
  named mask (``target``, ``goal``, ``arm``, ``background``, plus
  per-piece ``clutter_{i}`` if any). The per-piece clutter masks are also
  stacked along a leading clutter axis under ``segmentation_clutter`` as
  ``(T, K, H, W)`` for convenience (K is the scene's clutter count, fixed
  across the episode).

  Also derives ``object_uv`` ``(T, 2)`` float32 — the pixel centroid of
  the target mask, in (u, v) order. NaN per-step when the target is not
  visible in the frame.
  """
  images = np.stack([np.asarray(io.image, dtype=np.uint8) for io in image_obs_list], axis=0)
  target = np.stack([io.target_mask for io in image_obs_list], axis=0).astype(bool)
  goal   = np.stack([io.goal_mask   for io in image_obs_list], axis=0).astype(bool)

  out: dict[str, np.ndarray] = {
    "image":              images,
    "segmentation_target": target,
    "segmentation_goal":   goal,
    "object_uv":          mask_centroids_uv(target),
  }

  arm_list = [io.arm_mask for io in image_obs_list]
  if all(m is not None for m in arm_list):
    out["segmentation_arm"] = np.stack(arm_list, axis=0).astype(bool)  # type: ignore[arg-type]

  bg_list = [io.background_mask for io in image_obs_list]
  if all(m is not None for m in bg_list):
    out["segmentation_background"] = np.stack(bg_list, axis=0).astype(bool)  # type: ignore[arg-type]

  # Clutter count is fixed by the scene, so stack into (T, K, H, W).
  k = len(image_obs_list[0].clutter_masks)
  if k > 0:
    clutter = np.stack(
      [np.stack(io.clutter_masks, axis=0) for io in image_obs_list],
      axis=0,
    ).astype(bool)
    out["segmentation_clutter"] = clutter

  return out


def _stack_steps(steps: list[dict[str, Any]], action_kind: str) -> Episode:
  """Stack per-step records into a T-stacked :class:`Episode`, plus the action
  under its kind name.

  The per-step arrays + segmentation masks become normal (persisted) episode
  fields; ``step_info`` is attached as a non-persisted ("scratch") field — it is
  consumed in-memory by the replay buffer but never written to a recording, so
  it rides on the episode without reaching the writers."""
  arrays: dict[str, Any] = {
    k: np.stack([np.asarray(s[k]) for s in steps], axis=0) for k in _SHARED_KEYS
  }
  arrays.update(_stack_image_obs([s["image_obs"] for s in steps]))
  arrays[action_kind] = np.stack([np.asarray(s["action"]) for s in steps], axis=0)

  episode = Episode.from_arrays(arrays)
  episode.set("step_info", stack_step_infos([s["step_info"] for s in steps]),
              persist=False)
  return episode


class EpisodeCollector:
  """Drive any ``Policy`` in any ``PushingEnv``; record one episode."""

  def __init__(self, env: "PushingEnv", policy: "Policy") -> None:
    self.env    = env
    self.policy = policy
    self.scene: SceneConfig | None = None

  def reset(self) -> SceneConfig:
    """Sample a scene, reset env+policy, leave them ready for ``run()``."""
    scene: SceneConfig = self.env.generate_scene()
    self.scene = scene
    self.env.reset(scene=scene)
    self.policy.reset(scene)
    return scene

  def run(self) -> tuple[Episode | None, str]:
    """Roll one episode using ``scene.max_steps`` as the cap.

    Returns ``(episode, outcome)`` where ``episode`` is an :class:`Episode`
    (T-stacked array fields plus a non-persisted ``step_info``) or ``None`` if
    no steps were collected. Catches exceptions raised during ``policy.act`` or
    ``env.step`` (e.g. IK non-convergence) and reports ``"crashed"`` with
    whatever data was collected up to that point.
    """
    assert self.scene is not None, "Call reset() before run()."

    act_mode    = self.env.act_mode
    action_kind = "joint_action" if act_mode == "joint" else "ee_action"

    steps: list[dict[str, Any]] = []
    outcome = "timeout"
    try:
      obs, _ = self.env.reset(scene=self.scene)
      for _ in range(self.scene.max_steps):
        # Policies receive the full obs dict. State-mode policies (scripted)
        # read the named keys; modal policies (Dreamer in §6+) project
        # internally via env.policy_obs or by knowing their obs_mode.
        action                                        = self.policy.act(obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        step_info: StepInfo = info["step_info"]
        steps.append({
          "image_obs":  obs["image"],
          "action":     np.asarray(self.env.as_array(action), dtype=np.float32),
          "reward":     float(reward),
          "timestamp":  float(step_info.time),
          "joint_qpos": np.asarray(next_obs["arm_qpos"], dtype=np.float32),
          "ee_pos":     np.asarray(next_obs["ee_pos"],   dtype=np.float32),
          "ee_quat":    np.asarray(next_obs["ee_quat"],  dtype=np.float32),
          "object_xy":  np.asarray(next_obs["object_xy"], dtype=np.float32),
          "step_info":  step_info,
        })

        obs = next_obs
        if terminated:
          outcome = "done"
          break
        if truncated:
          outcome = "timeout"
          break
    except Exception as e:
      logger.warning("Simulation crashed after %d steps: %s", len(steps), e)
      outcome = "crashed"

    if not steps:
      return None, outcome
    return _stack_steps(steps, action_kind), outcome
