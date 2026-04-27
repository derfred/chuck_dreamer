import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from .scene_config import SceneConfig

if TYPE_CHECKING:
  from .data_collection import RandomPushPolicy

logger = logging.getLogger(__name__)


RawEpisode = dict[str, np.ndarray]


_STEP_KEYS = (
  "image", "action", "reward", "timestamp",
  "joint_qpos", "ee_pos", "ee_quat", "object_xy",
)


def _stack_steps(steps: list[dict[str, Any]]) -> RawEpisode:
  """Stack a list of per-step dicts into a dict of T-stacked arrays."""
  return {k: np.stack([np.asarray(s[k]) for s in steps], axis=0) for k in _STEP_KEYS}


class ScenePlayer:
  """
  Drives a PushingEnv with a RandomPushPolicy.

  Shared logic between the interactive viewer (show-scene) and the
  headless episode collector (generate-scenes).
  """

  def __init__(self, config, env, policy: "RandomPushPolicy") -> None:
    self.config = config
    self.env    = env
    self.policy = policy

  @property
  def state(self) -> str | None:
    return getattr(self.policy, "state", None)

  def advance_from_ready(self) -> None:
    if self.state == "ready":
      self.policy.state = "approach"

  def reset(self) -> SceneConfig:
    scene: SceneConfig = self.env.generate_scene()
    self.scene = scene
    self.env.reset(scene=scene)
    self.policy.reset(self.env.controller, scene)
    return scene

  def _step_once(self, obs: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray], bool, bool, str | None]:
    action, prev_state = self.policy.act(obs)
    next_obs, reward, terminated, truncated, _ = self.env.step(action)
    step = {
        "image":      obs["image"],
        "action":     np.asarray(action.qpos, dtype=np.float32),
        "reward":     float(reward),
        "timestamp":  float(next_obs["time"]),
        "joint_qpos": np.asarray(next_obs["arm_qpos"], dtype=np.float32),
        "ee_pos":     np.asarray(next_obs["ee_pos"],   dtype=np.float32),
        "ee_quat":    np.asarray(next_obs["ee_quat"],  dtype=np.float32),
        "object_xy":  np.asarray(next_obs["object_xy"], dtype=np.float32),
    }
    return step, next_obs, bool(terminated), bool(truncated), prev_state

  def run_interactive(self, viewer, step_delay: float) -> None:
    """Drive the simulation via a MuJoCo passive viewer.

    The env must already be reset before calling (the caller needs
    ``env.model``/``env.data`` to construct the viewer). The caller
    also owns any key_callback used to advance ``ready → approach``.
    Hints are drawn while the policy is in state "ready".
    """
    obs = self.env._get_obs()
    while viewer.is_running():
      step, obs, terminated, truncated, prev_state = self._step_once(obs)
      if prev_state is not None:
        print(f"Policy state changed: {prev_state} → {self.policy.state}")

      self.policy.insert_hints(viewer)

      viewer.sync()
      if step_delay > 0:
        time.sleep(step_delay)

      if terminated or truncated or self.policy.state == "done":
        break

  def run_headless(self, max_steps: int) -> tuple[RawEpisode | None, str]:
    """Run until completion, robust to simulation crashes.

    Auto-advances ``ready → approach`` (no manual step). Does not render
    hints. Catches exceptions raised during ``policy.act`` or
    ``env.step`` (e.g. IK non-convergence) and returns the data
    collected up to that point.

    Returns ``(episode, outcome)`` where ``episode`` is a ``RawEpisode``
    (dict of T-stacked arrays) or ``None`` if no steps were collected,
    and ``outcome`` is one of:
      - "done":       policy reached its done state
      - "terminated": env signaled terminated/truncated
      - "timeout":    ``max_steps`` reached
      - "crashed":    an exception was raised during stepping
    """
    steps: list[dict[str, Any]] = []
    outcome = "timeout"
    try:
      obs, _ = self.env.reset(scene=self.scene)
      for _ in range(max_steps):
        if self.policy.state == "ready":
          self.policy.state = "approach"

        step, obs, terminated, truncated, _ = self._step_once(obs)
        steps.append(step)

        if self.policy.state == "done":
          outcome = "done"
          break
        if terminated or truncated:
          outcome = "terminated"
          break
    except Exception as e:
      logger.warning("Simulation crashed after %d steps: %s", len(steps), e)
      outcome = "crashed"

    if not steps:
      return None, outcome
    return _stack_steps(steps), outcome
