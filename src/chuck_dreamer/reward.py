"""Reward functions.

A ``RewardFn`` maps a :class:`StepInfo` to a scalar reward. Reward is
recomputed in the replay buffer at sample time (not baked into recorded
episodes), so swapping ``cfg.env.reward`` at training start changes the
next batch without re-collecting.

Reward kinds:

  * ``goal_distance`` — legacy ``-‖object_xy − goal_xy‖``. Kept as a
    fallback that works against any episode carrying ``goal_xy``.
  * ``line_push``     — Eval-1: progress along the start→goal line
    + a goal-circle bonus + a corridor hinge penalty.
  * ``obstacle_push`` — Eval-2: progress + goal-bonus, plus two
    obstacle-clearance hinges (object→obstacle and gripper→obstacle).
    Obstacle position is a config parameter (per task), since the
    episode schema doesn't carry it.
  * ``teleop_proxy``  — Bridge reward for real teleop recordings
    where ``object_xy`` is still zero-filled. Reach + terminal bonus.

The shaped rewards anchor their "along the line" axis on the
``line_start_xy`` → ``goal_xy`` vector. ``line_start_xy`` is a
config-time parameter (the geometric centre of the start circle,
fixed per task); ``goal_xy`` comes from ``StepInfo`` (per-episode
metadata). Per-episode randomization of where the object actually
got placed is handled naturally — progress is measured against the
*line*, not against the object's initial position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
  from .sim.step_info import StepInfo


class RewardFn(Protocol):
  def __call__(self, info: "StepInfo") -> float: ...


# ---------------------------------------------------------------------------
# Legacy
# ---------------------------------------------------------------------------


@dataclass
class GoalDistanceReward:
  """Default reward: ``-‖object_xy - goal_xy‖``."""

  def __call__(self, info: "StepInfo") -> float:
    return -float(np.linalg.norm(info.object_xy - info.goal_xy))


# ---------------------------------------------------------------------------
# Shared geometry helper
# ---------------------------------------------------------------------------


def _project_onto_line(
  point_xy:  np.ndarray,
  start_xy:  np.ndarray,
  goal_xy:   np.ndarray,
) -> tuple[float, float, float]:
  """Project ``point_xy`` onto the start→goal segment.

  Returns ``(s_along, s_perp, length)`` where ``s_along`` is the signed
  scalar projection along the (start→goal) direction (0 at start, L at
  goal), ``s_perp`` is the absolute orthogonal distance from the line,
  and ``length`` is ``‖goal − start‖``. Degenerates to
  ``(0, ‖point − start‖, 0)`` when start == goal — keeps callers from
  dividing by zero.
  """
  line_vec = goal_xy - start_xy
  length   = float(np.linalg.norm(line_vec))
  if length < 1e-9:
    return 0.0, float(np.linalg.norm(point_xy - start_xy)), 0.0
  e_along = line_vec / length
  delta   = point_xy - start_xy
  s_along = float(np.dot(delta, e_along))
  perp    = delta - s_along * e_along
  s_perp  = float(np.linalg.norm(perp))
  return s_along, s_perp, length


# ---------------------------------------------------------------------------
# Eval-1: straight-line corridor push
# ---------------------------------------------------------------------------


@dataclass
class StraightLinePushReward:
  """Eval-1: progress along the line + goal bonus + corridor hinge.

  Per-step contributions:
    +w_progress · clip(s_along / L, 0, 1)             progress 0..1 along line
    +w_goal     · 𝟙[‖object − goal‖ < goal_radius]
    −w_corridor · max(0, s_perp − corridor_half_width)
  """

  line_start_xy:       tuple[float, float] = (-0.15, 0.18)
  corridor_half_width: float = 0.05
  goal_radius:         float = 0.05
  w_progress:          float = 1.0
  w_goal:              float = 2.0
  w_corridor:          float = 10.0

  def __call__(self, info: "StepInfo") -> float:
    start = np.asarray(self.line_start_xy, dtype=np.float32)
    s_along, s_perp, length = _project_onto_line(
      info.object_xy, start, info.goal_xy,
    )
    progress        = float(np.clip(s_along / max(length, 1e-6), 0.0, 1.0))
    corridor_excess = max(0.0, s_perp - self.corridor_half_width)
    in_goal         = float(np.linalg.norm(info.object_xy - info.goal_xy) < self.goal_radius)
    return (
        self.w_progress * progress
      + self.w_goal     * in_goal
      - self.w_corridor * corridor_excess
    )


# ---------------------------------------------------------------------------
# Eval-2: push around obstacle
# ---------------------------------------------------------------------------


@dataclass
class ObstaclePushReward:
  """Eval-2: progress + goal-bonus + two obstacle-clearance hinges.

  Same progress / goal-bonus mechanic as :class:`StraightLinePushReward`,
  but the corridor penalty is replaced by two soft barriers that
  discourage either the pushed object or the gripper from getting close
  to the obstacle.

  ``obstacle_xy`` is a config-time parameter (per task). The episode
  schema doesn't carry obstacle position, so the reward function takes
  it from the config and treats it as a fixed point. For the project
  setup, that's the midpoint of the start→goal line.

  ``r_safe_obj`` ≈ object_radius + obstacle_radius + small margin;
  ``r_safe_ee`` ≈ obstacle_radius + 4 cm gripper clearance.
  """

  line_start_xy: tuple[float, float] = (-0.15, 0.18)
  obstacle_xy:   tuple[float, float] = (0.0, 0.18)
  goal_radius:   float = 0.05
  r_safe_obj:    float = 0.06
  r_safe_ee:     float = 0.07
  w_progress:    float = 1.0
  w_goal:        float = 2.0
  w_obs_obj:     float = 20.0
  w_obs_ee:      float = 10.0

  def __call__(self, info: "StepInfo") -> float:
    start    = np.asarray(self.line_start_xy, dtype=np.float32)
    obstacle = np.asarray(self.obstacle_xy,   dtype=np.float32)

    s_along, _s_perp, length = _project_onto_line(
      info.object_xy, start, info.goal_xy,
    )
    progress  = float(np.clip(s_along / max(length, 1e-6), 0.0, 1.0))
    in_goal   = float(np.linalg.norm(info.object_xy - info.goal_xy) < self.goal_radius)
    d_obj_obs = float(np.linalg.norm(info.object_xy  - obstacle))
    d_ee_obs  = float(np.linalg.norm(info.ee_pos[:2] - obstacle))
    return (
        self.w_progress * progress
      + self.w_goal     * in_goal
      - self.w_obs_obj  * max(0.0, self.r_safe_obj - d_obj_obs)
      - self.w_obs_ee   * max(0.0, self.r_safe_ee  - d_ee_obs)
    )


# ---------------------------------------------------------------------------
# Real teleop recordings while object_xy is still zero-filled
# ---------------------------------------------------------------------------


@dataclass
class TeleopProxyReward:
  """Reach-distance proxy + terminal teleop-success bonus.

  Designed for real teleop recordings imported via
  ``main.py import-lerobot`` before the camera-calibration pipeline
  back-fills ``object_xy``. Two terms:

    +w_reach   · -‖ee_pos[:2] − goal_xy‖           reach toward the goal
    +w_success · 𝟙[step ≥ terminal_step]           terminal bonus

  The terminal bonus fires near the end of every teleop episode because
  every human demonstration succeeded by assumption — that's the
  contract you get from teleoperation. The reach term provides dense
  gradient from a quantity the recordings actually carry. Together they
  teach a coherent (if approximate) reward for CEM to plan against
  without waiting on the object localization.

  Once ``object_xy`` lands on real frames, swap to ``line_push`` /
  ``obstacle_push`` and ``--resume`` from this checkpoint — only the
  reward head needs to relearn against the new shape.
  """

  goal_xy:       tuple[float, float] = (0.15, 0.18)
  w_reach:       float = 1.0
  w_success:     float = 5.0
  terminal_step: int | None = None

  def __call__(self, info: "StepInfo") -> float:
    goal      = np.asarray(self.goal_xy, dtype=np.float32)
    reach_err = float(np.linalg.norm(info.ee_pos[:2] - goal))
    reach     = -reach_err * self.w_reach
    success   = 0.0
    if self.terminal_step is not None and int(info.step) >= self.terminal_step:
      success = self.w_success
    return reach + success


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _coerce_params(params: Any) -> dict[str, Any]:
  """Coerce an OmegaConf node (or None / mapping) to a plain dict.

  ``build_reward_fn`` can be called with an :class:`omegaconf.DictConfig`
  ``env`` block whose ``reward_params`` is a nested ``DictConfig``. We
  unwrap it to a plain dict so the dataclass constructors don't fight
  OmegaConf's lazy proxies on tuple-typed fields.
  """
  if params is None:
    return {}
  try:
    from omegaconf import OmegaConf  # type: ignore[import-not-found]

    return OmegaConf.to_container(params, resolve=True)  # type: ignore[return-value]
  except Exception:
    return dict(params)


def build_reward_fn(kind_or_env_cfg: Any) -> RewardFn:
  """Construct a :class:`RewardFn` from either a kind string or env config.

  Two call styles:

    * ``build_reward_fn("goal_distance")`` — legacy string form. The
      reward gets default-constructed; no params can be passed.
    * ``build_reward_fn(cfg.env)`` — preferred form. Reads
      ``cfg.env.reward`` for the kind and ``cfg.env.reward_params``
      (optional) as kwargs for the matching dataclass.

  Letting it accept either form means trainer.py can pass ``config.env``
  to enable parameterized rewards without breaking any callers that
  still pass the bare kind string.
  """
  if isinstance(kind_or_env_cfg, str):
    kind   = kind_or_env_cfg
    params = {}
  else:
    kind   = str(kind_or_env_cfg.reward)
    params = _coerce_params(getattr(kind_or_env_cfg, "reward_params", None))

  if kind == "goal_distance":
    return GoalDistanceReward()
  if kind == "line_push":
    return StraightLinePushReward(**params)
  if kind == "obstacle_push":
    return ObstaclePushReward(**params)
  if kind == "teleop_proxy":
    return TeleopProxyReward(**params)
  raise ValueError(f"Unknown reward kind: {kind!r}")
