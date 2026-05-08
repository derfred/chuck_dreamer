"""Reward functions.

A ``RewardFn`` maps a :class:`StepInfo` to a scalar reward. Reward is
recomputed in the replay buffer at sample time (not baked into recorded
episodes), so swapping ``cfg.env.reward`` at training start changes the
next batch without re-collecting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
  from .sim.step_info import StepInfo


class RewardFn(Protocol):
  def __call__(self, info: "StepInfo") -> float: ...


@dataclass
class GoalDistanceReward:
  """Default reward: ``-‖object_xy - goal_xy‖``."""

  def __call__(self, info: "StepInfo") -> float:
    return -float(np.linalg.norm(info.object_xy - info.goal_xy))


def build_reward_fn(kind: str) -> RewardFn:
  if kind == "goal_distance":
    return GoalDistanceReward()
  raise ValueError(f"Unknown reward kind: {kind!r}")
