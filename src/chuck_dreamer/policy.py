from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from chuck_dreamer.sim.scene_config import SceneConfig


@dataclass
class Action:
  mode: str        # "qpos", "qpos_delta", "ee_pos", "ee_pos_delta"
  qpos: np.ndarray  # shape (6,)
  ee_pos: np.ndarray  # shape (3,)

  @staticmethod
  def from_qpos(qpos: np.ndarray) -> "Action":
    assert qpos.shape == (6,), f"Expected qpos shape (6,), got {qpos.shape}"
    return Action(mode="qpos", qpos=qpos, ee_pos=np.zeros(3))

  @staticmethod
  def from_ee_pos(ee_pos: np.ndarray) -> "Action":
    assert ee_pos.shape == (3,), f"Expected ee_pos shape (3,), got {ee_pos.shape}"
    return Action(mode="ee_pos", qpos=np.zeros(6), ee_pos=ee_pos)


class Policy(Protocol):
  def reset(self, controller: Any, scene: SceneConfig) -> None: ...
  def act(self, obs: dict[str, np.ndarray]) -> tuple[Action, str | None]: ...
  def is_done(self) -> bool: ...
