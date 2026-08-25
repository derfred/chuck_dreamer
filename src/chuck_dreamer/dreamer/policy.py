"""DreamerPolicy stub.

The Dreamer model isn't yet wired to control. Until it is, this policy
returns a no-op action — the *current* EE pose (so the env's IK warm-start
holds the arm in place) or the current arm qpos in joint mode.
"""

import numpy as np

from ..policy import Action


class DreamerPolicy:
  def __init__(self, model, act_mode: str = "ee"):
    self.model = model
    self.act_mode = act_mode

  def reset(self, scene):
    pass

  def act(self, obs) -> Action:
    if self.act_mode == "joint":
      return Action(obs, q=np.asarray(obs["arm_qpos"], dtype=np.float64))
    return Action(obs, arm_pos=np.asarray(obs["ee_pos"], dtype=np.float64), quat=np.asarray(obs["ee_quat"], dtype=np.float64))
