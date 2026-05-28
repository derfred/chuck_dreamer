"""EE-pose stage: rescale joint_qpos to radians and run the MuJoCo FK."""
from __future__ import annotations

import numpy as np

from ...common import FK_MODEL_PATH
from .base import Requirement, StageContext

# MuJoCo [w, x, y, z] for a 180° rotation about x — gripper z points down.
_EE_QUAT_DOWN: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)

# Number of positioning joints consumed by FK (rest of the state vector is
# the gripper, which doesn't affect EE position).
_N_POSITIONING_JOINTS = 5


class EePosStage:
  name = "ee_pos"
  produces: tuple[str, ...] = ("joint_qpos", "ee_pos", "ee_quat", "ee_action")
  requires: tuple[str, ...] = ()

  def requirements(self, ctx: StageContext) -> list[Requirement]:
    return [Requirement(
      "FK MuJoCo model", FK_MODEL_PATH,
      "restore assets/mujoco/so101_arm.xml from git")]

  def apply(self, episode, metadata, ctx) -> None:
    qpos = episode.get("joint_qpos")
    if qpos is None or qpos.size == 0:
      return
    qpos = np.asarray(qpos, dtype=np.float32)
    n = min(qpos.shape[1], _N_POSITIONING_JOINTS)
    rescaled = qpos.copy()
    rescaled[:, :n] = np.deg2rad(rescaled[:, :n])
    episode["joint_qpos"] = rescaled

    fk = ctx.fk()
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
      [ee_pose[1:], ee_pose[-1:]], axis=0).astype(np.float32)
