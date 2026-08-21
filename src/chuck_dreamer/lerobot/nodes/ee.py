"""Native pipeline nodes (spec §7): the end-effector chain.

``normalize_joints`` → ``ee_pos_arm`` → (``ee_rotation``) → ``ee_action_arm``.
Each node declares its typed inputs and outputs; the numeric results match
the legacy stage (same ops, same dtypes), except that every coordinate track
is persisted in **millimetres** with an explicit frame suffix:

  joint_qpos_rad (T, J) rad (positioning joints), gripper channel unitless
  ee_pos_arm     (T, 3) mm, arm frame
  ee_quat_arm    (T, 4) unitless, arm frame
  ee_action_arm  (T, 7) mm + unitless, arm frame

The declared output :class:`TrackSpec` tuples are *templates* — per-frame
shapes that depend on the dataset (joint count) are finalized on the runtime
track; the runner matches outputs by name/frame/validity.

The table-frame variants (``ee_pos_table`` / ``ee_quat_table`` /
``ee_action_table``) are produced by the nodes in
:mod:`chuck_dreamer.lerobot.nodes.table_frame`, which consume the
``table_to_arm`` transform extracted per dataset.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from chuck_dreamer.common.tracks import (
  ChannelUnits,
  Frame,
  Persist,
  Track,
  TrackSpec,
  Unit,
)

from ..harness import InputDecl
from ..context import RunContext

if TYPE_CHECKING:
  from ..trackset import NodeView

# MuJoCo [w, x, y, z] for a 180° rotation about x — gripper z points down.
EE_QUAT_DOWN: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)

# Number of positioning joints consumed by FK (rest of the state vector is
# the gripper, which doesn't affect EE position).
N_POSITIONING_JOINTS = 5

# unit=mixed for a 7-D pose/action: mm position + unitless quaternion.
POSE_MM_UNITS = ChannelUnits({(0, 3): Unit.MM, (3, 7): Unit.UNITLESS})


def _joint_units(n_joints: int, positioning: Unit) -> Unit | ChannelUnits:
  n_pos = min(N_POSITIONING_JOINTS, n_joints)
  if n_pos == 0:
    return Unit.UNITLESS
  groups: dict[tuple[int, int], Unit] = {(0, n_pos): positioning}
  if n_joints > n_pos:
    groups[(n_pos, n_joints)] = Unit.UNITLESS
  return ChannelUnits(groups)


def shift_action(pose: np.ndarray) -> np.ndarray:
  """Spec §7.9: ``action[t]`` is the pose for step ``t+1``; the final frame
  repeats. Shared by the arm- and table-frame action nodes."""
  if len(pose) == 0:
    return pose.astype(np.float32)
  out: np.ndarray = np.concatenate([pose[1:], pose[-1:]], axis=0)
  return out.astype(np.float32)


class _EeNodeBase:
  scope_level = "episode"
  lane = "producer"          # decode-adjacent, cheap; runs with the GPU lane

  def __init__(self, ctx: RunContext) -> None:
    self.ctx = ctx


class NormalizeJointsNode(_EeNodeBase):
  """Spec §7.3: positioning joints deg → rad, gripper channel untouched."""
  name    = "normalize_joints"
  inputs  = (InputDecl("joint_qpos"),)
  outputs = (TrackSpec("joint_qpos_rad", np.float32, (), Frame.NONE,
                       Unit.RAD, persist=Persist.TRAJECTORY),)

  def run(self, view: "NodeView") -> list[Track]:
    qpos     = np.asarray(view.track("joint_qpos").values, dtype=np.float32)
    n        = min(qpos.shape[1] if qpos.ndim > 1 else 0, N_POSITIONING_JOINTS)
    rescaled = qpos.copy()
    if n:
      rescaled[:, :n] = np.deg2rad(rescaled[:, :n])
    spec = TrackSpec("joint_qpos_rad", rescaled.dtype, tuple(rescaled.shape[1:]),
                     Frame.NONE, _joint_units(rescaled.shape[1] if rescaled.ndim > 1 else 0, Unit.RAD),
                     persist=Persist.TRAJECTORY)
    return [Track(spec, rescaled)]


class EePosArmNode(_EeNodeBase):
  """Spec §7.6: forward kinematics per frame, arm frame. FK works in metres;
  the persisted track is millimetres (the trajectory-format unit)."""
  name    = "ee_pos_arm"
  inputs  = (InputDecl("joint_qpos_rad"), InputDecl("fk_model"))
  outputs = (TrackSpec("ee_pos_arm", np.float32, (3,), Frame.ARM,
                       Unit.MM, persist=Persist.TRAJECTORY),)

  def run(self, view: "NodeView") -> list[Track]:
    qpos      = np.asarray(view.track("joint_qpos_rad").values, dtype=np.float32)
    fk        = view.artifact("fk_model")
    T         = qpos.shape[0]
    ee_pos_mm = np.zeros((T, 3), dtype=np.float32)
    if qpos.ndim > 1 and qpos.shape[1] >= 1:
      for t in range(T):
        ee_pos_mm[t] = (fk(qpos[t, :N_POSITIONING_JOINTS]) * 1000.0).astype(np.float32)
    return [Track(self.outputs[0], ee_pos_mm)]


class EeRotationNode(_EeNodeBase):
  """Spec §7.8: constant gripper-down orientation per frame. A node so a
  real orientation source can replace it without structural change."""
  name    = "ee_rotation"
  inputs  = (InputDecl("timestamp"),)
  outputs = (TrackSpec("ee_quat_arm", np.float32, (4,), Frame.ARM,
                       Unit.UNITLESS, persist=Persist.TRAJECTORY),)

  def run(self, view: "NodeView") -> list[Track]:
    T = len(view.track("timestamp"))
    ee_quat = np.tile(np.asarray(EE_QUAT_DOWN, dtype=np.float32), (T, 1))
    return [Track(self.outputs[0], ee_quat)]


class EeActionNode(_EeNodeBase):
  """Spec §7.9: action[t] is the EE target pose for step t+1 (last frame
  repeats), concatenation of position and orientation."""
  name    = "ee_action_arm"
  inputs  = (InputDecl("ee_pos_arm", frame=Frame.ARM, unit=Unit.MM),
             InputDecl("ee_quat_arm", frame=Frame.ARM))
  outputs = (TrackSpec("ee_action_arm", np.float32, (7,), Frame.ARM,
                       POSE_MM_UNITS, persist=Persist.TRAJECTORY),)

  def run(self, view: "NodeView") -> list[Track]:
    ee_pos  = view.track("ee_pos_arm").values
    ee_quat = view.track("ee_quat_arm").values
    action = shift_action(np.concatenate([ee_pos, ee_quat], axis=-1))
    return [Track(self.outputs[0], action)]


def build_ee_nodes(ctx: RunContext) -> list[_EeNodeBase]:
  """The dependency-ordered arm-frame EE chain."""
  return [NormalizeJointsNode(ctx), EePosArmNode(ctx),
          EeRotationNode(ctx), EeActionNode(ctx)]
