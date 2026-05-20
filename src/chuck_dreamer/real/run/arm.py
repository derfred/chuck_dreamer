"""SO-101 follower + leader wrappers for the real-arm run loop.

:class:`Arm` owns the lerobot ``SOFollower`` bus and exposes joint
readings/commands in radians as a ``(6,)`` numpy array, ordered to
match the physical chain (shoulder pan → gripper). :class:`LeaderArm`
is a read-only twin that pulls joint positions from an SO-100/101
*leader* (teleop) bus — used by the manual policy so the operator can
move the leader arm and have the follower mirror it.

The lerobot bus ships with ``use_degrees=True`` by default; we convert
at the boundary so the rest of the pipeline stays in radians.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# lerobot's SO follower exposes the motors under these names, in the
# physical chain order.
_LEROBOT_MOTORS = (
  "shoulder_pan",
  "shoulder_lift",
  "elbow_flex",
  "wrist_flex",
  "wrist_roll",
  "gripper",
)

# Simulator-style joint names, in the same order as ``_LEROBOT_MOTORS``,
# for callers that prefer the canonical names.
JOINT_NAMES = (
  "Rotation",
  "Pitch",
  "Elbow",
  "Wrist_Pitch",
  "Wrist_Roll",
  "Jaw",
)

# Gripper range in the chuck_dreamer sim model (see
# ``assets/mujoco/so101_arm.xml``, joint class ``Jaw``). The real
# follower/leader expose the gripper motor in
# ``MotorNormMode.RANGE_0_100`` — *not* degrees — so the body's
# rad↔deg conversion must NOT be applied to the gripper channel.
# Without this special-case the gripper would be scrambled by a
# factor of (180/π) × (percent units interpreted as degrees), which
# pegs it against its travel limits on both reads and writes.
_GRIPPER_RAD_LO = -0.174
_GRIPPER_RAD_HI =  1.75
_GRIPPER_RAD_SPAN = _GRIPPER_RAD_HI - _GRIPPER_RAD_LO


def _bus_to_rad(vec_bus: np.ndarray, *, use_degrees: bool) -> np.ndarray:
  """Convert a 6-vector of lerobot motor units to chuck_dreamer radians.

  Body joints (indices 0..4) are degrees when ``use_degrees`` is set;
  the gripper (index 5) is always a percent in [0, 100] regardless of
  ``use_degrees`` (the lerobot bus pins it to ``RANGE_0_100``).
  """
  out = vec_bus.astype(np.float32, copy=True)
  if use_degrees:
    out[:5] = out[:5] * (np.pi / 180.0)
  pct = float(np.clip(vec_bus[5], 0.0, 100.0))
  out[5] = _GRIPPER_RAD_LO + (pct / 100.0) * _GRIPPER_RAD_SPAN
  return out


def _rad_to_bus(vec_rad: np.ndarray, *, use_degrees: bool) -> np.ndarray:
  """Inverse of :func:`_bus_to_rad`."""
  out = vec_rad.astype(np.float32, copy=True)
  if use_degrees:
    out[:5] = out[:5] * (180.0 / np.pi)
  pct = (float(vec_rad[5]) - _GRIPPER_RAD_LO) / _GRIPPER_RAD_SPAN * 100.0
  out[5] = float(np.clip(pct, 0.0, 100.0))
  return out


@dataclass
class ArmConfig:
  port:                          str | None
  calibration_id:                str | None = None
  disable_torque_on_disconnect:  bool = True
  max_relative_target:           float | None = 5.0
  use_degrees:                   bool = True


class Arm:
  """Real SO-101 follower; thin wrapper around lerobot's SOFollower.

  If ``config.port`` is None, the arm operates in *dry-run* mode: no
  serial bus is opened, ``read_joint_qpos`` returns a virtual pose, and
  ``send_joint_qpos`` accepts commands and stores them locally so the
  rest of the run loop (camera, HUD, IK, policy) behaves identically.
  """

  def __init__(self, config: ArmConfig) -> None:
    self.config = config
    self.joint_names = JOINT_NAMES
    self._follower = None
    # In dry-run we still need a coherent virtual pose so policies that
    # warm-start from the current joints see a stable value across ticks.
    self._virtual_qpos = np.zeros(6, dtype=np.float32)

  @property
  def is_dry_run(self) -> bool:
    return self.config.port is None

  # ------------------------------------------------------------------
  # Lifecycle
  # ------------------------------------------------------------------

  def connect(self) -> None:
    if self.is_dry_run:
      return
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    cfg = SOFollowerRobotConfig(
      port=self.config.port,
      id=self.config.calibration_id,
      disable_torque_on_disconnect=self.config.disable_torque_on_disconnect,
      max_relative_target=self.config.max_relative_target,
      use_degrees=self.config.use_degrees,
    )
    self._follower = SOFollower(cfg)
    self._follower.connect()

  def disconnect(self) -> None:
    if self._follower is not None and self._follower.is_connected:
      self._follower.disconnect()

  def __enter__(self) -> "Arm":
    self.connect()
    return self

  def __exit__(self, *_exc) -> None:
    self.disconnect()

  # ------------------------------------------------------------------
  # I/O
  # ------------------------------------------------------------------

  def read_joint_qpos(self) -> np.ndarray:
    """Return the current joint positions in radians, in simulator order.

    In dry-run mode this echoes back whatever was last sent (initially
    zeros), so policies and IK have a stable, monotonically-updating pose.
    """
    if self.is_dry_run:
      return self._virtual_qpos.copy()
    assert self._follower is not None, "Arm.read_joint_qpos() before connect()"
    obs = self._follower.get_observation()
    vec = np.array([obs[f"{m}.pos"] for m in _LEROBOT_MOTORS], dtype=np.float32)
    return _bus_to_rad(vec, use_degrees=self.config.use_degrees)

  def send_joint_qpos(self, qpos: np.ndarray) -> np.ndarray:
    """Command the follower to ``qpos`` (radians, simulator order).

    Returns the action lerobot actually sent (possibly clipped by
    ``max_relative_target``), in the same order/units as the input.
    In dry-run mode the command is stored as the new virtual pose and
    returned unchanged.
    """
    q = np.asarray(qpos, dtype=np.float32)
    if q.shape != (6,):
      raise ValueError(f"expected (6,) qpos in radians, got {q.shape}")
    if self.is_dry_run:
      self._virtual_qpos = q.copy()
      return q
    assert self._follower is not None, "Arm.send_joint_qpos() before connect()"
    out = _rad_to_bus(q, use_degrees=self.config.use_degrees)
    action = {f"{m}.pos": float(v) for m, v in zip(_LEROBOT_MOTORS, out)}
    sent = self._follower.send_action(action)
    sent_vec = np.array([sent[f"{m}.pos"] for m in _LEROBOT_MOTORS], dtype=np.float32)
    return _bus_to_rad(sent_vec, use_degrees=self.config.use_degrees)


@dataclass
class LeaderArmConfig:
  port:            str
  calibration_id:  str | None = None
  use_degrees:     bool = True


class LeaderArm:
  """Read-only SO-100/101 leader; wraps lerobot's ``SOLeader``.

  Used by the manual policy to mirror operator-driven joint motion onto
  the follower. ``read_joint_qpos`` is the only meaningful operation —
  the leader is torque-disabled by lerobot, so the bus only reports
  positions back.
  """

  def __init__(self, config: LeaderArmConfig) -> None:
    self.config = config
    self.joint_names = JOINT_NAMES
    self._leader = None

  # ------------------------------------------------------------------
  # Lifecycle
  # ------------------------------------------------------------------

  def connect(self) -> None:
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

    cfg = SOLeaderTeleopConfig(
      port=self.config.port,
      id=self.config.calibration_id,
      use_degrees=self.config.use_degrees,
    )
    self._leader = SOLeader(cfg)
    self._leader.connect()

  def disconnect(self) -> None:
    if self._leader is not None and self._leader.is_connected:
      self._leader.disconnect()

  def __enter__(self) -> "LeaderArm":
    self.connect()
    return self

  def __exit__(self, *_exc) -> None:
    self.disconnect()

  # ------------------------------------------------------------------
  # I/O
  # ------------------------------------------------------------------

  def read_joint_qpos(self) -> np.ndarray:
    """Return the leader's current joint positions in radians.

    Returns a ``(6,)`` numpy array ordered to match :data:`JOINT_NAMES` /
    the follower's :meth:`Arm.read_joint_qpos`.
    """
    assert self._leader is not None, "LeaderArm.read_joint_qpos() before connect()"
    action = self._leader.get_action()
    vec = np.array([action[f"{m}.pos"] for m in _LEROBOT_MOTORS], dtype=np.float32)
    return _bus_to_rad(vec, use_degrees=self.config.use_degrees)
