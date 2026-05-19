"""SO-101 follower wrapper for the real-arm run loop.

Owns the lerobot ``SOFollower`` bus and exposes joint readings/commands
in radians as a ``(6,)`` numpy array, ordered to match the physical
chain (shoulder pan → gripper).

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
    if self.config.use_degrees:
      vec = vec * (np.pi / 180.0)
    return vec

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
    out = q
    if self.config.use_degrees:
      out = q * (180.0 / np.pi)
    action = {f"{m}.pos": float(v) for m, v in zip(_LEROBOT_MOTORS, out)}
    sent = self._follower.send_action(action)
    sent_vec = np.array([sent[f"{m}.pos"] for m in _LEROBOT_MOTORS], dtype=np.float32)
    if self.config.use_degrees:
      sent_vec = sent_vec * (np.pi / 180.0)
    return sent_vec
