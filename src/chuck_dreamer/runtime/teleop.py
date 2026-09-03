from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# lerobot motor order, verified against the SO-101 leader's
# lerobot.teleoperators.so_leader.so_leader and the SO-101 follower's
# lerobot.robots.so_follower.so_follower (both share this bus.motors insertion
# order). Shared by the leader reader (M2) and the FeetechBackend (M1 hardware).
_LEADER_MOTORS = (
  "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
)
# Follower envelope order (configs/default.yaml runtime.safety.joint_names). The
# first five line up 1:1 with the leader's angular motors; the sixth is the jaw.
_FOLLOWER_JOINTS = (
  "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw",
)
# Motors 0..4 are angular: MotorNormMode.DEGREES when use_degrees=True, so a
# degree<->radian conversion applies. The gripper (index 5) is *always*
# MotorNormMode.RANGE_0_100 on the SO-101 — a 0..100 percentage, NOT degrees —
# so it is mapped separately to/from the follower jaw range. Blanket np.deg2rad
# over all six would silently mishandle the jaw.
_N_ANGULAR = 5


def leader_action_to_follower_qpos(
  action: dict[str, float],
  *,
  jaw_lower: float,
  jaw_upper: float,
) -> np.ndarray:
  """Map a lerobot ``get_action()`` dict to follower-ordered radians, shape ``(6,)``.

  - joints 0..4: ``"<motor>.pos"`` in degrees -> radians via :func:`numpy.deg2rad`.
  - joint 5 (gripper -> Jaw): ``"gripper.pos"`` in 0..100 (a percentage, clamped)
    -> linearly mapped into ``[jaw_lower, jaw_upper]`` radians.

  Raises:
    ValueError: if any expected ``"<motor>.pos"`` key is missing (a partial read
      or a leader with a different motor set fails loudly rather than misaligning).
  """
  q = np.empty(len(_LEADER_MOTORS), dtype=np.float64)
  for i, motor in enumerate(_LEADER_MOTORS):
    key = f"{motor}.pos"
    if key not in action:
      raise ValueError(f"leader action missing {key!r}; got keys {sorted(action)}")
    raw = float(action[key])
    if i < _N_ANGULAR:
      q[i] = np.deg2rad(raw)
    else:
      frac = max(0.0, min(1.0, raw / 100.0))  # clamp the percentage into [0, 1]
      q[i] = jaw_lower + frac * (jaw_upper - jaw_lower)
  return q


def follower_qpos_to_action(
  q: np.ndarray,
  *,
  jaw_lower: float,
  jaw_upper: float,
) -> dict[str, float]:
  """Inverse of :func:`leader_action_to_follower_qpos`: radians -> lerobot action dict.

  Maps a follower-ordered radian vector, shape ``(6,)``, to the
  ``{"<motor>.pos": value}`` dict the lerobot SO-101 follower's ``send_action``
  consumes (motors in :data:`_LEADER_MOTORS` order):

  - joints 0..4: radians -> degrees via :func:`numpy.rad2deg`.
  - joint 5 (Jaw -> gripper): radians in ``[jaw_lower, jaw_upper]`` -> a 0..100
    percentage (clamped), the inverse of the leader's gripper mapping.

  Used by :class:`~chuck_dreamer.runtime.feetech_backend.FeetechBackend` to send
  the kernel's commanded joint setpoint to the real arm in the same convention
  the leader reader maps the other way, so the two never disagree.

  Raises:
    ValueError: if ``q`` is not shape ``(6,)`` or ``jaw_upper == jaw_lower``
      (no jaw range to map the percentage into).
  """
  q = np.asarray(q, dtype=np.float64)
  if q.shape != (len(_LEADER_MOTORS),):
    raise ValueError(
      f"follower_qpos_to_action expects shape ({len(_LEADER_MOTORS)},); got {q.shape}")
  span = jaw_upper - jaw_lower
  if span == 0.0:
    raise ValueError("jaw_upper must differ from jaw_lower")
  action: dict[str, float] = {}
  for i, motor in enumerate(_LEADER_MOTORS):
    if i < _N_ANGULAR:
      action[f"{motor}.pos"] = float(np.rad2deg(q[i]))
    else:
      frac = max(0.0, min(1.0, (q[i] - jaw_lower) / span))  # clamp into [0, 1]
      action[f"{motor}.pos"] = frac * 100.0
  return action

@dataclass(frozen=True, slots=True)
class LeaderState:
  q:   np.ndarray   # (n,) rad, follower order
  now: float        # time.monotonic() when this reading was taken
  age: float = 0.0  # seconds since ``now``, filled in at read time
  ok:  bool  = True # the read that produced this value succeeded

  def aged(self, at: float) -> "LeaderState":
    """This reading with :attr:`age` measured against monotonic time ``at``."""
    return LeaderState(q=self.q, now=self.now, age=at - self.now, ok=self.ok)


@runtime_checkable
class LeaderReader(Protocol):
  def start(self) -> None:
    """Open the leader and begin polling (idempotent)."""
    ...

  def stop(self) -> None:
    """Release the leader (safe to call even if never started)."""
    ...

  def latest(self) -> LeaderState | None:
    """The current reading, or ``None`` before the first successful one."""
    ...


class LerobotLeaderReader:
  """Real SO-101 leader, read synchronously by whoever calls :meth:`latest`.

  ``jaw_lower`` / ``jaw_upper`` are the follower jaw bounds (radians) the gripper
  percentage maps into; the runtime injects them from the resolved safety
  envelope so the mapping always agrees with the kernel's clamp.
  """

  def __init__(
    self,
    *,
    port:           str,
    jaw_lower:      float,
    jaw_upper:      float,
    calibration_id: str | None = None,
    use_degrees:    bool = True,
    calibrate:      bool = True,
  ) -> None:
    self._port                       = port
    self._jaw_lower                  = float(jaw_lower)
    self._jaw_upper                  = float(jaw_upper)
    self._calibration_id             = calibration_id
    self._use_degrees                = bool(use_degrees)
    self._calibrate                  = bool(calibrate)
    self._latest: LeaderState | None = None
    self._leader: Any = None  # lerobot SO101Leader (untyped), constructed in start()

  @classmethod
  def from_config(
    cls,
    cfg,
    *,
    jaw_lower:      float,
    jaw_upper:      float,
    port:           str | None = None,
    calibration_id: str | None = None,
    use_degrees:    bool = True,
    calibrate:      bool = True,
    **ignored,
  ) -> "LerobotLeaderReader":
    if ignored:
      logger.warning(
        "LerobotLeaderReader ignoring leader params it does not use: %s",
        sorted(ignored))
    if not port:
      raise ValueError(
        "LerobotLeaderReader needs runtime.leader.params.port (or --leader-port)")
    return cls(
      port=port, jaw_lower=jaw_lower, jaw_upper=jaw_upper,
      calibration_id=calibration_id, use_degrees=use_degrees,
      calibrate=calibrate)

  def start(self) -> None:
    # Lazy import: keeps module import hardware-free (no serial libs loaded
    # unless a real leader is actually constructed).
    from lerobot.teleoperators.so_leader import (  # type: ignore[import-untyped]
      SO101Leader,
      SOLeaderTeleopConfig,
    )

    cfg          = SOLeaderTeleopConfig(port=self._port, use_degrees=self._use_degrees, id=self._calibration_id)
    self._leader = SO101Leader(cfg)
    self._leader.connect(calibrate=self._calibrate)

  def stop(self) -> None:
    if self._leader is not None:
      try:
        self._leader.disconnect()
      except Exception:  # never raise on shutdown
        logger.exception("leader disconnect failed")
      self._leader = None

  def latest(self) -> LeaderState | None:
    """Read the leader now and return it; ``None`` before the first success."""
    if self._leader is None:
      return None
    try:
      now    = time.monotonic()
      action = self._leader.get_action()
      q      = leader_action_to_follower_qpos(action, jaw_lower=self._jaw_lower, jaw_upper=self._jaw_upper)
      q.flags.writeable = False  # handed to the observation; never mutated
      self._latest = LeaderState(q=q, now=now, ok=True)
    except Exception:
      # Hold the last good value rather than propagating: this runs on the
      # policy thread, and an exception here would stop setpoints entirely.
      # `now` is deliberately not refreshed, so `age` keeps growing while the
      # leader stays unreachable -- a hiccup and a dead leader look different.
      logger.exception("leader read failed; holding last reading")
      if self._latest is not None and self._latest.ok:
        self._latest = LeaderState(q=self._latest.q, now=self._latest.now, ok=False)
    return None if self._latest is None else self._latest.aged(time.monotonic())


class FakeLeaderReader:
  def __init__(self, pose: np.ndarray | None = None, *, n_joints: int = 6) -> None:
    self._n      = int(n_joints)
    self._lock   = threading.Lock()
    self._latest: LeaderState | None = None
    self.set_pose(pose)

  def set_pose(self, pose: np.ndarray | None, *, ok: bool = True) -> None:
    """Set (or clear) the pose :meth:`latest` reports, as if freshly polled."""
    if pose is None:
      with self._lock:
        self._latest = None
      return
    q = np.asarray(pose, dtype=np.float64).copy()
    if q.shape != (self._n,):
      raise ValueError(f"pose must be shape ({self._n},); got {q.shape}")
    q.flags.writeable = False
    with self._lock:
      self._latest = LeaderState(q=q, now=time.monotonic(), ok=ok)

  def start(self) -> None:
    pass

  def stop(self) -> None:
    pass

  def latest(self) -> LeaderState | None:
    with self._lock:
      state = self._latest
    return None if state is None else state.aged(time.monotonic())

  @classmethod
  def from_config(cls, cfg, *, pose=None, **_) -> "FakeLeaderReader":
    return cls(pose=pose)
