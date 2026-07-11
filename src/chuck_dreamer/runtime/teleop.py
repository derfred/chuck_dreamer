"""Leader-arm reader: SO-101 leader joints as an observation modality (M2, §3.8).

The leader is the real lerobot ``SO101Leader`` over serial; there is no
software leader source. A :class:`LeaderReader` runs a background thread
continuously polling the leader bus and stores the latest reading (latest-value,
lock-protected) in *follower joint order and radians* — the policy loop samples
it non-blocking and never waits on serial. :class:`ManualPolicy`
(:mod:`chuck_dreamer.runtime.sources`) reads that reading off the observation and
returns it as the joint-space action; the M1 kernel's slew + clamp bounds the
motion to the safety envelope exactly as it does for a scripted setpoint.

Importing this module never touches hardware: ``lerobot`` is imported lazily
inside :meth:`LerobotLeaderReader.start` (the
:mod:`chuck_dreamer.runtime.feetech_backend` philosophy), so the
:class:`FakeLeaderReader` test double can stand in with no device present.
"""

from __future__ import annotations

import logging
import threading
import time
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


@runtime_checkable
class LeaderReader(Protocol):
  """Background source of leader joint state in follower-ordered radians.

  Construction convention: readers expose a ``from_config(cfg, *, jaw_lower,
  jaw_upper, **params)`` classmethod. The runtime always injects the resolved
  follower jaw bounds (radians) alongside ``runtime.leader.params``; readers
  take what they need and ignore (with a warning) params meant for another
  implementation, so swapping ``runtime.leader.target`` never crashes on
  leftover keys merged from the previous target's params.
  """

  def start(self) -> None:
    """Open the leader and begin polling (idempotent)."""
    ...

  def stop(self) -> None:
    """Stop polling and release the leader (safe to call even if never started)."""
    ...

  def latest(self) -> np.ndarray | None:
    """Freshest reading, shape ``(n_joints,)`` radians, or ``None`` before the first read.

    Non-blocking: returns a copy of the last stored value and never touches the
    bus. ``None`` until the first successful poll.
    """
    ...


class LerobotLeaderReader:
  """Real SO-101 leader on a background polling thread (latest-value, locked).

  Mirrors :class:`~chuck_dreamer.runtime.mujoco_backend.MujocoBackend`'s
  threading model: a daemon thread named ``runtime-leader``, a ``_stop`` event,
  and a lock guarding ``_latest``. The thread calls ``leader.get_action()``, maps
  it to follower-ordered radians, and stores it; :meth:`latest` returns a copy
  without blocking. ``lerobot`` is imported lazily in :meth:`start`, so importing
  this module is hardware-free.

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
    poll_rate_hz:   float = 100.0,
    calibrate:      bool = True,
  ) -> None:
    if poll_rate_hz <= 0:
      raise ValueError("poll_rate_hz must be positive")
    self._port                            = port
    self._jaw_lower                       = float(jaw_lower)
    self._jaw_upper                       = float(jaw_upper)
    self._calibration_id                  = calibration_id
    self._use_degrees                     = bool(use_degrees)
    self._period                          = 1.0 / float(poll_rate_hz)
    self._calibrate                       = bool(calibrate)
    self._lock                            = threading.Lock()
    self._latest: np.ndarray | None       = None
    self._stop                            = threading.Event()
    self._thread: threading.Thread | None = None
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
    poll_rate_hz:   float = 100.0,
    calibrate:      bool = True,
    **ignored,
  ) -> "LerobotLeaderReader":
    """Config factory (see :class:`LeaderReader` construction convention).

    Params meant for another reader (e.g. ``FakeLeaderReader``'s ``pose``
    surviving a target swap in the merged config) are ignored with a warning
    rather than crashing ``__init__``.
    """
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
      poll_rate_hz=poll_rate_hz, calibrate=calibrate)

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
    self._stop.clear()
    self._thread = threading.Thread(
      target=self._run, name="runtime-leader", daemon=True)
    self._thread.start()

  def stop(self, join_timeout: float = 5.0) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=join_timeout)
      self._thread = None
    if self._leader is not None:
      try:
        self._leader.disconnect()
      except Exception:  # never raise on shutdown
        logger.exception("leader disconnect failed")
      self._leader = None

  def latest(self) -> np.ndarray | None:
    with self._lock:
      return None if self._latest is None else self._latest.copy()

  def _run(self) -> None:
    next_deadline = time.monotonic()
    while not self._stop.is_set():
      try:
        action = self._leader.get_action()
        q = leader_action_to_follower_qpos(
          action, jaw_lower=self._jaw_lower, jaw_upper=self._jaw_upper)
        with self._lock:
          self._latest = q
      except Exception:
        # A transient serial hiccup must not kill the thread: keep the last
        # good value (the policy holds), log, and retry next tick.
        logger.exception("leader poll failed; holding last reading")
      next_deadline += self._period
      sleep_for = next_deadline - time.monotonic()
      if sleep_for > 0:
        self._stop.wait(sleep_for)
      else:
        next_deadline = time.monotonic()


class FakeLeaderReader:
  """Programmable :class:`LeaderReader` test double — no hardware, no lerobot import.

  :meth:`latest` returns the currently-set pose (``None`` until a pose is given
  via the constructor or :meth:`set_pose`). :meth:`start` / :meth:`stop` are
  no-ops — tests set the pose directly. Used for sim / headless validation and
  unit tests, and selected from config exactly like the real reader.
  """

  def __init__(self, pose: np.ndarray | None = None, *, n_joints: int = 6) -> None:
    self._n    = int(n_joints)
    self._lock = threading.Lock()
    self._latest = None if pose is None else np.asarray(pose, dtype=np.float64).copy()
    if self._latest is not None and self._latest.shape != (self._n,):
      raise ValueError(f"pose must be shape ({self._n},); got {self._latest.shape}")

  def set_pose(self, pose: np.ndarray | None) -> None:
    with self._lock:
      self._latest = None if pose is None else np.asarray(pose, dtype=np.float64).copy()

  def start(self) -> None:
    pass

  def stop(self) -> None:
    pass

  def latest(self) -> np.ndarray | None:
    with self._lock:
      return None if self._latest is None else self._latest.copy()

  @classmethod
  def from_config(cls, cfg, *, pose=None, **_) -> "FakeLeaderReader":
    """Config factory (see :class:`LeaderReader` construction convention).

    Swallows the always-injected ``jaw_lower`` / ``jaw_upper`` (irrelevant to a
    fake that returns a pose verbatim) along with any params meant for another
    reader.
    """
    return cls(pose=pose)
