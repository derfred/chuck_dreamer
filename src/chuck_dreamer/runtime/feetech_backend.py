"""``FeetechBackend`` — the real SO-101 follower as a :class:`RobotBackend`.

The hardware swap-in for the M1 control kernel (project plan, keystone
decision): the kernel, control loop, and everything above them are developed
against ``FakeBackend`` / ``MujocoBackend`` and this is the only seam to the
real arm — 6 STS3215 servos on one TTL bus, driven through lerobot's
``SO101Follower``. The control thread owns the bus exclusively: a sync write
(``send_action``) every tick and a sync read (``get_observation``) on a
decimated/diagnostic schedule (the ~2.5 ms write / ~7 ms read budget from the
plan).

Joint convention mirrors :mod:`chuck_dreamer.runtime.teleop` exactly: lerobot
exposes each motor as ``"<motor>.pos"`` (joints 0..4 in degrees, the gripper as
a 0..100 percentage), and this backend maps to/from follower-ordered radians
with the *same* helpers the leader reader uses, so the two never disagree.
``read_positions`` / ``write_positions`` therefore speak radians like every
other backend, and the kernel's slew + clamp shape motion identically in sim and
on hardware.

Importing this module never touches hardware: ``lerobot`` is imported lazily
inside :meth:`start` (the leader-reader philosophy), so the module is safe to
import on a machine with no arm and the tests stand in a stub for the bus.

The authoritative joint-space envelope (``joint_limits``) comes from config, not
from the follower's calibration counts — the kernel already prefers the
configured envelope, and keeping it in one place (``runtime.safety``) means the
clamp the kernel enforces and the bounds the backend reports always agree. The
follower jaw bounds the gripper percentage maps into are the last joint's
configured bounds, injected at construction the same way the leader gets them.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from .teleop import (
  _FOLLOWER_JOINTS,
  _LEADER_MOTORS,
  follower_qpos_to_action,
  leader_action_to_follower_qpos,
)

logger = logging.getLogger(__name__)

_N_JOINTS = len(_LEADER_MOTORS)


class FeetechBackend:
  """Real SO-101 follower over lerobot, behind the backend protocol.

  ``lower`` / ``upper`` are the configured per-joint position bounds (radians,
  follower order) reported verbatim by :meth:`joint_limits`; their last entry is
  the jaw range the gripper percentage maps into. ``read_decimation`` hits the
  bus only every Nth :meth:`read_positions` and returns the cached reading in
  between (the diagnostic-read schedule from the plan); ``1`` reads every tick.
  """

  def __init__(
    self,
    *,
    port: str,
    lower: np.ndarray,
    upper: np.ndarray,
    calibration_id: str | None = None,
    use_degrees: bool = True,
    read_decimation: int = 1,
    calibrate: bool = True,
    max_relative_target: float | None = None,
    disable_torque_on_disconnect: bool = True,
  ) -> None:
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.shape != (_N_JOINTS,) or hi.shape != (_N_JOINTS,):
      raise ValueError(
        f"FeetechBackend drives {_N_JOINTS} joints ({', '.join(_FOLLOWER_JOINTS)}); "
        f"got limit shapes lower={lo.shape}, upper={hi.shape}")
    if np.any(hi < lo):
      raise ValueError("every upper bound must be >= its lower bound")
    if int(read_decimation) < 1:
      raise ValueError("read_decimation must be >= 1")

    self._port                            = port
    self._lower                           = lo.copy()
    self._upper                           = hi.copy()
    self._jaw_lower                       = float(lo[-1])
    self._jaw_upper                       = float(hi[-1])
    self._calibration_id                  = calibration_id
    self._use_degrees                     = bool(use_degrees)
    self._read_decimation                 = int(read_decimation)
    self._calibrate                       = bool(calibrate)
    self._max_relative_target             = max_relative_target
    self._disable_torque_on_disconnect    = bool(disable_torque_on_disconnect)

    # Bus state. The control thread owns the bus (read_positions /
    # write_positions); other threads observe via last_positions (cache only).
    # _lock is held across every bus transaction so that even a misbehaving
    # extra caller cannot interleave packets on the half-duplex bus — two
    # concurrent sync-reads corrupt each other's status packets.
    self._lock                            = threading.Lock()
    self._read_count                      = 0
    self._cached: np.ndarray | None       = None
    self._follower: Any = None  # lerobot SO101Follower (untyped), built in start()

  @classmethod
  def from_config(cls, cfg, *, port: str | None = None, **params) -> "FeetechBackend":
    """Build from ``cfg.runtime.safety`` (the harness backend convention).

    The harness calls ``backend.from_config(cfg, **runtime.backend.params)``; the
    configured ``params`` (``port``, ``calibration_id``, ``read_decimation``, …)
    flow through as keyword arguments. The joint-space envelope and jaw bounds
    are pulled from ``runtime.safety`` so the backend reports exactly the bounds
    the kernel clamps to. ``port`` may also live in ``params``.
    """
    from omegaconf import OmegaConf

    safety = cfg.runtime.safety
    names = list(safety.joint_names)
    if len(names) != _N_JOINTS:
      raise ValueError(
        f"FeetechBackend drives {_N_JOINTS} joints; runtime.safety.joint_names "
        f"has {len(names)}")
    if safety.joint_lower is None or safety.joint_upper is None:
      raise ValueError(
        "FeetechBackend needs explicit runtime.safety.joint_lower/joint_upper "
        "(radians) — there is no model to read joint limits from")

    def _arr(node) -> np.ndarray:
      return np.asarray(
        OmegaConf.to_container(node, resolve=True)
        if OmegaConf.is_config(node) else node, dtype=np.float64)

    resolved_port = port if port is not None else params.pop("port", None)
    if resolved_port is None:
      raise ValueError("FeetechBackend requires a serial `port` (set runtime.backend.params.port)")
    return cls(
      port=str(resolved_port),
      lower=_arr(safety.joint_lower),
      upper=_arr(safety.joint_upper),
      **params,
    )

  # -- backend protocol -----------------------------------------------------

  @property
  def n_joints(self) -> int:
    return _N_JOINTS

  def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
    return self._lower.copy(), self._upper.copy()

  def start(self) -> None:
    # Lazy import: keeps module import hardware-free (no serial libs loaded
    # unless a real follower is actually constructed). Mirrors the leader reader.
    from lerobot.robots.so_follower import (  # type: ignore[import-untyped]
      SO101Follower,
      SO101FollowerConfig,
    )

    cfg = SO101FollowerConfig(
      port=self._port,
      id=self._calibration_id,
      use_degrees=self._use_degrees,
      max_relative_target=self._max_relative_target,
      disable_torque_on_disconnect=self._disable_torque_on_disconnect,
    )
    self._follower = SO101Follower(cfg)
    self._follower.connect(calibrate=self._calibrate)
    # Prime the cache with one bus read on the (single) starting thread:
    # last_positions() is then always answerable without touching the bus,
    # regardless of which loop thread runs first.
    with self._lock:
      self._read_count = 0
      self._cached = None
      self._bus_read_locked()

  def stop(self) -> None:
    # Safe to call during shutdown even if never started; never raise.
    follower = self._follower
    self._follower = None
    if follower is not None:
      try:
        follower.disconnect()
      except Exception:
        logger.exception("follower disconnect failed")

  def _bus_read_locked(self) -> np.ndarray:
    """One sync-read transaction; caller must hold ``self._lock``."""
    obs = self._follower.get_observation()
    q = leader_action_to_follower_qpos(
      {k: v for k, v in obs.items() if k.endswith(".pos")},
      jaw_lower=self._jaw_lower, jaw_upper=self._jaw_upper)
    self._cached = q
    return q.copy()

  def read_positions(self) -> np.ndarray:
    """Latest measured joint positions in follower-ordered radians, shape ``(6,)``.

    Control-thread path. Hits the bus once every ``read_decimation`` calls (the
    diagnostic-read schedule); between bus reads it returns the cached reading.
    The whole bus transaction runs under ``self._lock`` (see ``_lock`` comment).
    """
    if self._follower is None:
      raise RuntimeError("FeetechBackend.read_positions before start()")
    with self._lock:
      due = self._cached is None or (self._read_count % self._read_decimation == 0)
      self._read_count += 1
      if not due and self._cached is not None:
        return self._cached.copy()
      return self._bus_read_locked()

  def last_positions(self) -> np.ndarray:
    """Freshest cached reading, no bus I/O — the observer (policy-loop) path.

    ``start()`` primes the cache, so this is always answerable after start.
    """
    if self._follower is None:
      raise RuntimeError("FeetechBackend.last_positions before start()")
    with self._lock:
      assert self._cached is not None, "start() must prime the cache"
      return self._cached.copy()

  def write_positions(self, q: np.ndarray) -> None:
    """Command joint setpoints (follower-ordered radians) to the servos."""
    if self._follower is None:
      raise RuntimeError("FeetechBackend.write_positions before start()")
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (_N_JOINTS,):
      raise ValueError(f"write_positions expects ({_N_JOINTS},); got {q.shape}")
    action = follower_qpos_to_action(
      q, jaw_lower=self._jaw_lower, jaw_upper=self._jaw_upper)
    with self._lock:
      self._follower.send_action(action)
