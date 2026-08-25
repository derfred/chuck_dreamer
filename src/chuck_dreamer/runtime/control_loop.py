from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np
from omegaconf import DictConfig

from .backend import RobotBackend
from .control_mode import ControlMode, ControlModeMachine
from .control_state import ControlState, FaultFlags, parse_read_budget
from .control_trajectory import ControlTrajectory, ControlTrajectoryConfig
from .setpoint_channel import SetpointChannel
from .telemetry import TelemetryQueue, TelemetryRecord


def build_control_config(rt: DictConfig, *, n_joints: int) -> DictConfig:
  if "control_loop" not in rt:
    raise ValueError("runtime config has no `control_loop` section")
  cl = rt.control_loop
  for key in ("rate_hz", "safety"):
    if key not in cl:
      raise ValueError(f"runtime.control_loop is missing `{key}`")
  if "policy_rate_hz" not in rt:
    raise ValueError("runtime config has no `policy_rate_hz`")
  del n_joints  # the loop takes its joint count from the backend
  return rt


class ControlLoop:
  """Fixed-rate control thread: channel -> trajectory -> backend, with telemetry."""

  def __init__(
    self,
    cfg: DictConfig,
    backend: RobotBackend,
    channel: SetpointChannel,
    telemetry: TelemetryQueue
  ) -> None:
    if float(cfg.control_loop.rate_hz) <= 0:
      raise ValueError("rate_hz must be positive")

    read_cfg = cfg.control_loop.get("read", {}) or {}

    self._cfg                             = cfg
    self._backend                         = backend
    self._channel                         = channel
    self._telemetry                       = telemetry
    self._period                          = 1.0 / float(cfg.control_loop.rate_hz)
    self._stop                            = threading.Event()
    self._thread: threading.Thread | None = None
    self._ticks                           = 0
    self._modes                           = ControlModeMachine()
    self._budget_s                        = parse_read_budget(read_cfg)
    # A stale measurement must not silently become an open-loop command; the
    # limiter refuses to extend motion past this bound (see _safety_limit).
    self._stale_hold_s                    = float(read_cfg.get("stale_hold_s", 0.100))
    self._traj_cfg                        = ControlTrajectoryConfig.build(cfg, n_joints=backend.n_joints)

  @property
  def ticks(self) -> int:
    return self._ticks

  # -- mode ------------------------------------------------------------------
  #
  # The machine lives in control_mode.py; the loop owns an instance and wraps
  # it so callers drive the *loop*, not its internals. Requests are latched:
  # nothing resumes until release() is called.

  @property
  def mode(self) -> ControlMode:
    """What the loop is currently doing."""
    return self._modes.mode

  def request_estop(self, timeout: float | None = 1.0) -> bool:
    """Latch an immediate freeze; block until the loop has stopped commanding.

    Returns whether the loop acknowledged within ``timeout``. A ``False`` return
    means the control thread is not running or is wedged
    """
    self._modes.request_estop()
    return self._await(ControlMode.ESTOP, timeout)

  def request_brake(self, timeout: float | None = 1.0) -> bool:
    """Ask for a graceful coast-to-stop; block until the arm has stopped.

    Resolves when the loop reaches BRAKED — the coast-down segment has actually
    played out. Returns whether that happened within ``timeout``.
    """
    self._modes.request_brake()
    return self._await(ControlMode.BRAKED, timeout)

  def release(self, timeout: float | None = 1.0) -> bool:
    """Clear a latched stop and resume normal operation.

    Raises :class:`~chuck_dreamer.runtime.control_mode.InvalidTransition` if the
    arm is still coasting to a stop
    """
    self._modes.release()
    return self._await(ControlMode.NORMAL, timeout)

  def _await(self, mode: ControlMode, timeout: float | None) -> bool:
    """Wait for the loop to reach ``mode``, unless no thread is running."""
    if self._thread is None or not self._thread.is_alive():
      return self._modes.mode is mode
    return self._modes.wait_for(mode, timeout)

  @property
  def budget_s(self) -> float:
    """Seconds per tick the backend may spend measuring (``inf`` = unlimited)."""
    return self._budget_s

  def start(self) -> None:
    self._stop.clear()
    self._thread = threading.Thread(target=self._run, name="runtime-control", daemon=True)
    self._thread.start()

  def stop(self, join_timeout: float = 5.0) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=join_timeout)
      self._thread = None

  def _safety_limit(
    self,
    state: ControlState,
    q_ref: np.ndarray,
    qd_ref: np.ndarray,
    qdd_ref: np.ndarray,
  ) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """Constrain the trajectory's reference against the measured state.

    Returns ``(q_cmd, limits)``; ``q_cmd is None`` aborts the tick's write.

    The measurement's *age* is a first-class input here, which is the whole
    point of budgeting the read: when the backend could not refresh ``q``, the
    reference is clamped to a small envelope around the last known position
    instead of being trusted to keep extending a planned motion. Without this,
    a starved read degrades the loop to open-loop control silently — exactly
    the failure the staleness tracking exists to catch.
    """
    lower, upper = self._backend.joint_limits()
    q_cmd        = np.clip(q_ref, lower, upper)
    clamped      = bool(np.any(q_cmd != q_ref))

    limits: dict[str, Any] = {
      "clamped": clamped,
      "stale":   state.stale,
      "faults":  int(state.faults),
    }

    if state.stale:
      # Trust the measurement only as far as the arm can have moved since it
      # was taken: bound the command to the distance the velocity limit allows
      # over the measurement's age, capped at stale_hold_s so an indefinitely
      # starved loop converges to a hold rather than drifting.
      v_max                 = np.asarray(self._traj_cfg.v_max, dtype=np.float64)
      reach                 = v_max * min(state.q_age, self._stale_hold_s)
      q_cmd                 = np.clip(q_cmd, state.q - reach, state.q + reach)
      limits["stale_reach"] = float(np.max(reach)) if reach.size else 0.0

    return q_cmd, limits

  def _check_stops(self, mode: ControlMode, entered: bool, trajectory: ControlTrajectory,
                   state: ControlState) -> ControlTrajectory | None:
    """The trajectory that stops the arm, or ``None`` if it should keep running.

    ``entered`` marks the first tick in ``mode`` (see ``ControlModeMachine.take``)
    """
    if mode is ControlMode.ESTOP:
      return trajectory.stop(state.q)

    if mode is ControlMode.BRAKED:
      return trajectory                      # already stopped; just keep holding

    if mode is ControlMode.BRAKING:
      if entered:
        return trajectory.brake()
      if trajectory.stationary:
        self._modes.braked()
      return trajectory

    return None

  def _run(self) -> None:
    state          = self._backend.read_state(math.inf)
    last_time      = state.now
    next_deadline  = time.monotonic()
    current_action = None
    trajectory     = ControlTrajectory.hold(state.q, self._traj_cfg)

    while not self._stop.is_set():
      metrics = TelemetryRecord(t_wall=time.time(), t_mono=time.monotonic(), tick=self._ticks)

      with metrics.timed("read_state"):
        state = self._backend.read_state(self._budget_s)

      dt        = state.now - last_time
      last_time = state.now

      if state.faults:
        self._telemetry.emit_event("read_degraded", tick=self._ticks, detail=f"{FaultFlags(state.faults)!s} q_age={state.q_age:.4f}")

      mode, entered = self._modes.take()
      if stop_trajectory := self._check_stops(mode, entered, trajectory, state):
        trajectory = stop_trajectory
      else:
        action = self._channel.get()
        if action is not None and action != current_action:
          trajectory     = trajectory.update(action, state)
          current_action = action

      metrics.update(state=state, dt=dt, mode=mode)
      if trajectory is not None:
        ref = trajectory.tick(dt)
        if ref is None:
          # The segment ran out with nothing behind it: coast to a stop and hold
          # there rather than freezing mid-motion. Under BRAKING this is the
          # coast-down finishing, and _plan promotes to BRAKED next tick.
          trajectory = trajectory.brake()
          ref        = trajectory.tick(dt)

        if ref is not None:
          q_cmd, limits = self._safety_limit(state, *ref)
          if q_cmd is not None:
            with metrics.timed("write_positions"):
              self._backend.write_positions(q_cmd)
          metrics.update(ref=ref, q_cmd=q_cmd, limits=limits)

      self._telemetry.emit(metrics)
      self._ticks += 1

      next_deadline += self._period
      sleep_for      = next_deadline - time.monotonic()
      if sleep_for > 0:
        self._stop.wait(sleep_for)  # interruptible: shutdown is immediate
      else:
        # Overrun: we missed the deadline. Log it and re-anchor so we don't
        # spiral trying to catch up an unbounded backlog.
        self._telemetry.emit_event("control_overrun", tick=self._ticks, detail=f"late_by={-sleep_for:.4f}s")
        next_deadline = time.monotonic()
