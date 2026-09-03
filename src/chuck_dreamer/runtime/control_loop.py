from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np
from omegaconf import DictConfig

from .backend import RobotBackend
from .control_channel import ControlChannel
from .control_mode import ControlMode, ControlModeMachine
from .control_state import ControlState, FaultFlags, parse_read_budget
from .control_trajectory import ControlTrajectory, ControlTrajectoryConfig
from .telemetry import TelemetryQueue, TelemetryRecord
from .workspace import WorkspaceLimit, WorkspaceLimiter, build_workspace_limiter


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
    channel: ControlChannel,
    telemetry: TelemetryQueue,
    workspace: WorkspaceLimiter | None = None,
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
    # Cartesian envelope (None unless configured); see workspace.py. Built here
    # rather than injected so a misconfigured box fails at construction, before
    # any thread has started commanding an arm.
    self._workspace: WorkspaceLimiter | None = workspace if workspace is not None else build_workspace_limiter(cfg)
    if self._workspace is not None:
      self._check_home_inside_workspace(backend)

  def _check_home_inside_workspace(self, backend: RobotBackend) -> None:
    """Refuse a box that does not contain the arm's rest pose."""
    assert self._workspace is not None
    ee  = self._workspace.ee_pos(backend.home_qpos)
    box = self._workspace.box
    if not box.contains(ee):
      raise ValueError(
        f"the workspace box does not contain the arm's home pose: home tip is at "
        f"{np.round(ee, 4).tolist()} m (arm frame), box is "
        f"{np.round(box.lower, 4).tolist()} .. {np.round(box.upper, 4).tolist()}. "
        f"The runtime would e-stop on its first tick with no way to release; widen "
        f"the box or move the arm's home pose inside it.")

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

  def stop_for_shutdown(self, timeout: float | None = 1.0) -> bool:
    """Bring the arm to rest for teardown; block until it is. Returns whether that happened within ``timeout``."""
    if self.mode is ControlMode.ESTOP:
      return True
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

    if self._workspace is not None:
      # The Cartesian envelope goes last: it reasons about where the *tip* ends
      # up, so it must see the command that the joint-space limits above have
      # already settled on, not the raw reference.
      q_cmd, ws             = self._workspace.limit(state.q, q_cmd)
      limits["ws_scale"]    = ws.scale
      limits["ws_breach_m"] = ws.breach_m
      if ws.limited:
        limits["clamped"] = True
      if ws.breached:
        # Measured tip outside the box. No command can undo that, so latch the
        # e-stop (only an explicit release() clears it) and hold where we are.
        self._breach(ws)
        return None, limits

    return q_cmd, limits

  def _breach(self, ws: "WorkspaceLimit") -> None:
    """Latch an e-stop for a workspace breach, announcing it once.

    Called from the control thread each tick the tip is measured outside the
    box, so the event is emitted only on the transition: a latched e-stop that
    re-announced itself at the control rate would bury the rest of the log.
    """
    already = self._modes.mode is ControlMode.ESTOP
    self._modes.request_estop()
    if not already:
      self._telemetry.emit_event(
        "workspace_breach", tick=self._ticks,
        detail=f"ee={np.round(ws.ee_pos, 4).tolist()} outside_by={ws.breach_m:.4f}m; release() required")

  def _check_stops(self, mode: ControlMode, entered: bool, trajectory: ControlTrajectory, state: ControlState) -> ControlTrajectory | None:
    """The trajectory that *replaces* the running one, or ``None`` to keep it.

    ``entered`` marks the first tick in ``mode`` (see ``ControlModeMachine.take``)
    """
    if mode is ControlMode.ESTOP:
      if entered:
        return trajectory.stop(state.q)

    if mode is ControlMode.BRAKING:
      if entered:
        return trajectory.brake()
      if trajectory.stationary:
        self._modes.braked()

    return None

  def _run(self) -> None:
    state          = self._backend.read_state(math.inf)
    last_time      = state.now
    next_deadline  = time.monotonic()
    current_action = None
    trajectory     = ControlTrajectory.hold(state.q, self._traj_cfg)
    q_cmd_last     = None

    while not self._stop.is_set():
      metrics = TelemetryRecord(t_wall=time.time(), t_mono=time.monotonic(), tick=self._ticks)

      with metrics.timed("read_state"):
        state = self._backend.read_state(self._budget_s).at_tick(self._ticks)
      self._channel.publish_state(state)

      dt        = state.now - last_time
      last_time = state.now

      if state.faults:
        self._telemetry.emit_event("read_degraded", tick=self._ticks, detail=f"{FaultFlags(state.faults)!s} q_age={state.q_age:.4f}")

      mode, entered = self._modes.take()
      if entered:
        self._telemetry.emit_event("mode_change", tick=self._ticks, detail=f"{mode.name} q_age={state.q_age:.4f}")
      if stop_trajectory := self._check_stops(mode, entered, trajectory, state):
        trajectory = stop_trajectory
      elif not mode.stopping:
        action = self._channel.get()
        if action is not None and action != current_action:
          trajectory     = trajectory.update(action, state, q_ref_actual=q_cmd_last)
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
            q_cmd_last = q_cmd
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
