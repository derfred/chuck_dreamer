"""The trusted control kernel: mode machine + synchronized slew + clamp.

This is the safety core of the runtime (spec §3.7, §10). It is deliberately
*pure and synchronous*: it owns no clock, no threads, and no backend. The
threaded driver (:class:`~chuck_dreamer.runtime.control_loop.ControlLoop`)
reads the backend and the setpoint channel, calls :meth:`ControlKernel.tick`
with the elapsed wall time, and writes the result back. Keeping the logic
clock-free is what makes every M1 exit criterion — clamp at the boundary,
the joint velocity limit, the e-stop latch, the mode table — testable by
calling :meth:`tick` in a loop with hand-chosen ``dt`` values.

Three modes (spec §10.4):

- ``NORMAL``  — follow the requested target through the slew + clamp.
- ``HOLD``    — ignore the target; hold the last commanded position.
- ``ESTOP``   — latched hold. The *only* way out is :meth:`request_hold`;
                :meth:`request_normal` is refused while latched, so resume
                is always the explicit two step ESTOP → HOLD → NORMAL.

The slew reference is the last *commanded* position ``q_cmd`` (not the
measured ``q_meas``): interpolation-only mode is open loop, which keeps the
commanded trajectory continuous even when the arm lags (spec §3.7, §3.5
step 2). The final clamp is unconditional and applied last — it is the
authoritative enforcement point for joint limits (spec §10.3).
"""

from __future__ import annotations

import enum
import threading
from dataclasses import dataclass

import numpy as np


class Mode(enum.Enum):
  HOLD = "hold"
  NORMAL = "normal"
  ESTOP = "estop"


@dataclass(frozen=True)
class KernelLimits:
  """Authoritative joint-space safety envelope.

  ``lower``/``upper`` are per-joint position bounds (rad); ``max_velocity``
  is the per-joint joint-space velocity cap (rad/s). All shape ``(n,)``.
  """

  lower: np.ndarray
  upper: np.ndarray
  max_velocity: np.ndarray

  @classmethod
  def build(
    cls,
    lower: np.ndarray,
    upper: np.ndarray,
    max_velocity: float | np.ndarray,
  ) -> "KernelLimits":
    """Construct, broadcasting a scalar ``max_velocity`` to per-joint."""
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.shape != hi.shape:
      raise ValueError(f"lower/upper shape mismatch: {lo.shape} vs {hi.shape}")
    if np.any(hi < lo):
      raise ValueError("every upper bound must be >= its lower bound")
    vmax = np.broadcast_to(
      np.asarray(max_velocity, dtype=np.float64), lo.shape).copy()
    if np.any(vmax <= 0):
      raise ValueError("max_velocity must be positive")
    return cls(lower=lo.copy(), upper=hi.copy(), max_velocity=vmax)

  @property
  def n_joints(self) -> int:
    return int(self.lower.shape[0])


@dataclass(frozen=True)
class KernelOutput:
  """Result of one :meth:`ControlKernel.tick`."""

  q_cmd: np.ndarray
  mode: Mode
  clamped: bool
  velocity_limited: bool
  dt: float


class ControlKernel:
  """Pure safety kernel. Thread-safe mode requests; clock-free :meth:`tick`."""

  def __init__(
    self,
    limits: KernelLimits,
    q_init: np.ndarray,
    mode: Mode = Mode.HOLD,
  ) -> None:
    self._limits = limits
    q0           = np.asarray(q_init, dtype=np.float64)
    if q0.shape != (limits.n_joints,):
      raise ValueError(
        f"q_init shape {q0.shape} != ({limits.n_joints},)")

    self._q_cmd = np.clip(q0, limits.lower, limits.upper)
    self._mode  = mode
    self._lock  = threading.Lock()

  # -- mode machine ---------------------------------------------------------

  @property
  def mode(self) -> Mode:
    with self._lock:
      return self._mode

  def request_estop(self) -> bool:
    """Latch into ESTOP (idempotent). Always succeeds."""
    with self._lock:
      self._mode = Mode.ESTOP
    return True

  def request_estop_at(self, q: np.ndarray) -> bool:
    """Latch into ESTOP with the slew reference snapped to ``q`` (idempotent).

    Used by the following-error watchdog (spec §3.14): unlike
    :meth:`request_estop`, which freezes at whatever ``q_cmd`` currently is,
    this snaps to the *measured* position passed in. A following-error trip
    means ``q_cmd`` has drifted away from reality (that drift is exactly
    what tripped it), so latching there would, on release, slam the arm
    back toward the obstruction instead of holding where it actually sits.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (self._limits.n_joints,):
      raise ValueError(f"q shape {q.shape} != ({self._limits.n_joints},)")
    with self._lock:
      self._q_cmd = np.clip(q, self._limits.lower, self._limits.upper)
      self._mode = Mode.ESTOP
    return True

  def request_normal(self) -> bool:
    """Enter NORMAL. Allowed only from HOLD; refused while ESTOP latched.

    Returns whether the request was honored — a refused resume returns
    ``False`` (the caller logs it as a refusal event).
    """
    with self._lock:
      if self._mode is Mode.ESTOP:
        return False
      self._mode = Mode.NORMAL
      return True

  def request_hold(self) -> bool:
    """Enter HOLD. Allowed from any mode; the only exit from ESTOP."""
    with self._lock:
      self._mode = Mode.HOLD
    return True

  def reset(self, q: np.ndarray) -> None:
    """Snap the slew reference to ``q`` (startup / backend resync)."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (self._limits.n_joints,):
      raise ValueError(f"reset q shape {q.shape} != ({self._limits.n_joints},)")
    with self._lock:
      self._q_cmd = np.clip(q, self._limits.lower, self._limits.upper)

  @property
  def q_cmd(self) -> np.ndarray:
    with self._lock:
      return self._q_cmd.copy()

  # -- the safety step ------------------------------------------------------

  def tick(
    self,
    target: np.ndarray | None,
    q_meas: np.ndarray,  # noqa: ARG002 — open-loop; reserved for PID mode (M1 fallback)
    dt: float,
  ) -> KernelOutput:
    """Advance one control step and return the command to issue.

    ``q_meas`` is accepted for the interface (and future PID mode) but the
    interpolation-only kernel slews from ``q_cmd``, not the measured state.
    """
    lim = self._limits
    with self._lock:
      mode  = self._mode
      q_cmd = self._q_cmd

      # 1. Mode resolves the desired target. HOLD/ESTOP — and NORMAL before
      #    the first setpoint — collapse to "stay where commanded".
      if mode is Mode.NORMAL and target is not None:
        desired = np.asarray(target, dtype=np.float64)
        if desired.shape != (lim.n_joints,):
          raise ValueError(
            f"target shape {desired.shape} != ({lim.n_joints},)")
      else:
        desired = q_cmd

      # 2. Synchronized slew: one shared scale so the step stays a straight
      #    line in joint space. alpha is the largest fraction of the full
      #    delta that respects every joint's per-tick cap.
      delta    = desired - q_cmd
      max_step = lim.max_velocity * max(dt, 0.0)
      with np.errstate(divide="ignore", invalid="ignore"):
        need = np.abs(delta) / np.where(max_step > 0.0, max_step, np.inf)

      need_max         = float(np.max(need)) if need.size else 0.0
      velocity_limited = need_max > 1.0
      alpha            = 1.0 if need_max <= 1.0 else 1.0 / need_max
      q_next           = q_cmd + alpha * delta

      # 3. Authoritative final clamp (last line of defense, spec §10.3).
      q_clamped = np.clip(q_next, lim.lower, lim.upper)
      clamped   = bool(np.any(q_clamped != q_next))

      # 4. Commit: q_cmd is the slew reference for the next tick.
      self._q_cmd = q_clamped

    return KernelOutput(
      q_cmd=q_clamped.copy(),
      mode=mode,
      clamped=clamped,
      velocity_limited=velocity_limited,
      dt=float(dt),
    )
