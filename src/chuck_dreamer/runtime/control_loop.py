"""The threaded control driver and its watchdog (spec §3.7, §10.5, §3.14).

:class:`ControlLoop` is the threaded wrapper around the pure
:class:`~chuck_dreamer.runtime.kernel.ControlKernel`: it owns the
absolute-deadline timing, reads the backend and the setpoint channel, calls
``kernel.tick`` with the *measured* elapsed time, writes the command back,
observes the watchdog, and emits telemetry. It contains no safety logic of
its own — that all lives in the kernel and the watchdog.

:class:`Watchdog` combines two independent checks behind one trip latch
(spec §10.5, §3.14):

- **Liveness**: an external heartbeat thread trips if the control loop goes
  silent past ``timeout_s``. It runs on its own thread so that a hung
  control loop cannot also prevent this trip.
- **Following error**: on every :meth:`Watchdog.observe` call (i.e. every
  control tick, on the control-loop thread), if configured with
  ``fe_limits``, checks the per-joint tracking error against a
  velocity-scaled threshold and trips if any joint's breach has been
  continuous for longer than ``fe_limits.T``. This check is synchronous —
  it needs the exact (q_cmd, q_meas) pair the kernel just produced, so it
  cannot live on the independent poll thread the way the liveness check
  does.

Either check latches the same ``_tripped`` flag and fires the same
``on_trip`` callback; the two are otherwise unrelated conditions sharing one
trip mechanism.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from .backend import RobotBackend
from .kernel import ControlKernel
from .setpoint_channel import SetpointChannel
from .telemetry import TelemetryQueue, TelemetryRecord


@contextmanager
def _timed(acc: list[float]) -> Iterator[None]:
  """Append the wall time spent in the block (seconds) to ``acc``."""
  t0 = time.perf_counter()
  try:
    yield
  finally:
    acc.append(time.perf_counter() - t0)


@dataclass(frozen=True)
class FollowingErrorLimits:
  """Following-error watchdog thresholds (spec §3.14).

  ``theta_0`` and ``T`` are global (they derive from the shared servo model
  and its stall/transient behavior, not from any one joint's mounting);
  ``k`` is per-joint (distal joints, with lower gear ratio and higher
  reflected load inertia, need a larger allowance per unit commanded
  velocity than proximal joints).
  """

  theta_0: float
  k: np.ndarray  # (n,) s
  T: float

  @classmethod
  def build(cls, theta_0: float, k: float | np.ndarray, T: float, n_joints: int) -> "FollowingErrorLimits":
    if theta_0 < 0:
      raise ValueError("theta_0 must be non-negative")
    if T <= 0:
      raise ValueError("T must be positive")
    k_arr = np.broadcast_to(np.asarray(k, dtype=np.float64), (n_joints,)).copy()
    if np.any(k_arr < 0):
      raise ValueError("k must be non-negative")
    return cls(theta_0=float(theta_0), k=k_arr, T=float(T))


class Watchdog:
  """Heartbeat + following-error monitor: trips ``on_trip`` on either fault.

  ``observe(q_cmd, q_meas, v_cmd)`` replaces the old ``kick()`` as the
  per-tick call from :class:`ControlLoop`: it always records liveness, and
  — when constructed with ``fe_limits`` — also runs the following-error
  check synchronously, on the caller's (control-loop) thread.
  """

  def __init__(
    self,
    timeout_s: float,
    on_trip: Callable[[], None],
    *,
    fe_limits: FollowingErrorLimits | None = None,
    poll_interval_s: float | None = None,
    grace_s: float | None = None,
  ) -> None:
    if timeout_s <= 0:
      raise ValueError("timeout_s must be positive")
    self._timeout                         = float(timeout_s)
    self._on_trip                         = on_trip
    self._fe_limits                       = fe_limits
    self._poll                            = poll_interval_s if poll_interval_s is not None else timeout_s / 4.0
    # required for safe startup of a slow-booting backend like the real Feetech bus or MuJoCo sim
    self._grace                           = float(grace_s) if grace_s is not None else max(self._timeout, 1.0)
    self._lock                            = threading.Lock()
    self._last_kick                       = time.monotonic()
    self._armed_at                        = time.monotonic()
    self._tripped                         = False
    self._trip_reason                     = ""
    self._trip_q_meas: np.ndarray | None   = None
    # Per-joint contiguous-breach start time (NaN = not currently breaching).
    self._fe_breach_since                 = (
      np.full(fe_limits.k.shape, np.nan) if fe_limits is not None else np.zeros(0)
    )
    self._stop                            = threading.Event()
    self._thread: threading.Thread | None = None

  def observe(self, q_cmd: np.ndarray, q_meas: np.ndarray, v_cmd: np.ndarray) -> None:
    """Per-tick call: records liveness, then (if configured) checks following error.

    ``v_cmd`` is the commanded joint velocity (finite-differenced by the
    caller from consecutive ``q_cmd`` samples, or read from the
    interpolator's own state) — this is *not* recomputed here so the
    watchdog stays a pure consumer of whatever the control loop already
    tracks.
    """
    with self._lock:
      self._last_kick = time.monotonic()

    if self._fe_limits is None:
      return
    self._check_following_error(q_cmd, q_meas, v_cmd)

  def _check_following_error(self, q_cmd: np.ndarray, q_meas: np.ndarray, v_cmd: np.ndarray) -> None:
    assert self._fe_limits is not None
    lim = self._fe_limits
    now = time.monotonic()
    e = np.asarray(q_cmd, dtype=np.float64) - np.asarray(q_meas, dtype=np.float64)
    theta = lim.theta_0 + lim.k * np.abs(v_cmd)
    over = np.abs(e) > theta

    tripped_joint = None
    with self._lock:
      if self._tripped:
        return  # already latched (by either check); nothing more to do
      for j in range(len(e)):
        if over[j]:
          if np.isnan(self._fe_breach_since[j]):
            self._fe_breach_since[j] = now
          elif now - self._fe_breach_since[j] > lim.T:
            tripped_joint = j
            break
        else:
          self._fe_breach_since[j] = np.nan
      if tripped_joint is not None:
        self._tripped = True
        self._trip_reason = (
          f"following_error: joint={tripped_joint} "
          f"|e|={abs(float(e[tripped_joint])):.4f} theta={float(theta[tripped_joint]):.4f}"
        )
        # Spec §3.14: freeze at the *measured* position, not the drifted
        # q_cmd — the whole point of this trip is that q_cmd ran away from
        # reality, so latching there would (on release) slam the arm back
        # toward the obstruction instead of holding where it actually is.
        self._trip_q_meas = np.asarray(q_meas, dtype=np.float64).copy()

    if tripped_joint is not None:
      self._on_trip()

  @property
  def tripped(self) -> bool:
    with self._lock:
      return self._tripped

  @property
  def trip_q_meas(self) -> np.ndarray | None:
    """Measured position at the moment of a following-error trip, else ``None``."""
    with self._lock:
      return None if self._trip_q_meas is None else self._trip_q_meas.copy()

  @property
  def trip_reason(self) -> str:
    """Human-readable cause of the most recent trip (``""`` if untripped or liveness-tripped)."""
    with self._lock:
      return self._trip_reason

  def start(self) -> None:
    now = time.monotonic()
    with self._lock:
      self._last_kick = now
      self._armed_at = now
      self._tripped = False
      self._trip_reason = ""
      self._trip_q_meas = None
      self._fe_breach_since[:] = np.nan
    self._stop.clear()
    self._thread = threading.Thread(target=self._run, name="runtime-watchdog", daemon=True)
    self._thread.start()

  def stop(self, join_timeout: float = 2.0) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=join_timeout)
      self._thread = None

  def _run(self) -> None:
    while not self._stop.wait(self._poll):
      now = time.monotonic()
      with self._lock:
        silent_for = now - self._last_kick
        since_armed = now - self._armed_at
        already_tripped = self._tripped
      if since_armed < self._grace:
        continue  # still within the startup arm delay
      if silent_for > self._timeout and not already_tripped:
        with self._lock:
          self._tripped = True
          self._trip_reason = f"loop_stalled: silent_for={silent_for:.3f}s"
        self._on_trip()


class ControlLoop:
  """Fixed-rate control thread: channel -> kernel -> backend, with telemetry."""

  def __init__(
    self,
    kernel: ControlKernel,
    backend: RobotBackend,
    channel: SetpointChannel,
    telemetry: TelemetryQueue,
    *,
    rate_hz: float,
    watchdog: Watchdog | None = None,
  ) -> None:
    if rate_hz <= 0:
      raise ValueError("rate_hz must be positive")
    self._kernel                          = kernel
    self._backend                         = backend
    self._channel                         = channel
    self._telemetry                       = telemetry
    self._period                          = 1.0 / float(rate_hz)
    self._watchdog                        = watchdog
    self._stop                            = threading.Event()
    self._thread: threading.Thread | None = None
    self._ticks                           = 0

  @property
  def ticks(self) -> int:
    return self._ticks

  def start(self) -> None:
    self._stop.clear()
    self._thread = threading.Thread(target=self._run, name="runtime-control", daemon=True)
    self._thread.start()

  def stop(self, join_timeout: float = 5.0) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=join_timeout)
      self._thread = None

  def _run(self) -> None:
    # Snap the kernel to the measured pose so the first slew starts from
    # where the arm actually is (no jump on the first command).
    self._kernel.reset(self._backend.read_positions())
    next_deadline = time.monotonic()
    last_mono     = time.monotonic()
    q_cmd_prev    = self._kernel.q_cmd

    while not self._stop.is_set():
      now       = time.monotonic()
      dt        = now - last_mono
      last_mono = now

      backend_s = []
      with _timed(backend_s):
        q_meas = self._backend.read_positions()

      target, seq = self._channel.get_with_seq()
      tick_dt     = dt if dt > 0 else self._period
      out         = self._kernel.tick(target, q_meas, tick_dt)

      with _timed(backend_s):
        self._backend.write_positions(out.q_cmd)

      if self._watchdog is not None:
        v_cmd = (out.q_cmd - q_cmd_prev) / tick_dt
        self._watchdog.observe(out.q_cmd, q_meas, v_cmd)
      q_cmd_prev = out.q_cmd

      self._telemetry.emit(TelemetryRecord(
        t_wall=time.time(), t_mono=now, tick=self._ticks, mode=out.mode.value,
        clamped=out.clamped, velocity_limited=out.velocity_limited, dt=out.dt,
        q_meas=q_meas, q_cmd=out.q_cmd,
        target=target if target is not None else None, seq=seq,
        backend_s=sum(backend_s),
      ))
      self._ticks += 1

      next_deadline += self._period
      sleep_for = next_deadline - time.monotonic()
      if sleep_for > 0:
        self._stop.wait(sleep_for)  # interruptible: shutdown is immediate
      else:
        # Overrun: we missed the deadline. Log it and re-anchor so we don't
        # spiral trying to catch up an unbounded backlog.
        self._telemetry.emit_event(
          "control_overrun", tick=self._ticks,
          detail=f"late_by={-sleep_for:.4f}s")
        next_deadline = time.monotonic()
