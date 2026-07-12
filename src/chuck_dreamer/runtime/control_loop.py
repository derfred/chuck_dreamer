"""The threaded control driver and its watchdog (spec §3.7, §10.5).

:class:`ControlLoop` is the threaded wrapper around the pure
:class:`~chuck_dreamer.runtime.kernel.ControlKernel`: it owns the
absolute-deadline timing, reads the backend and the setpoint channel, calls
``kernel.tick`` with the *measured* elapsed time, writes the command back,
kicks the watchdog, and emits telemetry. It contains no safety logic of its
own — that all lives in the kernel.

:class:`Watchdog` is an *external* heartbeat thread (spec §10.5): the
control loop must ``kick`` it every tick, and if it goes silent past the
timeout the watchdog latches the kernel into ESTOP. It runs on its own
thread so that a hung control loop cannot also prevent the trip.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator

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


class Watchdog:
  """Heartbeat monitor: trips ``on_trip`` if not :meth:`kick`ed in time."""

  def __init__(
    self,
    timeout_s: float,
    on_trip: Callable[[], None],
    *,
    poll_interval_s: float | None = None,
    grace_s: float | None = None,
  ) -> None:
    if timeout_s <= 0:
      raise ValueError("timeout_s must be positive")
    self._timeout                         = float(timeout_s)
    self._on_trip                         = on_trip
    self._poll                            = poll_interval_s if poll_interval_s is not None else timeout_s / 4.0
    # required for safe startup of a slow-booting backend like the real Feetech bus or MuJoCo sim
    self._grace                           = float(grace_s) if grace_s is not None else max(self._timeout, 1.0)
    self._lock                            = threading.Lock()
    self._last_kick                       = time.monotonic()
    self._armed_at                        = time.monotonic()
    self._tripped                         = False
    self._stop                            = threading.Event()
    self._thread: threading.Thread | None = None

  def kick(self) -> None:
    with self._lock:
      self._last_kick = time.monotonic()

  @property
  def tripped(self) -> bool:
    with self._lock:
      return self._tripped

  def start(self) -> None:
    now = time.monotonic()
    with self._lock:
      self._last_kick = now
      self._armed_at = now
      self._tripped = False
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

    while not self._stop.is_set():
      now       = time.monotonic()
      dt        = now - last_mono
      last_mono = now

      backend_s = []
      with _timed(backend_s):
        q_meas = self._backend.read_positions()

      target, seq = self._channel.get_with_seq()
      out         = self._kernel.tick(target, q_meas, dt if dt > 0 else self._period)

      with _timed(backend_s):
        self._backend.write_positions(out.q_cmd)

      if self._watchdog is not None:
        self._watchdog.kick()

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
