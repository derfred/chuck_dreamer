from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from .backend import RobotBackend
from .kernel import ControlKernel
from .control_trajectory import ControlTrajectory
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
  ) -> None:
    if rate_hz <= 0:
      raise ValueError("rate_hz must be positive")
    self._kernel                          = kernel
    self._backend                         = backend
    self._channel                         = channel
    self._telemetry                       = telemetry
    self._period                          = 1.0 / float(rate_hz)
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

      backend_s: list[float] = []
      with _timed(backend_s):
        q_meas = self._backend.read_positions()

      target, seq = self._channel.get_with_seq()
      tick_dt     = dt if dt > 0 else self._period
      out         = self._kernel.tick(target, q_meas, tick_dt)

      with _timed(backend_s):
        self._backend.write_positions(out.q_cmd)

      self._telemetry.emit(TelemetryRecord(
        t_wall=time.time(), t_mono=now, tick=self._ticks, mode=out.mode.value,
        clamped=out.clamped, velocity_limited=out.velocity_limited, dt=out.dt,
        q_meas=q_meas, q_cmd=out.q_cmd,
        target=target if target is not None else None, seq=seq,
        backend_s=sum(backend_s),
      ))
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
