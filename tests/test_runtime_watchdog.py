"""Tests for :class:`chuck_dreamer.runtime.control_loop.Watchdog`.

Threaded, but bounded: timeouts are tiny and we poll a condition with a
deadline rather than sleeping a fixed time and asserting.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from chuck_dreamer.runtime.control_loop import Watchdog
from chuck_dreamer.runtime.kernel import ControlKernel, KernelLimits, Mode


def _wait_until(predicate, timeout=2.0, interval=0.005):
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if predicate():
      return True
    time.sleep(interval)
  return predicate()


def test_kicks_keep_alive():
  tripped = threading.Event()
  # Generous timeout vs kick interval (10x) so scheduling jitter under load
  # doesn't trip a healthy watchdog.
  wd = Watchdog(timeout_s=0.2, on_trip=tripped.set, grace_s=0.0)
  wd.start()
  try:
    # Kick well inside the timeout for a window longer than the timeout.
    end = time.monotonic() + 1.0
    while time.monotonic() < end:
      wd.kick()
      time.sleep(0.02)
    assert not tripped.is_set()
    assert not wd.tripped
  finally:
    wd.stop()


def test_trips_on_stall():
  count = {"n": 0}
  wd = Watchdog(
    timeout_s=0.03, grace_s=0.0,
    on_trip=lambda: count.__setitem__("n", count["n"] + 1),
  )
  wd.start()
  try:
    assert _wait_until(lambda: wd.tripped)
    # Trip fires exactly once and latches (idempotent thereafter).
    time.sleep(0.15)
    assert count["n"] == 1
  finally:
    wd.stop()


def test_trip_latches_kernel_into_estop():
  limits = KernelLimits.build(np.full(3, -1.0), np.full(3, 1.0), 1.0)
  kernel = ControlKernel(limits, np.zeros(3), mode=Mode.NORMAL)
  wd = Watchdog(timeout_s=0.03, on_trip=kernel.request_estop, grace_s=0.0)
  wd.start()
  try:
    assert _wait_until(lambda: kernel.mode is Mode.ESTOP)
  finally:
    wd.stop()


def test_grace_suppresses_early_trip():
  # With a grace window longer than the timeout, an immediate stall must NOT
  # trip until the grace elapses.
  tripped = threading.Event()
  wd = Watchdog(timeout_s=0.02, on_trip=tripped.set, grace_s=0.3)
  wd.start()
  try:
    time.sleep(0.1)  # well past timeout, still inside grace
    assert not tripped.is_set()
    # After the grace elapses, the still-unkicked watchdog trips.
    assert _wait_until(tripped.is_set, timeout=1.0)
  finally:
    wd.stop()


def test_rejects_bad_timeout():
  with pytest.raises(ValueError, match="timeout_s"):
    Watchdog(timeout_s=0.0, on_trip=lambda: None)
