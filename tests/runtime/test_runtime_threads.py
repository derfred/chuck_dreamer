from __future__ import annotations

import threading
import time

import pytest

from chuck_dreamer.runtime.threads import ManagedThread, PacedLoop


class _Counter(PacedLoop):
  def __init__(self, rate_hz=200.0, work_s=0.0):
    super().__init__("test-paced", rate_hz)
    self.ticks = 0
    self.seen: list[int] = []
    self.overruns: list[float] = []
    self._work_s = work_s

  def _run(self) -> None:
    for i in self.paced():
      self.ticks += 1
      self.seen.append(i)
      if self._work_s:
        time.sleep(self._work_s)

  def _on_overrun(self, late_by_s: float) -> None:
    self.overruns.append(late_by_s)


class _Hooked(ManagedThread):
  def __init__(self):
    super().__init__("test-managed")
    self.events: list[str] = []
    self.ran = threading.Event()

  def _on_start(self) -> None:
    self.events.append("start")

  def _on_stop(self) -> None:
    self.events.append("stop")

  def _run(self) -> None:
    self.ran.set()
    self._stop.wait()


def test_lifecycle_runs_hooks_in_order_around_the_thread():
  t = _Hooked()
  assert not t.running
  t.start()
  assert t.ran.wait(1.0)
  assert t.running
  t.stop()
  assert not t.running
  assert t.events == ["start", "stop"]


def test_stop_is_idempotent_and_safe_before_start():
  t = _Hooked()
  t.stop()                      # never started
  t.start()
  t.stop()
  t.stop()                      # already stopped
  assert not t.running


def test_the_yielded_index_counts_ticks_from_zero():
  loop = _Counter(rate_hz=200.0)
  loop.start()
  time.sleep(0.1)
  loop.stop()
  assert loop.seen == list(range(len(loop.seen)))


def test_paced_loop_ticks_at_about_the_configured_rate():
  loop = _Counter(rate_hz=100.0)
  loop.start()
  time.sleep(0.3)
  loop.stop()
  # ~30 ticks; generous bounds so a loaded machine does not fail the suite.
  assert 5 <= loop.ticks <= 60
  assert loop.overruns == []


def test_a_slow_tick_reports_an_overrun_and_reanchors():
  """A tick longer than the period must be reported, not silently absorbed.

  This is the regression the extraction exists to prevent: the policy loop
  and webcam poll previously let a missed deadline slide the schedule
  forward with no signal at all.
  """
  loop = _Counter(rate_hz=100.0, work_s=0.05)   # 50 ms of work per 10 ms period
  loop.start()
  time.sleep(0.25)
  loop.stop()

  assert loop.overruns, "a tick that outran its period reported nothing"
  assert all(late > 0 for late in loop.overruns)
  # Re-anchoring means it does not try to catch up a backlog: with 50 ms
  # ticks in 250 ms there can only be a handful of them.
  assert loop.ticks <= 10


def test_a_nonpositive_rate_is_refused_at_construction():
  for bad in (0.0, -1.0):
    with pytest.raises(ValueError, match="rate_hz must be positive"):
      _Counter(rate_hz=bad)


def test_period_is_the_inverse_of_the_rate():
  assert _Counter(rate_hz=50.0).period == pytest.approx(0.02)


def test_restart_clears_the_stop_event():
  loop = _Counter(rate_hz=200.0)
  loop.start()
  time.sleep(0.05)
  loop.stop()
  first = loop.ticks

  loop.start()                  # a stopped loop can be started again
  time.sleep(0.05)
  loop.stop()
  assert loop.ticks > first


def test_loop_body_keeps_its_locals_across_ticks():
  class _Accumulator(PacedLoop):
    def __init__(self):
      super().__init__("test-locals", 500.0)
      self.final = None

    def _run(self) -> None:
      carried = 0                       # a plain local, not self._carried
      for _ in self.paced():
        carried += 1              # noqa: SIM113 - a manual local is the point
        self.final = carried

  loop = _Accumulator()
  loop.start()
  time.sleep(0.1)
  loop.stop()
  assert loop.final is not None and loop.final > 1
