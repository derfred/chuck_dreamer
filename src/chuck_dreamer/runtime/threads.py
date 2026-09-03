from __future__ import annotations

import threading
import time
from collections.abc import Iterator

__all__ = ["ManagedThread", "PacedLoop"]


class ManagedThread:
  """A daemon thread with a stop event and a join-on-stop teardown.

  Subclasses implement :meth:`_run` (the whole thread body) and may override
  :meth:`_on_start` / :meth:`_on_stop` for setup and teardown that must happen
  around the thread's life rather than inside it. ``_run`` returning ends the
  thread; it should loop until ``self._stop`` is set.
  """

  def __init__(self, name: str) -> None:
    self._thread_name                     = name
    self._stop                            = threading.Event()
    self._thread: threading.Thread | None = None

  @property
  def running(self) -> bool:
    """Whether the thread is alive (a stopped or never-started loop is not)."""
    return self._thread is not None and self._thread.is_alive()

  def start(self) -> None:
    self._on_start()
    self._stop.clear()
    self._thread = threading.Thread(target=self._run, name=self._thread_name, daemon=True)
    self._thread.start()

  def stop(self, join_timeout: float = 5.0) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=join_timeout)
      self._thread = None
    self._on_stop()

  # -- subclass hooks ---------------------------------------------------------

  def _on_start(self) -> None:
    """Prepare state before the thread spawns. Raising here starts nothing."""

  def _on_stop(self) -> None:
    """Tear down after the thread has been joined (flush, close, drain)."""

  def _run(self) -> None:                                    # pragma: no cover
    raise NotImplementedError


class PacedLoop(ManagedThread):
  """A :class:`ManagedThread` whose body iterates :meth:`ticks` at a fixed rate.

  The schedule is absolute -- each deadline is the previous one plus the
  period, so the loop does not accumulate the cost of its own ticks. When a
  tick misses its deadline the schedule is re-anchored to now: catching up
  would mean running a burst of back-to-back ticks against a robot, which is
  worse than simply being late.
  """

  def __init__(self, name: str, rate_hz: float) -> None:
    if rate_hz <= 0:
      raise ValueError("rate_hz must be positive")
    super().__init__(name)
    self._period = 1.0 / float(rate_hz)

  @property
  def period(self) -> float:
    return self._period

  def paced(self) -> Iterator[int]:
    """Yield once per period until stopped, pacing the caller between yields."""
    next_deadline = time.monotonic()
    i             = 0
    while not self._stop.is_set():
      yield i
      i += 1

      next_deadline += self._period
      sleep_for      = next_deadline - time.monotonic()
      if sleep_for > 0:
        # Interruptible, so a stop() lands within one tick instead of one period.
        self._stop.wait(sleep_for)
      else:
        self._on_overrun(-sleep_for)
        next_deadline = time.monotonic()

  # -- subclass hooks ---------------------------------------------------------

  def _on_overrun(self, late_by_s: float) -> None:
    """Report a missed deadline. ``late_by_s`` is the overshoot in seconds."""
