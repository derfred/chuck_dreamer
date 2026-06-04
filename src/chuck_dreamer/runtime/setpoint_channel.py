"""The single-slot, latest-wins setpoint channel (spec §4.2).

The only cross-thread handoff of a setpoint: the policy/scripted thread
publishes, the control thread reads. There is exactly one slot — a publish
overwrites the previous value, and the control loop reads whatever is most
recent. Old setpoints are dropped by design; the control loop never blocks
and never waits for a producer (when the producer stalls the kernel simply
holds the last command).

Arrays are copied on both ``publish`` and ``get`` so neither side aliases
the other's buffer. A monotonic sequence number lets telemetry and tests
observe how many distinct setpoints the control loop has seen.
"""

from __future__ import annotations

import threading

import numpy as np


class SetpointChannel:
  """Thread-safe single-slot latest-wins channel of joint setpoints."""

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._value: np.ndarray | None = None
    self._seq = 0

  def publish(self, q: np.ndarray) -> None:
    """Overwrite the slot with a copy of ``q`` and bump the sequence."""
    q = np.asarray(q, dtype=np.float64).copy()
    with self._lock:
      self._value = q
      self._seq += 1

  def get(self) -> np.ndarray | None:
    """Most recent setpoint (a copy), or ``None`` before any publish."""
    with self._lock:
      return None if self._value is None else self._value.copy()

  def get_with_seq(self) -> tuple[np.ndarray | None, int]:
    """:meth:`get` plus the current sequence number, read atomically."""
    with self._lock:
      value = None if self._value is None else self._value.copy()
      return value, self._seq

  @property
  def seq(self) -> int:
    with self._lock:
      return self._seq
