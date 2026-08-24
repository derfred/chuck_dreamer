from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import IntFlag

import numpy as np


class FaultFlags(IntFlag):
  NONE          = 0
  STALE_Q       = 1 << 0   # q is older than the configured staleness bound
  READ_STARVED  = 1 << 1   # budget did not cover even a position read
  BUS_ERROR     = 1 << 2   # a transaction raised; cached values were reused
  SLOW_DEFERRED = 1 << 3   # tier-2 block was due but did not fit the budget
  OVER_TEMP     = 1 << 4   # a servo exceeded the temperature limit
  OVER_LOAD     = 1 << 5   # a servo exceeded the load limit
  UNDER_VOLT    = 1 << 6   # bus voltage sagged below the limit


@dataclass(frozen=True, slots=True)
class ControlState:
  now:      float                    # time.monotonic() at the start of the read
  q:        np.ndarray               # (n,) rad, follower order — always present
  q_age:    float = 0.0              # seconds since q was measured on the wire

  # Tier 1
  qd:       np.ndarray | None = None # (n,) rad/s
  load:     np.ndarray | None = None # (n,) normalised torque, [-1, 1]

  # Tier 2
  volt:     np.ndarray | None = None # (n,) volts
  temp:     np.ndarray | None = None # (n,) degrees C
  curr:     np.ndarray | None = None # (n,) amps
  slow_age: float = math.inf         # seconds since the tier-2 block was read

  # Bookkeeping the loop feeds back into the budget / telemetry.
  read_s:   float = 0.0              # wall cost of this read's transactions
  faults:   FaultFlags = FaultFlags.NONE

  @property
  def stale(self) -> bool:
    """Whether the safety layer should treat ``q`` as untrustworthy."""
    return bool(self.faults & (FaultFlags.STALE_Q | FaultFlags.READ_STARVED))

  @property
  def healthy(self) -> bool:
    return self.faults == FaultFlags.NONE

  def with_faults(self, extra: FaultFlags) -> ControlState:
    """Copy with ``extra`` flags OR-ed in (the safety layer annotates states)."""
    return replace(self, faults=self.faults | extra)


def parse_read_budget(read_cfg) -> float:
  """Seconds per tick the backend may spend measuring, from ``control_loop.read``.

  ``budget_s`` accepts a number of seconds, or ``"max"`` / ``null`` for
  unlimited (``math.inf``) — the default, which reads everything every tick.

  A typo is rejected here rather than coerced, because every plausible
  coercion is worse than a crash: ``float("fast")`` failing silently to 0.0
  would starve every read for the whole run, and the loop would keep going on
  ever-staler cached positions.
  """
  raw = None if read_cfg is None else read_cfg.get("budget_s", "max")
  if raw is None or (isinstance(raw, str) and raw.strip().lower() == "max"):
    return math.inf
  try:
    budget = float(raw)
  except (TypeError, ValueError):
    raise ValueError(
      f'control_loop.read.budget_s must be a number or "max"; got {raw!r}'
    ) from None
  if budget < 0:
    raise ValueError(f"control_loop.read.budget_s must be non-negative; got {budget}")
  return budget
