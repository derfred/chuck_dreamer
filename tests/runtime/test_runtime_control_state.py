"""Tests for ``ControlState`` and the configured read budget.

Both are pure (no bus, no threads), so these tests pin the behaviour the
budgeted-read design rests on: a state that is honest about its own staleness,
and a budget that is a predictable constant — including the ``max`` default,
which reads everything every tick.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from chuck_dreamer.runtime.control_state import (
  ControlState,
  FaultFlags,
  parse_read_budget,
)

# -- ControlState ------------------------------------------------------------


def test_control_state_defaults_to_positions_only():
  # q is the only guaranteed field; every diagnostic is absent until measured.
  st = ControlState(now=1.0, q=np.zeros(6))
  assert st.qd is None and st.load is None
  assert st.volt is None and st.temp is None and st.curr is None
  assert st.slow_age == math.inf
  assert st.healthy and not st.stale


def test_control_state_is_immutable():
  st = ControlState(now=1.0, q=np.zeros(6))
  with pytest.raises(AttributeError):
    st.q_age = 5.0                     # frozen: a consumer cannot rewrite history


@pytest.mark.parametrize("flag", [FaultFlags.STALE_Q, FaultFlags.READ_STARVED])
def test_control_state_stale_covers_both_untrustworthy_cases(flag):
  # Either "the read was skipped" or "the reading aged out" must make the
  # safety layer treat q as untrustworthy; they are different causes of the
  # same hazard.
  st = ControlState(now=1.0, q=np.zeros(6), faults=flag)
  assert st.stale and not st.healthy


def test_control_state_non_staleness_faults_do_not_imply_stale():
  # An over-temperature servo is a health problem, not a measurement problem:
  # the position is still trustworthy and the command must not be clamped.
  st = ControlState(now=1.0, q=np.zeros(6), faults=FaultFlags.OVER_TEMP)
  assert not st.stale
  assert not st.healthy


def test_control_state_with_faults_accumulates():
  st = ControlState(now=1.0, q=np.zeros(6), faults=FaultFlags.OVER_TEMP)
  st2 = st.with_faults(FaultFlags.STALE_Q)
  assert st2.faults == (FaultFlags.OVER_TEMP | FaultFlags.STALE_Q)
  assert st.faults == FaultFlags.OVER_TEMP    # the original is untouched


# -- parse_read_budget -------------------------------------------------------


@pytest.mark.parametrize("raw", ["max", "MAX", " max ", None])
def test_read_budget_treats_max_and_null_as_unlimited(raw):
  # "read everything every tick" is the default posture: rationing is only
  # worth it once real hardware shows the reads do not fit.
  assert parse_read_budget({"budget_s": raw}) == math.inf


def test_read_budget_defaults_to_unlimited_when_unset():
  assert parse_read_budget({}) == math.inf
  assert parse_read_budget(None) == math.inf


def test_read_budget_reads_a_number():
  assert parse_read_budget({"budget_s": 0.008}) == 0.008
  assert parse_read_budget({"budget_s": "0.008"}) == 0.008


def test_read_budget_allows_zero():
  # A zero budget is meaningful: it starves every read, which is exactly what
  # the starvation path is there to handle.
  assert parse_read_budget({"budget_s": 0.0}) == 0.0


def test_read_budget_rejects_a_negative_budget():
  with pytest.raises(ValueError, match="non-negative"):
    parse_read_budget({"budget_s": -0.001})


def test_read_budget_rejects_nonsense():
  # A typo must fail loudly rather than coerce: silently becoming 0.0 would
  # starve every read for the whole run while the loop kept going on
  # ever-staler cached positions.
  with pytest.raises(ValueError, match="budget_s"):
    parse_read_budget({"budget_s": "fast"})
