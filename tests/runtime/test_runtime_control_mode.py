"""Tests for :mod:`chuck_dreamer.runtime.control_mode`.

The machine is deliberately free of trajectory / backend / clock, so the whole
transition table is asserted here without threads. The one threaded case covers
``wait_for``, which is the loop's owner observing that a request landed.
"""

from __future__ import annotations

import threading
import time

import pytest

from chuck_dreamer.runtime.control_mode import (
  ControlMode,
  ControlModeMachine,
  InvalidTransition,
)


def test_starts_normal():
  assert ControlModeMachine().mode is ControlMode.NORMAL


# -- the graceful path -------------------------------------------------------


def test_brake_then_braked_then_release():
  m = ControlModeMachine()
  assert m.request_brake() is ControlMode.BRAKING
  assert m.braked() is ControlMode.BRAKED
  assert m.release() is ControlMode.NORMAL


def test_brake_is_idempotent_while_stopping():
  m = ControlModeMachine()
  m.request_brake()
  assert m.request_brake() is ControlMode.BRAKING   # no re-entry of the coast-down
  m.braked()
  assert m.request_brake() is ControlMode.BRAKED    # already stopped: nothing to do


def test_release_is_refused_mid_coast_down():
  """Resuming mid-brake would hand a partly-stopped trajectory to the policy."""
  m = ControlModeMachine()
  m.request_brake()
  with pytest.raises(InvalidTransition, match="while braking"):
    m.release()
  assert m.mode is ControlMode.BRAKING              # and the mode is unchanged


def test_braked_outside_braking_is_ignored():
  """A late completion must not pull an e-stopped arm back to merely-braked."""
  m = ControlModeMachine()
  m.request_estop()
  assert m.braked() is ControlMode.ESTOP
  assert ControlModeMachine().braked() is ControlMode.NORMAL


# -- the latched freeze ------------------------------------------------------


def test_estop_from_every_mode():
  """Escalating to a hard stop must never depend on what the loop was doing."""
  for setup in (
    lambda m: None,
    lambda m: m.request_brake(),
    lambda m: (m.request_brake(), m.braked()),
  ):
    m = ControlModeMachine()
    setup(m)
    assert m.request_estop() is ControlMode.ESTOP


def test_estop_outranks_a_graceful_stop():
  """Downgrading to BRAKING would command motion on an e-stopped arm."""
  m = ControlModeMachine()
  m.request_estop()
  assert m.request_brake() is ControlMode.ESTOP


def test_estop_release_returns_to_normal():
  m = ControlModeMachine()
  m.request_estop()
  assert m.release() is ControlMode.NORMAL


def test_release_when_already_normal_is_a_no_op():
  assert ControlModeMachine().release() is ControlMode.NORMAL


def test_illegal_direct_transition_raises():
  m = ControlModeMachine()
  with pytest.raises(InvalidTransition):
    m._to(ControlMode.BRAKED)      # NORMAL cannot jump the coast-down


# -- wait_for ----------------------------------------------------------------


def test_wait_for_returns_immediately_when_already_there():
  m = ControlModeMachine()
  assert m.wait_for(ControlMode.NORMAL, timeout=0.0)


def test_wait_for_times_out_and_reports_it():
  m = ControlModeMachine()
  t0 = time.monotonic()
  assert not m.wait_for(ControlMode.BRAKED, timeout=0.05)
  assert time.monotonic() - t0 >= 0.05


def test_wait_for_wakes_on_the_transition():
  m = ControlModeMachine()
  m.request_brake()

  def finish():
    time.sleep(0.02)
    m.braked()

  threading.Thread(target=finish, daemon=True).start()
  assert m.wait_for(ControlMode.BRAKED, timeout=1.0)
  assert m.mode is ControlMode.BRAKED
