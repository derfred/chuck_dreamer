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


def test_only_normal_consumes_setpoints():
  """``stopping`` is about the stop having been *requested*, not motion ending.

  BRAKING is the case that matters: the arm is still coasting, so it is not
  "stopped" in any physical sense, but a producer that has not yet noticed the
  request must not be able to feed it a new setpoint and drive it on.
  """
  assert not ControlMode.NORMAL.stopping
  assert ControlMode.BRAKING.stopping           # still moving, still stopping
  assert ControlMode.BRAKED.stopping
  assert ControlMode.ESTOP.stopping


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


# -- take(): mode plus the "just entered" edge -------------------------------


def test_take_reports_the_edge_once_per_transition():
  """The control loop plans on the edge and executes on every tick after it."""
  m = ControlModeMachine()
  # A fresh machine has not transitioned into anything yet.
  assert m.take() == (ControlMode.NORMAL, False)

  m.request_brake()
  assert m.take() == (ControlMode.BRAKING, True)    # the edge, reported once
  assert m.take() == (ControlMode.BRAKING, False)   # ...and consumed by the read
  assert m.take() == (ControlMode.BRAKING, False)

  m.braked()
  assert m.take() == (ControlMode.BRAKED, True)
  assert m.take() == (ControlMode.BRAKED, False)


def test_take_does_not_flag_an_idempotent_request():
  """A request that changes nothing is not an edge -- re-braking must not replan."""
  m = ControlModeMachine()
  m.request_brake()
  assert m.take() == (ControlMode.BRAKING, True)
  m.request_brake()                                  # already braking: a no-op
  assert m.take() == (ControlMode.BRAKING, False)


def test_take_edge_survives_until_read():
  """Transitions between reads still surface: the flag latches, it does not decay."""
  m = ControlModeMachine()
  m.request_estop()
  m.release()                                        # two transitions, no read
  assert m.take() == (ControlMode.NORMAL, True)
  assert m.take() == (ControlMode.NORMAL, False)


def test_mode_property_does_not_consume_the_edge():
  """Observers reading `.mode` must not steal the control thread's edge."""
  m = ControlModeMachine()
  m.request_brake()
  assert m.mode is ControlMode.BRAKING              # an outside observer peeks
  assert m.take() == (ControlMode.BRAKING, True)    # the loop still sees the edge
