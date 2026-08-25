"""Tests for the budgeted read path through :class:`ControlLoop`.

These run against fake backends -- no bus, no hardware -- and pin the two
properties the budget mechanism exists to provide: the loop hands the backend
the configured per-tick allowance (unlimited by default), and a measurement the
backend could not refresh constrains the command instead of silently becoming
open-loop.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from omegaconf import OmegaConf

from chuck_dreamer.runtime.backend import FakeBackend
from chuck_dreamer.runtime.control_loop import ControlLoop
from chuck_dreamer.policy import Action
from chuck_dreamer.runtime.control_mode import ControlMode
from chuck_dreamer.runtime.control_state import ControlState, FaultFlags
from chuck_dreamer.runtime.setpoint_channel import SetpointChannel
from chuck_dreamer.runtime.telemetry import TelemetryQueue

_N = 6
_LOWER = np.full(_N, -2.0)
_UPPER = np.full(_N, 2.0)


def _cfg(rate_hz=200.0, **read):
  return OmegaConf.create({
    "policy_rate_hz": 10,
    "control_loop": {
      "rate_hz": rate_hz,
      "safety": {"max_velocity": 1.5, "max_acceleration": 3.0,
                 "coast_to_stop_time": 0.1},
      "read": {"budget_s": "max", "stale_hold_s": 0.1, **read},
    },
  })


class _RecordingBackend(FakeBackend):
  """FakeBackend that records the budget each read was given."""

  def __init__(self, *a, **kw):
    super().__init__(*a, **kw)
    self.budgets: list[float] = []

  def read_state(self, budget_s=math.inf):
    self.budgets.append(budget_s)
    return super().read_state(budget_s)


class _Obs:
  def __init__(self, t=0.0):
    self.t = t


def _loop(backend, cfg=None):
  return ControlLoop(cfg or _cfg(), backend, SetpointChannel(), TelemetryQueue())


# -- budget plumbing ---------------------------------------------------------


def test_loop_defaults_to_an_unlimited_budget():
  # The shipped default reads everything every tick; rationing is opt-in until
  # hardware proves it necessary.
  b = _RecordingBackend(_N, lower=_LOWER, upper=_UPPER)
  b.start()
  loop = _loop(b)
  loop.start()
  time.sleep(0.15)
  loop.stop()

  assert loop.ticks > 5
  assert loop.budget_s == math.inf
  assert all(x == math.inf for x in b.budgets)


def test_loop_hands_the_backend_the_configured_budget_every_tick():
  b = _RecordingBackend(_N, lower=_LOWER, upper=_UPPER)
  b.start()
  loop = _loop(b, _cfg(budget_s=0.002))
  loop.start()
  time.sleep(0.15)
  loop.stop()

  assert loop.ticks > 5
  assert b.budgets[0] == math.inf          # the priming read has no deadline
  # Constant, not adaptive: every subsequent tick gets exactly what was
  # configured, so a log is enough to know what the backend was allowed to do.
  assert set(b.budgets[1:]) == {0.002}


def test_loop_runs_at_approximately_the_configured_rate():
  b = FakeBackend(_N, lower=_LOWER, upper=_UPPER)
  b.start()
  loop = _loop(b, _cfg(rate_hz=100.0))
  loop.start()
  time.sleep(0.3)
  loop.stop()
  assert 20 <= loop.ticks <= 40            # ~30 ticks, generous for CI jitter


# -- staleness reaches the safety limiter ------------------------------------


def _state(q, *, q_age=0.0, faults=FaultFlags.NONE):
  return ControlState(now=time.monotonic(), q=np.asarray(q, float),
                      q_age=q_age, faults=faults)


def _limiter_loop():
  return _loop(FakeBackend(_N, lower=_LOWER, upper=_UPPER))


def test_safety_limit_clamps_to_the_joint_envelope():
  loop = _limiter_loop()
  q_cmd, limits = loop._safety_limit(
    _state(np.zeros(_N)), np.full(_N, 9.0), np.zeros(_N), np.zeros(_N))
  np.testing.assert_array_equal(q_cmd, _UPPER)
  assert limits["clamped"] is True


def test_safety_limit_passes_a_fresh_in_range_reference_through():
  loop = _limiter_loop()
  ref = np.full(_N, 0.5)
  q_cmd, limits = loop._safety_limit(
    _state(np.zeros(_N)), ref, np.zeros(_N), np.zeros(_N))
  np.testing.assert_array_equal(q_cmd, ref)
  assert limits["clamped"] is False
  assert limits["stale"] is False


def test_safety_limit_bounds_the_command_when_the_measurement_is_stale():
  # This is the payoff of tracking q_age: a starved read must not let the
  # trajectory keep extending a motion against an old position.
  loop = _limiter_loop()
  st = _state(np.zeros(_N), q_age=0.010, faults=FaultFlags.READ_STARVED)
  q_cmd, limits = loop._safety_limit(
    st, np.full(_N, 1.0), np.zeros(_N), np.zeros(_N))

  reach = 1.5 * 0.010                  # v_max * age
  np.testing.assert_allclose(q_cmd, np.full(_N, reach))
  assert limits["stale"] is True
  assert limits["stale_reach"] == pytest.approx(reach)


def test_safety_limit_stale_envelope_is_capped_by_stale_hold():
  # Past the hold bound the envelope stops growing, so an indefinitely starved
  # loop converges to holding position rather than drifting.
  loop = _limiter_loop()
  st = _state(np.zeros(_N), q_age=10.0, faults=FaultFlags.STALE_Q)
  q_cmd, _ = loop._safety_limit(st, np.full(_N, 2.0), np.zeros(_N), np.zeros(_N))
  np.testing.assert_allclose(q_cmd, np.full(_N, 1.5 * 0.1))


def test_safety_limit_ignores_health_faults_that_are_not_about_staleness():
  # An over-temperature servo must not silently shrink the motion envelope --
  # that is a separate concern from whether the position is trustworthy.
  loop = _limiter_loop()
  st = _state(np.zeros(_N), faults=FaultFlags.OVER_TEMP)
  ref = np.full(_N, 1.0)
  q_cmd, limits = loop._safety_limit(st, ref, np.zeros(_N), np.zeros(_N))
  np.testing.assert_array_equal(q_cmd, ref)
  assert limits["stale"] is False
  assert limits["faults"] == int(FaultFlags.OVER_TEMP)


def test_loop_rejects_a_nonpositive_rate():
  b = FakeBackend(_N)
  with pytest.raises(ValueError, match="rate_hz"):
    _loop(b, _cfg(rate_hz=0.0))


# -- mode wrappers -----------------------------------------------------------
#
# The transition table itself is covered in test_runtime_control_mode.py; these
# pin what the *loop* adds -- that a request is actually carried out by the
# running control thread, and resolves only once it has been.


def _running(backend=None):
  loop = _loop(backend or FakeBackend(_N, lower=_LOWER, upper=_UPPER))
  loop.start()
  deadline = time.monotonic() + 1.0
  while loop.ticks == 0 and time.monotonic() < deadline:
    time.sleep(0.005)
  return loop


def test_loop_starts_in_normal():
  loop = _running()
  try:
    assert loop.mode is ControlMode.NORMAL
  finally:
    loop.stop()


def test_brake_resolves_once_the_arm_has_stopped():
  """request_brake returns when the coast-down has actually played out.

  A stationary arm's brake() degrades to an infinite hold whose segment never
  expires, so completion cannot be "the segment ran out" -- this pins that the
  no-op case still reports BRAKED rather than hanging until the timeout.
  """
  loop = _running()
  try:
    assert loop.request_brake(timeout=2.0)
    assert loop.mode is ControlMode.BRAKED
  finally:
    loop.stop()


def test_brake_of_a_moving_arm_completes_within_the_coast_down():
  """A *moving* arm must reach BRAKED in about `coast_to_stop_time`.

  Regression: BRAKING used to replan `trajectory.brake()` on every tick, so the
  coast-down segment restarted from s=0 against an ever-smaller velocity. The
  reference decayed geometrically instead of running to rest, and BRAKED landed
  an order of magnitude late (~1.2 s for a 0.1 s coast) -- which made the
  harness's 1 s shutdown brake time out on every real run. The idle-arm test
  above cannot catch this: a stationary arm is already at rest on entry.
  """
  backend = FakeBackend(_N, lower=_LOWER, upper=_UPPER)
  channel = SetpointChannel()
  loop = ControlLoop(_cfg(), backend, channel, TelemetryQueue())
  loop.start()
  try:
    deadline = time.monotonic() + 1.0
    while loop.ticks == 0 and time.monotonic() < deadline:
      time.sleep(0.005)
    # Get the arm genuinely moving before asking it to stop.
    channel.publish(Action(_Obs(t=0.0), q=np.full(_N, 1.5)))
    time.sleep(0.1)

    t0 = time.monotonic()
    assert loop.request_brake(timeout=2.0)
    elapsed = time.monotonic() - t0
    assert loop.mode is ControlMode.BRAKED
    # coast_to_stop_time is 0.1 s; allow generous slack for scheduling jitter
    # while still failing the ~1.2 s geometric-decay behaviour.
    assert elapsed < 0.6, f"coast-down took {elapsed:.3f}s"
  finally:
    loop.stop()


def test_braked_arm_ignores_new_setpoints():
  """A policy loop still draining must not re-accelerate an arm being stopped."""
  backend = FakeBackend(_N, lower=_LOWER, upper=_UPPER)
  channel = SetpointChannel()
  loop = ControlLoop(_cfg(), backend, channel, TelemetryQueue())
  loop.start()
  try:
    deadline = time.monotonic() + 1.0
    while loop.ticks == 0 and time.monotonic() < deadline:
      time.sleep(0.005)
    assert loop.request_brake(timeout=2.0)
    held = backend.last_positions()

    # a late setpoint, far from the hold
    channel.publish(Action(_Obs(t=1.0), q=np.full(_N, 1.5)))
    time.sleep(0.05)
    np.testing.assert_allclose(backend.last_positions(), held, atol=1e-9)
  finally:
    loop.stop()


def test_estop_then_release_returns_to_normal():
  loop = _running()
  try:
    assert loop.request_estop(timeout=2.0)
    assert loop.mode is ControlMode.ESTOP
    assert loop.release(timeout=2.0)
    assert loop.mode is ControlMode.NORMAL
  finally:
    loop.stop()


def test_release_after_brake_resumes_normal():
  loop = _running()
  try:
    assert loop.request_brake(timeout=2.0)
    assert loop.release(timeout=2.0)
    assert loop.mode is ControlMode.NORMAL
  finally:
    loop.stop()


def test_mode_request_on_a_stopped_loop_does_not_block():
  """No thread to carry the request out -- report the truth, don't hang."""
  loop = _loop(FakeBackend(_N, lower=_LOWER, upper=_UPPER))
  t0 = time.monotonic()
  assert not loop.request_brake(timeout=5.0)   # never started
  assert time.monotonic() - t0 < 1.0
