"""Tests for :mod:`chuck_dreamer.runtime.kernel` — the pure safety core.

These are clock-free: every M1 safety criterion (tracking, velocity limit,
synchronized slew, boundary clamp, e-stop latch + refuse-resume, the mode
table) is asserted by calling ``kernel.tick`` directly with chosen ``dt``.
"""

from __future__ import annotations

import numpy as np
import pytest

from chuck_dreamer.runtime.kernel import ControlKernel, KernelLimits, Mode


def _kernel(
  n: int = 3,
  lo: float = -2.0,
  hi: float = 2.0,
  vmax: float = 1.0,
  q0: float = 0.0,
  mode: Mode = Mode.NORMAL,
) -> ControlKernel:
  limits = KernelLimits.build(np.full(n, lo), np.full(n, hi), vmax)
  return ControlKernel(limits, np.full(n, q0), mode=mode)


def _run_to_steady(kernel, target, dt=0.1, steps=500):
  out = None
  for _ in range(steps):
    out = kernel.tick(np.asarray(target, float), kernel.q_cmd, dt)
  return out


# -- tracking + slew ---------------------------------------------------------


def test_normal_tracks_reachable_target():
  k = _kernel(vmax=1.0)
  target = np.array([1.0, -0.5, 0.3])
  out = _run_to_steady(k, target, dt=0.1)
  np.testing.assert_allclose(out.q_cmd, target, atol=1e-9)
  assert out.mode is Mode.NORMAL
  assert not out.clamped


def test_velocity_limit_per_tick():
  vmax, dt = 1.0, 0.02
  k = _kernel(vmax=vmax)
  out = k.tick(np.array([10.0, 10.0, 10.0]), k.q_cmd, dt)
  step = np.abs(out.q_cmd)  # started at 0
  assert out.velocity_limited
  assert np.all(step <= vmax * dt + 1e-12)
  np.testing.assert_allclose(step, vmax * dt, atol=1e-12)


def test_synchronized_slew_shares_one_alpha():
  # Two joints with different deltas should move in proportion to their
  # deltas (single shared alpha) — a straight line in joint space.
  limits = KernelLimits.build(np.array([-5.0, -5.0]), np.array([5.0, 5.0]), 1.0)
  k = ControlKernel(limits, np.zeros(2), mode=Mode.NORMAL)
  target = np.array([2.0, 1.0])  # joint 0 is the limiting joint
  out = k.tick(target, k.q_cmd, dt=0.1)
  # Limiting joint 0 moves vmax*dt = 0.1; joint 1 moves half as far.
  np.testing.assert_allclose(out.q_cmd, [0.1, 0.05], atol=1e-12)
  assert out.q_cmd[0] / out.q_cmd[1] == pytest.approx(target[0] / target[1])


def test_larger_dt_takes_larger_step():
  small = _kernel(vmax=1.0).tick(np.full(3, 10.0), np.zeros(3), dt=0.01)
  large = _kernel(vmax=1.0).tick(np.full(3, 10.0), np.zeros(3), dt=0.05)
  assert np.all(np.abs(large.q_cmd) > np.abs(small.q_cmd))


# -- clamp -------------------------------------------------------------------


def test_clamp_at_boundary():
  k = _kernel(lo=-1.0, hi=1.0, vmax=1.0)
  out = _run_to_steady(k, np.full(3, 5.0), dt=0.1)
  np.testing.assert_allclose(out.q_cmd, np.full(3, 1.0), atol=1e-9)
  assert out.clamped


def test_clamp_flag_false_when_in_box():
  k = _kernel(lo=-1.0, hi=1.0, vmax=10.0)
  out = k.tick(np.full(3, 0.5), k.q_cmd, dt=1.0)
  assert not out.clamped


# -- mode machine ------------------------------------------------------------


def test_estop_latches_and_holds():
  k = _kernel(vmax=1.0)
  k.tick(np.full(3, 1.0), k.q_cmd, dt=0.1)  # move a bit
  held = k.q_cmd
  assert k.request_estop()
  assert k.mode is Mode.ESTOP
  for _ in range(20):
    out = k.tick(np.full(3, 5.0), k.q_cmd, dt=0.1)  # target ignored
  np.testing.assert_allclose(out.q_cmd, held, atol=1e-12)
  assert out.mode is Mode.ESTOP


def test_estop_refuses_resume_until_hold():
  k = _kernel()
  k.request_estop()
  assert k.request_normal() is False  # refused while latched
  assert k.mode is Mode.ESTOP
  assert k.request_hold() is True     # the only exit
  assert k.mode is Mode.HOLD
  assert k.request_normal() is True   # now allowed
  assert k.mode is Mode.NORMAL


def test_hold_ignores_target():
  k = _kernel(mode=Mode.HOLD)
  start = k.q_cmd
  for _ in range(10):
    out = k.tick(np.full(3, 2.0), k.q_cmd, dt=0.1)
  np.testing.assert_allclose(out.q_cmd, start, atol=1e-12)
  assert out.mode is Mode.HOLD


def test_normal_with_no_target_holds():
  k = _kernel(mode=Mode.NORMAL)
  start = k.q_cmd
  out = k.tick(None, k.q_cmd, dt=0.1)
  np.testing.assert_allclose(out.q_cmd, start, atol=1e-12)


@pytest.mark.parametrize(
  "start,request_fn,expected",
  [
    (Mode.HOLD, "request_normal", Mode.NORMAL),
    (Mode.HOLD, "request_hold", Mode.HOLD),
    (Mode.HOLD, "request_estop", Mode.ESTOP),
    (Mode.NORMAL, "request_normal", Mode.NORMAL),
    (Mode.NORMAL, "request_hold", Mode.HOLD),
    (Mode.NORMAL, "request_estop", Mode.ESTOP),
    (Mode.ESTOP, "request_normal", Mode.ESTOP),  # refused
    (Mode.ESTOP, "request_hold", Mode.HOLD),
    (Mode.ESTOP, "request_estop", Mode.ESTOP),
  ],
)
def test_mode_transition_table(start, request_fn, expected):
  k = _kernel(mode=start)
  getattr(k, request_fn)()
  assert k.mode is expected


# -- construction guards -----------------------------------------------------


def test_limits_reject_inverted_bounds():
  with pytest.raises(ValueError, match="upper"):
    KernelLimits.build(np.array([1.0]), np.array([0.0]), 1.0)


def test_limits_reject_nonpositive_velocity():
  with pytest.raises(ValueError, match="max_velocity"):
    KernelLimits.build(np.array([-1.0]), np.array([1.0]), 0.0)


def test_reset_clamps_into_box():
  k = _kernel(lo=-1.0, hi=1.0)
  k.reset(np.full(3, 9.0))
  np.testing.assert_allclose(k.q_cmd, np.full(3, 1.0))
