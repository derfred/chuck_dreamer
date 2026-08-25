"""Tests for :meth:`ControlTrajectory.stop` -- the hard-stop constructor.

``brake`` (the smooth coast-down) is exercised through the control loop; this
file pins the property that makes ``stop`` safe to call at any point in a
segment, which is easy to get wrong and expensive on real hardware.
"""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from chuck_dreamer.runtime.control_state import ControlState
from chuck_dreamer.runtime.control_trajectory import (
  ControlTrajectory,
  ControlTrajectoryConfig,
)

_N = 6


@pytest.fixture
def cfg():
  return ControlTrajectoryConfig.build(OmegaConf.create({
    "policy_rate_hz": 10,
    "control_loop": {"safety": {"max_velocity": 1.5, "max_acceleration": 3.0,
                                "coast_to_stop_time": 0.1}},
  }), n_joints=_N)


class _Action:
  def __init__(self, q, t):
    self.q = np.asarray(q, dtype=float)
    self.obs = type("Obs", (), {"t": t})()


def _moving(cfg, goal=1.0):
  """A trajectory part-way through a long segment toward ``goal``."""
  start = ControlTrajectory.hold(np.zeros(_N), cfg)
  return start.update(_Action(np.full(_N, goal), 0.0),
                      ControlState(now=0.0, q=np.zeros(_N)))


@pytest.mark.parametrize("ticks", [1, 5, 30])
def test_stop_is_continuous_with_the_live_reference(cfg, ticks):
  """Stopping must not step the command.

  Defaulting to ``q_end`` would command the entire remaining travel in one
  tick -- ~1 rad when stopping 20 ms into a 1.4 s segment. The hold must sit
  exactly on the reference the arm was already being sent to.
  """
  traj = _moving(cfg)
  for _ in range(ticks):
    traj.tick(0.02)
  q_ref, _, _ = traj.eval()

  np.testing.assert_allclose(traj.stop().q_end, q_ref, atol=1e-12)


def test_stop_does_not_hold_at_the_far_end_of_the_segment(cfg):
  """Guards the specific regression: q_end is not the default."""
  traj = _moving(cfg)
  traj.tick(0.02)                      # barely started
  assert np.max(np.abs(traj.stop().q_end - traj.q_end)) > 0.5


def test_stop_honours_an_explicit_pose(cfg):
  """ESTOP holds at the *measured* position, not the reference."""
  traj = _moving(cfg)
  traj.tick(0.1)
  q = np.full(_N, 0.5)
  np.testing.assert_allclose(traj.stop(q).q_end, q)


def test_stop_is_stationary_and_carries_the_config(cfg):
  traj = _moving(cfg)
  traj.tick(0.1)
  stopped = traj.stop()
  assert not np.any(stopped.qd_end)    # a hold commands no motion
  assert stopped.cfg is cfg            # caller need not pass it back in


def test_stop_of_a_hold_is_a_no_op(cfg):
  """brake() routes stationary trajectories here, so this must not drift."""
  held = ControlTrajectory.hold(np.full(_N, 0.3), cfg)
  np.testing.assert_allclose(held.stop().q_end, np.full(_N, 0.3))
  np.testing.assert_allclose(held.brake().q_end, np.full(_N, 0.3))


# -- brake / stationary ------------------------------------------------------


def test_brake_is_continuous_with_the_live_reference(cfg):
  """A mid-flight brake coasts from where the arm is, not from the endpoint.

  ``q_end``/``qd_end`` describe the segment's far end, which has not been
  reached yet -- anchoring there would command the whole remaining travel
  before slowing down.
  """
  traj = _moving(cfg)
  for _ in range(20):
    traj.tick(0.02)
  q_ref, _, _ = traj.eval()

  braking = traj.brake()
  np.testing.assert_allclose(braking.eval(0.0)[0], q_ref, atol=1e-12)


def test_brake_mid_segment_actually_coasts(cfg):
  """Regression: qd_end is zero by construction on a plain plan, so branching
  on it collapsed a mid-motion brake into a hold at the far endpoint."""
  traj = _moving(cfg)
  for _ in range(20):
    traj.tick(0.02)
  q_ref, _, _ = traj.eval()
  braking = traj.brake()

  assert not braking.stationary                        # still moving
  assert braking.T == pytest.approx(cfg.t_brake)       # a real coast-down
  # Rests a short distance past where it started -- not at the segment's end.
  travel = float(np.max(np.abs(braking.q_end - q_ref)))
  assert 0.0 < travel < 0.2
  assert np.max(np.abs(braking.q_end - traj.q_end)) > 0.5


def test_brake_comes_to_rest(cfg):
  traj = _moving(cfg)
  for _ in range(20):
    traj.tick(0.02)
  braking = traj.brake()
  while braking.tick(0.02) is not None:
    pass
  assert braking.stationary


def test_brake_of_a_resting_reference_holds_in_place(cfg):
  held = ControlTrajectory.hold(np.full(_N, 0.3), cfg)
  assert held.brake().stationary
  np.testing.assert_allclose(held.brake().q_end, np.full(_N, 0.3))


def test_stationary_asks_the_reference_not_the_endpoint(cfg):
  """A hold never expires (T = inf) yet is stationary; a moving segment is not."""
  held = ControlTrajectory.hold(np.zeros(_N), cfg)
  assert held.stationary and not held.expired

  traj = _moving(cfg)
  traj.tick(0.02)
  assert not traj.stationary
  # ...even though its terminal velocity is zero, which is what made qd_end
  # the wrong thing to ask.
  assert not np.any(traj.qd_end)


# -- stationary: numerical robustness ----------------------------------------


def test_stationary_is_a_tolerance_not_an_exact_zero(cfg):
  """A quintic decays asymptotically; exact-zero would call rest "moving".

  At s = 0.9999 the reference is ~2e-8 rad/s -- physically stopped, far below
  anything a servo acts on, but not 0.0.
  """
  traj = _moving(cfg)
  for _ in range(20):
    traj.tick(0.02)
  braking = traj.brake()

  v_tail = np.max(np.abs(braking.eval(0.9999)[1]))
  assert v_tail > 0.0                       # genuinely non-zero...
  assert v_tail <= np.max(cfg.v_eps)        # ...yet counts as at rest


def test_stationary_reports_rest_on_the_completing_tick(cfg):
  """No off-by-one-tick lag: the tick that finishes the coast-down is at rest.

  This used to hold only because eval() clamps s to 1.0, so the *next* tick
  re-evaluated to exact zeros -- an accident of the clamp, not a guarantee.
  """
  traj = _moving(cfg)
  for _ in range(20):
    traj.tick(0.02)
  braking = traj.brake()
  while braking.tick(0.02) is not None:
    pass
  assert braking.stationary


def test_stationary_still_rejects_real_motion(cfg):
  """The tolerance must not swallow motion the arm can actually perform."""
  traj = _moving(cfg)
  for _ in range(20):
    traj.tick(0.02)
  braking = traj.brake()
  assert not braking.stationary                      # at the start of the coast
  assert not braking.eval(0.5)[1].round(9).all() == 0  # and half-way through
  assert np.max(np.abs(braking.eval(0.5)[1])) > np.max(cfg.v_eps)


def test_v_eps_scales_with_the_velocity_limit(cfg):
  """The threshold is relative to what the joint can do, not an absolute guess."""
  np.testing.assert_allclose(cfg.v_eps, cfg.v_eps_frac * cfg.v_max)
  # Far below a servo's resolution, far above float round-off.
  assert 1e-12 < np.max(cfg.v_eps) < 1e-3
