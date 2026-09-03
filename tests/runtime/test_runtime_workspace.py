"""Tests for the Cartesian workspace envelope (``runtime/workspace.py``).

Two halves, matching the feature's two promises: the *limiter* tapers a
command's Cartesian speed to zero at the box wall and never lets the tip
cross it, and the *control loop* latches an e-stop the moment the measured tip
is outside — one that an explicit ``release()`` only clears once the arm is
back inside.

Most of these run against a hand-written planar FK stub rather than the real
URDF: the property under test is the limiter's arithmetic, and a stub makes
the geometry legible (``ee = q[:3]``, identity Jacobian) instead of requiring
the reader to hold a five-link chain in their head. The tests that do exercise
the real Pinocchio model are marked, and skip when the URDF is absent.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from omegaconf import OmegaConf

from chuck_dreamer.common import FK_MODEL_PATH
from chuck_dreamer.policy import Action
from chuck_dreamer.runtime.backend import FakeBackend
from chuck_dreamer.runtime.control_loop import ControlLoop
from chuck_dreamer.runtime.control_mode import ControlMode
from chuck_dreamer.runtime.control_state import ControlState, FaultFlags
from chuck_dreamer.runtime.control_channel import ControlChannel
from chuck_dreamer.runtime.telemetry import TelemetryQueue
from chuck_dreamer.runtime.workspace import (
  WorkspaceBox,
  WorkspaceLimiter,
  build_workspace_limiter,
)

_N = 6


class _PlanarFK:
  """FK stub: the tip *is* the first three joints, so the Jacobian is I.

  Keeps the limiter's geometry readable — a box in "joint space" is the same
  box in "tip space", so an assertion about the tip is an assertion about the
  command.
  """

  def __call__(self, q, dq=None):
    q = np.asarray(q, dtype=np.float64)
    if dq is not None:
      q = q + np.asarray(dq, dtype=np.float64)
    return q[:3].copy()

  def jacobian(self, q, dq=None):
    J = np.zeros((3, 5))
    J[:3, :3] = np.eye(3)
    return J


def _q(x=0.0, y=0.0, z=0.0):
  """A full follower-ordered joint vector whose stub-FK tip is ``(x, y, z)``."""
  return np.array([x, y, z, 0.0, 0.0, 0.0])


def _limiter(**kw):
  box = kw.pop("box", WorkspaceBox(lower=np.zeros(3), upper=np.ones(3)))
  kw.setdefault("margin_m", 0.1)
  return WorkspaceLimiter(_PlanarFK(), box, **kw)


def _ws_cfg(workspace=None, **safety):
  """A runtime-shaped config with the given envelope under control_loop.safety."""
  if workspace is not None:
    safety["workspace"] = workspace
  return OmegaConf.create({"control_loop": {"safety": safety}})


# -- the box -----------------------------------------------------------------


def test_box_rejects_an_empty_axis():
  # A box with no interior would clamp every command to a plane; that is a
  # config error, not a very tight envelope.
  with pytest.raises(ValueError, match="empty on some axis"):
    WorkspaceBox(lower=np.zeros(3), upper=np.array([1.0, 0.0, 1.0]))


def test_a_box_is_spanned_by_its_corners_in_either_order():
  # Config states two opposite corners; which one is "lower" is not the
  # operator's problem.
  a = WorkspaceBox.from_corners([0.4, -0.2, 0.3], [0.1, 0.2, 0.0])
  np.testing.assert_allclose(a.lower, [0.1, -0.2, 0.0])
  np.testing.assert_allclose(a.upper, [0.4, 0.2, 0.3])

  b = WorkspaceBox.from_corners([0.1, 0.2, 0.0], [0.4, -0.2, 0.3])
  np.testing.assert_allclose(b.lower, a.lower)
  np.testing.assert_allclose(b.upper, a.upper)


def test_box_measures_how_far_outside_a_point_is():
  box = WorkspaceBox(lower=np.zeros(3), upper=np.ones(3))
  assert box.outside_by(np.array([0.5, 0.5, 0.5])) == 0.0
  assert box.outside_by(np.array([0.5, 0.5, 1.25])) == pytest.approx(0.25)
  # The reported breach is the worst axis, not their sum.
  assert box.outside_by(np.array([-0.1, 0.5, 1.3])) == pytest.approx(0.3)


# -- the velocity taper ------------------------------------------------------


def test_a_step_far_from_every_wall_passes_untouched():
  lim = _limiter()
  cmd = _q(0.5, 0.5, 0.55)
  out, info = lim.limit(_q(0.5, 0.5, 0.5), cmd)

  assert info.scale == 1.0
  assert not info.limited
  np.testing.assert_allclose(out, cmd)


def test_the_step_is_damped_inside_the_margin():
  # margin 0.1, tip 0.05 from the wall: the outward budget is d * (d/margin)
  # = 0.05 * 0.5 = 0.025, so a 0.05 step is halved.
  lim  = _limiter(margin_m=0.1)
  meas = _q(0.5, 0.5, 0.95)
  out, info = lim.limit(meas, _q(0.5, 0.5, 1.0))

  assert info.scale == pytest.approx(0.5)
  assert info.limited
  assert lim.ee_pos(out)[2] == pytest.approx(0.975)


def test_the_taper_reaches_zero_at_the_wall():
  # Exactly on the wall there is no outward budget left, so the command holds.
  lim  = _limiter(margin_m=0.1)
  meas = _q(0.5, 0.5, 1.0)
  out, info = lim.limit(meas, _q(0.5, 0.5, 1.5))

  assert info.scale == 0.0
  np.testing.assert_allclose(out, meas)


def test_retreat_from_a_wall_is_never_slowed():
  # The arm must always be able to come back from a boundary it is pressed
  # against, or a release could never recover.
  lim  = _limiter(margin_m=0.1)
  meas = _q(0.5, 0.5, 1.0)
  cmd  = _q(0.5, 0.5, 0.5)
  out, info = lim.limit(meas, cmd)

  assert info.scale == 1.0
  np.testing.assert_allclose(out, cmd)


def test_the_binding_axis_scales_the_whole_step():
  # One scalar for every joint: the path keeps its direction instead of being
  # bent sideways along the wall it is approaching.
  lim  = _limiter(margin_m=0.1)
  meas = _q(0.5, 0.5, 0.95)
  out, _ = lim.limit(meas, _q(0.6, 0.7, 1.0))     # z binds at 0.5

  step = lim.ee_pos(out) - lim.ee_pos(meas)
  np.testing.assert_allclose(step, np.array([0.05, 0.10, 0.025]))


def test_repeated_commands_at_a_wall_converge_without_crossing():
  lim  = _limiter(margin_m=0.1)
  q    = _q(0.5, 0.5, 0.5)
  for _ in range(500):
    q, _ = lim.limit(q, q + _q(z=0.05))           # keep pushing at the wall
    assert lim.box.contains(lim.ee_pos(q))
  # It should have made it deep into the margin, just never through the wall.
  assert lim.ee_pos(q)[2] > 0.99


def test_the_jaw_is_passed_through_unscaled_when_the_tip_is_free():
  # The jaw does not move the tip, so a gripper command far from any wall is
  # not something the envelope has an opinion about.
  lim  = _limiter()
  meas = _q(0.5, 0.5, 0.5)
  cmd  = meas.copy()
  cmd[5] = 1.0
  out, info = lim.limit(meas, cmd)

  assert info.scale == 1.0
  assert out[5] == 1.0


# -- breach ------------------------------------------------------------------


def test_a_measured_tip_outside_the_box_is_a_breach_and_holds():
  lim  = _limiter()
  meas = _q(0.5, 0.5, 1.4)
  out, info = lim.limit(meas, _q(0.5, 0.5, 1.5))

  assert info.breached
  assert info.breach_m == pytest.approx(0.4)
  assert info.scale == 0.0
  np.testing.assert_allclose(out, meas)           # hold, do not "fix" it


# -- against the real arm model ----------------------------------------------


@pytest.mark.skipif(not FK_MODEL_PATH.exists(), reason="SO-101 URDF not present")
def test_real_fk_never_lets_the_tip_leave_the_box():
  # The Jacobian is a first-order estimate, so a big enough step can predict an
  # in-box endpoint that the true kinematics put outside; the limiter verifies
  # against real FK for exactly this reason. Steps here are 5x a control tick's.
  from chuck_dreamer.common.fk import FK

  fk  = FK(FK_MODEL_PATH)
  rng = np.random.default_rng(0)
  p0  = fk(np.zeros(5))
  box = WorkspaceBox(lower=p0 - 0.08, upper=p0 + 0.08)
  lim = WorkspaceLimiter(fk, box, margin_m=0.03)

  q = np.zeros(6)
  assert box.contains(lim.ee_pos(q))
  for _ in range(300):
    q, _ = lim.limit(q, q + np.concatenate([rng.uniform(-0.15, 0.15, 5), [0.0]]))
    assert box.outside_by(lim.ee_pos(q)) == 0.0


# -- construction from config ------------------------------------------------


def test_no_limiter_without_a_box():
  # The envelope is opt-in, and stating a box is the opt-in -- there is no
  # separate enable flag to leave switched off by accident.
  assert build_workspace_limiter(OmegaConf.create({"control_loop": {}})) is None
  assert build_workspace_limiter(_ws_cfg()) is None


@pytest.mark.skipif(not FK_MODEL_PATH.exists(), reason="SO-101 URDF not present")
def test_a_box_is_read_from_its_two_corners():
  lim = build_workspace_limiter(_ws_cfg([[0.1, -0.2, 0.0], [0.4, 0.2, 0.3]]))

  assert lim is not None
  np.testing.assert_allclose(lim.box.lower, [0.1, -0.2, 0.0])
  np.testing.assert_allclose(lim.box.upper, [0.4, 0.2, 0.3])


def test_a_box_that_is_not_two_corners_is_a_config_error():
  with pytest.raises(ValueError, match="two opposite corners"):
    build_workspace_limiter(_ws_cfg([0.1, -0.2, 0.0]))
  with pytest.raises(ValueError, match="two opposite corners"):
    build_workspace_limiter(_ws_cfg([[0.1, -0.2, 0.0], [0.4, 0.2, 0.3], [0.0, 0.0, 0.0]]))


@pytest.mark.skipif(not FK_MODEL_PATH.exists(), reason="SO-101 URDF not present")
def test_the_margin_is_read_from_the_safety_block():
  cfg = _ws_cfg([[0.1, -0.2, 0.0], [0.4, 0.2, 0.3]], workspace_margin_m=0.02)
  lim = build_workspace_limiter(cfg)

  assert lim is not None
  assert lim.margin_m == pytest.approx(0.02)


# -- in the control loop -----------------------------------------------------


def _loop_cfg(rate_hz=200.0):
  return OmegaConf.create({
    "policy_rate_hz": 10,
    "control_loop": {
      "rate_hz": rate_hz,
      "safety": {"max_velocity": 1.5, "max_acceleration": 3.0,
                 "coast_to_stop_time": 0.1},
      "read": {"budget_s": "max", "stale_hold_s": 0.1},
    },
  })


class _Obs:
  def __init__(self, t=0.0):
    self.t = t


def _running_loop(q_init, limiter):
  """A started ControlLoop over a FakeBackend seeded at ``q_init``."""
  backend = FakeBackend(_N, lower=np.full(_N, -2.0), upper=np.full(_N, 2.0), q_init=q_init)
  backend.start()
  loop = ControlLoop(_loop_cfg(), backend, ControlChannel(), TelemetryQueue(), workspace=limiter)
  loop.start()
  return backend, loop


def _teleported_out(limiter, *, to=None):
  """A running loop whose arm has been moved outside the box behind its back.

  Boots inside the envelope (the loop refuses to start otherwise) and then
  writes the joints directly, which is the software stand-in for the arm being
  shoved, slipping, or coming back from a power cycle somewhere else.
  """
  backend, loop = _running_loop(_q(0.5, 0.5, 0.5), limiter)
  assert _wait_for(lambda: loop.ticks > 2)
  backend.write_positions(to if to is not None else _q(0.5, 0.5, 1.5))
  return backend, loop


def _wait_for(predicate, timeout=2.0):
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if predicate():
      return True
    time.sleep(0.005)
  return False


def test_the_loop_estops_when_the_measured_tip_is_outside_the_box():
  backend, loop = _teleported_out(_limiter())
  try:
    assert _wait_for(lambda: loop.mode is ControlMode.ESTOP)
  finally:
    loop.stop()
    backend.stop()


def test_a_breach_aborts_the_write_on_the_tick_it_is_detected():
  """A breached tick writes nothing at all, and says why.

  The breach is found in ``_safety_limit``, which runs *after* the mode was
  taken for this tick -- so the ESTOP it latches only governs the *next* one.
  Returning ``None`` is what stops the loop spending one further tick of
  planned motion on an arm already outside the box.

  Asserted here rather than through a running loop because the limiter hands
  back the measured pose on a breach, so against an echo backend that stray
  write is indistinguishable from no write at all. It only displaces the arm
  when an earlier stage has already altered the command -- a stale-measurement
  clamp, say -- which is exactly when it is least affordable.
  """
  backend, loop = _running_loop(_q(0.5, 0.5, 0.5), _limiter())
  loop.stop()
  try:
    st = ControlState(now=time.monotonic(), q=_q(0.5, 0.5, 1.4), q_age=0.0,
                      faults=FaultFlags.NONE)
    q_cmd, limits = loop._safety_limit(st, _q(0.5, 0.5, 1.5), np.zeros(_N), np.zeros(_N))

    assert q_cmd is None                      # nothing is written this tick
    assert limits["ws_breach_m"] == pytest.approx(0.4)
    assert loop.mode is ControlMode.ESTOP     # and the freeze is latched
  finally:
    backend.stop()


def test_the_breach_estop_is_latched_against_release_until_the_arm_is_back_inside():
  backend, loop = _teleported_out(_limiter())
  try:
    assert _wait_for(lambda: loop.mode is ControlMode.ESTOP)

    # Releasing with the tip still outside re-latches on the next tick: nothing
    # commands motion outside the box, so recovery is physical.
    loop.release(timeout=0.5)
    assert _wait_for(lambda: loop.mode is ControlMode.ESTOP)

    # Move the arm back inside "by hand", then release for real.
    backend.write_positions(_q(0.5, 0.5, 0.5))
    assert loop.release(timeout=1.0)
    time.sleep(0.05)
    assert loop.mode is ControlMode.NORMAL
  finally:
    loop.stop()
    backend.stop()


def test_the_loop_holds_the_tip_inside_the_box_while_a_policy_drives_it_out():
  # An unconstrained command straight through the +z wall; the envelope must
  # hold the arm inside without ever tripping the breach e-stop.
  box     = WorkspaceBox(lower=np.zeros(3), upper=np.ones(3))
  limiter = WorkspaceLimiter(_PlanarFK(), box, margin_m=0.1)
  backend, loop = _running_loop(_q(0.5, 0.5, 0.5), limiter)
  try:
    loop._channel.publish(Action(_Obs(0.0), q=_q(0.5, 0.5, 1.9)))
    # The taper is asymptotic, so give it enough ticks to press up to the wall.
    assert _wait_for(lambda: backend.last_positions()[2] > 0.9, timeout=3.0)

    assert loop.mode is ControlMode.NORMAL          # limited, not breached
    assert box.contains(limiter.ee_pos(backend.last_positions()))
  finally:
    loop.stop()
    backend.stop()


def test_a_box_that_excludes_the_home_pose_is_refused_at_construction():
  # Such a runtime would e-stop on its first tick with nothing able to release
  # it, so the mistyped bound is caught where the message can explain itself
  # rather than as a mysterious latched freeze at boot.
  backend = FakeBackend(_N, lower=np.full(_N, -2.0), upper=np.full(_N, 2.0),
                        q_init=_q(0.5, 0.5, 1.5))
  backend.start()
  try:
    with pytest.raises(ValueError, match="does not contain the arm's home pose"):
      ControlLoop(_loop_cfg(), backend, ControlChannel(), TelemetryQueue(),
                  workspace=_limiter())
  finally:
    backend.stop()


def test_the_loop_runs_unconstrained_when_no_envelope_is_configured():
  # The envelope is opt-in; without it the loop is exactly as it was.
  backend, loop = _running_loop(_q(0.5, 0.5, 0.5), None)
  try:
    loop._channel.publish(Action(_Obs(0.0), q=_q(0.5, 0.5, 1.9)))
    assert _wait_for(lambda: backend.last_positions()[2] > 1.5, timeout=3.0)
    assert loop.mode is ControlMode.NORMAL
  finally:
    loop.stop()
    backend.stop()
