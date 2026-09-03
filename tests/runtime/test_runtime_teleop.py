"""Tests for M2 teleop: leader-reader, mapping, ManualPolicy, and end-to-end.

No real hardware: the leader-action mapping is pure; the real
:class:`LerobotLeaderReader` thread is exercised with a monkeypatched lerobot
stub; everything else uses :class:`FakeLeaderReader`. The end-to-end tests
follow the harness conventions in ``test_runtime_harness.py`` — ``FakeBackend``,
bounded ``duration_s``, shutdown driven directly, telemetry read back from the
``fake_rerun`` capture, no leftover ``runtime-*`` threads.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from chuck_dreamer.config import load_config
from chuck_dreamer.policy import Action
from chuck_dreamer.runtime.harness import PolicyLoop, Runtime
from chuck_dreamer.runtime.control_mode import ControlMode
from chuck_dreamer.runtime.modalities import RuntimeObservation
from chuck_dreamer.runtime.sources import ManualPolicy

from .conftest import control_tick_rows
from .test_runtime_harness import merge_runtime
from chuck_dreamer.runtime.teleop import (
  FakeLeaderReader,
  LerobotLeaderReader,
  ScriptedLeaderReader,
  follower_qpos_to_action,
  leader_action_to_follower_qpos,
)

_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _action(angular_deg, gripper_pct) -> dict[str, float]:
  d = {f"{m}.pos": float(angular_deg[i]) for i, m in enumerate(_MOTORS[:5])}
  d["gripper.pos"] = float(gripper_pct)
  return d


# -- leader_action_to_follower_qpos (pure) -----------------------------------


def test_mapping_angular_deg_to_rad():
  action = _action([90.0, -45.0, 0.0, 180.0, -90.0], gripper_pct=0.0)
  q = leader_action_to_follower_qpos(action, jaw_lower=-0.174, jaw_upper=1.75)
  np.testing.assert_allclose(q[:5], np.deg2rad([90.0, -45.0, 0.0, 180.0, -90.0]))


def test_mapping_gripper_percent_to_jaw_range():
  lo, hi = -0.174, 1.75
  q0   = leader_action_to_follower_qpos(_action([0] * 5, 0.0), jaw_lower=lo, jaw_upper=hi)
  q50  = leader_action_to_follower_qpos(_action([0] * 5, 50.0), jaw_lower=lo, jaw_upper=hi)
  q100 = leader_action_to_follower_qpos(_action([0] * 5, 100.0), jaw_lower=lo, jaw_upper=hi)
  assert q0[5]   == pytest.approx(lo)
  assert q50[5]  == pytest.approx(lo + 0.5 * (hi - lo))
  assert q100[5] == pytest.approx(hi)


def test_mapping_gripper_percent_is_clamped():
  lo, hi = -0.174, 1.75
  q_neg = leader_action_to_follower_qpos(_action([0] * 5, -20.0), jaw_lower=lo, jaw_upper=hi)
  q_big = leader_action_to_follower_qpos(_action([0] * 5, 150.0), jaw_lower=lo, jaw_upper=hi)
  assert q_neg[5] == pytest.approx(lo)
  assert q_big[5] == pytest.approx(hi)


def test_mapping_raises_on_missing_motor():
  bad = _action([0] * 5, 0.0)
  del bad["wrist_roll.pos"]
  with pytest.raises(ValueError, match="wrist_roll.pos"):
    leader_action_to_follower_qpos(bad, jaw_lower=0.0, jaw_upper=1.0)


# -- follower_qpos_to_action (inverse, used by FeetechBackend) ----------------


def test_inverse_mapping_radians_to_deg_and_percent():
  lo, hi = -0.174, 1.75
  q = np.array([np.deg2rad(90.0), np.deg2rad(-45.0), 0.0, 0.0, 0.0, lo + 0.5 * (hi - lo)])
  action = follower_qpos_to_action(q, jaw_lower=lo, jaw_upper=hi)
  assert action["shoulder_pan.pos"] == pytest.approx(90.0)
  assert action["shoulder_lift.pos"] == pytest.approx(-45.0)
  assert action["gripper.pos"] == pytest.approx(50.0)        # jaw midpoint -> 50%


def test_inverse_mapping_round_trips_with_forward():
  lo, hi = -0.174, 1.75
  q = np.array([0.5, -1.0, 1.0, 0.3, -0.4, lo + 0.7 * (hi - lo)])
  back = leader_action_to_follower_qpos(
    follower_qpos_to_action(q, jaw_lower=lo, jaw_upper=hi), jaw_lower=lo, jaw_upper=hi)
  np.testing.assert_allclose(back, q, atol=1e-9)


def test_inverse_mapping_clamps_jaw_percentage():
  lo, hi = -0.174, 1.75
  below = follower_qpos_to_action(np.array([0, 0, 0, 0, 0, lo - 1.0]), jaw_lower=lo, jaw_upper=hi)
  above = follower_qpos_to_action(np.array([0, 0, 0, 0, 0, hi + 1.0]), jaw_lower=lo, jaw_upper=hi)
  assert below["gripper.pos"] == pytest.approx(0.0)
  assert above["gripper.pos"] == pytest.approx(100.0)


def test_inverse_mapping_rejects_wrong_shape():
  with pytest.raises(ValueError, match=r"\(6,\)"):
    follower_qpos_to_action(np.zeros(5), jaw_lower=0.0, jaw_upper=1.0)


def test_inverse_mapping_rejects_zero_jaw_span():
  with pytest.raises(ValueError, match="jaw_upper"):
    follower_qpos_to_action(np.zeros(6), jaw_lower=0.5, jaw_upper=0.5)


# -- FakeLeaderReader --------------------------------------------------------


def test_fake_leader_none_then_pose_and_is_immutable():
  r = FakeLeaderReader()
  assert r.latest() is None
  pose = np.arange(6, dtype=np.float64)
  r.set_pose(pose)
  got = r.latest()
  np.testing.assert_array_equal(got.q, pose)
  assert got.ok
  with pytest.raises(ValueError):     # the shared reading is not writeable
    got.q[0] = 999.0
  pose[0] = 999.0                     # nor does mutating the caller's array reach it
  np.testing.assert_array_equal(r.latest().q, np.arange(6, dtype=np.float64))


def test_fake_leader_reports_age_against_the_sampler_clock():
  r = FakeLeaderReader(pose=np.zeros(6))
  first = r.latest().age
  time.sleep(0.02)
  assert r.latest().age > first       # age grows between samples, storage unchanged


def test_fake_leader_can_report_a_failed_poll():
  r = FakeLeaderReader(pose=np.zeros(6))
  assert r.latest().ok
  r.set_pose(np.zeros(6), ok=False)
  assert not r.latest().ok


def test_fake_leader_rejects_wrong_shape():
  with pytest.raises(ValueError):
    FakeLeaderReader(pose=np.zeros(3))


# -- ScriptedLeaderReader ----------------------------------------------------

# `pose_at` is pure (elapsed seconds -> pose), so the script is asserted without
# threads or sleeps -- the same convention the scripted policies use for
# `target_at`. Only the clock-anchoring tests actually start the reader.

_WPS = [
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
  [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
]


def test_scripted_leader_interpolates_between_waypoints():
  r = ScriptedLeaderReader(_WPS, durations=2.0)
  np.testing.assert_allclose(r.pose_at(0.0), _WPS[0])
  np.testing.assert_allclose(r.pose_at(1.0), [0.5, 1.0, 0.0, 0.0, 0.0, 0.0])
  np.testing.assert_allclose(r.pose_at(2.0), _WPS[1])
  np.testing.assert_allclose(r.pose_at(3.0), [1.0, 2.0, 1.5, 0.0, 0.0, 0.0])
  np.testing.assert_allclose(r.pose_at(4.0), _WPS[2])
  assert r.total_duration_s == pytest.approx(4.0)


def test_scripted_leader_holds_the_last_waypoint_by_default():
  r = ScriptedLeaderReader(_WPS, durations=1.0)
  np.testing.assert_allclose(r.pose_at(2.0), _WPS[2])
  np.testing.assert_allclose(r.pose_at(1e6), _WPS[2])


def test_scripted_leader_loops_back_to_the_start():
  r = ScriptedLeaderReader(_WPS, durations=1.0, loop=True)
  # One full pass is 2s, so t=2 wraps to the script's start, not its end.
  np.testing.assert_allclose(r.pose_at(2.0), _WPS[0])
  np.testing.assert_allclose(r.pose_at(2.5), r.pose_at(0.5))


def test_scripted_leader_accepts_per_leg_durations():
  r = ScriptedLeaderReader(_WPS, durations=[1.0, 3.0])
  np.testing.assert_allclose(r.pose_at(1.0), _WPS[1])       # first leg is short
  np.testing.assert_allclose(r.pose_at(2.5), [1.0, 2.0, 1.5, 0.0, 0.0, 0.0])
  assert r.total_duration_s == pytest.approx(4.0)


def test_scripted_leader_clamps_negative_time_to_the_start():
  r = ScriptedLeaderReader(_WPS, durations=1.0)
  np.testing.assert_allclose(r.pose_at(-5.0), _WPS[0])


def test_scripted_leader_single_waypoint_is_a_constant_pose():
  r = ScriptedLeaderReader([[0.1] * 6])
  assert r.total_duration_s == 0.0
  np.testing.assert_allclose(r.pose_at(0.0), [0.1] * 6)
  np.testing.assert_allclose(r.pose_at(99.0), [0.1] * 6)


def test_scripted_leader_accepts_omegaconf_params():
  """Waypoints arrive from YAML as a ListConfig, not a sequence numpy takes."""
  cfg = OmegaConf.create({"waypoints": _WPS, "durations": [1.0, 3.0]})
  r   = ScriptedLeaderReader(cfg.waypoints, durations=cfg.durations)
  np.testing.assert_allclose(r.pose_at(1.0), _WPS[1])


def test_scripted_leader_is_none_before_start_then_tracks_the_clock():
  r = ScriptedLeaderReader(_WPS, durations=10.0)
  assert r.latest() is None            # no motion until the clock is anchored
  r.start()
  first = r.latest()
  np.testing.assert_allclose(first.q, _WPS[0], atol=1e-2)
  assert first.ok and first.age < 0.1  # synthesised fresh, so never stale
  with pytest.raises(ValueError):      # the shared reading is not writeable
    first.q[0] = 999.0


def test_scripted_leader_start_is_idempotent():
  r = ScriptedLeaderReader(_WPS, durations=10.0)
  r.start()
  time.sleep(0.05)
  before = r.latest().q.copy()
  r.start()                            # a second start must not rewind the script
  assert r.latest().q[0] >= before[0]


def test_scripted_leader_rejects_bad_scripts():
  with pytest.raises(ValueError):
    ScriptedLeaderReader([])                             # no waypoints
  with pytest.raises(ValueError):
    ScriptedLeaderReader(_WPS, durations=[1.0])          # wrong per-leg count
  with pytest.raises(ValueError):
    ScriptedLeaderReader(_WPS, durations=0.0)            # non-positive duration


def test_scripted_leader_from_config_requires_waypoints():
  with pytest.raises(ValueError, match="waypoints"):
    ScriptedLeaderReader.from_config(None)


def test_scripted_leader_from_config_ignores_stale_params(caplog):
  """A target swap can leak the fake leader's `pose` in; it is dropped loudly."""
  r = ScriptedLeaderReader.from_config(None, waypoints=_WPS, durations=1.0, pose=None)
  assert r.total_duration_s == pytest.approx(2.0)
  assert "pose" in caplog.text


# -- ManualPolicy ------------------------------------------------------------


def _obs(t=0.0, q_meas=None, leader=None, n=6) -> RuntimeObservation:
  return RuntimeObservation(
    t=t,
    q_meas=np.zeros(n) if q_meas is None else np.asarray(q_meas, dtype=np.float64),
    leader_qpos=None if leader is None else np.asarray(leader, dtype=np.float64),
  )


def test_manual_policy_passes_leader_through():
  p = ManualPolicy()
  p.reset(np.zeros(6))
  leader = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.6])
  np.testing.assert_array_equal(p.act(_obs(leader=leader)).q, leader)


def test_manual_policy_holds_last_leader_when_absent():
  p = ManualPolicy()
  p.reset(np.zeros(6))
  leader = np.full(6, 0.42)
  p.act(_obs(leader=leader))            # remembers it
  # Next step has no leader reading -> hold the last commanded pose.
  np.testing.assert_array_equal(p.act(_obs(leader=None)).q, leader)


def test_manual_policy_falls_back_to_qmeas_at_boot():
  p = ManualPolicy()
  q_meas = np.full(6, -0.3)
  # No reset, no leader yet: must still return something sane (measured pose).
  np.testing.assert_array_equal(p.act(_obs(q_meas=q_meas, leader=None)).q, q_meas)


# -- RuntimeObservation backward-compat --------------------------------------


def test_observation_two_arg_construct_has_no_leader():
  obs = RuntimeObservation(t=1.0, q_meas=np.zeros(6))
  assert obs.has_leader is False
  assert obs.leader_qpos is None


def test_observation_with_leader_present():
  obs = RuntimeObservation(t=1.0, q_meas=np.zeros(6), leader_qpos=np.ones(6))
  assert obs.has_leader is True


# -- LerobotLeaderReader (monkeypatched lerobot, no device) ------------------


class _StubLeader:
  """Stand-in for lerobot SO101Leader: programmable get_action, no serial."""

  def __init__(self, config):
    self.config = config
    self.connected = False
    self._action = _action([10.0, 20.0, 30.0, 40.0, 50.0], gripper_pct=100.0)
    self._raise = False
    self.calls = 0

  def connect(self, calibrate=True):
    self.connected = True

  def disconnect(self):
    self.connected = False

  def get_action(self):
    self.calls += 1
    if self._raise:
      raise RuntimeError("simulated serial hiccup")
    return dict(self._action)


@pytest.fixture
def stub_lerobot(monkeypatch):
  """Install a fake ``lerobot.teleoperators.so_leader`` module for the lazy import."""
  created: list[_StubLeader] = []

  def _factory(config):
    leader = _StubLeader(config)
    created.append(leader)
    return leader

  class _Config:
    def __init__(self, *, port, use_degrees=True, id=None):
      self.port = port
      self.use_degrees = use_degrees
      self.id = id

  mod = types.ModuleType("lerobot.teleoperators.so_leader")
  mod.SO101Leader = _factory                     # type: ignore[attr-defined]
  mod.SOLeaderTeleopConfig = _Config             # type: ignore[attr-defined]
  monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
  monkeypatch.setitem(sys.modules, "lerobot.teleoperators", types.ModuleType("lerobot.teleoperators"))
  monkeypatch.setitem(sys.modules, "lerobot.teleoperators.so_leader", mod)
  return created


def _await(predicate, timeout=2.0, interval=0.005):
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if predicate():
      return True
    time.sleep(interval)
  return False


def test_lerobot_reader_reads_on_demand_and_maps(stub_lerobot):
  r = LerobotLeaderReader(port="/dev/null", jaw_lower=-0.174, jaw_upper=1.75)
  assert r.latest() is None                   # nothing before connect
  r.start()
  try:
    state = r.latest()                        # reads the bus on this thread
    assert state is not None
    np.testing.assert_allclose(state.q[:5], np.deg2rad([10.0, 20.0, 30.0, 40.0, 50.0]))
    assert state.q[5] == pytest.approx(1.75)  # gripper 100% -> jaw upper
    assert state.ok
    assert state.age < 0.1                    # read inline: essentially fresh
  finally:
    r.stop()
  # No polling thread exists to leak, and the reader is closed for business.
  assert not any(t.name == "runtime-leader" for t in threading.enumerate())
  assert r.latest() is None


def test_lerobot_reader_reads_every_call(stub_lerobot):
  """No caching between calls: each latest() is a fresh bus transaction."""
  r = LerobotLeaderReader(port="/dev/null", jaw_lower=0.0, jaw_upper=1.0)
  r.start()
  try:
    before = stub_lerobot[0].calls
    r.latest()
    r.latest()
    assert stub_lerobot[0].calls == before + 2
  finally:
    r.stop()


def test_lerobot_reader_holds_last_value_over_a_failed_read(stub_lerobot):
  """A serial error must not raise: this runs on the policy thread."""
  r = LerobotLeaderReader(port="/dev/null", jaw_lower=0.0, jaw_upper=1.0)
  r.start()
  try:
    good = r.latest()
    assert good.ok
    stub_lerobot[0]._raise = True               # subsequent reads now throw
    held = r.latest()                           # returns rather than raising
    assert held is not None
    assert not held.ok                          # and says the read failed
    np.testing.assert_array_equal(held.q, good.q)   # last good value held
    # `now` is not refreshed over a failure, so age keeps growing: that is what
    # separates "one hiccup" from "the leader is gone".
    later = r.latest()
    assert later.age > held.age
  finally:
    r.stop()


def test_lerobot_reader_returns_none_if_it_never_succeeded(stub_lerobot):
  """A leader that fails from the first read has no value to hold."""
  r = LerobotLeaderReader(port="/dev/null", jaw_lower=0.0, jaw_upper=1.0)
  r.start()
  stub_lerobot[0]._raise = True
  try:
    assert r.latest() is None
  finally:
    r.stop()


# -- End-to-end Runtime with ManualPolicy + FakeLeaderReader -----------------


def _manual_policy() -> dict:
  """Runtime overrides selecting ManualPolicy.

  ``params: {}`` is not redundant: OmegaConf merges the runtime block key-wise,
  so the default config's SineSweep params would otherwise survive the target
  swap. The harness drops them with a warning either way; clearing them here
  keeps these tests off that path.
  """
  return {"policy": {"target": "chuck_dreamer.runtime.sources:ManualPolicy",
                     "params": {}}}


def _manual_cfg(tmp_path: Path, pose, **runtime_overrides):
  cfg = load_config()
  base = {
    "duration_s": 0.4,
    "control_loop": {"rate_hz": 200},
    "policy_rate_hz": 100,
    "sensors": [],   # FakeBackend has no MuJoCo scene for SimCameraSensor
    "logging": {"rerun": {"rrd_dir": str(tmp_path / "rrd")}},
    "viewer": {"enabled": False},
    **_manual_policy(),
    "leader": {
      "enabled": True,
      "target": "chuck_dreamer.runtime.teleop:FakeLeaderReader",
      "params": {"pose": list(pose)} if pose is not None else {"pose": None},
    },
  }
  cfg.runtime = merge_runtime(cfg.runtime, base, runtime_overrides)
  return cfg


def _runtime_threads():
  return [t for t in threading.enumerate() if t.name.startswith("runtime-")]


def test_manual_teleop_follower_tracks_leader(tmp_path, fake_rerun):
  pose = [0.5, -1.0, 1.0, 0.3, -0.4, 0.2]      # all within the default envelope
  # High caps so the slew reaches the (in-box) leader pose within the bounded
  # run; ManualPolicy publishes the leader pose verbatim as the target. The
  # planner reads control_loop.safety -- a bare `safety=` lands on a dead key,
  # leaving the run at the 1.5 rad/s default, far too slow to converge here.
  # Acceleration is the binding limit at this scale, so both are raised.
  rt = Runtime(_manual_cfg(
    tmp_path, pose, duration_s=0.6,
    control_loop={"rate_hz": 200,
                  "safety": {"max_velocity": 50.0, "max_acceleration": 200.0}}))
  home = rt.backend.home_qpos.copy()           # boot pose, before any setpoint
  rt.run()

  rows = control_tick_rows(fake_rerun.rec)
  assert rows
  # ManualPolicy publishes the leader pose verbatim, so the trajectory converges
  # on it: `target` is the planner's *reference*, which approaches the goal
  # asymptotically rather than jumping to it. It lands within ~1e-3 here, but
  # the run can end mid-segment on a slow tick, so the bound is deliberately
  # loose -- this asserts convergence, not a particular settling time.
  last = rows[-1]
  for i in range(6):
    assert last["target"][i] == pytest.approx(pose[i], abs=5e-2)
    assert last["q_cmd"][i] == pytest.approx(pose[i], abs=5e-2)
    # And it genuinely moved from home toward the leader (not coincidentally there).
    assert abs(pose[i] - last["q_cmd"][i]) < abs(pose[i] - home[i]) + 1e-9
  assert _runtime_threads() == []


def test_scripted_teleop_drives_a_preprogrammed_motion(tmp_path, fake_rerun):
  """End-to-end: a scripted leader moves the follower through the whole runtime.

  The point of routing a script through the *leader* rather than a scripted
  policy is that this exercises the real teleop path -- leader construction,
  the per-step `latest()` poll, the leader_qpos modality, ManualPolicy -- so
  the only thing swapped out relative to a human is the pose source.
  """
  start = [0.0, -0.2, 0.2, 0.0, 0.0, 0.2]
  end   = [0.6, -0.2, 0.2, 0.0, 0.0, 0.2]
  # Acceleration is the binding limit at this scale, so both caps are raised to
  # let the follower track the ramp closely within a short bounded run.
  cfg = _manual_cfg(
    tmp_path, None, duration_s=1.2,
    control_loop={"rate_hz": 200,
                  "safety": {"max_velocity": 20.0, "max_acceleration": 200.0}})
  cfg.runtime.leader = OmegaConf.create({
    "enabled": True,
    "target": "chuck_dreamer.runtime.teleop:ScriptedLeaderReader",
    "params": {"waypoints": [start, end], "durations": 0.3},
  })
  rt = Runtime(cfg)
  rt.run()

  rows = control_tick_rows(fake_rerun.rec)
  assert rows
  # The script ramps joint 0 from 0.0 to 0.6 and then holds: the follower must
  # both END near the final waypoint and have PASSED THROUGH the middle, which
  # a constant-pose leader could not produce.
  assert rows[-1]["q_cmd"][0] == pytest.approx(end[0], abs=5e-2)
  mid = [r["q_cmd"][0] for r in rows if 0.15 < r["q_cmd"][0] < 0.45]
  assert mid, "follower jumped instead of tracking the ramp"
  # Advances along the script rather than wandering. Asserted as a shape, not
  # tick-to-tick monotonicity: the planner is second-order, so it overshoots the
  # final waypoint by ~1e-2 rad and settles back onto it -- a real reversal that
  # a strict monotone check would flag. Start low, pass through, end on target
  # is what actually distinguishes tracking from any other trajectory.
  seq = [float(r["q_cmd"][0]) for r in rows]
  assert seq[0] < 0.15                       # began at the first waypoint
  assert max(seq) <= end[0] + 0.05           # never ran past the script
  assert all(v > 0.45 for v in seq[-3:])     # and stayed on the held final one
  # Never reverses meaningfully: no tick undoes more than a fraction of the ramp.

  assert _runtime_threads() == []


def test_leader_health_reaches_the_policy_observation(tmp_path, fake_rerun):
  """The leader arrives as a LeaderState, so `age`/`ok` are visible to a policy.

  The leader is pulled straight off its own reader rather than routed through
  the control channel (separate, uncontended bus), so it is sampled on a
  different clock than `obs.control` -- `age` is what lets a consumer tell how
  far apart the two are.
  """
  seen: list = []

  class _Recorder:
    def reset(self, start):
      pass

    def act(self, obs):
      seen.append(obs)
      return Action(obs, q=np.asarray(obs.q_meas, dtype=np.float64))

  pose = [0.1, -0.2, 0.3, 0.0, 0.0, 0.1]
  rt   = Runtime(_manual_cfg(tmp_path, pose))
  rt.policy = _Recorder()
  rt.policy_loop = PolicyLoop(
    rt.policy, rt.channel, rt.backend,
    rate_hz=float(rt.cfg.runtime.policy_rate_hz), leader=rt.leader,
    sensors=rt.sensors, perception=rt.perception, rerun_sink=rt.sink)
  rt.run()

  assert seen, "policy never ran"
  obs = seen[-1]
  assert obs.leader is not None
  np.testing.assert_allclose(obs.leader.q, pose)
  np.testing.assert_allclose(obs.leader_qpos, pose)   # the flat view agrees
  assert obs.leader.ok
  assert obs.leader.age >= 0.0
  assert obs.has_leader


def test_manual_teleop_out_of_box_is_clamped(tmp_path, fake_rerun):
  pose = [50.0] * 6                             # far outside the joint box
  rt = Runtime(_manual_cfg(
    tmp_path, pose,
    control_loop={"rate_hz": 200, "safety": {"max_velocity": 1000.0}}))
  lower, upper = rt.backend.joint_limits()
  rt.run()

  rows = control_tick_rows(fake_rerun.rec)
  assert rows
  for r in rows:
    q = r["q_cmd"]
    assert np.all(q >= lower - 1e-9) and np.all(q <= upper + 1e-9)
  assert any(r["clamped"] == 1 for r in rows)


def test_estop_freezes_teleop(tmp_path, fake_rerun):
  # A moving fake leader, but e-stop latched mid-run -> q_cmd must stop following.
  pose = [0.6, -1.0, 1.0, 0.3, -0.4, 0.2]
  rt = Runtime(_manual_cfg(tmp_path, pose, duration_s=None))
  failure: list[BaseException] = []

  def drive():
    # Let teleop track for a bit, latch e-stop, then move the leader and stop.
    try:
      time.sleep(0.15)
      assert rt.control_loop.request_estop(), "loop did not acknowledge the e-stop"
      assert rt.control_loop.mode is ControlMode.ESTOP
      rt.leader.set_pose(np.full(6, 0.0))       # leader jumps; the loop must ignore it
      time.sleep(0.15)
    except BaseException as exc:                # noqa: BLE001 - reported below
      failure.append(exc)
    finally:
      # Always release the main thread: duration_s is None, so without this the
      # run would block forever on a failed assertion.
      rt.request_shutdown()

  killer = threading.Thread(target=drive)
  killer.start()
  rt.run()
  killer.join()
  if failure:
    raise failure[0]

  # Everything commanded after the e-stop latched must be the frozen pose: the
  # loop holds where the arm was, regardless of where the leader went.
  rows = [r for r in control_tick_rows(fake_rerun.rec) if not r["is_event"]]
  estopped = [r for r in rows if r["mode"] == ControlMode.ESTOP.value]
  assert estopped, "no ticks were recorded under ESTOP"
  frozen = estopped[0]["q_cmd"]
  for r in estopped:
    np.testing.assert_allclose(r["q_cmd"], frozen, atol=1e-9)
  assert _runtime_threads() == []


# -- _build_leader behavior --------------------------------------------------


def _build_only_cfg(tmp_path, **runtime_overrides):
  cfg = load_config()
  base = {"duration_s": 0.1, "sensors": [],
          "logging": {"rerun": {"rrd_dir": str(tmp_path / "rrd")}}}
  cfg.runtime = OmegaConf.merge(cfg.runtime, OmegaConf.create(base),
                                OmegaConf.create(runtime_overrides))
  return cfg


def test_build_leader_none_when_disabled(tmp_path):
  # Default config has the leader block present but disabled -> no reader.
  rt = Runtime(_build_only_cfg(tmp_path, **_manual_policy()))
  assert rt.leader is None


def test_build_leader_decoupled_from_policy(tmp_path):
  # An enabled leader builds the reader even with a *scripted* policy: the
  # leader is a sensor, not tied to ManualPolicy.
  rt = Runtime(_build_only_cfg(
    tmp_path,
    policy={"target": "chuck_dreamer.runtime.sources:SineSweep"},
    leader={"enabled": True, "target": "chuck_dreamer.runtime.teleop:FakeLeaderReader",
            "params": {"pose": None}},
  ))
  assert rt.leader is not None


def test_build_leader_survives_params_leaked_from_target_swap(tmp_path):
  # Regression (B2): `--leader-port` swaps runtime.leader.target to the real
  # reader, but OmegaConf's key-wise merge leaves the fake's `pose: null` in
  # params. from_config must absorb the leaked key (warn, not crash), and the
  # jaw bounds must be injected on the from_config path too.
  rt = Runtime(_build_only_cfg(
    tmp_path,
    **_manual_policy(),
    leader={"enabled": True,
            "target": "chuck_dreamer.runtime.teleop:LerobotLeaderReader",
            "params": {"pose": None, "port": "/dev/ttyFAKE"}},
  ))
  assert isinstance(rt.leader, LerobotLeaderReader)
  assert rt.leader._port == "/dev/ttyFAKE"
  # Jaw bounds come from the resolved safety envelope, not config params.
  jaw_lower, jaw_upper = rt.backend.joint_limits()
  assert rt.leader._jaw_lower == pytest.approx(float(jaw_lower[-1]))
  assert rt.leader._jaw_upper == pytest.approx(float(jaw_upper[-1]))


def test_build_leader_real_reader_requires_port(tmp_path):
  # Without a port the real reader must fail loudly at build time, not at
  # start() on the bench.
  with pytest.raises(ValueError, match="port"):
    Runtime(_build_only_cfg(
      tmp_path,
      **_manual_policy(),
      leader={"enabled": True,
              "target": "chuck_dreamer.runtime.teleop:LerobotLeaderReader",
              "params": {}},
    ))


def test_manual_policy_without_leader_holds(tmp_path):
  # ManualPolicy no longer requires a leader; it just holds.
  rt = Runtime(_build_only_cfg(tmp_path, **_manual_policy()))
  assert rt.leader is None
  assert isinstance(rt.policy, ManualPolicy)


# -- Lifecycle ordering ------------------------------------------------------


class _SpyLeader(FakeLeaderReader):
  """Records the order of start/stop relative to a shared event log."""

  def __init__(self, log):
    super().__init__(pose=np.zeros(6))
    self._log = log

  def start(self):
    self._log.append("leader.start")

  def stop(self):
    self._log.append("leader.stop")


def test_leader_lifecycle_ordering(tmp_path):
  log: list[str] = []
  rt = Runtime(_manual_cfg(tmp_path, [0.0] * 6, duration_s=0.2))
  spy = _SpyLeader(log)
  rt.leader = spy
  rt.policy_loop._leader = spy
  # Wrap policy_loop start/stop to observe relative ordering.
  orig_start, orig_stop = rt.policy_loop.start, rt.policy_loop.stop
  rt.policy_loop.start = lambda: (log.append("policy.start"), orig_start())[1]   # type: ignore[assignment]
  rt.policy_loop.stop = lambda *a, **k: (log.append("policy.stop"), orig_stop(*a, **k))[1]  # type: ignore[assignment]
  rt.run()

  assert log.index("leader.start") < log.index("policy.start")
  assert log.index("policy.stop") < log.index("leader.stop")
