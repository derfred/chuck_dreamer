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
from chuck_dreamer.runtime.harness import Runtime
from chuck_dreamer.runtime.kernel import Mode
from chuck_dreamer.runtime.modalities import RuntimeObservation
from chuck_dreamer.runtime.sources import ManualPolicy

from .conftest import control_tick_rows
from chuck_dreamer.runtime.teleop import (
  FakeLeaderReader,
  LerobotLeaderReader,
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


def test_fake_leader_none_then_pose_and_copy():
  r = FakeLeaderReader()
  assert r.latest() is None
  pose = np.arange(6, dtype=np.float64)
  r.set_pose(pose)
  got = r.latest()
  np.testing.assert_array_equal(got, pose)
  got[0] = 999.0                      # mutating the return must not change storage
  np.testing.assert_array_equal(r.latest(), pose)


def test_fake_leader_rejects_wrong_shape():
  with pytest.raises(ValueError):
    FakeLeaderReader(pose=np.zeros(3))


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
  np.testing.assert_array_equal(p.act(_obs(leader=leader)), leader)


def test_manual_policy_holds_last_leader_when_absent():
  p = ManualPolicy()
  p.reset(np.zeros(6))
  leader = np.full(6, 0.42)
  p.act(_obs(leader=leader))            # remembers it
  # Next step has no leader reading -> hold the last commanded pose.
  np.testing.assert_array_equal(p.act(_obs(leader=None)), leader)


def test_manual_policy_falls_back_to_qmeas_at_boot():
  p = ManualPolicy()
  q_meas = np.full(6, -0.3)
  # No reset, no leader yet: must still return something sane (measured pose).
  np.testing.assert_array_equal(p.act(_obs(q_meas=q_meas, leader=None)), q_meas)


# -- RuntimeObservation backward-compat --------------------------------------


def test_observation_two_arg_construct_has_no_leader():
  obs = RuntimeObservation(t=1.0, q_meas=np.zeros(6))
  assert obs.has_leader is False
  assert obs.leader_qpos is None


def test_observation_with_leader_present():
  obs = RuntimeObservation(t=1.0, q_meas=np.zeros(6), leader_qpos=np.ones(6))
  assert obs.has_leader is True


# -- LerobotLeaderReader thread (monkeypatched lerobot, no device) -----------


class _StubLeader:
  """Stand-in for lerobot SO101Leader: programmable get_action, no serial."""

  def __init__(self, config):
    self.config = config
    self.connected = False
    self._action = _action([10.0, 20.0, 30.0, 40.0, 50.0], gripper_pct=100.0)
    self._raise = False

  def connect(self, calibrate=True):
    self.connected = True

  def disconnect(self):
    self.connected = False

  def get_action(self):
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


def test_lerobot_reader_populates_latest_and_maps(stub_lerobot):
  r = LerobotLeaderReader(port="/dev/null", jaw_lower=-0.174, jaw_upper=1.75, poll_rate_hz=200)
  r.start()
  try:
    assert _await(lambda: r.latest() is not None), "reader never populated latest()"
    q = r.latest()
    np.testing.assert_allclose(q[:5], np.deg2rad([10.0, 20.0, 30.0, 40.0, 50.0]))
    assert q[5] == pytest.approx(1.75)        # gripper 100% -> jaw upper
  finally:
    r.stop()
  assert not any(t.name == "runtime-leader" for t in threading.enumerate())


def test_lerobot_reader_survives_poll_exception(stub_lerobot):
  r = LerobotLeaderReader(port="/dev/null", jaw_lower=0.0, jaw_upper=1.0, poll_rate_hz=200)
  r.start()
  try:
    assert _await(lambda: r.latest() is not None)
    good = r.latest()
    stub_lerobot[0]._raise = True               # subsequent polls now throw
    time.sleep(0.1)                             # thread keeps spinning, holds last value
    assert r.latest() is not None               # not killed
    np.testing.assert_array_equal(r.latest(), good)
    assert any(t.name == "runtime-leader" for t in threading.enumerate())
  finally:
    r.stop()


# -- End-to-end Runtime with manual source + FakeLeaderReader ----------------


def _manual_cfg(tmp_path: Path, pose, **runtime_overrides):
  cfg = load_config()
  base = {
    "duration_s": 0.4,
    "control_rate_hz": 200,
    "policy_rate_hz": 100,
    "sensors": [],   # FakeBackend has no MuJoCo scene for SimCameraSensor
    "logging": {"rerun": {"rrd_dir": str(tmp_path / "rrd")}},
    "viewer": {"enabled": False},
    "source": {"kind": "manual"},
    "leader": {
      "enabled": True,
      "target": "chuck_dreamer.runtime.teleop:FakeLeaderReader",
      "params": {"pose": list(pose)} if pose is not None else {"pose": None},
    },
  }
  cfg.runtime = OmegaConf.merge(cfg.runtime, OmegaConf.create(base),
                                OmegaConf.create(runtime_overrides))
  return cfg


def _runtime_threads():
  return [t for t in threading.enumerate() if t.name.startswith("runtime-")]


def test_manual_teleop_follower_tracks_leader(tmp_path, fake_rerun):
  pose = [0.5, -1.0, 1.0, 0.3, -0.4, 0.2]      # all within the default envelope
  # High velocity cap so the slew reaches the (in-box) leader pose within the
  # bounded run; ManualPolicy publishes the leader pose verbatim as the target.
  rt = Runtime(_manual_cfg(tmp_path, pose, duration_s=0.6, safety={"max_velocity": 50.0}))
  home = rt.kernel.q_cmd.copy()                # boot pose, before any setpoint
  rt.run()

  rows = control_tick_rows(fake_rerun.rec)
  assert rows
  # The published target IS the leader pose (pass-through), and q_cmd slews to it.
  last = rows[-1]
  for i in range(6):
    assert last["target"][i] == pytest.approx(pose[i], abs=1e-9)
    assert last["q_cmd"][i] == pytest.approx(pose[i], abs=1e-2)
    # And it genuinely moved from home toward the leader (not coincidentally there).
    assert abs(pose[i] - last["q_cmd"][i]) < abs(pose[i] - home[i]) + 1e-9
  assert _runtime_threads() == []


def test_manual_teleop_out_of_box_is_clamped(tmp_path, fake_rerun):
  pose = [50.0] * 6                             # far outside the joint box
  rt = Runtime(_manual_cfg(tmp_path, pose, safety={"max_velocity": 1000.0}))
  lower, upper = rt._limits.lower, rt._limits.upper
  rt.run()

  rows = control_tick_rows(fake_rerun.rec)
  assert rows
  for r in rows:
    q = r["q_cmd"]
    assert np.all(q >= lower - 1e-9) and np.all(q <= upper + 1e-9)
  assert any(r["clamped"] == 1 for r in rows)


def test_estop_freezes_teleop(tmp_path):
  # A moving fake leader, but e-stop latched mid-run -> q_cmd must stop following.
  pose = [0.6, -1.0, 1.0, 0.3, -0.4, 0.2]
  rt = Runtime(_manual_cfg(tmp_path, pose, duration_s=None))

  def drive():
    # Let teleop track for a bit, latch e-stop, then move the leader and stop.
    time.sleep(0.15)
    rt.kernel.request_estop()
    frozen = rt.kernel.q_cmd.copy()
    rt.leader.set_pose(np.full(6, 0.0))         # leader jumps; kernel must ignore it
    time.sleep(0.15)
    after = rt.kernel.q_cmd.copy()
    assert rt.kernel.mode is Mode.ESTOP
    np.testing.assert_allclose(after, frozen, atol=1e-9)
    rt.request_shutdown()

  killer = threading.Thread(target=drive)
  killer.start()
  rt.run()
  killer.join()
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
  rt = Runtime(_build_only_cfg(tmp_path, source={"kind": "manual"}))
  assert rt.leader is None


def test_build_leader_decoupled_from_policy(tmp_path):
  # An enabled leader builds the reader even with a *scripted* source: the
  # leader is a sensor, not tied to ManualPolicy.
  rt = Runtime(_build_only_cfg(
    tmp_path,
    source={"kind": "sine_sweep"},
    leader={"enabled": True, "target": "chuck_dreamer.runtime.teleop:FakeLeaderReader",
            "params": {"pose": None}},
  ))
  assert rt.leader is not None


def test_manual_source_without_leader_holds(tmp_path):
  # source.kind=manual no longer requires a leader; ManualPolicy just holds.
  rt = Runtime(_build_only_cfg(tmp_path, source={"kind": "manual"}))
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
