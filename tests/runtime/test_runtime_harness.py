"""Integration tests for :class:`chuck_dreamer.runtime.harness.Runtime`.

FakeBackend only, bounded runs, no real SIGINT (we drive the shutdown event
directly). These cover the M0 end-to-end exit criterion (boots, both loops run,
telemetry logged, clean shutdown), the M1 envelope criterion (an out-of-box
scripted sweep is clamped on every tick), and the M3 additions (the .rrd is
written; startup fails fast on an unproduced required modality).

The runtime logs to Rerun now, not CSV. The ``fake_rerun`` fixture (conftest.py)
captures logged entities; :func:`control_tick_rows` reconstructs the per-tick
row view the CSV-era assertions used. ``sensors`` defaults to ``[]`` (cameras
are backend-specific and enabled explicitly in config).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from chuck_dreamer.config import load_config
from chuck_dreamer.policy import Action
from chuck_dreamer.runtime.backend import FakeBackend
from chuck_dreamer.runtime.control_channel import ControlChannel
from chuck_dreamer.runtime.harness import PolicyLoop, Runtime
from chuck_dreamer.runtime.modalities import ModalityError

from .conftest import control_tick_rows


def merge_runtime(runtime, *overrides):
  """Merge override dicts into a runtime config, rejecting unknown keys.

  Plain :func:`OmegaConf.merge` *creates* a key that is not already in the
  config, so a mistyped or mis-nested override lands somewhere nothing reads
  and the test silently runs on defaults instead. (This is not hypothetical:
  ``safety={...}`` here reads as ``runtime.safety``, but the planner reads
  ``runtime.control_loop.safety`` -- so the cap never applied.) Struct mode
  turns that class of typo into a loud failure at merge time.

  The ``params`` blocks under ``policy`` / ``leader`` / ``backend`` are exempt:
  their keys are defined by whichever registry target is selected, so an
  unknown key there is the normal case, not a typo.
  """
  merged = OmegaConf.create(runtime)
  OmegaConf.set_struct(merged, True)
  for node in ("policy", "leader", "backend"):
    params = OmegaConf.select(merged, f"{node}.params")
    if params is not None:
      OmegaConf.set_struct(params, False)
  for override in overrides:
    merged = OmegaConf.merge(merged, OmegaConf.create(override))
  OmegaConf.set_struct(merged, False)   # the runtime itself may add keys
  return merged


def _cfg(tmp_path: Path, **runtime_overrides):
  """Full default config with the runtime block patched for a fast test."""
  cfg = load_config()
  base = {
    "duration_s": 0.3,
    "control_loop": {"rate_hz": 200},
    "policy_rate_hz": 50,
    "sensors": [],   # explicit: no camera in these FakeBackend tests
    "logging": {"rerun": {"rrd_dir": str(tmp_path / "rrd")}},
    "viewer": {"enabled": False},
  }
  cfg.runtime = merge_runtime(cfg.runtime, base, runtime_overrides)
  return cfg


class _ZeroPolicy:
  """A custom out-of-tree policy, to prove `policy.target` takes any path."""

  def reset(self, start):
    self._n = len(np.asarray(start))

  def act(self, obs):
    return Action(obs, q=np.zeros(self._n))


def _runtime_threads():
  return [t for t in threading.enumerate() if t.name.startswith("runtime-")]


# -- PolicyLoop drives via policy.act(obs) -----------------------------------


class _RecordingPolicy:
  """Policy that records the obs it sees and returns a fixed action."""

  def __init__(self, action):
    self.action = np.asarray(action, dtype=np.float64)
    self.reset_arg = None
    self.seen = []

  def reset(self, scene):
    self.reset_arg = scene

  def act(self, obs):
    self.seen.append(obs)
    return Action(obs, q=self.action)


def test_policy_loop_resets_and_publishes_act_output():
  policy = _RecordingPolicy([0.1, 0.2])
  channel = ControlChannel()
  home = np.array([0.3, -0.4])
  backend = FakeBackend(2, lower=np.full(2, -1.0), upper=np.full(2, 1.0),
                        q_init=home)
  loop = PolicyLoop(policy, channel, backend, rate_hz=200)

  # No control loop here: stand in for its per-tick publish, since the policy
  # loop now takes its measurement off the channel rather than the backend.
  channel.publish_state(backend.read_state())

  loop.start()
  deadline = time.monotonic() + 1.0
  while channel.seq == 0 and time.monotonic() < deadline:
    time.sleep(0.005)
  loop.stop()

  # reset is anchored on the backend's home pose, not a threaded-in argument
  np.testing.assert_array_equal(policy.reset_arg, home)
  # act() output published, wrapped as an Action carrying the obs it answers
  published = channel.get()
  np.testing.assert_array_equal(published.q, [0.1, 0.2])
  assert published.obs is policy.seen[-1]
  assert policy.seen                                       # act() was called
  obs = policy.seen[-1]
  assert obs.t >= 0.0                                      # obs carries elapsed time
  assert obs.q_meas.shape == (2,)                          # and measured joints
  assert obs.present() >= {"t", "q_meas"}                  # M3 modality dict present
  # q_meas is a projection of the channel state, not an independent sample
  assert obs.control is not None
  np.testing.assert_array_equal(obs.q_meas, obs.control.q)
  np.testing.assert_array_equal(obs.q_meas, home)


def test_policy_is_not_called_until_a_control_state_exists():
  """No state means nothing to build an observation around, so the policy waits."""
  policy  = _RecordingPolicy([0.1, 0.2])
  channel = ControlChannel()
  backend = FakeBackend(2, lower=np.full(2, -1.0), upper=np.full(2, 1.0))
  loop    = PolicyLoop(policy, channel, backend, rate_hz=200)

  loop.start()                               # no control loop running
  try:
    time.sleep(0.05)
    assert not policy.seen                   # never called with a missing measurement
    assert channel.seq == 0                  # so nothing was published either

    channel.publish_state(backend.read_state())   # the control loop comes up
    deadline = time.monotonic() + 1.0
    while channel.seq == 0 and time.monotonic() < deadline:
      time.sleep(0.005)
    assert channel.seq > 0                   # and the loop picks up from there
  finally:
    loop.stop()


def test_policy_observes_the_control_loops_own_state(tmp_path, fake_rerun):
  """End-to-end: the obs a policy sees carries the control thread's measurement.

  The point of the two-way channel. Before it, the policy sampled
  ``backend.last_positions()`` -- a bare vector on its own clock, with no age
  and no fault flags. Now the object on the observation is the one the safety
  layer was handed.
  """
  seen: list = []

  class _Recorder:
    def reset(self, start):
      pass

    def act(self, obs):
      seen.append(obs)
      return Action(obs, q=np.asarray(obs.q_meas, dtype=np.float64))

  cfg = _cfg(tmp_path)
  rt  = Runtime(cfg)
  rt.policy = _Recorder()
  rt.policy_loop = PolicyLoop(
    rt.policy, rt.channel, rt.backend,
    rate_hz=float(cfg.runtime.policy_rate_hz), leader=rt.leader,
    sensors=rt.sensors, perception=rt.perception, rerun_sink=rt.sink)
  rt.run()

  assert seen, "policy never ran"
  for obs in seen:
    assert obs.control is not None                 # a real ControlState, not None
    assert obs.q_meas is obs.control.q             # q_meas projects it, not a resample
    assert obs.control.q_age >= 0.0                # and the age rides along
  assert obs.leader is None                        # no leader configured here


def test_boots_runs_and_shuts_down_cleanly(tmp_path, fake_rerun):
  cfg = _cfg(tmp_path)
  rt = Runtime(cfg)
  rt.run()  # bounded by duration_s

  assert control_tick_rows(fake_rerun.rec)   # control loop logged ticks
  assert rt.control_loop.ticks > 0
  assert rt.channel.seq > 0                   # policy loop published
  assert fake_rerun.rec.flushed              # .rrd flushed on shutdown
  # No leftover runtime threads.
  time.sleep(0.05)
  assert _runtime_threads() == []


def test_writes_rrd_file(tmp_path):
  # Real rerun (no fake): a non-empty .rrd lands in the configured dir.
  rt = Runtime(_cfg(tmp_path))
  rt.run()
  assert rt.sink.path is not None and rt.sink.path.exists()
  assert rt.sink.path.stat().st_size > 0
  assert rt.sink.path.parent == tmp_path / "rrd"


def test_control_rows_carry_per_joint_commands(tmp_path, fake_rerun):
  rt = Runtime(_cfg(tmp_path))
  rt.run()
  rows = control_tick_rows(fake_rerun.rec)
  assert rows
  assert rows[0]["q_cmd"].shape == (6,)       # all 6 SO-101 joints logged


def test_simulated_sigint_shuts_down_and_flushes(tmp_path, fake_rerun):
  # duration None would block forever; emulate Ctrl-C by setting the same
  # event the installed handler sets, from a timer thread.
  cfg = _cfg(tmp_path, duration_s=None)
  rt = Runtime(cfg)

  def fire():
    time.sleep(0.2)
    rt.request_shutdown()

  killer = threading.Thread(target=fire)
  killer.start()
  rt.run()
  killer.join()

  assert control_tick_rows(fake_rerun.rec)
  assert fake_rerun.rec.flushed
  assert _runtime_threads() == []


# -- policy construction is one registry path --------------------------------


def test_policy_built_from_registry_spec(tmp_path):
  # The scripted policies get no special case: they are constructed by the same
  # {target, params} path a learned policy will use at M6.
  from chuck_dreamer.runtime.sources import GoToPose

  rt = Runtime(_cfg(tmp_path, policy={
    "target": "chuck_dreamer.runtime.sources:GoToPose",
    "params": {"q_goal": None, "duration_s": 1.0}}))
  assert isinstance(rt.policy, GoToPose)


def test_policy_target_can_be_any_import_path(tmp_path):
  rt = Runtime(_cfg(tmp_path, policy={"target": f"{__name__}:_ZeroPolicy",
                                      "params": {}}))
  assert isinstance(rt.policy, _ZeroPolicy)


def test_policy_params_the_target_cannot_take_are_dropped(tmp_path, caplog):
  # OmegaConf merges key-wise, so switching target leaves the previous policy's
  # params behind. Building must warn and ignore them, not crash.
  from chuck_dreamer.runtime.sources import ManualPolicy

  with caplog.at_level(logging.WARNING):
    rt = Runtime(_cfg(tmp_path, policy={
      "target": "chuck_dreamer.runtime.sources:ManualPolicy"}))
  assert isinstance(rt.policy, ManualPolicy)
  assert "amplitude" in caplog.text


def test_policy_without_target_fails_loudly(tmp_path):
  cfg = _cfg(tmp_path)
  cfg.runtime.policy = OmegaConf.create({"params": {}})
  with pytest.raises(ValueError, match="target"):
    Runtime(cfg)


def test_out_of_envelope_sweep_is_clamped_every_tick(tmp_path, fake_rerun):
  # Sine amplitude far exceeds the joint box -> every commanded position must
  # stay in-box and clamp events must appear.
  cfg = _cfg(
    tmp_path,
    policy={"target": "chuck_dreamer.runtime.sources:SineSweep",
            "params": {"amplitude": 50.0, "freq_hz": 2.0, "phase": 0.0}},
    # let it race to the boundary fast. Note the full path: the planner reads
    # control_loop.safety, so a bare `safety=` would land on a dead key.
    control_loop={"rate_hz": 200, "safety": {"max_velocity": 1000.0}},
  )
  rt = Runtime(cfg)
  lower, upper = rt.backend.joint_limits()
  rt.run()

  rows = control_tick_rows(fake_rerun.rec)
  assert rows
  for r in rows:
    q = r["q_cmd"]
    assert np.all(q >= lower - 1e-9) and np.all(q <= upper + 1e-9)
  assert any(r["clamped"] == 1 for r in rows)


def test_fake_backend_runtime_brakes_to_a_stop_on_shutdown(tmp_path, fake_rerun):
  """The trajectory-era replacement for "starts in NORMAL, ends held".

  There is no mode to assert on any more, so assert the observable behaviour
  the old HOLD existed to produce: the run ends with the arm stationary at the
  last commanded position, not mid-segment.
  """
  rt = Runtime(_cfg(tmp_path, duration_s=0.3))
  rt.run()

  rows = control_tick_rows(fake_rerun.rec)
  assert rows
  # The run ends braked: the tail is a coast-down settling onto a hold, so
  # assert it has converged rather than that it is bit-identical -- a quintic
  # approaches its endpoint asymptotically and the last steps are ~1e-6 rad.
  tail  = [np.asarray(r["q_cmd"], dtype=float) for r in rows[-3:]]
  steps = [float(np.max(np.abs(b - a))) for a, b in zip(tail, tail[1:])]
  assert all(s < 1e-4 for s in steps), f"still moving at shutdown: {steps}"


def test_shutdown_of_an_estopped_runtime_is_prompt_and_quiet(tmp_path, fake_rerun, caplog):
  """An e-stopped runtime tears down at once, without a spurious warning.

  Shutdown used to ask for a graceful brake unconditionally. The mode machine
  refuses one under ESTOP -- a latched freeze outranks a graceful stop -- so
  the call blocked out its whole timeout waiting for a BRAKED that could never
  arrive, and then warned. The cleanest shutdown there is (arm already frozen)
  was the slowest, and the only one that looked like a fault in the log.

  How the loop decides what "come to rest" means is its own business and is
  covered in test_runtime_control_loop.py; what the harness owes is a teardown
  that neither dawdles nor cries wolf.
  """
  rt = Runtime(_cfg(tmp_path, duration_s=None))
  rt.start()
  try:
    deadline = time.monotonic() + 2.0
    while rt.control_loop.ticks == 0 and time.monotonic() < deadline:
      time.sleep(0.005)
    assert rt.control_loop.request_estop(timeout=2.0)
  finally:
    with caplog.at_level(logging.WARNING, logger="chuck_dreamer.runtime.harness"):
      t0 = time.monotonic()
      rt.shutdown()
      elapsed = time.monotonic() - t0

  # The default stop timeout is 1 s; the bug spent all of it every time.
  assert elapsed < 0.5, f"shutdown blocked for {elapsed:.3f}s on an e-stopped arm"
  assert not [r for r in caplog.records if "did not come to rest" in r.message]
  assert not _runtime_threads(), "threads outlived shutdown"


# -- M3: modality composition + fail-fast ------------------------------------


def test_startup_fails_fast_on_unproduced_required_modality(tmp_path):
  # Require object_xy but configure no perception that produces it -> the
  # runtime must refuse to construct (before any thread starts).
  cfg = _cfg(tmp_path, required_modalities=["object_xy"])
  with pytest.raises(ModalityError, match="object_xy"):
    Runtime(cfg)
  # And nothing was spun up.
  assert _runtime_threads() == []


def test_startup_passes_when_required_modality_is_produced(tmp_path):
  # ee is produced by EePoseModule; but EePoseModule needs a backend with FK.
  # FakeBackend has none, so require only the base modalities here — the point
  # is that a satisfiable required set constructs cleanly.
  cfg = _cfg(tmp_path, required_modalities=["q_meas", "t"])
  rt = Runtime(cfg)
  assert "q_meas" in rt.available and "t" in rt.available


# -- M3: sim camera path must not starve the control loop --------------------


def test_sim_camera_does_not_stall_control_loop(tmp_path):
  """Regression: SimCameraSensor renders on the policy thread; if it held the
  physics lock across the full GL render it starved the control loop. Run the
  real MuJoCo + camera + perception path and assert the control loop still
  achieved most of its configured rate (spec §3.2/§4.1: slow perception must
  not affect the control loop)."""
  pytest.importorskip("mujoco")
  cfg = load_config()
  base = {
    "duration_s": 1.5,
    "backend": {"target": "chuck_dreamer.runtime.mujoco_backend:MujocoBackend",
                "params": {"realtime": True}},
    "sensors": [{"target": "chuck_dreamer.runtime.sensors:SimCameraSensor",
                 "params": {"camera": "main_camera", "name": "camera/front",
                            "render_size": "240x320"}}],
    "perception": [{"target": "chuck_dreamer.runtime.perception:EePoseModule"}],
    "required_modalities": ["image", "ee"],
    "logging": {"rerun": {"rrd_dir": str(tmp_path / "rrd")}},
    "viewer": {"enabled": False},
  }
  cfg.runtime = OmegaConf.merge(cfg.runtime, OmegaConf.create(base))
  rt = Runtime(cfg)
  assert {"image", "ee"} <= rt.available
  rt.run()

  # The starvation this guards against showed up as a collapsed tick rate, so
  # assert the rate directly rather than merely that some ticks happened.
  expected = base["duration_s"] * float(cfg.runtime.control_loop.rate_hz)
  assert rt.control_loop.ticks > 0.5 * expected
  assert rt.channel.seq > 0                  # observations flowed despite rendering
  assert rt.sink.path is not None and rt.sink.path.stat().st_size > 0
  time.sleep(0.05)
  assert _runtime_threads() == []


def test_policy_loop_emits_its_own_telemetry_rows():
  """The policy thread reports its cost the way the control thread does."""
  from chuck_dreamer.runtime.telemetry import TelemetryQueue

  policy   = _RecordingPolicy([0.1, 0.2])
  channel  = ControlChannel()
  backend  = FakeBackend(2, lower=np.full(2, -1.0), upper=np.full(2, 1.0))
  telem    = TelemetryQueue(maxsize=100)
  loop     = PolicyLoop(policy, channel, backend, rate_hz=200, telemetry=telem)

  channel.publish_state(backend.read_state())
  loop.start()
  deadline = time.monotonic() + 1.0
  while channel.seq == 0 and time.monotonic() < deadline:
    time.sleep(0.005)
  loop.stop()

  rows = [r for r in telem.drain() if r.kind == "policy"]
  assert rows, "policy loop emitted no telemetry"
  r = rows[0]
  assert r.policy_s > 0                      # the step was timed
  assert r.act_s > 0                         # and so was policy.act
  assert r.seq > 0                           # the published action is identified


def test_policy_telemetry_is_optional():
  """Omitting the queue leaves the loop working — it is instrumentation only."""
  policy  = _RecordingPolicy([0.1, 0.2])
  channel = ControlChannel()
  backend = FakeBackend(2, lower=np.full(2, -1.0), upper=np.full(2, 1.0))
  loop    = PolicyLoop(policy, channel, backend, rate_hz=200)   # no telemetry

  channel.publish_state(backend.read_state())
  loop.start()
  deadline = time.monotonic() + 1.0
  while channel.seq == 0 and time.monotonic() < deadline:
    time.sleep(0.005)
  loop.stop()
  assert policy.seen


def test_policy_loop_reports_an_overrun():
  """A policy step slower than its period emits ``policy_overrun``.

  Before the loop shared PacedLoop it had no overrun path at all: a slow
  step silently slid the schedule forward, so a policy that could not hold
  its configured rate looked identical to one that could.
  """
  from chuck_dreamer.runtime.telemetry import TelemetryQueue

  class _SlowPolicy(_RecordingPolicy):
    def act(self, obs):
      time.sleep(0.05)                       # far longer than the 10 ms period
      return super().act(obs)

  channel = ControlChannel()
  backend = FakeBackend(2, lower=np.full(2, -1.0), upper=np.full(2, 1.0))
  telem   = TelemetryQueue(maxsize=200)
  loop    = PolicyLoop(_SlowPolicy([0.1, 0.2]), channel, backend,
                       rate_hz=100.0, telemetry=telem)

  channel.publish_state(backend.read_state())
  loop.start()
  time.sleep(0.25)
  loop.stop()

  events = [r.event for r in telem.drain() if r.event]
  assert "policy_overrun" in events
