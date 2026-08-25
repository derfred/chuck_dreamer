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

import threading
import time
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from chuck_dreamer.config import load_config
from chuck_dreamer.policy import Action
from chuck_dreamer.runtime.backend import FakeBackend
from chuck_dreamer.runtime.harness import PolicyLoop, Runtime
from chuck_dreamer.runtime.modalities import ModalityError
from chuck_dreamer.runtime.setpoint_channel import SetpointChannel

from .conftest import control_tick_rows


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
  cfg.runtime = OmegaConf.merge(cfg.runtime, OmegaConf.create(base),
                                OmegaConf.create(runtime_overrides))
  return cfg


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
  channel = SetpointChannel()
  home = np.array([0.3, -0.4])
  backend = FakeBackend(2, lower=np.full(2, -1.0), upper=np.full(2, 1.0),
                        q_init=home)
  loop = PolicyLoop(policy, channel, backend, rate_hz=200)

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


def test_out_of_envelope_sweep_is_clamped_every_tick(tmp_path, fake_rerun):
  # Sine amplitude far exceeds the joint box -> every commanded position must
  # stay in-box and clamp events must appear.
  cfg = _cfg(
    tmp_path,
    source={"kind": "sine_sweep", "sine_sweep": {"amplitude": 50.0, "freq_hz": 2.0, "phase": 0.0}},
    safety={"max_velocity": 1000.0},  # let it race to the boundary fast
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
