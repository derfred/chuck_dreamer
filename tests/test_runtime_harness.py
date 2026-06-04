"""Integration tests for :class:`chuck_dreamer.runtime.harness.Runtime`.

FakeBackend only, bounded runs, no real SIGINT (we drive the shutdown event
directly). These cover the M0 end-to-end exit criterion (boots, both loops
run, CSV written, clean shutdown) and the M1 envelope criterion (an
out-of-box scripted sweep is clamped on every tick).
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from chuck_dreamer.config import load_config
from chuck_dreamer.runtime.backend import FakeBackend
from chuck_dreamer.runtime.harness import PolicyLoop, Runtime
from chuck_dreamer.runtime.kernel import Mode
from chuck_dreamer.runtime.setpoint_channel import SetpointChannel


def _cfg(tmp_path: Path, **runtime_overrides):
  """Full default config with the runtime block patched for a fast test."""
  cfg = load_config()
  base = {
    "duration_s": 0.3,
    "control_rate_hz": 200,
    "policy_rate_hz": 50,
    "logging": {"csv_path": str(tmp_path / "telemetry.csv"), "flush_every_n": 10},
    "viewer": {"enabled": False},
  }
  cfg.runtime = OmegaConf.merge(cfg.runtime, OmegaConf.create(base),
                                OmegaConf.create(runtime_overrides))
  return cfg


def _read_rows(path: Path):
  with path.open() as f:
    return list(csv.DictReader(f))


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
    return self.action


def test_policy_loop_resets_and_publishes_act_output():
  policy = _RecordingPolicy([0.1, 0.2])
  channel = SetpointChannel()
  backend = FakeBackend(2, lower=np.full(2, -1.0), upper=np.full(2, 1.0))
  home = np.array([0.3, -0.4])
  loop = PolicyLoop(policy, channel, backend, home, rate_hz=200)

  loop.start()
  deadline = time.monotonic() + 1.0
  while channel.seq == 0 and time.monotonic() < deadline:
    time.sleep(0.005)
  loop.stop()

  np.testing.assert_array_equal(policy.reset_arg, home)   # reset got the start pose
  np.testing.assert_array_equal(channel.get(), [0.1, 0.2])  # act() output published
  assert policy.seen                                       # act() was called
  obs = policy.seen[-1]
  assert obs.t >= 0.0                                      # obs carries elapsed time
  assert obs.q_meas.shape == (2,)                          # and measured joints


def test_boots_runs_and_shuts_down_cleanly(tmp_path):
  cfg = _cfg(tmp_path)
  rt = Runtime(cfg)
  rt.run()  # bounded by duration_s

  rows = _read_rows(tmp_path / "telemetry.csv")
  tick_rows = [r for r in rows if r["event"] == ""]
  assert len(tick_rows) > 0          # control loop ran
  assert rt.control_loop.ticks > 0
  assert rt.channel.seq > 0          # policy loop published
  # No leftover runtime threads.
  time.sleep(0.05)
  assert _runtime_threads() == []


def test_csv_has_header_and_parses(tmp_path):
  rt = Runtime(_cfg(tmp_path))
  rt.run()
  rows = _read_rows(tmp_path / "telemetry.csv")
  assert rows
  # Per-joint columns exist for all 6 SO-101 joints.
  for i in range(6):
    assert f"q_cmd_{i}" in rows[0]


def test_simulated_sigint_shuts_down_and_flushes(tmp_path):
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

  rows = _read_rows(tmp_path / "telemetry.csv")
  assert len(rows) > 0
  assert _runtime_threads() == []


def test_out_of_envelope_sweep_is_clamped_every_tick(tmp_path):
  # Sine amplitude far exceeds the joint box -> every commanded position must
  # stay in-box and clamp events must appear.
  cfg = _cfg(
    tmp_path,
    source={"kind": "sine_sweep", "sine_sweep": {"amplitude": 50.0, "freq_hz": 2.0, "phase": 0.0}},
    safety={"max_velocity": 1000.0},  # let it race to the boundary fast
  )
  rt = Runtime(cfg)
  lower, upper = rt._limits.lower, rt._limits.upper
  rt.run()

  rows = _read_rows(tmp_path / "telemetry.csv")
  tick_rows = [r for r in rows if r["event"] == ""]
  assert tick_rows
  for r in tick_rows:
    for i in range(6):
      q = float(r[f"q_cmd_{i}"])
      assert lower[i] - 1e-9 <= q <= upper[i] + 1e-9
  assert any(r["clamped"] == "1" for r in tick_rows)


def test_fake_backend_runtime_starts_in_normal(tmp_path):
  rt = Runtime(_cfg(tmp_path))
  assert rt.kernel.mode is Mode.NORMAL
