"""Unit tests for :class:`chuck_dreamer.runtime.rerun_sink.RerunSink`.

The sink imports ``rerun`` lazily inside its methods, so the shared
``fake_rerun`` fixture (tests/conftest.py) monkeypatches the ``rerun`` module
with a fake that captures every ``log`` call. That lets us assert exactly which
entity paths the sink emits from control telemetry and from observations —
without parsing a ``.rrd`` (the experimental reader doesn't expose entity
contents). A final test exercises the real ``rerun`` path and asserts a
non-empty ``.rrd`` is written and flushed.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from chuck_dreamer.runtime.modalities import EE, IMAGE, OBJECT_XY, RuntimeObservation
from chuck_dreamer.runtime.rerun_sink import RerunSink
from chuck_dreamer.runtime.telemetry import TelemetryQueue, TelemetryRecord


def _wait_until(pred, timeout=2.0):
  deadline = time.monotonic() + timeout
  while not pred() and time.monotonic() < deadline:
    time.sleep(0.005)
  assert pred(), "condition not met before timeout"


# -- control telemetry path ---------------------------------------------------


def test_telemetry_tick_row_logs_control_entities(fake_rerun):
  q = TelemetryQueue()
  sink = RerunSink("unused", q, n_joints=2)
  sink.start("ep")
  q.emit(TelemetryRecord(
    t_wall=0.0, t_mono=0.1, tick=3, mode="normal", clamped=True,
    velocity_limited=False, dt=0.01,
    q_meas=np.array([0.0, 0.1]), q_cmd=np.array([0.0, 0.2]),
    target=np.array([0.0, 0.3]), seq=5,
  ))
  _wait_until(lambda: "control/q_cmd" in fake_rerun.rec.logged_paths())
  sink.stop()

  paths = fake_rerun.rec.logged_paths()
  assert "control/q_meas" in paths
  assert "control/q_cmd" in paths
  assert "control/target" in paths
  assert "control/clamped" in paths


def test_event_row_logs_textlog(fake_rerun):
  q = TelemetryQueue()
  sink = RerunSink("unused", q, n_joints=1)
  sink.start("ep")
  q.emit_event("watchdog_trip", tick=7, detail="stalled")
  _wait_until(lambda: "events" in fake_rerun.rec.logged_paths())
  sink.stop()
  events = fake_rerun.rec.logged("events")
  assert events and "watchdog_trip" in events[0].value


# -- observation path ---------------------------------------------------------


def test_log_observation_logs_camera_and_vectors(fake_rerun):
  q = TelemetryQueue()
  sink = RerunSink("unused", q, n_joints=6)
  sink.start("ep")
  obs = RuntimeObservation(
    t=0.2, q_meas=np.zeros(6),
    modalities={
      "t": 0.2, "q_meas": np.zeros(6),
      IMAGE: np.zeros((4, 4, 3), np.uint8),
      EE: np.arange(7, dtype=np.float32),
      OBJECT_XY: np.array([1.0, 2.0], np.float32),
    },
  )
  sink.log_observation(obs, step=2, t=0.2)
  _wait_until(lambda: "camera/image" in fake_rerun.rec.logged_paths())
  sink.stop()

  paths = fake_rerun.rec.logged_paths()
  assert "camera/image" in paths
  assert "obs/joint_qpos" in paths
  assert "obs/ee" in paths
  assert "obs/object_xy" in paths


def test_observation_without_image_skips_camera(fake_rerun):
  q = TelemetryQueue()
  sink = RerunSink("unused", q, n_joints=6)
  sink.start("ep")
  obs = RuntimeObservation(t=0.0, q_meas=np.zeros(6),
                           modalities={"t": 0.0, "q_meas": np.zeros(6)})
  sink.log_observation(obs, step=0, t=0.0)
  _wait_until(lambda: "obs/joint_qpos" in fake_rerun.rec.logged_paths())
  sink.stop()
  assert "camera/image" not in fake_rerun.rec.logged_paths()


def test_log_observation_never_blocks_and_counts_drops(fake_rerun):
  q = TelemetryQueue()
  sink = RerunSink("unused", q, n_joints=1, obs_queue_maxsize=1)
  # Don't start the logger thread -> the obs queue fills, exercising the
  # drop-on-full guard without a draining consumer.
  obs = RuntimeObservation(t=0.0, q_meas=np.zeros(1),
                           modalities={"t": 0.0, "q_meas": np.zeros(1)})
  for i in range(5):
    sink.log_observation(obs, step=i, t=float(i))   # must never raise
  assert sink.dropped >= 1


# -- real rerun end-to-end ----------------------------------------------------


def test_writes_nonempty_rrd_with_real_rerun(tmp_path: Path):
  q = TelemetryQueue()
  sink = RerunSink(tmp_path, q, n_joints=2)
  sink.start("episode-test")
  q.emit(TelemetryRecord(
    t_wall=0.0, t_mono=0.0, tick=0, mode="normal",
    q_meas=np.zeros(2), q_cmd=np.zeros(2), target=np.zeros(2), seq=0))
  obs = RuntimeObservation(t=0.0, q_meas=np.zeros(2),
                           modalities={"t": 0.0, "q_meas": np.zeros(2),
                                       IMAGE: np.zeros((4, 4, 3), np.uint8)})
  sink.log_observation(obs, step=0, t=0.0)
  time.sleep(0.2)
  sink.stop()

  assert sink.path is not None
  assert sink.path.exists()
  assert sink.path.stat().st_size > 0
