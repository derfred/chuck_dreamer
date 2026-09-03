"""Tests for :mod:`chuck_dreamer.runtime.telemetry`."""

from __future__ import annotations

import csv
import time

import numpy as np

from chuck_dreamer.runtime.telemetry import (
  CsvSink,
  TelemetryQueue,
  TelemetryRecord,
  record_fieldnames,
)


def _tick_record(tick: int, n: int = 3) -> TelemetryRecord:
  return TelemetryRecord(
    t_wall=time.time(), t_mono=time.monotonic(), tick=tick, mode="normal",
    clamped=False, stale=False, dt=0.02,
    q_meas=np.full(n, 0.1 * tick), q_cmd=np.full(n, 0.1 * tick),
    target=np.full(n, 0.5), seq=tick,
  )


def test_csv_written_with_header_and_rows(tmp_path):
  n = 3
  q = TelemetryQueue(maxsize=100)
  sink = CsvSink(tmp_path / "telemetry.csv", q, n_joints=n, flush_every_n=2)
  sink.start()
  for i in range(5):
    q.emit(_tick_record(i, n))
  sink.stop()

  path = tmp_path / "telemetry.csv"
  assert path.exists()
  with path.open() as f:
    rows = list(csv.DictReader(f))
  assert len(rows) == 5
  assert set(record_fieldnames(n)).issubset(rows[0].keys())
  # Spot-check a per-joint column and the tick index parse cleanly.
  assert int(rows[4]["tick"]) == 4
  assert float(rows[2]["q_cmd_0"]) == 0.2


def test_event_rows_recorded(tmp_path):
  q = TelemetryQueue(maxsize=100)
  sink = CsvSink(tmp_path / "t.csv", q, n_joints=2)
  sink.start()
  q.emit_event("watchdog_trip", tick=7, detail="stall")
  sink.stop()
  with (tmp_path / "t.csv").open() as f:
    rows = list(csv.DictReader(f))
  assert rows[0]["event"] == "watchdog_trip"
  assert rows[0]["detail"] == "stall"
  # Per-joint columns are blank for event rows.
  assert rows[0]["q_cmd_0"] == ""


def test_emit_never_blocks_and_counts_drops():
  q = TelemetryQueue(maxsize=2)
  for i in range(10):
    q.emit(_tick_record(i))  # would block on a plain Queue once full
  assert q.dropped == 8


def test_stop_drains_remaining_records(tmp_path):
  # Many records, slow-ish flush: stop() must drain the tail, not truncate.
  q = TelemetryQueue(maxsize=10000)
  sink = CsvSink(tmp_path / "t.csv", q, n_joints=1, flush_every_n=1000)
  sink.start()
  for i in range(500):
    q.emit(_tick_record(i, n=1))
  sink.stop()
  with (tmp_path / "t.csv").open() as f:
    rows = list(csv.DictReader(f))
  assert len(rows) == 500


# -- policy-thread rows -------------------------------------------------------


def test_policy_row_totals_its_stage_timings():
  """``update_policy`` folds the timed blocks into the flat policy fields."""
  rec = TelemetryRecord(t_wall=time.time(), t_mono=time.monotonic(), tick=7)
  with rec.timed("step"):
    with rec.timed("sensors"):
      time.sleep(0.002)
    with rec.timed("perception"):
      time.sleep(0.002)
    with rec.timed("act"):
      time.sleep(0.002)
  rec.update_policy(dt=0.1)

  assert rec.kind == "policy"
  assert rec.sensor_s > 0 and rec.perception_s > 0 and rec.act_s > 0
  # The stages happened inside the step, so the step bounds their sum.
  assert rec.policy_s >= rec.sensor_s + rec.perception_s + rec.act_s
  assert rec.dt == 0.1


def test_control_rows_default_to_control_kind():
  """Existing control rows keep their shape; ``kind`` discriminates them."""
  assert _tick_record(0).kind == "control"


def test_policy_and_control_rows_share_one_queue(tmp_path):
  """Both loops' rows survive the same queue and sink, tagged by ``kind``."""
  n = 3
  q = TelemetryQueue(maxsize=100)
  sink = CsvSink(tmp_path / "t.csv", q, n_joints=n, flush_every_n=1)
  sink.start()
  q.emit(_tick_record(0, n))
  policy = TelemetryRecord(t_wall=time.time(), t_mono=time.monotonic(), tick=0)
  policy.update_policy(dt=0.1)
  q.emit(policy)
  sink.stop()

  with (tmp_path / "t.csv").open() as f:
    rows = list(csv.DictReader(f))
  kinds = [r["kind"] for r in rows]
  assert kinds == ["control", "policy"]


def test_policy_fields_are_in_the_csv_header():
  cols = record_fieldnames(3)
  for name in ("kind", "sensor_s", "perception_s", "act_s", "policy_s", "action_age_s"):
    assert name in cols


def test_action_age_defaults_to_absent_on_a_control_row():
  """Ticks that consumed no new setpoint report -1, not a stale age."""
  assert _tick_record(0).action_age_s == -1.0
