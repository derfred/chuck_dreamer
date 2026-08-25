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
