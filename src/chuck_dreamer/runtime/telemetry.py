"""Telemetry: a flat record, a non-blocking queue, and logger sinks.

The record shape (spec §3.11) is intentionally flat — one row per control
tick, plus sparse "event" rows (overrun, e-stop, mode change).
The control loop emits :class:`TelemetryRecord`s onto a :class:`TelemetryQueue`;
a sink thread drains and writes them.

As of M3 the *active* runtime sink is the Rerun
:class:`~chuck_dreamer.runtime.rerun_sink.RerunSink` (one ``.rrd`` per episode,
control telemetry + the policy thread's observations on a shared timeline) —
this fulfils the "upgrade the sink to Rerun without changing the record shape"
plan. :class:`CsvSink` is kept here as a standalone, dependency-free sink (and
the canonical documentation of the flat row layout) but is no longer wired into
:class:`Runtime`.

The control loop must never block on logging, so :meth:`TelemetryQueue.emit`
is a non-blocking ``put_nowait`` that drops (and counts) on a full queue. A
sink drains the queue on its own thread and flushes on a clean ``stop`` — that
is the "flushed logs on shutdown" guarantee.
"""

from __future__ import annotations

import csv
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TelemetryRecord:
  """One telemetry row. Either a per-tick row or a sparse event row.

  Per-joint quantities are stored as arrays and expanded into
  ``<name>_<i>`` columns by :func:`record_fieldnames` / :func:`record_to_row`.
  """

  t_wall: float
  t_mono: float
  tick: int
  mode: str = ""
  clamped: bool = False
  velocity_limited: bool = False
  dt: float = 0.0
  q_meas: np.ndarray | None = None
  q_cmd: np.ndarray | None = None
  target: np.ndarray | None = None
  seq: int = -1
  event: str = ""
  detail: str = ""
  backend_s: float = 0.0


_SCALAR_FIELDS = [
  "t_wall", "t_mono", "tick", "mode", "clamped",
  "velocity_limited", "dt", "seq", "event", "detail", "backend_s",
]


def record_fieldnames(n_joints: int) -> list[str]:
  """CSV column order for a runtime with ``n_joints`` joints."""
  cols = list(_SCALAR_FIELDS)
  for name in ("q_meas", "q_cmd", "target"):
    cols.extend(f"{name}_{i}" for i in range(n_joints))
  return cols


def record_to_row(rec: TelemetryRecord, n_joints: int) -> dict[str, Any]:
  """Flatten a record into a ``{column: value}`` dict for ``csv.DictWriter``."""
  row: dict[str, Any] = {
    "t_wall": rec.t_wall,
    "t_mono": rec.t_mono,
    "tick": rec.tick,
    "mode": rec.mode,
    "clamped": int(rec.clamped),
    "velocity_limited": int(rec.velocity_limited),
    "dt": rec.dt,
    "seq": rec.seq,
    "event": rec.event,
    "detail": rec.detail,
    "backend_s": rec.backend_s,
  }
  for name, arr in (("q_meas", rec.q_meas), ("q_cmd", rec.q_cmd), ("target", rec.target)):
    for i in range(n_joints):
      row[f"{name}_{i}"] = "" if arr is None else float(arr[i])
  return row


class TelemetryQueue:
  """Bounded queue with a non-blocking emit that drops + counts on overflow."""

  def __init__(self, maxsize: int = 10000) -> None:
    self._q: "queue.Queue[TelemetryRecord]" = queue.Queue(maxsize=maxsize)
    self._dropped = 0
    self._dropped_lock = threading.Lock()

  def emit(self, rec: TelemetryRecord) -> None:
    try:
      self._q.put_nowait(rec)
    except queue.Full:
      with self._dropped_lock:
        self._dropped += 1

  def emit_event(self, name: str, *, tick: int = -1, detail: str = "") -> None:
    """Emit a sparse event row stamped with the current wall/mono clocks."""
    self.emit(TelemetryRecord(
      t_wall=time.time(), t_mono=time.monotonic(),
      tick=tick, event=name, detail=detail,
    ))

  def get(self, timeout: float) -> TelemetryRecord | None:
    try:
      return self._q.get(timeout=timeout)
    except queue.Empty:
      return None

  def drain(self) -> list[TelemetryRecord]:
    out: list[TelemetryRecord] = []
    while True:
      try:
        out.append(self._q.get_nowait())
      except queue.Empty:
        break
    return out

  @property
  def dropped(self) -> int:
    with self._dropped_lock:
      return self._dropped


class CsvSink:
  """Logger thread draining a :class:`TelemetryQueue` to a CSV file."""

  def __init__(
    self,
    path: str | Path,
    queue_: TelemetryQueue,
    n_joints: int,
    *,
    flush_every_n: int = 50,
  ) -> None:
    self._path                            = Path(path)
    self._queue                           = queue_
    self._n_joints                        = n_joints
    self._flush_every_n                   = max(1, flush_every_n)
    self._stop                            = threading.Event()
    self._thread: threading.Thread | None = None

  def start(self) -> None:
    self._path.parent.mkdir(parents=True, exist_ok=True)
    self._thread = threading.Thread(target=self._run, name="runtime-csv", daemon=True)
    self._thread.start()

  def stop(self, join_timeout: float = 5.0) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=join_timeout)
      self._thread = None

  def _run(self) -> None:
    fieldnames = record_fieldnames(self._n_joints)
    with self._path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      f.flush()
      since_flush = 0

      def write(rec: TelemetryRecord) -> None:
        nonlocal since_flush
        writer.writerow(record_to_row(rec, self._n_joints))
        since_flush += 1

      while not self._stop.is_set():
        rec = self._queue.get(timeout=0.1)
        if rec is not None:
          write(rec)
        if since_flush >= self._flush_every_n:
          f.flush()
          since_flush = 0

      # Drain whatever is left so shutdown never loses tail records.
      for rec in self._queue.drain():
        write(rec)
      f.flush()
