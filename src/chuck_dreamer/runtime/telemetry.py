"""Telemetry: a flat record, a non-blocking queue, and logger sinks.

The record shape (spec §3.11) is intentionally flat — one row per control
tick, plus sparse "event" rows (overrun, e-stop, mode change).
The control loop emits :class:`TelemetryRecord`s onto a :class:`TelemetryQueue`;
a sink thread drains and writes them.
"""

from __future__ import annotations

import csv
import queue
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .threads import ManagedThread


@dataclass
class TelemetryRecord:
  """One telemetry row. Either a per-tick row or a sparse event row.

  Per-joint quantities are stored as arrays and expanded into
  ``<name>_<i>`` columns by :func:`record_fieldnames` / :func:`record_to_row`.
  """

  t_wall: float
  t_mono: float
  tick: int
  kind: str = "control" # Which loop produced this row.
  mode: str = ""
  clamped: bool = False
  stale: bool = False
  dt: float = 0.0
  q_meas: np.ndarray | None = None
  q_cmd: np.ndarray | None = None
  target: np.ndarray | None = None
  seq: int = -1
  event: str = ""
  detail: str = ""
  backend_s: float = 0.0
  action_age_s: float = -1.0 # End-to-end setpoint latency

  # Measurement health, from the tick's ControlState (see control_state.py).
  q_age: float = 0.0
  read_s: float = 0.0
  faults: int = 0
  read_budget_s: float = 0.0

  # workspace limiter
  ws_scale: float = 1.0
  ws_breach_m: float = 0.0

  # -- policy rows (kind="policy") --------------------------------------------
  # The policy loop's analogue of ``backend_s``: where its step time went.
  # ``policy_s`` is the whole step, and the three below are its stages, so
  # ``policy_s - (sensor_s + perception_s + act_s)`` is the loop's own
  # overhead (observation build, publish, logging).
  sensor_s: float = 0.0
  perception_s: float = 0.0
  act_s: float = 0.0
  policy_s: float = 0.0

  _timed: dict[str, float] = field(default_factory=dict)

  @contextmanager
  def timed(self, name: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
      yield
    finally:
      self._timed[name] = time.perf_counter() - t0

  def elapsed(self, name: str, default: float = 0.0) -> float:
    """Duration recorded by a :meth:`timed` block, or ``default`` if it never ran."""
    return self._timed.get(name, default)

  def update(
    self,
    *,
    state=None,
    action=None,
    q_cmd=None,
    ref=None,
    dt: float = 0.0,
    limits=None,
    mode=None,
    read_budget_s: float = 0.0,
    action_age_s: float | None = None,
  ) -> None:
    """Fold one tick's outcome into the record.

    Everything is optional because a tick can legitimately end early (no
    trajectory, a refused command), and a partial row is more useful than a
    missing one.
    """
    self.dt            = float(dt)
    self.read_budget_s = float(read_budget_s)
    if state is not None:
      self.q_meas    = state.q
      self.q_age     = float(state.q_age)
      self.read_s    = float(state.read_s)
      self.faults    = int(state.faults)
      self.backend_s = self.elapsed("read_state") + self.elapsed("write_positions")
    if q_cmd is not None:
      self.q_cmd = np.asarray(q_cmd, dtype=np.float64)
    if ref is not None:
      self.target = np.asarray(ref[0], dtype=np.float64)
    if action is not None:
      self.seq = int(getattr(action, "seq", -1))
    if action_age_s is not None:
      self.action_age_s = float(action_age_s)
    if mode is not None:
      # Accept the enum or its value, so callers need not unwrap it.
      self.mode = str(getattr(mode, "value", mode))
    if limits:
      self.clamped     = bool(limits.get("clamped", False))
      self.stale       = bool(limits.get("stale", False))
      self.ws_scale    = float(limits.get("ws_scale", 1.0))
      self.ws_breach_m = float(limits.get("ws_breach_m", 0.0))

  def update_policy(self, *, action=None, dt: float = 0.0) -> None:
    """Fold one policy step's outcome into the record (policy thread).

    The stage timings come from :meth:`timed` blocks the caller already ran,
    so this only has to total them — the same shape as ``backend_s`` on the
    control side.
    """
    self.kind         = "policy"
    self.dt           = float(dt)
    self.sensor_s     = self.elapsed("sensors")
    self.perception_s = self.elapsed("perception")
    self.act_s        = self.elapsed("act")
    self.policy_s     = self.elapsed("step")
    if action is not None:
      self.seq = int(getattr(action, "seq", -1))


_SCALAR_FIELDS = [
  "t_wall", "t_mono", "tick", "kind", "mode", "clamped",
  "stale", "dt", "seq", "event", "detail", "backend_s",
  "q_age", "read_s", "faults", "read_budget_s", "ws_scale", "ws_breach_m",
  "action_age_s", "sensor_s", "perception_s", "act_s", "policy_s",
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
    "kind": rec.kind,
    "mode": rec.mode,
    "clamped": int(rec.clamped),
    "stale": int(rec.stale),
    "dt": rec.dt,
    "seq": rec.seq,
    "event": rec.event,
    "detail": rec.detail,
    "backend_s": rec.backend_s,
    "q_age": rec.q_age,
    "read_s": rec.read_s,
    "faults": int(rec.faults),
    "read_budget_s": rec.read_budget_s,
    "ws_scale": rec.ws_scale,
    "ws_breach_m": rec.ws_breach_m,
    "sensor_s": rec.sensor_s,
    "perception_s": rec.perception_s,
    "act_s": rec.act_s,
    "policy_s": rec.policy_s,
    "action_age_s": rec.action_age_s,
  }
  for name, arr in (("q_meas", rec.q_meas), ("q_cmd", rec.q_cmd), ("target", rec.target)):
    for i in range(n_joints):
      row[f"{name}_{i}"] = "" if arr is None else float(arr[i])
  return row


class TelemetryQueue:
  """Bounded queue with a non-blocking emit that drops + counts on overflow."""

  def __init__(self, maxsize: int = 10000) -> None:
    self._q: queue.Queue[TelemetryRecord] = queue.Queue(maxsize=maxsize)
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


class CsvSink(ManagedThread):
  """Logger thread draining a :class:`TelemetryQueue` to a CSV file."""

  def __init__(
    self,
    path: str | Path,
    queue_: TelemetryQueue,
    n_joints: int,
    *,
    flush_every_n: int = 50,
  ) -> None:
    super().__init__("runtime-csv")
    self._path          = Path(path)
    self._queue         = queue_
    self._n_joints      = n_joints
    self._flush_every_n = max(1, flush_every_n)

  def _on_start(self) -> None:
    self._path.parent.mkdir(parents=True, exist_ok=True)

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
