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
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from .threads import ManagedThread


@dataclass
class TelemetryRecord:
  """One telemetry row. Either a per-tick row or a sparse event row."""

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
  read_state_s: float = 0.0
  write_positions_s: float = 0.0
  action_age_s: float = -1.0 # End-to-end setpoint latency

  # Measurement health, from the tick's ControlState (see control_state.py).
  q_age: float = 0.0
  read_s: float = 0.0
  faults: int = 0

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

  @property
  def backend_s(self) -> float:
    return self.read_state_s + self.write_positions_s

  @contextmanager
  def timed(self, field_name: str) -> Iterator[None]:
    """Time a block straight into the named duration field."""
    if not hasattr(self, field_name):
      raise AttributeError(f"TelemetryRecord has no duration field {field_name!r}")
    t0 = time.perf_counter()
    try:
      yield
    finally:
      setattr(self, field_name, time.perf_counter() - t0)

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
    action_age_s: float | None = None,
  ) -> None:
    """Fold one tick's outcome into the record."""
    self.dt = float(dt)
    if state is not None:
      self.q_meas    = state.q
      self.q_age     = float(state.q_age)
      self.read_s    = float(state.read_s)
      self.faults    = int(state.faults)
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

    The stage timings (`sensor_s` / `perception_s` / `act_s` / `policy_s`) are
    already on the record: :meth:`timed` wrote them as the caller's blocks
    ran.
    """
    self.kind         = "policy"
    self.dt           = float(dt)
    if action is not None:
      self.seq = int(getattr(action, "seq", -1))


# Per-joint quantities: stored as arrays, expanded to `<name>_<i>` columns.
_ARRAY_FIELDS = ("q_meas", "q_cmd", "target")


def _scalar_fields() -> list[str]:
  """Every scalar field, in declaration order, derived from the dataclass."""
  return [f.name for f in fields(TelemetryRecord) if f.name not in _ARRAY_FIELDS] + ["backend_s"]


def record_fieldnames(n_joints: int) -> list[str]:
  """CSV column order for a runtime with ``n_joints`` joints."""
  cols = _scalar_fields()
  for name in _ARRAY_FIELDS:
    cols.extend(f"{name}_{i}" for i in range(n_joints))
  return cols


def record_to_row(rec: TelemetryRecord, n_joints: int) -> dict[str, Any]:
  """Flatten a record into a ``{column: value}`` dict for ``csv.DictWriter``."""
  row: dict[str, Any] = {}
  for name in _scalar_fields():
    value = getattr(rec, name)
    # bool is a subclass of int, so this must precede any numeric handling;
    # CSV wants 0/1 rather than True/False.
    row[name] = int(value) if isinstance(value, bool) else value
  for name in _ARRAY_FIELDS:
    arr = getattr(rec, name)
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
