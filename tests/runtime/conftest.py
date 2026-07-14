"""Shared runtime-test fixtures.

The M3 runtime logs to Rerun, not CSV. The experimental ``.rrd`` reader doesn't
expose entity contents, so to assert *what* the runtime logged we monkeypatch
the ``rerun`` module (imported lazily inside the sink) with a fake that captures
every ``set_time`` / ``log`` call. :func:`control_tick_rows` reconstructs the
old per-tick "row" view from that capture so migrated tests read almost like
their CSV ancestors.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


class _FakeArchetype:
  def __init__(self, kind, value):
    self.kind = kind
    self.value = value


class _FakeRec:
  """Captures the ordered stream of set_time + log calls on a recording."""

  def __init__(self):
    self.events: list[tuple] = []   # ("time", timeline, kw) | ("log", path, archetype)
    self.saved_path = None
    self.flushed = False
    self.application_id = None
    self.recording_id = None

  def save(self, path):
    self.saved_path = path

  def set_time(self, timeline, **kw):
    self.events.append(("time", timeline, kw))

  def log(self, path, archetype, **_kw):
    self.events.append(("log", path, archetype))

  def flush(self, timeout_sec=0.0):
    self.flushed = True

  # -- read helpers used by tests -----------------------------------------

  def logged_paths(self) -> list[str]:
    return [p for kind, p, *_ in self.events if kind == "log"]

  def logged(self, path: str) -> list:
    return [a for kind, p, a in self.events if kind == "log" and p == path]


class _FakeRerun(types.ModuleType):
  def __init__(self):
    super().__init__("rerun")
    self.rec = _FakeRec()

  def RecordingStream(self, application_id, recording_id):  # noqa: N802
    self.rec.application_id = application_id
    self.rec.recording_id = recording_id
    return self.rec

  def Image(self, img):  # noqa: N802
    return _FakeArchetype("image", np.asarray(img))

  def Scalars(self, v):  # noqa: N802
    return _FakeArchetype("scalars", v)

  def TextLog(self, s):  # noqa: N802
    return _FakeArchetype("textlog", s)


@pytest.fixture
def fake_rerun(monkeypatch):
  """Replace the ``rerun`` module with a capturing fake; yields it."""
  fake = _FakeRerun()
  monkeypatch.setitem(sys.modules, "rerun", fake)
  return fake


def control_tick_rows(rec: _FakeRec) -> list[dict]:
  """Reconstruct per-control-tick rows from a captured recording.

  Each control tick emits a ``set_time("step", sequence=...)`` then a burst of
  ``control/*`` logs (or a single ``events`` log). We segment on the step
  marker and collect the control entities that follow, returning a list of
  ``{seq, mode, clamped, q_meas, q_cmd, target, is_event, event}`` dicts —
  enough to port the CSV assertions.
  """
  rows: list[dict] = []
  cur: dict | None = None
  for kind, a, b in rec.events:
    if kind == "time" and a == "step":
      if cur is not None:
        rows.append(cur)
      cur = {"seq": b.get("sequence"), "is_event": False, "event": "",
             "clamped": None, "mode": None,
             "q_meas": None, "q_cmd": None, "target": None}
      continue
    if kind == "log" and cur is not None:
      path, arch = a, b
      if path == "events":
        cur["is_event"] = True
        cur["event"] = arch.value
      elif path == "control/mode":
        cur["mode"] = arch.value
      elif path == "control/clamped":
        cur["clamped"] = int(arch.value)
      elif path in ("control/q_meas", "control/q_cmd", "control/target"):
        cur[path.split("/")[1]] = np.asarray(arch.value, dtype=float)
  if cur is not None:
    rows.append(cur)
  # Per-tick rows are those carrying control scalars (not the bare event rows
  # and not the observation-only steps from the policy thread).
  return [r for r in rows if not r["is_event"] and r["q_cmd"] is not None]
