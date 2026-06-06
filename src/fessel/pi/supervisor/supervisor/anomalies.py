"""Slice-5 sensing ingest: live vision/audio values + the in-memory anomaly log.

supervisor subscribes to the vision/audio summary, heartbeat, level, event, and
anomaly-recording topics (S5.1). This class caches the latest live values, keeps
a bounded in-memory log of anomaly events (S5.1/S5.3), and computes the
`vision`/`audio`/`recent_anomalies` fields of /state (S5.2). It does NOT drive
any intervention — that is Slice 6; supervisor receives, caches, and surfaces.

Health (`vision.healthy` / `audio.healthy`) is supervisor's interpretation of
heartbeat freshness: a heartbeat (vision) / level message (audio) seen within
`staleness_s` => healthy. The clock is injectable so the freshness logic is
deterministic in tests.

The log is in-memory — a supervisor restart loses it. Durable storage is Loki
via promtail in Slice 7; this is the convenient live view. Thread-safe: the
relay feeds it from the MQTT loop thread; /state and /anomalies read it from the
HTTP threads.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

from fessel_schemas import (
  AnomalyLogEntry,
  AnomalyType,
  AudioState,
  VisionState,
)

# The violation event types (vs. their `*_cleared` twins). Only violations open
# a new log entry; a `*_cleared` flips the matching open entry's `cleared` flag.
_VIOLATIONS = {
  AnomalyType.safe_box_violation,
  AnomalyType.rest_violation,
  AnomalyType.audio_spike,
}
_CLEAR_OF = {
  AnomalyType.safe_box_cleared: AnomalyType.safe_box_violation,
  AnomalyType.rest_violation_cleared: AnomalyType.rest_violation,
  AnomalyType.audio_spike_cleared: AnomalyType.audio_spike,
}


class AnomalyTracker:
  def __init__(
    self,
    *,
    log_cap: int = 100,
    recent_cap: int = 10,
    staleness_s: float = 6.0,
    now: Callable[[], float] = time.monotonic,
  ) -> None:
    self._lock = threading.Lock()
    self._log: deque[AnomalyLogEntry] = deque(maxlen=log_cap)
    self._recent_cap = recent_cap
    self._staleness_s = staleness_s
    self._now = now

    # Live vision values (from the ~1 Hz summary) + heartbeat freshness.
    self._activity_score_ema = 0.0
    self._last_frame_at: str | None = None
    self._vision_hb_at: float | None = None

    # Live audio value (from the ~2 Hz retained level) + freshness.
    self._audio_level_db: float | None = None
    self._audio_at: float | None = None

  # --- MQTT inputs (called by the relay) -------------------------------------

  def note_vision_summary(self, payload: dict) -> None:
    with self._lock:
      ema = payload.get("activity_score_ema")
      if isinstance(ema, (int, float)):
        self._activity_score_ema = float(ema)

  def note_vision_heartbeat(self, payload: dict) -> None:
    with self._lock:
      self._vision_hb_at = self._now()
      lf = payload.get("last_frame_at")
      if isinstance(lf, str):
        self._last_frame_at = lf

  def note_audio_level(self, payload: dict) -> None:
    with self._lock:
      rms = payload.get("rms_db")
      if isinstance(rms, (int, float)):
        self._audio_level_db = float(rms)
      self._audio_at = self._now()

  def note_event(self, event_type: AnomalyType, payload: dict) -> None:
    """A vision/audio anomaly event (violation or its clear). Violations open a
    log entry; clears flip the matching open entry's `cleared` flag."""
    with self._lock:
      if event_type in _VIOLATIONS:
        ts = payload.get("ts")
        self._log.appendleft(
          AnomalyLogEntry(
            ts=ts if isinstance(ts, str) else "",
            type=event_type,
            payload=payload,
          )
        )
      elif event_type in _CLEAR_OF:
        opened = _CLEAR_OF[event_type]
        for entry in self._log:
          if entry.type is opened and not entry.cleared:
            entry.cleared = True
            break

  def note_anomaly_recording(self, payload: dict) -> None:
    """An anomaly-recording event from video: link the most recent matching open
    log entry to its recording id (S5.1). Matched by anomaly_event_type +
    trigger_ts, falling back to the newest entry of that type."""
    rid = payload.get("anomaly_recording_id")
    etype = payload.get("anomaly_event_type")
    trigger_ts = payload.get("trigger_ts")
    if not isinstance(rid, str):
      return
    with self._lock:
      # Prefer an exact (type, ts) match; else the newest unlinked of that type.
      candidate = None
      for entry in self._log:
        if entry.type.value != etype:
          continue
        if entry.ts == trigger_ts and entry.anomaly_recording_id is None:
          entry.anomaly_recording_id = rid
          return
        if candidate is None and entry.anomaly_recording_id is None:
          candidate = entry
      if candidate is not None:
        candidate.anomaly_recording_id = rid

  # --- read path (for /state and /anomalies) ---------------------------------

  def vision_state(self) -> VisionState:
    with self._lock:
      return VisionState(
        activity_score_ema=round(self._activity_score_ema, 4),
        last_frame_at=self._last_frame_at,
        healthy=self._fresh(self._vision_hb_at),
      )

  def audio_state(self) -> AudioState:
    with self._lock:
      return AudioState(
        level_db=self._audio_level_db,
        healthy=self._fresh(self._audio_at),
      )

  def recent(self) -> list[AnomalyLogEntry]:
    """The newest `recent_cap` entries (newest first) for /state."""
    with self._lock:
      return [e.model_copy() for e in list(self._log)[: self._recent_cap]]

  def full_log(self) -> list[AnomalyLogEntry]:
    """The full bounded log (newest first) for /anomalies."""
    with self._lock:
      return [e.model_copy() for e in self._log]

  def _fresh(self, at: float | None) -> bool:
    if at is None:
      return False
    return (self._now() - at) < self._staleness_s
