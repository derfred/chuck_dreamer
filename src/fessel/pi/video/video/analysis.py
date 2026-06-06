"""Analysis coordinator: the bridge between the Slice-5 detectors and the rest
of the system (MQTT publishes + anomaly-recording triggers).

It owns the pure `VisionAnalyzer` and `AudioAnalyzer` and the `AnomalyRecorder`,
and turns their outputs into:
  - periodic MQTT publishes (vision summary ~1 Hz, audio level ~2 Hz, vision
    heartbeat ~0.5 Hz), rate-limited here so the appsink/level callback rate
    doesn't flood MQTT;
  - per-event MQTT publishes on the vision/audio event topics (with self-clear
    twins);
  - anomaly-recording triggers for the three *violation* event types (not the
    `*_cleared` twins), wired in-process to the recorder (no MQTT round-trip).

Rest-period state (V5.4) is updated from MQTT: Jetson episode events and
supervisor safety state. The Jetson/supervisor subscriptions live in main.py;
they forward into `note_jetson_event` / `note_supervisor_state` here.

This class is GStreamer-free and unit-testable: the GstVision/GstAudio handles
call its `on_vision_*` / `on_audio_*` methods; tests call them directly. The
only injected dependencies are the MQTT publish callable and the AnomalyRecorder
(itself testable with a fake forward capture).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone

from fessel_schemas import (
  AnomalyType,
  AudioLevel,
  AudioSpikeEvent,
  RestViolationEvent,
  SafeBoxEvent,
  VisionSummary,
)
from fessel_shared import topics

from .anomaly_recording import AnomalyRecorder
from .audio import AudioAnalyzer
from .vision import FrameResult, VisionAnalyzer

log = logging.getLogger(__name__)

# Safety states that imply a rest period (V5.4). Operator-driven rest; episode
# boundaries are the other source. SHUTDOWN_JETSON is not "arm rest" — the arm
# may still be powered — but the arm-relevant ones are.
_REST_SAFETY_STATES = {"PAUSED", "STOPPED", "SHUTDOWN_ARM"}

# Jetson event types that bracket the active (non-rest) phase. Between an
# episode_end and the next episode_start the arm is at rest (plan §V5.4).
_EPISODE_END = {"episode_end"}
_EPISODE_START = {"episode_start"}

# Map a violation event type to its MQTT topic. The `*_cleared` twins map too.
_EVENT_TOPICS = {
  AnomalyType.safe_box_violation: topics.VISION_EVENT_SAFE_BOX,
  AnomalyType.safe_box_cleared: topics.VISION_EVENT_SAFE_BOX_CLEARED,
  AnomalyType.rest_violation: topics.VISION_EVENT_REST,
  AnomalyType.rest_violation_cleared: topics.VISION_EVENT_REST_CLEARED,
  AnomalyType.audio_spike: topics.AUDIO_EVENT_SPIKE,
  AnomalyType.audio_spike_cleared: topics.AUDIO_EVENT_SPIKE_CLEARED,
}

# The three violation types that trigger an anomaly recording (V5.7); their
# `*_cleared` twins do not.
_VIOLATIONS = {
  AnomalyType.safe_box_violation,
  AnomalyType.rest_violation,
  AnomalyType.audio_spike,
}

# Build the right event model for a (type, details) pair so the published
# payload is schema-shaped (and the ts/type stamped consistently).
_EVENT_MODELS = {
  AnomalyType.safe_box_violation: SafeBoxEvent,
  AnomalyType.safe_box_cleared: SafeBoxEvent,
  AnomalyType.rest_violation: RestViolationEvent,
  AnomalyType.rest_violation_cleared: RestViolationEvent,
  AnomalyType.audio_spike: AudioSpikeEvent,
  AnomalyType.audio_spike_cleared: AudioSpikeEvent,
}


def _now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


PublishFn = Callable[[str, dict, int, bool], None]


class AnalysisCoordinator:
  def __init__(
    self,
    *,
    vision: VisionAnalyzer,
    audio: AudioAnalyzer,
    recorder: AnomalyRecorder | None,
    publish: PublishFn,
    summary_hz: float = 1.0,
    level_hz: float = 2.0,
    now: Callable[[], float] = time.monotonic,
  ) -> None:
    self._vision = vision
    self._audio = audio
    self._recorder = recorder
    self._publish = publish
    self._summary_period = 1.0 / max(0.1, summary_hz)
    self._level_period = 1.0 / max(0.1, level_hz)
    self._now = now
    self._lock = threading.Lock()

    # -inf so the first frame/level always publishes (a zero start would
    # suppress the very first call when now() also starts at 0).
    self._last_summary_at = float("-inf")
    self._summary_frame_count = 0
    self._activity_max_since_last = 0.0
    self._last_level_at = float("-inf")

    # Rest-period inputs: latest known Jetson episode phase + supervisor state.
    self._in_episode = True  # assume active until told otherwise
    self._safety_rest_reason: str | None = None

  # --- vision callbacks (from GstVisionPipeline) -----------------------------

  def on_vision_summary(self, result: FrameResult) -> None:
    """Per-frame; rate-limited to ~summary_hz for the MQTT publish (V5.2)."""
    now = self._now()
    with self._lock:
      self._summary_frame_count += 1
      self._activity_max_since_last = max(self._activity_max_since_last, result.activity_score)
      due = (now - self._last_summary_at) >= self._summary_period
      if not due:
        return
      summary = VisionSummary(
        ts=_now_iso(),
        activity_score_ema=round(result.activity_score_ema, 4),
        activity_score_max_since_last=round(self._activity_max_since_last, 4),
        frame_count=self._summary_frame_count,
      )
      self._last_summary_at = now
      self._summary_frame_count = 0
      self._activity_max_since_last = 0.0
    self._publish(topics.VISION_SUMMARY, summary.model_dump(), topics.QOS_VISION_SUMMARY, False)

  def on_vision_events(self, events: list) -> None:
    for etype, details in events:
      self._handle_event(etype, details)

  # --- audio callbacks (from GstAudioPipeline) -------------------------------

  def on_audio_level(self, rms_db: float, peak_db: float) -> None:
    now = self._now()
    with self._lock:
      due = (now - self._last_level_at) >= self._level_period
      if not due:
        return
      self._last_level_at = now
    level = AudioLevel(rms_db=round(rms_db, 1), peak_db=round(peak_db, 1))
    # Retained (the live meter value the dashboard wants on connect, V5.6).
    self._publish(topics.AUDIO_LEVEL, level.model_dump(), topics.QOS_AUDIO_LEVEL, True)

  def on_audio_events(self, events: list) -> None:
    for etype, details in events:
      self._handle_event(etype, details)

  # --- the event bridge: publish + (for violations) trigger a recording ------

  def _handle_event(self, etype: AnomalyType, details: dict) -> None:
    ts = _now_iso()
    # Trigger the anomaly recording FIRST for violations so the recording id is
    # known and can be... (note: the recording id is async to the event here;
    # supervisor links them via the separate anomaly-recording event topic).
    anomaly_recording_id: str | None = None
    if etype in _VIOLATIONS and self._recorder is not None:
      try:
        anomaly_recording_id = self._recorder.trigger(etype, ts, details)
      except Exception:  # noqa: BLE001 — a recording failure must not drop the event
        log.exception("anomaly recording trigger failed for %s", etype.value)

    model_cls = _EVENT_MODELS[etype]
    payload = model_cls(ts=ts, type=etype, **details)
    topic = _EVENT_TOPICS[etype]
    qos = topics.QOS_AUDIO_EVENT if "audio" in topic else topics.QOS_VISION_EVENT
    self._publish(topic, payload.model_dump(), qos, False)
    log.info(
      '{"event":"anomaly","type":"%s","recording":"%s"}',
      etype.value,
      anomaly_recording_id or "",
    )

  # --- rest-period inputs (V5.4) ---------------------------------------------

  def note_jetson_event(self, event_type: str) -> None:
    """A Jetson episode boundary event. episode_end -> rest; episode_start ->
    active. Other event types are ignored for rest purposes."""
    et = (event_type or "").lower()
    if et in _EPISODE_END:
      self._in_episode = False
    elif et in _EPISODE_START:
      self._in_episode = True
    else:
      return
    self._recompute_rest()

  def note_supervisor_state(self, safety_state: str | None) -> None:
    """Supervisor's safety state. PAUSED/STOPPED/SHUTDOWN_ARM imply rest."""
    st = (safety_state or "").upper()
    self._safety_rest_reason = _safety_rest_reason(st)
    self._recompute_rest()

  def _recompute_rest(self) -> None:
    # In rest if the operator state says so OR we're between episodes.
    if self._safety_rest_reason is not None:
      self._vision.set_rest(True, self._safety_rest_reason)
    elif not self._in_episode:
      self._vision.set_rest(True, "between_episodes")
    else:
      self._vision.set_rest(False, None)

  # --- SIGHUP reload ---------------------------------------------------------

  def update_rates(self, summary_hz: float, level_hz: float) -> None:
    with self._lock:
      self._summary_period = 1.0 / max(0.1, summary_hz)
      self._level_period = 1.0 / max(0.1, level_hz)


def _safety_rest_reason(state: str) -> str | None:
  if state not in _REST_SAFETY_STATES:
    return None
  return {
    "PAUSED": "paused",
    "STOPPED": "stopped",
    "SHUTDOWN_ARM": "shutdown_arm",
  }[state]
