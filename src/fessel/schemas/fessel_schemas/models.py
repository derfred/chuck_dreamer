"""Pydantic models for the Slice-1 wire shapes.

These models are the source of truth. JSON Schema is exported from them
(see tools/generate-types.sh) and TypeScript is generated for the frontend.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field

# Canonical mode string: "<W>x<H>@<fps>@<bitrate_bps>", e.g. "1280x720@30@2500000".
# '@' separates the three fields; 'x' separates width/height. URL-safe.
_MODE_RE = re.compile(r"^(?P<w>\d+)x(?P<h>\d+)@(?P<fps>\d+)@(?P<bps>\d+)$")


class ModeTriplet(BaseModel):
  """A concrete video operating point.

  `resolution` is the canonical "<W>x<H>" string; fps and bitrate_bps are
  integers. The full canonical form (incl. fps and bitrate) is produced by
  mode_to_canonical; recordings use it (the live path is a fixed profile).
  """

  resolution: str = Field(pattern=r"^\d+x\d+$", description='"<W>x<H>", e.g. "1280x720"')
  fps: int = Field(gt=0)
  bitrate_bps: int = Field(gt=0)


class LiveStateValue(str, Enum):
  off = "off"
  starting = "starting"
  running = "running"
  stopping = "stopping"


class Capabilities(BaseModel):
  """Retained payload on arm/video/capabilities.

  `recording_mode` is the operating point the deploy is configured to record
  with (from `video.recording` config, §2.2) — recording resolution is a deploy
  setting, not a per-request choice, so the operator UI no longer picks it.
  `max_lookback_seconds` is how far a recording can reach back into the always-on
  ring buffer (= the ring window); the record dialog uses it to bound its
  look-back timeline.
  """

  recording_mode: ModeTriplet
  max_lookback_seconds: float


class LiveActivate(BaseModel):
  """Payload of arm/video/cmd/live/activate.

  Parameterless (architecture §2.2): the live profile is a deploy-time config
  setting on video, not a per-session parameter. The message carries no
  fields; it means "a viewer wants the live stream; attach the sender."
  """


class LiveDeactivate(BaseModel):
  """Payload of arm/video/cmd/live/deactivate. Parameterless."""


class LiveViewErrorCode(str, Enum):
  """Error codes of a rejected POST /whep (webui relay, architecture §2.3)."""

  live_unavailable = "live_unavailable"  # health-gate fast reject (503)
  live_timeout = "live_timeout"  # ingest not up within live_activation_timeout (504)
  live_activation_failed = "live_activation_failed"  # supervisor activate failed (502)
  live_error = "live_error"  # unexpected controller failure (500)
  whep_negotiate_failed = "whep_negotiate_failed"  # SDP negotiation failed (500)


class LiveViewError(BaseModel):
  """Error body of a rejected POST /whep.

  Authored by the Go webui (generated Go struct) and consumed by the
  frontend's WHEP client (generated Zod validator), so both ends of the
  live-view error contract derive from this one definition.
  """

  error: LiveViewErrorCode
  reason: str


class LiveState(BaseModel):
  """Retained payload on arm/video/state/live."""

  state: LiveStateValue
  path: str | None = None
  mode: ModeTriplet | None = None


# --- Slice 3 control-plane shapes --------------------------------------------
# supervisor's /state aggregation, forwarded verbatim by the backend's
# /api/state and consumed by the dashboard. The safety state is a placeholder
# in Slice 3 (Slice 6 replaces the placeholder with the real state machine).


class SafetyState(str, Enum):
  """Abbreviated supervisor safety state (architecture §2.4).

  Slice 3 only ever holds whatever the operator's last command implies; the
  real transitions land in Slice 6. The full set is declared here so the
  enum doesn't have to change later.
  """

  INIT = "INIT"
  IDLE = "IDLE"
  RUNNING = "RUNNING"
  PAUSED = "PAUSED"
  STOPPED = "STOPPED"
  SHUTDOWN_ARM = "SHUTDOWN_ARM"
  SHUTDOWN_JETSON = "SHUTDOWN_JETSON"
  DEGRADED = "DEGRADED"


class PlugState(BaseModel):
  """A WiZ plug's last *verified* state (S3.1/S3.3).

  `on` reflects the state read back via getPilot after the last action or
  background refresh — not the last command issued. `verified` is False when
  the most recent verify failed (the command may have been sent but could not
  be confirmed); `verified_at` is the ISO-8601 timestamp of that read.
  """

  on: bool | None = None
  verified: bool = False
  verified_at: str | None = None


class CameraState(BaseModel):
  """Camera up/down, derived from the retained arm/video/state/* topics."""

  up: bool | None = None


# --- Slice 4 recording / upload shapes ---------------------------------------
# The explicit-recording lifecycle (video's recording state machine, V4.3), the
# retained recording-state topic (V4.5), the per-recording metadata file (V4.4),
# the upload progress topic (V4.7), the recordings listing (S4.4), and the
# upload-backlog gauges (V4.8). The recording branch is a manual operator
# gesture: at most one recording is active at a time.


class RecordingStateValue(str, Enum):
  """video's explicit-recording branch state machine (V4.3).

  Separate from the live branch's LiveStateValue: the two branches share the
  encoder hardware but are independent. `idle` is the at-rest state (no
  recording), mirroring the live branch's `off`.
  """

  idle = "idle"
  starting = "starting"
  recording = "recording"
  stopping = "stopping"


class UploadStateValue(str, Enum):
  """Where a recording is in the flag-for-upload pipeline (V4.6/V4.7).

  `none` — never flagged. `queued` — marker written, uploader not yet acting.
  `uploading` — transfer in progress. `uploaded` — in MinIO, marker deleted.
  `failed` — non-retryable failure; marker renamed `.failed`.
  """

  none = "none"
  queued = "queued"
  uploading = "uploading"
  uploaded = "uploaded"
  failed = "failed"


class RecordingStartCmd(BaseModel):
  """Payload of arm/video/cmd/recording/start (V4.5).

  `recording_id` is a UUID chosen by the caller (supervisor). Recording reuses
  the always-on ring's encode (§2.2), so there is no per-recording `mode`: the
  resolution/bitrate come from the deploy's `video.recording` config, not the
  request. `lookback_seconds` reaches the recording's start BACK into the ring
  buffer (0 = start now); it is clamped by video to what the ring holds.
  `upload_when_done` flags the finished recording for upload automatically on
  finalise. `operator` is the GitHub user passed through from webui-backend so
  metadata.json (V4.4) records who started the recording.
  """

  recording_id: str
  lookback_seconds: float = Field(default=0.0, ge=0)
  upload_when_done: bool = False
  operator: str | None = None


class RecordingFlagUploadCmd(BaseModel):
  """Payload of arm/video/cmd/recording/flag-upload (V4.5)."""

  recording_id: str


class RecordingState(BaseModel):
  """Retained payload on arm/video/state/recording (V4.5).

  The single source of truth for "is a recording active right now"; the
  dashboard derives its Start/Stop button from this (via supervisor /state).
  """

  state: RecordingStateValue = RecordingStateValue.idle
  active_recording_id: str | None = None
  started_at: str | None = None


class RecordingMetadata(BaseModel):
  """The per-recording metadata.json written on finalisation (V4.4).

  `flagged_for_upload` and `upload_state` are mutable in place: flag-upload
  (V4.5) and the uploader (V4.7) rewrite this file. It is the recording's
  source of truth for the rest of the system (listing, playback routing).
  """

  id: str
  started_at: str
  ended_at: str | None = None
  operator: str | None = None
  mode: ModeTriplet | None = None
  duration_seconds: int | None = None
  segments: int | None = None
  flagged_for_upload: bool = False
  upload_state: UploadStateValue = UploadStateValue.none
  uploaded_at: str | None = None


class UploadProgress(BaseModel):
  """Payload on arm/video/upload/<recording-id> (V4.7), surfaced read-only by
  the backend (B4.6) so the dashboard can show per-recording upload status."""

  recording_id: str
  state: UploadStateValue = UploadStateValue.queued
  attempts: int = 0
  last_attempt_at: str | None = None
  last_error: str | None = None


class UploadBacklog(BaseModel):
  """Upload-queue health gauges (V4.8): how many markers are pending and how
  old the oldest one is. The Slice 7 ">3h" P2 alert lands against these.

  `healthy` is the uploader's own liveness, derived from heartbeat freshness
  (arm/video/upload/heartbeat), NOT from queue depth: an uploader that never
  started (or is crash-looping before it ever publishes anything) has
  count=0/oldest=None too, indistinguishable from a genuinely idle uploader
  unless liveness is tracked separately. None = no heartbeat ever received."""

  count: int = 0
  oldest_pending_seconds: int | None = None
  healthy: bool | None = None


class RecordingListItem(BaseModel):
  """One row of supervisor GET /recordings (S4.4) — what's on the Pi.

  A projection of RecordingMetadata sufficient for the dashboard list; segment
  URLs are decided by the backend (B4.2/B4.3), not enumerated here.
  """

  recording_id: str
  started_at: str
  ended_at: str | None = None
  duration_seconds: int | None = None
  operator: str | None = None
  flagged_for_upload: bool = False
  upload_state: UploadStateValue = UploadStateValue.none


class UploadGate(BaseModel):
  """Payload of arm/video/cmd/upload/gate (architecture §2.12).

  The bandwidth coordinator publishes this (retained); the uploader honours it.
  `uploads_allowed=False` means a viewer (live feed or ring playback) is active,
  so recording uploads must suspend to free the cellular uplink (Requirements
  §4.2). `reason` is advisory, for logs/observability.
  """

  uploads_allowed: bool = True
  reason: str | None = None


# --- Slice 5 vision / audio analysis shapes ----------------------------------
# The sensing layer that feeds the safety system (Slice 6). video runs vision
# and audio analysis threads on the appsink/level branches and publishes a
# periodic summary plus discrete anomaly events; supervisor caches the live
# values, keeps a bounded anomaly log, and surfaces both in /state and a small
# /anomalies endpoint. None of this drives an intervention yet (that is Slice 6).


class AnomalyType(str, Enum):
  """The discrete anomaly events the Slice-5 detectors emit (and clear).

  `safe_box_violation` — foreground motion outside the configured safe box.
  `rest_violation` — motion during a rest period (between episodes / paused).
  `audio_spike` — a sound-level spike above baseline. The `*_cleared` variants
  are the self-clearing return-to-normal events the Slice-6 state machine wants
  to see paired with the entry.
  """

  safe_box_violation = "safe_box_violation"
  safe_box_cleared = "safe_box_cleared"
  rest_violation = "rest_violation"
  rest_violation_cleared = "rest_violation_cleared"
  audio_spike = "audio_spike"
  audio_spike_cleared = "audio_spike_cleared"


class VisionHeartbeat(BaseModel):
  """Payload on arm/video/vision/heartbeat (V5.1).

  Published periodically while the vision thread is alive; its absence is how
  supervisor marks vision degraded (`vision.healthy=false`). `last_frame_at` is
  the wall-clock of the most recent appsink frame.
  """

  ts: str
  last_frame_at: str | None = None
  frames_processed: int = 0
  frames_dropped: int = 0


class VisionSummary(BaseModel):
  """Payload on arm/video/vision/summary (V5.2), ~1 Hz.

  `activity_score_ema` is the smoothed foreground-area fraction in [0,1];
  `activity_score_max_since_last` is the peak per-frame score since the last
  summary, so a transient spike the EMA smoothed out is still visible.
  """

  ts: str
  activity_score_ema: float = 0.0
  activity_score_max_since_last: float = 0.0
  frame_count: int = 0


class SafeBox(BaseModel):
  """The image-space safe-box rectangle (V5.3), in 640x360 downscaled coords."""

  x: int = Field(ge=0)
  y: int = Field(ge=0)
  w: int = Field(gt=0)
  h: int = Field(gt=0)


class SafeBoxEvent(BaseModel):
  """Payload on arm/video/vision/events/safe-box[_cleared] (V5.3)."""

  ts: str
  type: AnomalyType = AnomalyType.safe_box_violation
  outside_fraction: float = 0.0
  threshold: float = 0.0
  duration_frames: int = 0
  ring_segment_hint: str | None = None


class RestViolationEvent(BaseModel):
  """Payload on arm/video/vision/events/rest-violation[_cleared] (V5.4)."""

  ts: str
  type: AnomalyType = AnomalyType.rest_violation
  activity_score: float = 0.0
  threshold: float = 0.0
  # Why this is a rest period: episode boundary vs operator-driven safety state.
  rest_reason: str | None = None
  ring_segment_hint: str | None = None


class AudioLevel(BaseModel):
  """Retained payload on arm/video/audio/level (V5.6), ~2 Hz, for the live
  meter. Retained so the dashboard sees a value immediately on connect."""

  rms_db: float = 0.0
  peak_db: float = 0.0


class AudioSpikeEvent(BaseModel):
  """Payload on arm/video/audio/events/spike[_cleared] (V5.6)."""

  ts: str
  type: AnomalyType = AnomalyType.audio_spike
  peak_db: float = 0.0
  baseline_db: float = 0.0
  duration_ms: int = 0
  ring_segment_hint: str | None = None


class AnomalyRecordingEvent(BaseModel):
  """Payload on arm/video/events/anomaly-recording (V5.7).

  Emitted by video when it has bracketed an anomaly with a finite HLS recording
  lifted from the ring buffer plus a forward capture, so supervisor and the
  dashboard can link the anomaly to its evidence.
  """

  anomaly_recording_id: str
  anomaly_event_type: AnomalyType
  trigger_ts: str


class AnomalyRecordingMetadata(BaseModel):
  """The per-anomaly-recording metadata.json (V5.7).

  Compatible with RecordingMetadata on the fields supervisor's listing reads
  (id, started/ended, duration, segments, flagged_for_upload, upload_state) —
  the `type` field is what discriminates an anomaly recording from an explicit
  one (S5.4). `trigger_events` carries every coalesced trigger (V5.7 coalesce
  rule), the first of which is the original trigger.
  """

  id: str
  type: str = "anomaly"
  anomaly_event_type: AnomalyType
  trigger_ts: str
  window: dict = Field(default_factory=dict)  # {before_seconds, after_seconds}
  trigger_events: list[dict] = Field(default_factory=list)
  started_at: str
  ended_at: str | None = None
  operator: str | None = None  # anomaly recordings have no operator
  mode: ModeTriplet | None = None
  duration_seconds: int | None = None
  segments: int | None = None
  flagged_for_upload: bool = False
  upload_state: UploadStateValue = UploadStateValue.none
  uploaded_at: str | None = None


class VisionState(BaseModel):
  """The `vision` field of supervisor /state (S5.2).

  `healthy` is supervisor's interpretation of the vision heartbeat: false when
  the heartbeat is stale. Not a separate safety state, just a flag Slice 6 reads.
  """

  activity_score_ema: float = 0.0
  last_frame_at: str | None = None
  healthy: bool = False


class AudioState(BaseModel):
  """The `audio` field of supervisor /state (S5.2)."""

  level_db: float | None = None
  healthy: bool = False


class VersionState(BaseModel):
  """The `version` field of supervisor /state (health-check spec §2.1).

  `component` is the AUTHORITATIVE Pi-side version used for the cluster/Pi
  compatibility comparison (all Pi components ship in one dpkg, so one version
  covers them). `supervisor`/`video` are optional per-process details, useful in
  the drill-down only if they ever diverge (e.g. a partial upgrade). The strings
  are semver-shaped MAJOR.MINOR.PATCH; the comparison uses MAJOR.MINOR only.
  """

  component: str
  supervisor: str | None = None
  video: str | None = None


class AnomalyLogEntry(BaseModel):
  """One entry of supervisor's bounded in-memory anomaly log (S5.1/S5.3).

  `payload` is the full original event so the dashboard can show details;
  `anomaly_recording_id` links to the evidence recording if one was created.
  `cleared` flips true when the matching `*_cleared` event arrives.
  """

  ts: str
  type: AnomalyType
  anomaly_recording_id: str | None = None
  cleared: bool = False
  payload: dict = Field(default_factory=dict)


class StateResponse(BaseModel):
  """Body of supervisor GET /state and backend GET /api/state (S3.3).

  `jetson` is whatever the Jetson's GET /state returned (opaque to the
  backend and dashboard — supervisor caches it and passes it through);
  `null` when the Jetson has not been reached yet.

  Slice 4 adds `recording` (the retained recording-state topic, cached) and
  `upload_backlog` (the V4.8 gauges, cached) so the dashboard's recording
  controls and backlog indicator read from the same /state poll as the rest.

  Slice 5 adds `vision` and `audio` (live sensing values + health) and
  `recent_anomalies` (a short tail of the in-memory anomaly log), so the
  dashboard's sensing strip and recent-anomalies panel read from the same poll.
  """

  safety_state: SafetyState = SafetyState.IDLE
  jetson: dict | None = None
  plugs: dict[str, PlugState] = Field(default_factory=dict)
  camera: CameraState = Field(default_factory=CameraState)
  recording: RecordingState = Field(default_factory=RecordingState)
  upload_backlog: UploadBacklog = Field(default_factory=UploadBacklog)
  vision: VisionState = Field(default_factory=VisionState)
  audio: AudioState = Field(default_factory=AudioState)
  recent_anomalies: list[AnomalyLogEntry] = Field(default_factory=list)
  version: VersionState | None = None


def mode_to_canonical(mode: ModeTriplet) -> str:
  """Serialise a ModeTriplet to the canonical "<W>x<H>@<fps>@<bitrate_bps>" form.

  This is the ONLY place the canonical string is built. The token signer
  depend on this exact form.
  """
  return f"{mode.resolution}@{mode.fps}@{mode.bitrate_bps}"


def mode_from_canonical(s: str) -> ModeTriplet:
  """Parse the canonical "<W>x<H>@<fps>@<bitrate_bps>" form into a ModeTriplet."""
  m = _MODE_RE.match(s)
  if not m:
    raise ValueError(f"invalid canonical mode string: {s!r}")
  return ModeTriplet(
    resolution=f"{m['w']}x{m['h']}",
    fps=int(m["fps"]),
    bitrate_bps=int(m["bps"]),
  )
