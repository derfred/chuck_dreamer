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
  mode_to_canonical and is what travels in the WHEP `mode` query parameter
  and the signed token.
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
  """Retained payload on arm/video/capabilities."""

  modes: list[ModeTriplet]


class LiveActivate(BaseModel):
  """Payload of arm/video/cmd/live/activate."""

  path: str
  mode: ModeTriplet


class LiveDeactivate(BaseModel):
  """Payload of arm/video/cmd/live/deactivate."""

  path: str


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

  `recording_id` is a UUID chosen by the caller (supervisor); `mode` is the
  triplet to encode with; `operator` is the GitHub user passed through from
  webui-backend so metadata.json (V4.4) records who started the recording.
  """

  recording_id: str
  mode: ModeTriplet
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
  old the oldest one is. The Slice 7 ">3h" P2 alert lands against these."""

  count: int = 0
  oldest_pending_seconds: int | None = None


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


class StateResponse(BaseModel):
  """Body of supervisor GET /state and backend GET /api/state (S3.3).

  `jetson` is whatever the Jetson's GET /state returned (opaque to the
  backend and dashboard — supervisor caches it and passes it through);
  `null` when the Jetson has not been reached yet.

  Slice 4 adds `recording` (the retained recording-state topic, cached) and
  `upload_backlog` (the V4.8 gauges, cached) so the dashboard's recording
  controls and backlog indicator read from the same /state poll as the rest.
  """

  safety_state: SafetyState = SafetyState.IDLE
  jetson: dict | None = None
  plugs: dict[str, PlugState] = Field(default_factory=dict)
  camera: CameraState = Field(default_factory=CameraState)
  recording: RecordingState = Field(default_factory=RecordingState)
  upload_backlog: UploadBacklog = Field(default_factory=UploadBacklog)


def mode_to_canonical(mode: ModeTriplet) -> str:
  """Serialise a ModeTriplet to the canonical "<W>x<H>@<fps>@<bitrate_bps>" form.

  This is the ONLY place the canonical string is built. The token signer
  (backend) and verifier (mediamtx) both depend on this exact form.
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
