"""Fessel canonical data shapes — single source of truth.

Both the Pi-side processes (video, supervisor) and the cluster-side
backend consume these. The frontend consumes the generated TypeScript
emitted by tools/generate-types.sh; it must never hand-define the shapes.

The canonical mode-triplet string form is fixed here once:

    "<W>x<H>@<fps>@<bitrate_bps>"

e.g. "1280x720@30@2500000". This exact form flows through:
  - the WHEP `mode` query parameter,
  - the HMAC-signed token (over path + mode + expiry),
  - the recording-start command and recording metadata.

It is parsed/serialised only via mode_to_canonical / mode_from_canonical
so every producer and consumer of the form agrees on it.
"""

from .models import (
    AnomalyLogEntry,
    AnomalyRecordingEvent,
    AnomalyRecordingMetadata,
    AnomalyType,
    AudioLevel,
    AudioSpikeEvent,
    AudioState,
    Capabilities,
    CameraState,
    LiveActivate,
    LiveDeactivate,
    LiveState,
    LiveStateValue,
    LiveViewError,
    LiveViewErrorCode,
    ModeTriplet,
    PlugState,
    RecordingFlagUploadCmd,
    RecordingListItem,
    RecordingMetadata,
    RecordingStartCmd,
    RecordingState,
    RecordingStateValue,
    RestViolationEvent,
    SafeBox,
    SafeBoxEvent,
    SafetyState,
    StateResponse,
    UploadBacklog,
    UploadGate,
    UploadProgress,
    UploadStateValue,
    VersionState,
    VisionHeartbeat,
    VisionState,
    VisionSummary,
    mode_from_canonical,
    mode_to_canonical,
)
from .version import fessel_version

__all__ = [
    "AnomalyLogEntry",
    "AnomalyRecordingEvent",
    "AnomalyRecordingMetadata",
    "AnomalyType",
    "AudioLevel",
    "AudioSpikeEvent",
    "AudioState",
    "Capabilities",
    "CameraState",
    "LiveActivate",
    "LiveDeactivate",
    "LiveState",
    "LiveStateValue",
    "LiveViewError",
    "LiveViewErrorCode",
    "ModeTriplet",
    "PlugState",
    "RecordingFlagUploadCmd",
    "RecordingListItem",
    "RecordingMetadata",
    "RecordingStartCmd",
    "RecordingState",
    "RecordingStateValue",
    "RestViolationEvent",
    "SafeBox",
    "SafeBoxEvent",
    "SafetyState",
    "StateResponse",
    "UploadBacklog",
    "UploadGate",
    "UploadProgress",
    "UploadStateValue",
    "VersionState",
    "VisionHeartbeat",
    "VisionState",
    "VisionSummary",
    "fessel_version",
    "mode_from_canonical",
    "mode_to_canonical",
]
