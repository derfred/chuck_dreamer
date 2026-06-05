"""MQTT topic map for Slice 1. The single source of truth for topic names.

MQTT is Pi-internal only — there is no cluster->Pi MQTT traffic. The
broker must not be exposed beyond the Pi.

Slice-1 topics:

| Topic                        | Publisher          | Subscriber         | Retained | QoS |
|------------------------------|--------------------|--------------------|----------|-----|
| arm/video/capabilities       | video              | supervisor, webui  | yes      | 1   |
| arm/video/cmd/live/activate  | supervisor         | video              | no       | 1   |
| arm/video/cmd/live/deactivate| supervisor         | video              | no       | 1   |
| arm/video/state/live         | video              | supervisor         | yes      | 1   |
| arm/video/heartbeat          | video, supervisor  | each other         | no       | 0   |
"""

SITE_PREFIX = "arm"

CAPABILITIES = f"{SITE_PREFIX}/video/capabilities"
CMD_LIVE_ACTIVATE = f"{SITE_PREFIX}/video/cmd/live/activate"
CMD_LIVE_DEACTIVATE = f"{SITE_PREFIX}/video/cmd/live/deactivate"
STATE_LIVE = f"{SITE_PREFIX}/video/state/live"
HEARTBEAT = f"{SITE_PREFIX}/video/heartbeat"

# Camera up/down, retained by video (architecture §7.4). supervisor reads this
# for the `camera` field of GET /state (S3.3).
STATE_CAMERA = f"{SITE_PREFIX}/video/state/camera"

# Verified WiZ plug state, published retained by supervisor on every confirmed
# state change (S3.1). `<name>` is the plug name (arm | jetson).
PLUG_STATE_PREFIX = f"{SITE_PREFIX}/supervisor/plug"


def plug_state(name: str) -> str:
  """Retained topic carrying the last *verified* state of plug `name`."""
  return f"{PLUG_STATE_PREFIX}/{name}"


# --- Slice 4: recording + upload topics (architecture §2.7) ------------------
# Explicit-recording command surface (supervisor -> video) and the retained
# recording-state topic (video -> supervisor). The upload pipeline publishes
# per-recording progress and periodic backlog gauges (video uploader ->
# supervisor). All Pi-internal; no cluster->Pi MQTT.

CMD_RECORDING_START = f"{SITE_PREFIX}/video/cmd/recording/start"
CMD_RECORDING_STOP = f"{SITE_PREFIX}/video/cmd/recording/stop"
CMD_RECORDING_FLAG_UPLOAD = f"{SITE_PREFIX}/video/cmd/recording/flag-upload"

# Retained: the dashboard's Start/Stop button derives from this via /state.
STATE_RECORDING = f"{SITE_PREFIX}/video/state/recording"

# Per-recording upload progress; `<recording-id>` is the recording UUID.
# Retained so a late subscriber (supervisor restart) still sees the last state.
UPLOAD_PREFIX = f"{SITE_PREFIX}/video/upload"
UPLOAD_BACKLOG_COUNT = f"{UPLOAD_PREFIX}/backlog_count"
UPLOAD_BACKLOG_OLDEST = f"{UPLOAD_PREFIX}/oldest_pending_seconds"


def upload_progress(recording_id: str) -> str:
  """Retained topic carrying the last UploadProgress for `recording_id`."""
  return f"{UPLOAD_PREFIX}/{recording_id}"


# Bandwidth coordinator gate (architecture §2.12): supervisor publishes whether
# recording uploads may run, the uploader honours it. Retained so the uploader
# sees the current gate immediately on (re)connect rather than defaulting to a
# guess. `uploads_allowed=false` means a viewer (live or ring) is active.
CMD_UPLOAD_GATE = f"{SITE_PREFIX}/video/cmd/upload/gate"


# QoS levels per the topic map.
QOS_CAPABILITIES = 1
QOS_CMD = 1
QOS_STATE = 1
QOS_HEARTBEAT = 0
QOS_PLUG = 1
QOS_UPLOAD = 1

# Retained flags per the topic map.
RETAIN_CAPABILITIES = True
RETAIN_STATE = True
RETAIN_PLUG = True
# Recording state is retained (matches STATE_LIVE). Upload progress + backlog
# gauges are retained so a restarted supervisor recovers the latest value
# without waiting for the next periodic publish.
RETAIN_RECORDING = True
RETAIN_UPLOAD = True
# The upload gate is retained (unlike the live activate/deactivate commands):
# it is a latched state, so a (re)connecting uploader must see the current value
# without waiting for the coordinator's next tick.
RETAIN_UPLOAD_GATE = True
