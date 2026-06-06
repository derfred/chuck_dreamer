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

# Supervisor safety state, published retained by supervisor on every state
# change. Architecture §2.7 lists this primarily as a /metrics source; Slice 5
# adds an actual retained MQTT publish so video's rest-period detection (V5.4)
# can learn the operator-driven safety state (PAUSED/STOPPED/SHUTDOWN_ARM).
STATE_SUPERVISOR = f"{SITE_PREFIX}/supervisor/state"

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


# --- Slice 5: vision + audio analysis topics (plan §7 X5.1) ------------------
# video runs vision/audio analysis on the appsink/level branches and publishes
# a periodic summary, a heartbeat, and discrete anomaly events; supervisor
# subscribes to all of them. The detectors self-clear (a `*_cleared` event on
# return-to-normal) so Slice 6 sees both the entry and the exit. All Pi-internal.

# Vision thread heartbeat (V5.1): absence => supervisor marks vision degraded.
VISION_HEARTBEAT = f"{SITE_PREFIX}/video/vision/heartbeat"
# ~1 Hz vision summary (V5.2): activity-score EMA + max-since-last for the
# dashboard's live sensing strip and Slice-6 fusion.
VISION_SUMMARY = f"{SITE_PREFIX}/video/vision/summary"
# Vision anomaly events (V5.3/V5.4). Each has a self-clearing `_cleared` twin.
VISION_EVENT_SAFE_BOX = f"{SITE_PREFIX}/video/vision/events/safe-box"
VISION_EVENT_SAFE_BOX_CLEARED = f"{SITE_PREFIX}/video/vision/events/safe-box_cleared"
VISION_EVENT_REST = f"{SITE_PREFIX}/video/vision/events/rest-violation"
VISION_EVENT_REST_CLEARED = f"{SITE_PREFIX}/video/vision/events/rest_violation_cleared"

# Audio level meter (V5.6), ~2 Hz, retained so a connecting dashboard sees a
# value immediately. Audio spike events have a self-clearing `_cleared` twin.
AUDIO_LEVEL = f"{SITE_PREFIX}/video/audio/level"
AUDIO_EVENT_SPIKE = f"{SITE_PREFIX}/video/audio/events/spike"
AUDIO_EVENT_SPIKE_CLEARED = f"{SITE_PREFIX}/video/audio/events/spike_cleared"

# video emits this when it has bracketed an anomaly with a finite HLS recording
# (V5.7). Not a command (`cmd/`) — an event video publishes; supervisor and the
# dashboard react. Minted this slice (not in Slice 0's map, which is fine — the
# slice introducing it owns adding it).
EVENT_ANOMALY_RECORDING = f"{SITE_PREFIX}/video/events/anomaly-recording"

# QoS for the Slice-5 topics (plan §7 X5.1 table). Summary/events at QoS 1
# (delivery matters); heartbeat + level at QoS 0 (cheap, loss-tolerant).
QOS_VISION_SUMMARY = 1
QOS_VISION_EVENT = 1
QOS_VISION_HEARTBEAT = 0
QOS_AUDIO_LEVEL = 0
QOS_AUDIO_EVENT = 1
QOS_ANOMALY_RECORDING = 1

# Retained flags (plan §7 X5.1): only the audio level meter is retained — it is
# the live value the dashboard wants on connect. Summaries, the heartbeat, and
# events are NOT retained (a stale anomaly event must not resurface on connect).
RETAIN_AUDIO_LEVEL = True


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
