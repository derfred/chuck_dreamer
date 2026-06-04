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


# QoS levels per the topic map.
QOS_CAPABILITIES = 1
QOS_CMD = 1
QOS_STATE = 1
QOS_HEARTBEAT = 0
QOS_PLUG = 1

# Retained flags per the topic map.
RETAIN_CAPABILITIES = True
RETAIN_STATE = True
RETAIN_PLUG = True
