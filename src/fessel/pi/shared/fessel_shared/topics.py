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

# QoS levels per the topic map.
QOS_CAPABILITIES = 1
QOS_CMD = 1
QOS_STATE = 1
QOS_HEARTBEAT = 0

# Retained flags per the topic map.
RETAIN_CAPABILITIES = True
RETAIN_STATE = True
