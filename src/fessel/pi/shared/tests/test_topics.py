from fessel_shared import topics


def test_topic_names_and_qos():
  assert topics.CAPABILITIES == "arm/video/capabilities"
  assert topics.CMD_LIVE_ACTIVATE == "arm/video/cmd/live/activate"
  assert topics.CMD_LIVE_DEACTIVATE == "arm/video/cmd/live/deactivate"
  assert topics.STATE_LIVE == "arm/video/state/live"
  assert topics.STATE_CAMERA == "arm/video/state/camera"
  assert topics.HEARTBEAT == "arm/video/heartbeat"
  # Retained per the topic map.
  assert topics.RETAIN_CAPABILITIES is True
  assert topics.RETAIN_STATE is True
  assert topics.QOS_HEARTBEAT == 0


def test_plug_state_topic():
  # Slice 3: verified plug state, one retained topic per plug name.
  assert topics.plug_state("arm") == "arm/supervisor/plug/arm"
  assert topics.plug_state("jetson") == "arm/supervisor/plug/jetson"
  assert topics.QOS_PLUG == 1
  assert topics.RETAIN_PLUG is True


def test_slice4_recording_and_upload_topics():
  assert topics.CMD_RECORDING_START == "arm/video/cmd/recording/start"
  assert topics.CMD_RECORDING_STOP == "arm/video/cmd/recording/stop"
  assert topics.CMD_RECORDING_FLAG_UPLOAD == "arm/video/cmd/recording/flag-upload"
  assert topics.STATE_RECORDING == "arm/video/state/recording"
  assert topics.RETAIN_RECORDING is True
  # Per-recording upload progress is under the upload prefix; backlog gauges are
  # exact siblings (so an exact-match subscriber wins over the wildcard).
  assert topics.upload_progress("abc") == "arm/video/upload/abc"
  assert topics.UPLOAD_BACKLOG_COUNT == "arm/video/upload/backlog_count"
  assert topics.UPLOAD_BACKLOG_OLDEST == "arm/video/upload/oldest_pending_seconds"
  assert topics.QOS_UPLOAD == 1
  assert topics.RETAIN_UPLOAD is True


def test_upload_gate_topic():
  # §2.12: the bandwidth coordinator's gate, retained so a (re)connecting
  # uploader sees the current value immediately.
  assert topics.CMD_UPLOAD_GATE == "arm/video/cmd/upload/gate"
  assert topics.RETAIN_UPLOAD_GATE is True


def test_slice5_vision_audio_topics():
  # Plan §7 X5.1: vision/audio analysis topic map.
  assert topics.VISION_HEARTBEAT == "arm/video/vision/heartbeat"
  assert topics.VISION_SUMMARY == "arm/video/vision/summary"
  assert topics.VISION_EVENT_SAFE_BOX == "arm/video/vision/events/safe-box"
  assert topics.VISION_EVENT_SAFE_BOX_CLEARED == "arm/video/vision/events/safe-box_cleared"
  assert topics.VISION_EVENT_REST == "arm/video/vision/events/rest-violation"
  assert topics.VISION_EVENT_REST_CLEARED == "arm/video/vision/events/rest_violation_cleared"
  assert topics.AUDIO_LEVEL == "arm/video/audio/level"
  assert topics.AUDIO_EVENT_SPIKE == "arm/video/audio/events/spike"
  assert topics.AUDIO_EVENT_SPIKE_CLEARED == "arm/video/audio/events/spike_cleared"
  assert topics.EVENT_ANOMALY_RECORDING == "arm/video/events/anomaly-recording"
  # Only the live audio meter is retained (the value the dashboard wants on
  # connect); summaries/heartbeats/events are not (no stale anomaly on connect).
  assert topics.RETAIN_AUDIO_LEVEL is True
  assert topics.QOS_AUDIO_LEVEL == 0
  assert topics.QOS_VISION_SUMMARY == 1
  assert topics.QOS_VISION_EVENT == 1
  assert topics.QOS_AUDIO_EVENT == 1
  assert topics.QOS_ANOMALY_RECORDING == 1
