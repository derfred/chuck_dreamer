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
