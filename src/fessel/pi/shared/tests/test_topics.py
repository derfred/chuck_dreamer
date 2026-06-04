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
