"""Camera-state publishing (§7.4).

The capture pipeline IS the camera, so video publishes a retained
arm/video/state/camera {up} on capture start (True) and clean shutdown
(False). supervisor caches it into /state.camera.up, which the webui health
gate consults before answering /whep — a silent topic left the fact unknown
and (before the gate fix) fast-rejected every live view in the test rig.

No GStreamer: _publish_camera_state only touches MQTT.
"""

from __future__ import annotations

from fessel_shared import topics

from video.main import VideoApp


class FakeMqtt:
  def __init__(self) -> None:
    self.published: list[tuple[str, dict, int, bool]] = []

  def publish(self, topic, payload, qos=0, retain=False):
    self.published.append((topic, payload, qos, retain))


def _app() -> tuple[VideoApp, FakeMqtt]:
  app = object.__new__(VideoApp)
  app._mqtt = FakeMqtt()
  return app, app._mqtt


def test_camera_state_publishes_retained_up():
  app, mqtt = _app()
  app._publish_camera_state(up=True)
  topic, payload, qos, retain = mqtt.published[0]
  assert topic == topics.STATE_CAMERA
  assert payload == {"up": True}
  assert qos == topics.QOS_STATE
  assert retain is True


def test_camera_state_publishes_down_on_shutdown():
  app, mqtt = _app()
  app._publish_camera_state(up=False)
  assert mqtt.published[0][1] == {"up": False}
