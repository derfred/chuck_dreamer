"""video process entrypoint.

Wires together:
  - capability probing -> retained publish on arm/video/capabilities (V1.3)
  - MQTT command handler for activate/deactivate (V1.5)
  - the live state machine (V1.4), which owns the GStreamer pipeline
  - heartbeat (Slice 0)

Config (video.yaml) keys:
  mqtt: {host, port}
  srt:  {host, port}        # mediamtx Tailscale magicDNS name + 8890
  camera: {device, use_test_source}
  live: {connect_timeout_s}
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time

from fessel_schemas import (
  Capabilities,
  LiveActivate,
  LiveDeactivate,
  LiveStateValue,
  ModeTriplet,
)
from fessel_shared import MqttClient, load_yaml_config
from fessel_shared import topics

from .capabilities import probe
from .gst_pipeline_handle import GstLivePipeline
from .state_machine import LiveStateMachine

log = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 5.0
CONFIG_PATH = os.environ.get("VIDEO_CONFIG", "/etc/fessel/video.yaml")


class VideoApp:
  def __init__(self, config: dict) -> None:
    self._cfg = config
    mqtt_cfg = config.get("mqtt", {})
    self._mqtt = MqttClient(
      client_id="video",
      host=mqtt_cfg.get("host", "127.0.0.1"),
      port=mqtt_cfg.get("port", 1883),
    )

    srt_cfg = config.get("srt", {})
    cam_cfg = config.get("camera", {})
    live_cfg = config.get("live", {})
    self._srt_host = srt_cfg.get("host", "mediamtx-srt")
    self._srt_port = srt_cfg.get("port", 8890)
    self._device = cam_cfg.get("device", "/dev/video0")
    self._use_test_source = bool(cam_cfg.get("use_test_source", False))

    self._sm = LiveStateMachine(
      pipeline_factory=self._make_pipeline,
      publish_state=self._publish_live_state,
      connect_timeout_s=float(live_cfg.get("connect_timeout_s", 15.0)),
    )
    self._stop = threading.Event()

  def _make_pipeline(self, mode: ModeTriplet, path: str) -> GstLivePipeline:
    return GstLivePipeline(
      sm=self._sm,
      mode=mode,
      path=path,
      srt_host=self._srt_host,
      srt_port=self._srt_port,
      device=self._device,
      use_test_source=self._use_test_source,
    )

  def _publish_live_state(
    self, state: LiveStateValue, path: str | None, mode: ModeTriplet | None
  ) -> None:
    from fessel_schemas import LiveState

    payload = LiveState(state=state, path=path, mode=mode)
    self._mqtt.publish(
      topics.STATE_LIVE,
      payload.model_dump(),
      qos=topics.QOS_STATE,
      retain=topics.RETAIN_STATE,
    )

  def _on_activate(self, _topic: str, payload: dict) -> None:
    try:
      cmd = LiveActivate.model_validate(payload)
    except Exception:  # noqa: BLE001
      log.exception("bad activate payload: %s", payload)
      return
    log.info("activate path=%s mode=%s", cmd.path, cmd.mode)
    self._sm.request_activate(cmd.mode, cmd.path)

  def _on_deactivate(self, _topic: str, payload: dict) -> None:
    try:
      cmd = LiveDeactivate.model_validate(payload)
    except Exception:  # noqa: BLE001
      log.exception("bad deactivate payload: %s", payload)
      return
    log.info("deactivate path=%s", cmd.path)
    self._sm.request_deactivate(cmd.path)

  def _publish_capabilities(self) -> None:
    modes = probe(self._device if not self._use_test_source else "")
    caps = Capabilities(modes=modes)
    self._mqtt.publish(
      topics.CAPABILITIES,
      caps.model_dump(),
      qos=topics.QOS_CAPABILITIES,
      retain=topics.RETAIN_CAPABILITIES,
    )
    log.info("published %d capability modes", len(modes))

  def run(self) -> None:
    self._mqtt.connect()
    self._mqtt.subscribe(topics.CMD_LIVE_ACTIVATE, self._on_activate, qos=topics.QOS_CMD)
    self._mqtt.subscribe(topics.CMD_LIVE_DEACTIVATE, self._on_deactivate, qos=topics.QOS_CMD)
    self._mqtt.loop_start()
    self._sm.start()
    self._publish_capabilities()

    signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
    signal.signal(signal.SIGINT, lambda *_: self._stop.set())

    while not self._stop.is_set():
      self._mqtt.publish(topics.HEARTBEAT, {"from": "video", "ts": time.time()}, qos=topics.QOS_HEARTBEAT)
      self._stop.wait(HEARTBEAT_INTERVAL_S)

    log.info("video shutting down")
    self._sm.shutdown()
    self._mqtt.loop_stop()
    self._mqtt.disconnect()


def main() -> None:
  logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
  )
  config = load_yaml_config(CONFIG_PATH)
  VideoApp(config).run()


if __name__ == "__main__":
  main()
