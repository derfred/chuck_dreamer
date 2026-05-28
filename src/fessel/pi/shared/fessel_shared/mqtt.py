"""Thin wrapper over paho-mqtt for the Pi-internal bus.

Both video and supervisor use this. It is intentionally small: connect,
publish JSON, subscribe with a typed callback, and a background loop. The
broker is always localhost (MQTT is Pi-internal only).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import paho.mqtt.client as mqtt

log = logging.getLogger(__name__)

MessageHandler = Callable[[str, dict[str, Any]], None]


class MqttClient:
  def __init__(self, client_id: str, host: str = "127.0.0.1", port: int = 1883) -> None:
    self._client = mqtt.Client(
      mqtt.CallbackAPIVersion.VERSION2, client_id=client_id
    )
    self._host = host
    self._port = port
    self._handlers: dict[str, tuple[MessageHandler, int]] = {}
    self._client.on_connect = self._on_connect
    self._client.on_message = self._on_message

  def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:  # noqa: ANN001
    if reason_code != 0:
      log.error("mqtt connect failed: %s", reason_code)
      return
    log.info("mqtt connected to %s:%d", self._host, self._port)
    # Re-subscribe on (re)connect so subscriptions survive a broker blip.
    for topic, (_, qos) in self._handlers.items():
      client.subscribe(topic, qos=qos)

  def _on_message(self, client, userdata, msg) -> None:  # noqa: ANN001
    handler = self._handlers.get(msg.topic)
    if handler is None:
      # Match wildcard subscriptions registered against their pattern.
      for pattern, (h, _) in self._handlers.items():
        if mqtt.topic_matches_sub(pattern, msg.topic):
          handler = (h, 0)
          break
    if handler is None:
      return
    try:
      payload = json.loads(msg.payload) if msg.payload else {}
    except json.JSONDecodeError:
      log.warning("non-JSON payload on %s; ignoring", msg.topic)
      return
    handler[0](msg.topic, payload)

  def connect(self) -> None:
    self._client.connect(self._host, self._port)

  def loop_start(self) -> None:
    self._client.loop_start()

  def loop_stop(self) -> None:
    self._client.loop_stop()

  def loop_forever(self) -> None:
    self._client.loop_forever()

  def disconnect(self) -> None:
    self._client.disconnect()

  def subscribe(self, topic: str, handler: MessageHandler, qos: int = 0) -> None:
    self._handlers[topic] = (handler, qos)
    self._client.subscribe(topic, qos=qos)

  def publish(self, topic: str, payload: dict[str, Any], qos: int = 0, retain: bool = False) -> None:
    self._client.publish(topic, json.dumps(payload), qos=qos, retain=retain)
