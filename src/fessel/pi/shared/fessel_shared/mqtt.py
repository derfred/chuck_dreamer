"""Thin wrapper over paho-mqtt for the Pi-internal bus.

Both video and supervisor use this. It is intentionally small: connect,
publish JSON, subscribe with a typed callback, and a background loop. The
broker is always localhost (MQTT is Pi-internal only).

Two subscribe flavours:

  - `subscribe`        — hands the raw decoded dict to the handler. Use for
                         schemaless topics (heartbeat) or topics whose payload
                         model does not exist yet.
  - `subscribe_model`  — validates the decoded dict through a parser
                         (`Model.model_validate`) before dispatch, so the
                         handler receives a validated object, never a raw
                         dict. A payload that fails validation is logged once
                         and dropped — it never reaches the handler.

`subscribe_model` takes a plain `parser: Callable[[dict], T]` rather than a
pydantic type so this module keeps zero schema dependency (the dependency
direction stays fessel_schemas -> consumers, never the reverse).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, TypeVar

import paho.mqtt.client as mqtt

log = logging.getLogger(__name__)

MessageHandler = Callable[[str, dict[str, Any]], None]

T = TypeVar("T")
# A validator: dict -> validated object. `pydantic.BaseModel.model_validate`
# is exactly this shape, so callers pass `MyModel.model_validate`.
Parser = Callable[[dict[str, Any]], T]
ModelHandler = Callable[[str, T], None]


class MqttClient:
  def __init__(self, client_id: str, host: str = "127.0.0.1", port: int = 1883) -> None:
    self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
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

  def subscribe_model(
    self, topic: str, parser: Parser[T], handler: ModelHandler[T], qos: int = 0
  ) -> None:
    """Subscribe with schema validation.

    `parser` (typically `SomeModel.model_validate`) is applied to the decoded
    dict; the handler receives the validated object. A payload that fails
    validation is logged once and dropped — it never reaches the handler, so
    a malformed message can't crash or mislead a subscriber. This is the
    consumer-side mirror of constructing-then-dumping a model on publish.
    """

    def _validating(t: str, payload: dict[str, Any]) -> None:
      try:
        obj = parser(payload)
      except Exception as e:  # noqa: BLE001 — model_validate raises ValidationError; stay parser-agnostic
        log.warning("dropping invalid payload on %s: %s", t, e)
        return
      handler(t, obj)

    self.subscribe(topic, _validating, qos=qos)

  def publish(
    self, topic: str, payload: dict[str, Any], qos: int = 0, retain: bool = False
  ) -> None:
    self._client.publish(topic, json.dumps(payload), qos=qos, retain=retain)
