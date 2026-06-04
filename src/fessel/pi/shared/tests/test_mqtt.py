"""subscribe_model validation tests.

The transport itself stays dumb (publish/decode JSON); subscribe_model adds the
consumer-side schema guard: a parser runs before dispatch, and a payload that
fails validation is dropped (logged once) instead of reaching the handler or
crashing the on-message callback. These tests drive the REAL decode->validate->
dispatch path via the paho on_message entry point with a fake message.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from fessel_shared import MqttClient


@dataclass
class FakeMsg:
  topic: str
  payload: bytes


def client_no_broker() -> MqttClient:
  # Constructing MqttClient builds a paho client but never connects; subscribe
  # touches self._client.subscribe (harmless offline), and we invoke the
  # decode/dispatch path directly via _on_message.
  return MqttClient(client_id="test")


def deliver(c: MqttClient, topic: str, body) -> None:
  payload = json.dumps(body).encode() if not isinstance(body, bytes) else body
  c._on_message(None, None, FakeMsg(topic, payload))


def parse_point(d: dict) -> tuple[int, int]:
  # A tiny stand-in validator with the (dict -> T) shape model_validate has;
  # raises on a missing/!int field, which is the failure subscribe_model drops.
  return (int(d["x"]), int(d["y"]))


def test_valid_payload_reaches_handler_as_parsed_object():
  got: list = []
  c = client_no_broker()
  c.subscribe_model("t/topic", parse_point, lambda _t, p: got.append(p))
  deliver(c, "t/topic", {"x": 1, "y": 2})
  assert got == [(1, 2)]


def test_invalid_payload_is_dropped_not_dispatched():
  got: list = []
  c = client_no_broker()
  c.subscribe_model("t/topic", parse_point, lambda _t, p: got.append(p))
  # Missing 'y' -> parser raises -> wrapper drops it; handler never called.
  deliver(c, "t/topic", {"x": 1})
  assert got == []
  # A subsequent valid message still lands (the drop didn't break the sub).
  deliver(c, "t/topic", {"x": 3, "y": 4})
  assert got == [(3, 4)]


def test_raw_subscribe_still_passes_dicts_through():
  got: list = []
  c = client_no_broker()
  c.subscribe("t/raw", lambda _t, p: got.append(p))
  deliver(c, "t/raw", {"any": "shape"})
  assert got == [{"any": "shape"}]
