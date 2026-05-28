"""supervisor control-plane tests with a fake MQTT relay."""

from unittest.mock import patch

from fastapi.testclient import TestClient


def make_client(monkeypatch):
  # Avoid touching a real broker: stub MqttClient methods used by Relay.
  import supervisor.app as appmod

  published: list = []

  class FakeMqtt:
    def __init__(self, *a, **k):
      pass

    def connect(self):
      pass

    def subscribe(self, topic, handler, qos=0):
      pass

    def loop_start(self):
      pass

    def loop_stop(self):
      pass

    def disconnect(self):
      pass

    def publish(self, topic, payload, qos=0, retain=False):
      published.append((topic, payload, qos, retain))

  monkeypatch.setattr(appmod, "MqttClient", FakeMqtt)
  app = appmod.create_app({"mqtt": {"host": "127.0.0.1"}})
  return TestClient(app), published


def test_healthz(monkeypatch):
  client, _ = make_client(monkeypatch)
  with client:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_activate_relays_to_mqtt(monkeypatch):
  client, published = make_client(monkeypatch)
  with client:
    r = client.post("/control/live/activate", json={"path": "pi", "mode": "1280x720@30@2500000"})
    assert r.status_code == 200
  topics = [p[0] for p in published]
  assert "arm/video/cmd/live/activate" in topics
  activate = next(p for p in published if p[0] == "arm/video/cmd/live/activate")
  # mode canonical string was parsed into the triplet object on the bus.
  assert activate[1]["mode"]["resolution"] == "1280x720"
  assert activate[1]["mode"]["bitrate_bps"] == 2_500_000
  assert activate[3] is False  # commands not retained


def test_activate_rejects_bad_mode(monkeypatch):
  client, _ = make_client(monkeypatch)
  with client:
    r = client.post("/control/live/activate", json={"path": "pi", "mode": "garbage"})
    assert r.status_code == 400


def test_activate_extracts_mode_from_raw_query(monkeypatch):
  # mediamtx runOnDemand posts the raw WHEP query; supervisor extracts mode.
  client, published = make_client(monkeypatch)
  with client:
    r = client.post(
      "/control/live/activate",
      json={"path": "pi", "query": "mode=640x480@30@1000000&token=abc.def.ghi"},
    )
    assert r.status_code == 200
  activate = next(p for p in published if p[0] == "arm/video/cmd/live/activate")
  assert activate[1]["mode"]["resolution"] == "640x480"


def test_deactivate_relays(monkeypatch):
  client, published = make_client(monkeypatch)
  with client:
    r = client.post("/control/live/deactivate", json={"path": "pi"})
    assert r.status_code == 200
  assert any(p[0] == "arm/video/cmd/live/deactivate" for p in published)


def test_state_live_tracks_history(monkeypatch):
  client, _ = make_client(monkeypatch)
  with client:
    relay = client.app.state.relay
    # Simulate retained live-state messages arriving from video.
    for st in ["off", "starting", "running", "starting", "off"]:
      relay._on_live_state("arm/video/state/live", {"state": st, "path": "pi"})
    r = client.get("/state/live")
    assert r.status_code == 200
    body = r.json()
    assert body["current"]["state"] == "off"
    # Consecutive duplicates collapse; transitions preserved in order.
    assert body["history"] == ["off", "starting", "running", "starting", "off"]
