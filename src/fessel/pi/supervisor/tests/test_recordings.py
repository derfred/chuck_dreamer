"""supervisor Slice-4 tests: recording control (S4.1), /state additions (S4.2),
ring proxy (S4.3), recordings listing (S4.4), recording-segment proxy, and the
upload-progress cache (B4.6). Uses a FakeMqtt (no broker) and a tmp SSD."""

import paho.mqtt.client as mqtt
import pytest
from fastapi.testclient import TestClient
from fessel_schemas import RecordingMetadata
from fessel_shared import MqttClient, Storage


class FakeMqtt(MqttClient):
  """Subclass that skips paho but keeps the real subscribe/subscribe_model
  dispatch, including wildcard matching, so the upload-progress topic routes
  exactly as production does."""

  def __init__(self, *a, **k):
    self._handlers = {}
    self._client = None

  def connect(self):
    pass

  def subscribe(self, topic, handler, qos=0):
    self._handlers[topic] = (handler, qos)

  def loop_start(self):
    pass

  def loop_stop(self):
    pass

  def disconnect(self):
    pass

  def publish(self, topic, payload, qos=0, retain=False):
    self.published.append((topic, payload, qos, retain))

  published: list = []

  def deliver(self, topic, payload):
    h = self._handlers.get(topic)
    if h is None:
      for pat, (hh, _) in self._handlers.items():
        if mqtt.topic_matches_sub(pat, topic):
          h = (hh, 0)
          break
    h[0](topic, payload)


@pytest.fixture
def client(tmp_path, monkeypatch):
  import supervisor.app as appmod

  FakeMqtt.published = []
  monkeypatch.setattr(appmod, "MqttClient", FakeMqtt)
  app = appmod.create_app({"mqtt": {"host": "127.0.0.1"}, "storage": {"ssd_path": str(tmp_path)}})
  c = TestClient(app)
  c._published = FakeMqtt.published  # type: ignore[attr-defined]
  c._ssd = Storage(str(tmp_path))  # type: ignore[attr-defined]
  c._ssd.ensure_layout()
  return c


def _seed_recording(storage: Storage, rid: str, **kw) -> None:
  d = storage.recording_dir(rid)
  d.mkdir(parents=True, exist_ok=True)
  (d / "index.m3u8").write_text("#EXTM3U")
  (d / "seg-00000.ts").write_bytes(b"TSDATA")
  storage.write_metadata(
    RecordingMetadata(id=rid, started_at="2026-06-04T00:00:00+00:00", **kw)
  )


# --- S4.1 recording control --------------------------------------------------


def test_recording_start_mints_id_and_publishes(client):
  with client:
    r = client.post("/recording/start", json={"mode": "1920x1080@30@8000000", "operator": "octocat"})
    assert r.status_code == 200
    rid = r.json()["recording_id"]
    assert rid
  pub = [p for p in client._published if p[0] == "arm/video/cmd/recording/start"]
  assert pub and pub[-1][1]["recording_id"] == rid
  assert pub[-1][1]["operator"] == "octocat"
  assert pub[-1][1]["mode"]["resolution"] == "1920x1080"


def test_recording_start_rejects_bad_mode(client):
  with client:
    assert client.post("/recording/start", json={"mode": "garbage"}).status_code == 400


def test_recording_stop_publishes(client):
  with client:
    assert client.post("/recording/stop").status_code == 200
  assert any(p[0] == "arm/video/cmd/recording/stop" for p in client._published)


def test_flag_upload_404_for_unknown(client):
  with client:
    assert client.post("/recording/flag-upload", json={"recording_id": "ghost"}).status_code == 404


def test_flag_upload_publishes_for_finalised(client):
  _seed_recording(client._ssd, "r1")
  with client:
    r = client.post("/recording/flag-upload", json={"recording_id": "r1"})
    assert r.status_code == 200
  assert any(p[0] == "arm/video/cmd/recording/flag-upload" for p in client._published)


def test_flag_upload_409_when_active(client):
  _seed_recording(client._ssd, "live1")
  with client:
    client.app.state.relay._mqtt.deliver(
      "arm/video/state/recording",
      {"state": "recording", "active_recording_id": "live1", "started_at": "t"},
    )
    r = client.post("/recording/flag-upload", json={"recording_id": "live1"})
    assert r.status_code == 409


# --- S4.2 /state additions ---------------------------------------------------


def test_state_includes_recording_and_backlog(client):
  with client:
    relay = client.app.state.relay
    relay._mqtt.deliver(
      "arm/video/state/recording",
      {"state": "recording", "active_recording_id": "rX", "started_at": "t"},
    )
    relay._mqtt.deliver("arm/video/upload/backlog_count", {"value": 4})
    relay._mqtt.deliver("arm/video/upload/oldest_pending_seconds", {"value": 99})
    body = client.get("/state").json()
  assert body["recording"]["state"] == "recording"
  assert body["recording"]["active_recording_id"] == "rX"
  assert body["upload_backlog"]["count"] == 4
  assert body["upload_backlog"]["oldest_pending_seconds"] == 99


def test_state_recording_defaults_idle(client):
  with client:
    body = client.get("/state").json()
  assert body["recording"]["state"] == "idle"
  assert body["upload_backlog"]["count"] == 0


# --- S4.4 recordings listing -------------------------------------------------


def test_list_recordings_newest_first(client):
  _seed_recording(client._ssd, "old", operator="a")
  _seed_recording(client._ssd, "new", operator="b")
  # Make "new" newer.
  client._ssd.write_metadata(
    RecordingMetadata(id="new", started_at="2026-06-04T05:00:00+00:00", operator="b")
  )
  with client:
    recs = client.get("/recordings").json()
  assert [r["recording_id"] for r in recs] == ["new", "old"]
  assert recs[0]["operator"] == "b"


# --- S4.3 ring proxy + traversal protection ----------------------------------


def test_ring_playlist_and_segment(client):
  (client._ssd.ring_dir / "index.m3u8").write_text("#EXTM3U\nseg-00000.ts\n")
  (client._ssd.ring_dir / "seg-00000.ts").write_bytes(b"X" * 50)
  with client:
    assert client.get("/ring/index.m3u8").status_code == 200
    seg = client.get("/ring/seg-00000.ts")
    assert seg.status_code == 200 and len(seg.content) == 50
    # Range request -> 206 (HLS scrubbing).
    rng = client.get("/ring/seg-00000.ts", headers={"Range": "bytes=0-9"})
    assert rng.status_code == 206


def test_ring_missing_playlist_is_404(client):
  with client:
    assert client.get("/ring/index.m3u8").status_code == 404


def test_ring_traversal_blocked(client):
  with client:
    assert client.get("/ring/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_recording_segment_proxy(client):
  _seed_recording(client._ssd, "r1")
  with client:
    assert client.get("/recordings/r1/index.m3u8").status_code == 200
    assert client.get("/recordings/r1/seg-00000.ts").status_code == 200
    assert client.get("/recordings/unknown/index.m3u8").status_code == 404


# --- B4.6 upload-progress cache ----------------------------------------------


def test_upload_progress_cache(client):
  with client:
    relay = client.app.state.relay
    relay._mqtt.deliver(
      "arm/video/upload/r1",
      {"recording_id": "r1", "state": "uploaded", "attempts": 2},
    )
    body = client.get("/recordings/r1/upload").json()
    assert body["state"] == "uploaded" and body["attempts"] == 2
    assert client.get("/recordings/none/upload").status_code == 404
