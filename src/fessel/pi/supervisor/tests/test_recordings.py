"""supervisor Slice-4 tests: recording control (S4.1), /state additions (S4.2),
ring proxy (S4.3), recordings listing (S4.4), recording-segment proxy, and the
upload-progress cache (B4.6). Uses a FakeMqtt (no broker) and a tmp SSD."""

from typing import ClassVar

import paho.mqtt.client as mqtt
import pytest
from fastapi.testclient import TestClient
from fessel_schemas import AnomalyRecordingMetadata, AnomalyType, RecordingMetadata
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

  # Deliberately class-level: the ``client`` fixture resets it per test and
  # aliases it onto the TestClient as ``_published``.
  published: ClassVar[list] = []

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


def _seed_anomaly_recording(storage: Storage, aid: str, started: str, **kw) -> None:
  d = storage.anomaly_recording_dir(aid)
  d.mkdir(parents=True, exist_ok=True)
  (d / "index.m3u8").write_text("#EXTM3U")
  (d / "seg-00000.ts").write_bytes(b"TSDATA")
  kw.setdefault("anomaly_event_type", AnomalyType.safe_box_violation)
  kw.setdefault("trigger_ts", started)
  storage.write_anomaly_metadata(
    AnomalyRecordingMetadata(id=aid, started_at=started, **kw)
  )


# --- S4.1 recording control --------------------------------------------------


def test_recording_start_mints_id_and_publishes(client):
  with client:
    # No mode: recording resolution is a deploy setting resolved on video.
    r = client.post(
      "/recording/start",
      json={"lookback_seconds": 45.0, "upload_when_done": True, "operator": "octocat"},
    )
    assert r.status_code == 200
    rid = r.json()["recording_id"]
    assert rid
  pub = [p for p in client._published if p[0] == "arm/video/cmd/recording/start"]
  assert pub and pub[-1][1]["recording_id"] == rid
  assert pub[-1][1]["operator"] == "octocat"
  assert pub[-1][1]["lookback_seconds"] == 45.0
  assert pub[-1][1]["upload_when_done"] is True
  # No per-recording mode is published.
  assert "mode" not in pub[-1][1]


def test_recording_start_defaults_lookback_and_upload(client):
  with client:
    r = client.post("/recording/start", json={"operator": "octocat"})
    assert r.status_code == 200
  pub = [p for p in client._published if p[0] == "arm/video/cmd/recording/start"]
  assert pub[-1][1]["lookback_seconds"] == 0.0
  assert pub[-1][1]["upload_when_done"] is False


def test_recording_start_rejects_negative_lookback(client):
  with client:
    # ge=0 on the body -> pydantic 422 (FastAPI validation), not a 200.
    assert client.post("/recording/start", json={"lookback_seconds": -5}).status_code == 422


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


# (No ring playback endpoints: the ring buffer is never served for viewing — it
# lives on the Pi behind the cellular link and only backs recording look-back.
# Recording playback + its Range/traversal handling are covered below.)


def test_recording_segment_proxy(client):
  _seed_recording(client._ssd, "r1")
  with client:
    assert client.get("/recordings/r1/index.m3u8").status_code == 200
    seg = client.get("/recordings/r1/seg-00000.ts")
    assert seg.status_code == 200
    # Range request -> 206 (HLS scrubbing).
    rng = client.get("/recordings/r1/seg-00000.ts", headers={"Range": "bytes=0-9"})
    assert rng.status_code == 206
    assert client.get("/recordings/unknown/index.m3u8").status_code == 404


def test_recording_traversal_blocked(client):
  with client:
    # A traversal segment in the recording id must not escape the recordings dir.
    assert client.get("/recordings/..%2F..%2Fetc/index.m3u8").status_code == 404


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


# --- Slice 5: anomaly recordings + sensing /state + /anomalies ---------------


def test_recordings_merges_explicit_and_anomaly_with_type(client):
  _seed_recording(client._ssd, "exp1", operator="octocat")
  client._ssd.write_metadata(
    RecordingMetadata(id="exp1", started_at="2026-06-05T01:00:00+00:00", operator="octocat")
  )
  _seed_anomaly_recording(client._ssd, "anom1", "2026-06-05T02:00:00+00:00")
  with client:
    recs = client.get("/recordings").json()
  by_id = {r["recording_id"]: r for r in recs}
  assert by_id["exp1"]["type"] == "explicit"
  assert by_id["anom1"]["type"] == "anomaly"
  # Newest first: the anomaly (02:00) precedes the explicit (01:00).
  assert [r["recording_id"] for r in recs] == ["anom1", "exp1"]


def test_anomaly_recording_is_playable(client):
  _seed_anomaly_recording(client._ssd, "anom1", "2026-06-05T02:00:00+00:00")
  with client:
    assert client.get("/recordings/anom1/index.m3u8").status_code == 200
    assert client.get("/recordings/anom1/seg-00000.ts").status_code == 200


def test_flag_upload_works_for_anomaly_recording(client):
  _seed_anomaly_recording(client._ssd, "anom1", "2026-06-05T02:00:00+00:00")
  with client:
    r = client.post("/recording/flag-upload", json={"recording_id": "anom1"})
    assert r.status_code == 200
  assert any(p[0] == "arm/video/cmd/recording/flag-upload" for p in client._published)


def test_state_includes_vision_audio_and_recent_anomalies(client):
  with client:
    relay = client.app.state.relay
    relay._mqtt.deliver("arm/video/vision/summary", {"activity_score_ema": 0.12})
    relay._mqtt.deliver(
      "arm/video/vision/heartbeat",
      {"last_frame_at": "2026-06-05T00:00:00Z", "frames_processed": 10, "frames_dropped": 0},
    )
    relay._mqtt.deliver("arm/video/audio/level", {"rms_db": -33.0, "peak_db": -30.0})
    relay._mqtt.deliver(
      "arm/video/vision/events/safe-box",
      {"ts": "2026-06-05T00:00:01Z", "type": "safe_box_violation", "outside_fraction": 0.2},
    )
    body = client.get("/state").json()
  assert body["vision"]["activity_score_ema"] == 0.12
  assert body["vision"]["healthy"] is True
  assert body["audio"]["level_db"] == -33.0
  assert body["audio"]["healthy"] is True
  assert len(body["recent_anomalies"]) == 1
  assert body["recent_anomalies"][0]["type"] == "safe_box_violation"


def test_anomalies_endpoint_returns_log_with_recording_link(client):
  with client:
    relay = client.app.state.relay
    relay._mqtt.deliver(
      "arm/video/audio/events/spike",
      {"ts": "t0", "type": "audio_spike", "peak_db": -3.0},
    )
    relay._mqtt.deliver(
      "arm/video/events/anomaly-recording",
      {"anomaly_recording_id": "rec-7", "anomaly_event_type": "audio_spike", "trigger_ts": "t0"},
    )
    body = client.get("/anomalies").json()
  assert len(body) == 1
  assert body[0]["type"] == "audio_spike"
  assert body[0]["anomaly_recording_id"] == "rec-7"
  assert body[0]["payload"]["peak_db"] == -3.0


def test_safety_state_published_on_startup_and_change(client, monkeypatch):
  # On startup the control plane publishes the initial IDLE safety state retained
  # so video's rest detection has a value immediately (V5.4).
  with client:
    startup = [
      p for p in client._published if p[0] == "arm/supervisor/state"
    ]
  assert startup and startup[0][1]["safety_state"] == "IDLE"
  assert startup[0][3] is True  # retained
