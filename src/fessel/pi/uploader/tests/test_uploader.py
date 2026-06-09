"""Uploader core tests (V4.6/V4.7/V4.8/V5.5.1): the directory-watch upload
pipeline against a tmp SSD + an in-memory FakeIngestClient. No MQTT, no real
backend."""

from fessel_schemas import RecordingMetadata, UploadStateValue
from fessel_shared import Storage
from uploader import FakeIngestClient, Uploader


def _setup(tmp_path, *, segments=2, with_metadata=True):
  s = Storage(str(tmp_path))
  s.ensure_layout()
  rid = "rec-1"
  d = s.recording_dir(rid)
  d.mkdir(parents=True)
  (d / "index.m3u8").write_text("#EXTM3U")
  for i in range(segments):
    (d / f"seg-{i:05d}.ts").write_bytes(b"X" * 10)
  if with_metadata:
    s.write_metadata(
      RecordingMetadata(
        id=rid,
        started_at="2026-06-04T00:00:00+00:00",
        flagged_for_upload=True,
        upload_state=UploadStateValue.queued,
      )
    )
  s.create_upload_marker(rid)
  return s, rid


def _publish_sink():
  pubs = []
  return pubs, lambda t, p, q, r: pubs.append((t, p))


def test_happy_path_uploads_all_files_and_clears_marker(tmp_path):
  s, rid = _setup(tmp_path)
  pubs, publish = _publish_sink()
  client = FakeIngestClient()
  up = Uploader(storage=s, client=client, publish=publish)

  outs = up.process_pending()
  assert len(outs) == 1 and outs[0].state is UploadStateValue.uploaded
  # Marker deleted; metadata advanced to uploaded with a timestamp.
  assert not s.upload_marker_path(rid).exists()
  meta = s.read_metadata(rid)
  assert meta.upload_state is UploadStateValue.uploaded and meta.uploaded_at
  # Every file PUT to the ingest endpoint, keyed by (recording_id, file_name).
  assert set(client.objects) == {
    (rid, "index.m3u8"),
    (rid, "metadata.json"),
    (rid, "seg-00000.ts"),
    (rid, "seg-00001.ts"),
  }
  # Progress was published with a retained "uploaded".
  states = [payload["state"] for topic, payload in pubs if topic == f"arm/video/upload/{rid}"]
  assert "uploaded" in states


def test_completion_order_metadata_last(tmp_path):
  # V5.5.1: segments first, then the playlist, then metadata.json LAST —
  # metadata.json's arrival is the backend's "fully uploaded" marker.
  s, rid = _setup(tmp_path, segments=2)
  _, publish = _publish_sink()
  client = FakeIngestClient()
  up = Uploader(storage=s, client=client, publish=publish)

  up.process_pending()
  names = [name for _rid, name in client.put_order]
  # Segments come before the playlist; metadata.json is strictly last.
  assert names[-1] == "metadata.json"
  assert names.index("index.m3u8") > names.index("seg-00001.ts")
  assert names.index("index.m3u8") < names.index("metadata.json")


def test_retryable_keeps_marker_then_succeeds(tmp_path):
  s, rid = _setup(tmp_path)
  _, publish = _publish_sink()
  client = FakeIngestClient(fail_n_then_ok=1)
  up = Uploader(storage=s, client=client, publish=publish)

  o1 = up.process_pending()
  assert o1[0].state is UploadStateValue.uploading  # transient -> stays
  assert s.upload_marker_path(rid).exists()
  assert s.read_metadata(rid).upload_state is UploadStateValue.uploading

  o2 = up.process_pending()
  assert o2[0].state is UploadStateValue.uploaded
  assert not s.upload_marker_path(rid).exists()


def test_backoff_increases_and_caps():
  up = Uploader(storage=None, client=None, publish=lambda *a: None, backoff_s=[30, 60, 300])
  assert up.backoff_for(1) == 30
  assert up.backoff_for(2) == 60
  assert up.backoff_for(3) == 300
  assert up.backoff_for(99) == 300  # capped at the last value


def test_permanent_failure_renames_marker_failed(tmp_path):
  s, rid = _setup(tmp_path)
  _, publish = _publish_sink()
  client = FakeIngestClient(permanent=True)
  up = Uploader(storage=s, client=client, publish=publish)

  o = up.process_pending()
  assert o[0].state is UploadStateValue.failed
  assert not s.upload_marker_path(rid).exists()
  assert s.upload_failed_path(rid).exists()
  assert s.read_metadata(rid).upload_state is UploadStateValue.failed


def test_missing_recording_dir_is_non_retryable(tmp_path):
  s = Storage(str(tmp_path))
  s.ensure_layout()
  # Marker with no recording directory -> .failed, not an infinite retry.
  s.create_upload_marker("ghost")
  _, publish = _publish_sink()
  up = Uploader(storage=s, client=FakeIngestClient(), publish=publish)
  o = up.process_pending()
  assert o[0].state is UploadStateValue.failed
  assert s.upload_failed_path("ghost").exists()


def test_gate_honoring(tmp_path):
  # §2.12: the uploader suspends scanning while the gate denies uploads, and
  # resumes when it re-allows. The marker stays on disk while paused (nothing
  # lost), then uploads on resume.
  from fessel_schemas import UploadGate
  from uploader.main import UploaderApp

  app = UploaderApp({"storage": {"ssd_path": str(tmp_path)}, "uploader": {"driver": "fake"}})
  # Default: allowed (data-safe until the retained gate arrives).
  assert app._allowed() is True
  # Gate denies -> uploads suspended.
  app._on_gate(None, UploadGate(uploads_allowed=False, reason="live viewer"))
  assert app._allowed() is False
  # Gate re-allows -> uploads resume.
  app._on_gate(None, UploadGate(uploads_allowed=True, reason="no viewer"))
  assert app._allowed() is True


def test_ingest_config_env_overrides(monkeypatch):
  # The integration env points the test-Pi's uploader at the in-cluster webui
  # ingest listener via env (the baked fessel.test.yaml defaults to driver:
  # fake), per the architecture's "config injected via env". Env overlays the
  # file block.
  from uploader.main import _ingest_config_with_env

  monkeypatch.setenv("FESSEL_INGEST_DRIVER", "http")
  monkeypatch.setenv("FESSEL_INGEST_URL_BASE", "http://webui:8001")
  monkeypatch.setenv("FESSEL_INGEST_VERIFY_TLS", "false")
  out = _ingest_config_with_env({"driver": "fake", "ingest_url_base": "old"})
  assert out["driver"] == "http"
  assert out["ingest_url_base"] == "http://webui:8001"
  assert out["verify_tls"] is False


def test_ingest_config_no_env_is_passthrough(monkeypatch):
  from uploader.main import _ingest_config_with_env

  for k in ("FESSEL_INGEST_DRIVER", "FESSEL_INGEST_URL_BASE", "FESSEL_INGEST_VERIFY_TLS"):
    monkeypatch.delenv(k, raising=False)
  assert _ingest_config_with_env({"driver": "fake"}) == {"driver": "fake"}


def test_reload_applies_upload_tunables(tmp_path):
  # §2.13: SIGHUP reload updates poll/backlog intervals + retry backoff.
  import pytest
  from uploader.main import UploaderApp

  app = UploaderApp({"storage": {"ssd_path": str(tmp_path)}, "uploader": {"driver": "fake"}})
  app.apply_config(
    {"upload": {"poll_interval_s": 3, "backlog_interval_s": 7, "backoff_s": [1, 2, 9]}}
  )
  assert app._poll_interval == 3.0
  assert app._backlog_interval == 7.0
  assert app._uploader.backoff_for(1) == 1
  assert app._uploader.backoff_for(99) == 9  # capped at last

  # validate rejects malformed tunables (reloader keeps old on raise).
  with pytest.raises(ValueError):
    UploaderApp.validate_config({"upload": {"poll_interval_s": 0}})
  with pytest.raises(ValueError):
    UploaderApp.validate_config({"upload": {"backoff_s": []}})
  # A good config validates clean.
  UploaderApp.validate_config({"upload": {"poll_interval_s": 5, "backoff_s": [30]}})


def test_backlog_gauges(tmp_path):
  s = Storage(str(tmp_path))
  s.ensure_layout()
  pubs, publish = _publish_sink()
  up = Uploader(storage=s, client=FakeIngestClient(), publish=publish)

  count, oldest = up.publish_backlog()
  assert count == 0 and oldest is None
  s.create_upload_marker("a")
  s.create_upload_marker("b")
  count, oldest = up.publish_backlog()
  assert count == 2 and oldest is not None and oldest >= 0
  # Both gauge topics were published.
  topics_seen = {t for t, _ in pubs}
  assert "arm/video/upload/backlog_count" in topics_seen
  assert "arm/video/upload/oldest_pending_seconds" in topics_seen
