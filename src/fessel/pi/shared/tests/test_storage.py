"""SSD storage layout + metadata + upload-marker contract tests (V4.4/V4.6)."""

from fessel_schemas import (
  AnomalyRecordingMetadata,
  AnomalyType,
  RecordingMetadata,
  UploadStateValue,
)
from fessel_shared import Storage


def _meta(rid: str, started: str, **kw) -> RecordingMetadata:
  return RecordingMetadata(id=rid, started_at=started, **kw)


def _anomaly_meta(aid: str, started: str, **kw) -> AnomalyRecordingMetadata:
  kw.setdefault("anomaly_event_type", AnomalyType.safe_box_violation)
  kw.setdefault("trigger_ts", started)
  return AnomalyRecordingMetadata(id=aid, started_at=started, **kw)


def test_ensure_layout_creates_dirs(tmp_path):
  s = Storage(str(tmp_path))
  s.ensure_layout()
  assert s.ring_dir.is_dir()
  assert s.explicit_dir.is_dir()
  assert s.anomaly_dir.is_dir()
  assert s.upload_queue_dir.is_dir()
  # The documented architecture paths.
  assert s.ring_dir == tmp_path / "ring"
  assert s.explicit_dir == tmp_path / "recordings" / "explicit"
  assert s.anomaly_dir == tmp_path / "recordings" / "anomaly"
  assert s.upload_queue_dir == tmp_path / "upload_queue"


def test_write_then_read_metadata_roundtrip(tmp_path):
  s = Storage(str(tmp_path))
  meta = _meta("r1", "2026-06-04T00:00:00+00:00", operator="octocat", segments=5)
  s.write_metadata(meta)
  got = s.read_metadata("r1")
  assert got is not None
  assert got.id == "r1" and got.operator == "octocat" and got.segments == 5
  # Defaults survive the round-trip.
  assert got.flagged_for_upload is False
  assert got.upload_state is UploadStateValue.none


def test_write_metadata_is_atomic_no_tmp_left(tmp_path):
  s = Storage(str(tmp_path))
  s.write_metadata(_meta("r1", "2026-06-04T00:00:00+00:00"))
  # No leftover .tmp after an atomic replace.
  files = {p.name for p in s.recording_dir("r1").iterdir()}
  assert "metadata.json" in files
  assert not any(f.endswith(".tmp") for f in files)


def test_read_metadata_missing_or_partial_returns_none(tmp_path):
  s = Storage(str(tmp_path))
  # Never written -> None (a partial recording that never finalised, V4.3).
  assert s.read_metadata("ghost") is None
  # A corrupt metadata file -> None, not a crash.
  d = s.recording_dir("bad")
  d.mkdir(parents=True)
  (d / "metadata.json").write_text("{ not json")
  assert s.read_metadata("bad") is None


def test_list_recordings_sorted_newest_first_skips_partial(tmp_path):
  s = Storage(str(tmp_path))
  s.write_metadata(_meta("old", "2026-06-04T00:00:00+00:00"))
  s.write_metadata(_meta("new", "2026-06-04T02:00:00+00:00"))
  # A dir with no metadata (partial) is skipped.
  s.recording_dir("partial").mkdir(parents=True)
  ids = [m.id for m in s.list_recordings()]
  assert ids == ["new", "old"]


def test_count_segments(tmp_path):
  s = Storage(str(tmp_path))
  d = s.recording_dir("r1")
  d.mkdir(parents=True)
  (d / "seg-00000.ts").write_bytes(b"a")
  (d / "seg-00001.ts").write_bytes(b"b")
  (d / "index.m3u8").write_text("#EXTM3U")  # not a segment
  assert s.count_segments("r1") == 2
  assert s.count_segments("ghost") == 0


def test_anomaly_metadata_roundtrip_and_listing(tmp_path):
  # Slice 5: anomaly recordings live under recordings/anomaly/<id>/ with their
  # own metadata shape; the listing mirrors list_recordings (newest first,
  # partials skipped).
  s = Storage(str(tmp_path))
  s.write_anomaly_metadata(
    _anomaly_meta("a-old", "2026-06-05T00:00:00+00:00", duration_seconds=90, segments=45)
  )
  s.write_anomaly_metadata(
    _anomaly_meta(
      "a-new",
      "2026-06-05T02:00:00+00:00",
      anomaly_event_type=AnomalyType.audio_spike,
    )
  )
  s.anomaly_recording_dir("a-partial").mkdir(parents=True)  # no metadata -> skipped
  got = s.read_anomaly_metadata("a-old")
  assert got is not None and got.type == "anomaly" and got.duration_seconds == 90
  ids = [m.id for m in s.list_anomaly_recordings()]
  assert ids == ["a-new", "a-old"]
  assert s.read_anomaly_metadata("ghost") is None


def test_count_anomaly_segments(tmp_path):
  s = Storage(str(tmp_path))
  d = s.anomaly_recording_dir("a1")
  d.mkdir(parents=True)
  (d / "seg-00000.ts").write_bytes(b"a")
  (d / "seg-00001.ts").write_bytes(b"b")
  (d / "index.m3u8").write_text("#EXTM3U")
  assert s.count_anomaly_segments("a1") == 2
  assert s.count_anomaly_segments("ghost") == 0


def test_upload_marker_lifecycle(tmp_path):
  s = Storage(str(tmp_path))
  s.ensure_layout()
  s.create_upload_marker("r1")
  assert s.upload_marker_path("r1").exists()
  assert s.upload_marker_path("r1").read_text() == ""  # empty: presence is the signal
  # Idempotent re-flag.
  s.create_upload_marker("r1")
  assert [p.name for p in s.list_upload_markers()] == ["r1.upload"]
  # .failed markers are NOT in the pending work list.
  s.upload_failed_path("r2").write_text("")
  assert [p.name for p in s.list_upload_markers()] == ["r1.upload"]
