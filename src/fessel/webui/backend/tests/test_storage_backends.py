"""Storage backend tests (B5.5.1/B5.5.3): the disk backend + the factory + the
shared range parser. The MinIO backend is exercised via the integration env's
optional variant and the FakeStorageBackend's presigned mode (test_recordings_
api) rather than a real minio here."""

import json

import pytest

from app.storage import DiskStorageBackend, build_storage_backend
from app.storage.base import ServeLocally, parse_byte_range


# --- disk backend round trip -------------------------------------------------


def test_store_then_read_full(tmp_path):
  b = DiskStorageBackend(str(tmp_path))
  b.store("rec-1", "seg-00000.ts", [b"AB", b"CD", b"EF"])
  assert b.exists("rec-1")
  res = b.read("rec-1", "seg-00000.ts")
  assert b"".join(res.chunks) == b"ABCDEF"
  assert res.content_type == "video/mp2t"
  assert res.content_length == 6
  assert res.byte_range is None


def test_store_is_idempotent_overwrite(tmp_path):
  b = DiskStorageBackend(str(tmp_path))
  b.store("rec-1", "index.m3u8", [b"first"])
  b.store("rec-1", "index.m3u8", [b"second-longer"])  # a retried PUT
  res = b.read("rec-1", "index.m3u8")
  assert b"".join(res.chunks) == b"second-longer"


def test_read_range_206(tmp_path):
  b = DiskStorageBackend(str(tmp_path))
  b.store("rec-1", "seg-00000.ts", [b"0123456789"])
  res = b.read("rec-1", "seg-00000.ts", "bytes=2-5")
  assert b"".join(res.chunks) == b"2345"
  assert res.content_length == 4
  assert res.byte_range.start == 2 and res.byte_range.end == 5 and res.byte_range.total == 10


def test_read_missing_is_none(tmp_path):
  b = DiskStorageBackend(str(tmp_path))
  assert b.read("nope", "index.m3u8") is None
  assert b.playback_url("nope", "index.m3u8") is None
  assert b.exists("nope") is False


def test_playback_url_is_serve_locally(tmp_path):
  b = DiskStorageBackend(str(tmp_path))
  b.store("rec-1", "index.m3u8", [b"#EXTM3U"])
  t = b.playback_url("rec-1", "index.m3u8")
  assert isinstance(t, ServeLocally)
  assert t.recording_id == "rec-1" and t.file_name == "index.m3u8"


def test_list_parses_metadata_and_sorts_newest_first(tmp_path):
  b = DiskStorageBackend(str(tmp_path))
  b.store("old", "metadata.json", [json.dumps({"started_at": "2026-06-01T00:00:00+00:00"}).encode()])
  b.store("new", "metadata.json", [json.dumps({"started_at": "2026-06-09T00:00:00+00:00"}).encode()])
  b.store("partial", "seg-00000.ts", [b"x"])  # no metadata.json
  views = b.list()
  ids = [v.recording_id for v in views]
  # newest first; the partial (no started_at) sorts last.
  assert ids[0] == "new" and ids[1] == "old" and ids[-1] == "partial"
  partial = next(v for v in views if v.recording_id == "partial")
  assert partial.metadata is None  # in-progress


def test_anomaly_type_lands_in_anomaly_subdir(tmp_path):
  b = DiskStorageBackend(str(tmp_path))
  # metadata.json names the type; a file stored before it lands under explicit,
  # but once metadata says anomaly the recording is found under anomaly/.
  b.store("a1", "metadata.json", [json.dumps({"type": "anomaly", "started_at": "x"}).encode()])
  views = b.list()
  v = next(v for v in views if v.recording_id == "a1")
  assert v.rec_type == "anomaly"


# --- path safety (the load-bearing one) --------------------------------------


@pytest.mark.parametrize(
  "rid,name",
  [
    ("../etc", "passwd"),
    ("rec", "../../etc/passwd"),
    ("rec", ".."),
    ("..", "index.m3u8"),
    ("rec/evil", "index.m3u8"),
    ("rec", "a/b"),
    ("", "index.m3u8"),
    ("rec", ""),
  ],
)
def test_traversal_is_rejected(tmp_path, rid, name):
  b = DiskStorageBackend(str(tmp_path))
  with pytest.raises(ValueError):
    b.store(rid, name, [b"x"])
  # read/playback of a traversal id/name never escape: they return None, not a
  # file outside the root.
  assert b.read(rid, name) is None
  assert b.playback_url(rid, name) is None


# --- range parser ------------------------------------------------------------


def test_parse_byte_range_forms():
  assert parse_byte_range(None, 100) is None
  assert (parse_byte_range("bytes=0-3", 100).start, parse_byte_range("bytes=0-3", 100).end) == (0, 3)
  assert (parse_byte_range("bytes=10-", 100).start, parse_byte_range("bytes=10-", 100).end) == (10, 99)
  assert (parse_byte_range("bytes=-5", 100).start, parse_byte_range("bytes=-5", 100).end) == (95, 99)
  # end past EOF clamps.
  assert parse_byte_range("bytes=90-999", 100).end == 99


@pytest.mark.parametrize("bad", ["bytes=200-300", "bytes=50-10", "items=0-3", "bytes=0-3,5-9"])
def test_parse_byte_range_rejects(bad):
  with pytest.raises(ValueError):
    parse_byte_range(bad, 100)


# --- factory -----------------------------------------------------------------


def test_factory_disk(monkeypatch, tmp_path):
  monkeypatch.setenv("FESSEL_RECORDINGS_BACKEND", "disk")
  monkeypatch.setenv("FESSEL_RECORDINGS_DISK_PATH", str(tmp_path))
  assert isinstance(build_storage_backend(), DiskStorageBackend)


def test_factory_disk_missing_path(monkeypatch):
  monkeypatch.setenv("FESSEL_RECORDINGS_BACKEND", "disk")
  monkeypatch.delenv("FESSEL_RECORDINGS_DISK_PATH", raising=False)
  with pytest.raises(RuntimeError):
    build_storage_backend()


def test_factory_unconfigured_raises(monkeypatch):
  for k in ("FESSEL_RECORDINGS_BACKEND", "FESSEL_MINIO_ENDPOINT"):
    monkeypatch.delenv(k, raising=False)
  with pytest.raises(RuntimeError):
    build_storage_backend()
