"""Anomaly-recording extraction tests (V5.7).

A fake forward-capture handle stands in for the GStreamer branch, and a tmp ring
dir with mtime-stamped segments exercises the copy-from-ring + coalesce +
metadata logic without GStreamer.
"""

import json
import os

from fessel_schemas import AnomalyType

from video.anomaly_recording import (
  AnomalyRecorder,
  AnomalyWindow,
  latest_ring_segment,
  select_ring_segments,
)


class FakeForward:
  """Records calls; fires on_closed only when the test asks it to."""

  def __init__(self, rec_dir, after_seconds, on_closed):
    self.rec_dir = rec_dir
    self.after_seconds = after_seconds
    self._on_closed = on_closed
    self.started = False
    self.extends: list[float] = []
    self.stopped = False

  def start(self):
    self.started = True
    # Simulate the forward capture writing a couple of fresh segments.
    (self.rec_dir / "seg-90000.ts").write_bytes(b"fwd0")
    (self.rec_dir / "seg-90001.ts").write_bytes(b"fwd1")

  def extend(self, extra_seconds):
    self.extends.append(extra_seconds)

  def stop(self):
    self.stopped = True

  def close(self):
    self._on_closed()


def _seed_ring(ring_dir, count, base_mtime):
  ring_dir.mkdir(parents=True, exist_ok=True)
  for i in range(count):
    p = ring_dir / f"seg-{i:05d}.ts"
    p.write_bytes(bytes([i]))
    # Space segments 2s apart, ending at base_mtime.
    mtime = base_mtime - (count - 1 - i) * 2
    os.utime(p, (mtime, mtime))


def _make_recorder(
  tmp_path, *, window=None, fakes=None, published=None, auto_upload=False, flagged=None
):
  ring = tmp_path / "ring"
  anomaly = tmp_path / "recordings" / "anomaly"
  fakes = fakes if fakes is not None else []
  published = published if published is not None else []
  flagged = flagged if flagged is not None else []

  def factory(rec_dir, after_seconds, on_closed):
    f = FakeForward(rec_dir, after_seconds, on_closed)
    fakes.append(f)
    return f

  ids = iter(["anom-1", "anom-2", "anom-3"])
  rec = AnomalyRecorder(
    ring_dir=ring,
    anomaly_dir=anomaly,
    forward_capture_factory=factory,
    publish_event=lambda aid, etype, ts: published.append((aid, etype, ts)),
    window=window or AnomalyWindow(before_seconds=30, after_seconds=60),
    count_segments=lambda aid: (
      sum(1 for p in (anomaly / aid).iterdir() if p.suffix == ".ts")
      if (anomaly / aid).is_dir()
      else 0
    ),
    auto_upload=auto_upload,
    flag_for_upload=lambda aid: flagged.append(aid),
    now=lambda: 1000.0,
    new_id=lambda: next(ids),
  )
  return rec, ring, anomaly, fakes, published, flagged


def test_select_ring_segments_by_mtime(tmp_path):
  ring = tmp_path / "ring"
  _seed_ring(ring, 10, base_mtime=1000.0)  # segments at t=982,984,...,1000
  # Window [970, 1000] should pick the segments with mtime in range.
  segs = select_ring_segments(ring, 970.0, 1000.0)
  names = [p.name for p in segs]
  # 982..1000 = segments 0..9 (all within), ordered by write time.
  assert names == [f"seg-{i:05d}.ts" for i in range(10)]


def test_latest_ring_segment(tmp_path):
  ring = tmp_path / "ring"
  _seed_ring(ring, 5, base_mtime=1000.0)
  assert latest_ring_segment(ring) == "seg-00004.ts"
  assert latest_ring_segment(tmp_path / "nope") is None


def test_trigger_copies_ring_and_spawns_forward(tmp_path):
  rec, ring, anomaly, fakes, published, _ = _make_recorder(tmp_path)
  _seed_ring(ring, 20, base_mtime=1000.0)  # 40s of 2s segments ending at now=1000
  aid = rec.trigger(AnomalyType.safe_box_violation, "2026-06-05T00:00:00Z", {"x": 1})
  assert aid == "anom-1"
  # Forward capture spawned and started.
  assert len(fakes) == 1 and fakes[0].started
  rec_dir = anomaly / "anom-1"
  # Before-segments copied from the ring (window=30s -> ~15 of the 20 segments).
  copied = sorted(
    p.name for p in rec_dir.iterdir() if p.suffix == ".ts" and not p.name.startswith("seg-9")
  )
  assert len(copied) >= 15
  # Not finalised until the forward capture closes.
  assert not (rec_dir / "metadata.json").exists()
  assert published == []


def test_finalise_writes_metadata_and_publishes(tmp_path):
  rec, ring, anomaly, fakes, published, _ = _make_recorder(tmp_path)
  _seed_ring(ring, 20, base_mtime=1000.0)
  rec.trigger(AnomalyType.audio_spike, "2026-06-05T00:00:00Z", {"peak_db": -3})
  fakes[0].close()  # forward capture finishes
  rec_dir = anomaly / "anom-1"
  meta = json.loads((rec_dir / "metadata.json").read_text())
  assert meta["type"] == "anomaly"
  assert meta["anomaly_event_type"] == "audio_spike"
  assert meta["operator"] is None
  assert meta["flagged_for_upload"] is False
  assert meta["segments"] >= 17  # before (~15) + 2 forward
  assert len(meta["trigger_events"]) == 1
  # Combined VOD playlist references all segments in order, with an end tag.
  pl = (rec_dir / "index.m3u8").read_text()
  assert "#EXT-X-ENDLIST" in pl
  assert "seg-90000.ts" in pl
  assert published == [("anom-1", AnomalyType.audio_spike, "2026-06-05T00:00:00Z")]


def test_coalesce_extends_instead_of_new_recording(tmp_path):
  rec, ring, anomaly, fakes, published, _ = _make_recorder(tmp_path)
  _seed_ring(ring, 20, base_mtime=1000.0)
  a1 = rec.trigger(AnomalyType.safe_box_violation, "t0", {"a": 1})
  a2 = rec.trigger(AnomalyType.rest_violation, "t1", {"b": 2})  # during capture
  # Same recording id; the second trigger extended the forward window.
  assert a1 == a2 == "anom-1"
  assert len(fakes) == 1  # no second forward capture spawned
  assert fakes[0].extends == [60]  # extended by after_seconds
  # Finalise: both triggers recorded in metadata.
  fakes[0].close()
  meta = json.loads((anomaly / "anom-1" / "metadata.json").read_text())
  assert [e["type"] for e in meta["trigger_events"]] == [
    "safe_box_violation",
    "rest_violation",
  ]
  # Only one publish (one recording).
  assert len(published) == 1


def test_next_anomaly_after_close_is_a_new_recording(tmp_path):
  rec, ring, anomaly, fakes, published, _ = _make_recorder(tmp_path)
  _seed_ring(ring, 10, base_mtime=1000.0)
  rec.trigger(AnomalyType.safe_box_violation, "t0", {})
  fakes[0].close()
  rec.trigger(AnomalyType.audio_spike, "t1", {})
  assert len(fakes) == 2  # a fresh forward capture for the second
  assert published[0][0] == "anom-1"


def test_auto_upload_flags_when_enabled(tmp_path):
  rec, ring, anomaly, fakes, published, flagged = _make_recorder(tmp_path, auto_upload=True)
  _seed_ring(ring, 10, base_mtime=1000.0)
  rec.trigger(AnomalyType.safe_box_violation, "t0", {})
  fakes[0].close()
  assert flagged == ["anom-1"]


def test_no_auto_upload_by_default(tmp_path):
  rec, ring, anomaly, fakes, published, flagged = _make_recorder(tmp_path)
  _seed_ring(ring, 10, base_mtime=1000.0)
  rec.trigger(AnomalyType.safe_box_violation, "t0", {})
  fakes[0].close()
  assert flagged == []
