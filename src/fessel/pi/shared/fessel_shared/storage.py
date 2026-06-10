"""USB-SSD storage layout + recording metadata + upload-queue contract (V4.4,
V4.6).

The architecture's on-disk layout (§2.8, X4.2):

  <ssd>/
    ring/
      index.m3u8
      seg-NNNNN.ts
    recordings/
      explicit/<recording-id>/
        index.m3u8
        seg-NNNNN.ts
        metadata.json
      anomaly/<event-id>/...        # Slice 5
      upload_queue/
      <recording-id>.upload         # flag-for-upload marker (presence = signal)
      <recording-id>.failed         # uploader renames on non-retryable failure

This module is the single place that knows those paths and the metadata.json
shape, so the state machine, the MQTT command handlers, and the uploader all
agree. It is pure filesystem + JSON (no GStreamer, no MQTT) and therefore
unit-testable against a tmp dir.

`metadata.json` is the recording's source of truth (V4.4): `flagged_for_upload`
and `upload_state` are mutated in place by flag-upload (V4.5) and the uploader
(V4.7). Updates are atomic (write-temp + os.replace) so a crash mid-write never
leaves a half-written, unparseable metadata file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fessel_schemas import AnomalyRecordingMetadata, RecordingMetadata

# Subdirectory names under the SSD mount. Stable strings shared with the
# uploader and with supervisor's proxy/listing (kept here as the one source).
RING_DIRNAME = "ring"
RECORDINGS_DIRNAME = "recordings"
EXPLICIT_DIRNAME = "explicit"
ANOMALY_DIRNAME = "anomaly"
UPLOAD_QUEUE_DIRNAME = "upload_queue"

METADATA_FILENAME = "metadata.json"
PLAYLIST_FILENAME = "index.m3u8"

# An explicit recording is assembled by copying ring segments into the recording
# dir as raw-NNNNN.ts (in write order); assemble_recording_playlist() renames them
# to the final seg-NNNNN.ts and builds index.m3u8.
REC_RAW_SEGMENT_GLOB = "raw-*.ts"

UPLOAD_MARKER_SUFFIX = ".upload"
UPLOAD_FAILED_SUFFIX = ".failed"


class Storage:
  """Resolves the on-disk paths for the ring, recordings, and upload queue
  under a configurable SSD mount (`/mnt/ssd` by convention)."""

  def __init__(self, ssd_root: str) -> None:
    self.root = Path(ssd_root)

  # --- directories -----------------------------------------------------------

  @property
  def ring_dir(self) -> Path:
    return self.root / RING_DIRNAME

  @property
  def explicit_dir(self) -> Path:
    return self.root / RECORDINGS_DIRNAME / EXPLICIT_DIRNAME

  @property
  def anomaly_dir(self) -> Path:
    # Slice 5: anomaly-triggered recordings sit alongside explicit ones under
    # recordings/, in their own subdirectory (§2.8, V5.7).
    return self.root / RECORDINGS_DIRNAME / ANOMALY_DIRNAME

  @property
  def upload_queue_dir(self) -> Path:
    return self.root / UPLOAD_QUEUE_DIRNAME

  def recording_dir(self, recording_id: str) -> Path:
    return self.explicit_dir / recording_id

  def anomaly_recording_dir(self, anomaly_id: str) -> Path:
    return self.anomaly_dir / anomaly_id

  def ensure_layout(self) -> None:
    """Create the ring / recordings / upload_queue directories if absent. The
    Pi deploy step does this too (X4.2); doing it here as well means the
    processes are robust to a fresh SSD without a manual mkdir."""
    self.ring_dir.mkdir(parents=True, exist_ok=True)
    self.explicit_dir.mkdir(parents=True, exist_ok=True)
    self.anomaly_dir.mkdir(parents=True, exist_ok=True)
    self.upload_queue_dir.mkdir(parents=True, exist_ok=True)

  # --- metadata --------------------------------------------------------------

  def metadata_path(self, recording_id: str) -> Path:
    return self.recording_dir(recording_id) / METADATA_FILENAME

  def write_metadata(self, meta: RecordingMetadata) -> None:
    """Atomically write a recording's metadata.json (V4.4).

    Write to a sibling temp file then os.replace, so a reader never sees a
    partially written file and a crash mid-write leaves the prior version
    intact.
    """
    path = self.metadata_path(meta.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta.model_dump(mode="json"), indent=2), encoding="utf-8")
    os.replace(tmp, path)

  def read_metadata(self, recording_id: str) -> RecordingMetadata | None:
    """Read + validate a recording's metadata.json, or None if it is missing
    or unparseable (a partial recording that never finalised, V4.3)."""
    path = self.metadata_path(recording_id)
    try:
      raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
      return None
    try:
      return RecordingMetadata.model_validate(raw)
    except Exception:  # noqa: BLE001 — a malformed file is "no metadata"
      return None

  def list_recordings(self) -> list[RecordingMetadata]:
    """Every finalised explicit recording with a readable metadata.json, newest
    first. Recordings whose metadata is missing/partial are skipped (S4.4)."""
    out: list[RecordingMetadata] = []
    if not self.explicit_dir.is_dir():
      return out
    for child in self.explicit_dir.iterdir():
      if not child.is_dir():
        continue
      meta = self.read_metadata(child.name)
      if meta is not None:
        out.append(meta)
    out.sort(key=lambda m: m.started_at, reverse=True)
    return out

  def count_segments(self, recording_id: str) -> int:
    """Number of .ts segments written for a recording (for metadata.segments)."""
    d = self.recording_dir(recording_id)
    if not d.is_dir():
      return 0
    return sum(1 for f in d.iterdir() if f.suffix == ".ts")

  # --- anomaly recordings (Slice 5, V5.7) ------------------------------------
  # Anomaly recordings live under recordings/anomaly/<id>/ with the same on-disk
  # shape as explicit ones (index.m3u8 + seg-NNNNN.ts + metadata.json), but the
  # metadata is AnomalyRecordingMetadata (carries a `type:"anomaly"` and the
  # trigger events). supervisor's /recordings merges both lists (S5.4).

  def anomaly_metadata_path(self, anomaly_id: str) -> Path:
    return self.anomaly_recording_dir(anomaly_id) / METADATA_FILENAME

  def write_anomaly_metadata(self, meta: AnomalyRecordingMetadata) -> None:
    """Atomically write an anomaly recording's metadata.json (V5.7), same
    temp-file + os.replace contract as the explicit-recording write."""
    path = self.anomaly_metadata_path(meta.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta.model_dump(mode="json"), indent=2), encoding="utf-8")
    os.replace(tmp, path)

  def read_anomaly_metadata(self, anomaly_id: str) -> AnomalyRecordingMetadata | None:
    """Read + validate an anomaly recording's metadata.json, or None if missing
    or unparseable (a partial recording that never finalised)."""
    path = self.anomaly_metadata_path(anomaly_id)
    try:
      raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
      return None
    try:
      return AnomalyRecordingMetadata.model_validate(raw)
    except Exception:  # noqa: BLE001 — a malformed file is "no metadata"
      return None

  def list_anomaly_recordings(self) -> list[AnomalyRecordingMetadata]:
    """Every finalised anomaly recording with a readable metadata.json, newest
    first. Partial/never-finalised ones are skipped (mirrors list_recordings)."""
    out: list[AnomalyRecordingMetadata] = []
    if not self.anomaly_dir.is_dir():
      return out
    for child in self.anomaly_dir.iterdir():
      if not child.is_dir():
        continue
      meta = self.read_anomaly_metadata(child.name)
      if meta is not None:
        out.append(meta)
    out.sort(key=lambda m: m.started_at, reverse=True)
    return out

  def count_anomaly_segments(self, anomaly_id: str) -> int:
    """Number of .ts segments written for an anomaly recording (before + after)."""
    d = self.anomaly_recording_dir(anomaly_id)
    if not d.is_dir():
      return 0
    return sum(1 for f in d.iterdir() if f.suffix == ".ts")

  # --- upload-queue markers (V4.6) ------------------------------------------

  def upload_marker_path(self, recording_id: str) -> Path:
    return self.upload_queue_dir / f"{recording_id}{UPLOAD_MARKER_SUFFIX}"

  def upload_failed_path(self, recording_id: str) -> Path:
    return self.upload_queue_dir / f"{recording_id}{UPLOAD_FAILED_SUFFIX}"

  def create_upload_marker(self, recording_id: str) -> None:
    """Create the empty `<recording-id>.upload` marker — presence is the
    uploader's input (V4.5/V4.6). Idempotent: re-flagging an already-marked
    recording is a no-op."""
    self.upload_queue_dir.mkdir(parents=True, exist_ok=True)
    self.upload_marker_path(recording_id).touch(exist_ok=True)

  def list_upload_markers(self) -> list[Path]:
    """Pending `.upload` markers (NOT `.failed`), the uploader's work list."""
    d = self.upload_queue_dir
    if not d.is_dir():
      return []
    return sorted(p for p in d.iterdir() if p.suffix == UPLOAD_MARKER_SUFFIX)


def assemble_recording_playlist(recording_dir: str | Path, segment_seconds: int) -> int:
  """Build a coherent VOD `index.m3u8` from the `raw-*.ts` segments staged in
  `recording_dir` (copied from the ring, in write order), and return the count.

  An explicit recording is assembled by copying the ring segments covering its
  window into the recording dir as `raw-NNNNN.ts` (the recording reuses the
  ring's encode — hlssink2 can't be added live). Here we rename them to the final
  `seg-NNNNN.ts` contiguously (the uploader PUTs every file in the dir, so the
  names must be the final HLS names) and write the finite VOD playlist. Pure
  filesystem; no GStreamer.
  """
  d = Path(recording_dir)
  raw = sorted(d.glob(REC_RAW_SEGMENT_GLOB), key=lambda p: int(p.stem.split("-")[-1]))
  names: list[str] = []
  for i, src in enumerate(raw):
    dst = d / f"seg-{i:05d}.ts"
    os.replace(src, dst)  # same dir -> atomic rename
    names.append(dst.name)
  lines = [
    "#EXTM3U",
    "#EXT-X-VERSION:3",
    "#EXT-X-MEDIA-SEQUENCE:0",
    f"#EXT-X-TARGETDURATION:{segment_seconds}",
    "#EXT-X-PLAYLIST-TYPE:VOD",
  ]
  for name in names:
    lines.append(f"#EXTINF:{float(segment_seconds)},")
    lines.append(name)
  lines.append("#EXT-X-ENDLIST")
  (d / PLAYLIST_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
  return len(names)
