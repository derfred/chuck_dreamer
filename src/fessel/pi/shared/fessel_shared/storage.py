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
    rec_staging/                    # scratch the always-on recording hlssink2
                                    # writes to between recordings (never uploaded)

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
REC_STAGING_DIRNAME = "rec_staging"

METADATA_FILENAME = "metadata.json"
PLAYLIST_FILENAME = "index.m3u8"

# The always-on recording hlssink2 (valve-gated, §2.2) writes raw fragments under
# this template; assemble_recording_playlist() renames them to seg-NNNNN.ts and
# builds the real index.m3u8. Kept distinct so the uploader never PUTs a raw-*.
REC_RAW_SEGMENT_GLOB = "raw-*.ts"
REC_RAW_SEGMENT_TEMPLATE = "raw-%05d.ts"
# hlssink2's own playlist for the recording sink is ignored (its segment counter
# desyncs from splitmuxsink after a live `location` retarget); we write our own.
REC_STAGING_PLAYLIST = "throwaway.m3u8"

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

  @property
  def rec_staging_dir(self) -> Path:
    # Scratch the always-on recording hlssink2 points at when NOT recording, so
    # its idle fragments never land in a real recording dir (the uploader PUTs
    # every file in a recording dir). A top-level sibling of recordings/, so it
    # is never walked by list_recordings (which iterates explicit_dir only).
    return self.root / REC_STAGING_DIRNAME

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
    self.rec_staging_dir.mkdir(parents=True, exist_ok=True)

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
  """Rebuild a coherent VOD `index.m3u8` from the `raw-*.ts` the always-on
  recording hlssink2 produced into `recording_dir`, and return the segment count.

  The recording sink is built cold into the capture pipeline and valve-gated
  (§2.2) because hlssink2/splitmuxsink cannot be added live to a running
  pipeline. We point its `location` at the recording dir per recording, but its
  OWN playlist (`throwaway.m3u8`) is unusable: after a live `location` retarget
  hlssink2's playlist counter and splitmuxsink's fragment counter desync, so the
  playlist names don't match the `.ts` on disk. So we ignore it and rebuild from
  the fragments actually written: rename `raw-NNNNN.ts` -> `seg-NNNNN.ts`
  contiguously (the uploader PUTs every file in the dir, so the names must be the
  final HLS names and `throwaway.m3u8` must not survive), then write a finite VOD
  playlist. Pure filesystem; no GStreamer.
  """
  d = Path(recording_dir)
  raw = sorted(d.glob(REC_RAW_SEGMENT_GLOB), key=lambda p: int(p.stem.split("-")[-1]))
  names: list[str] = []
  for i, src in enumerate(raw):
    dst = d / f"seg-{i:05d}.ts"
    os.replace(src, dst)  # same dir -> atomic rename
    names.append(dst.name)
  # Drop hlssink2's throwaway playlist so the uploader never PUTs it.
  (d / REC_STAGING_PLAYLIST).unlink(missing_ok=True)
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
