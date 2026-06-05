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

from fessel_schemas import RecordingMetadata

# Subdirectory names under the SSD mount. Stable strings shared with the
# uploader and with supervisor's proxy/listing (kept here as the one source).
RING_DIRNAME = "ring"
RECORDINGS_DIRNAME = "recordings"
EXPLICIT_DIRNAME = "explicit"
UPLOAD_QUEUE_DIRNAME = "upload_queue"

METADATA_FILENAME = "metadata.json"
PLAYLIST_FILENAME = "index.m3u8"

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
  def upload_queue_dir(self) -> Path:
    return self.root / UPLOAD_QUEUE_DIRNAME

  def recording_dir(self, recording_id: str) -> Path:
    return self.explicit_dir / recording_id

  def ensure_layout(self) -> None:
    """Create the ring / recordings / upload_queue directories if absent. The
    Pi deploy step does this too (X4.2); doing it here as well means the
    processes are robust to a fresh SSD without a manual mkdir."""
    self.ring_dir.mkdir(parents=True, exist_ok=True)
    self.explicit_dir.mkdir(parents=True, exist_ok=True)
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
