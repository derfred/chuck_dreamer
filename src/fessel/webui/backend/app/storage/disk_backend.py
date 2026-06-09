"""Disk storage backend (B5.5.3).

Writes recordings to a mounted directory (a PVC in the cluster); serves
playback itself as byte-range GETs. The simpler backend operationally, but
more code, because the bytes flow through the backend on the way out too.

On-store layout (identical to MinIO, architecture §4.3):
  <root>/recordings/<type>/<id>/{index.m3u8, seg-NNNNN.ts, metadata.json}

PATH SAFETY is load-bearing: every path is built from a caller-supplied id and
name, and the playback endpoint is browser-facing (through oauth2-proxy). A
directory traversal there would read arbitrary backend-pod files. `_safe_path`
rejects any id/name that isn't a single plain path component and confirms the
resolved path stays strictly under <root>/recordings/.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from .base import (
  ANOMALY,
  DEFAULT_TYPE,
  EXPLICIT,
  METADATA_FILENAME,
  ReadResult,
  RecordingMetadataView,
  ServeLocally,
  StorageBackend,
  content_type_for,
  parse_byte_range,
)

# Stream in 1 MiB chunks: large enough to keep syscall overhead low on
# multi-MB segments, small enough that a 200 MB file never lands in memory.
CHUNK_SIZE = 1024 * 1024
RECORDINGS_DIRNAME = "recordings"


def _is_plain_component(part: str) -> bool:
  """True iff `part` is a single, safe path component: non-empty, not `.`/`..`,
  no separators, no NUL. The defence against id/name traversal."""
  if not part or part in (".", ".."):
    return False
  if "/" in part or "\\" in part or "\x00" in part:
    return False
  return True


class DiskStorageBackend(StorageBackend):
  def __init__(self, root: str) -> None:
    # The mounted PVC path from config (recordings_storage.disk.path). Resolve
    # once so _safe_path can compare resolved children against a resolved base.
    self._recordings_root = (Path(root) / RECORDINGS_DIRNAME).resolve()

  # --- path safety -----------------------------------------------------------

  def _type_dir(self, rec_type: str) -> Path:
    # Only the two known types map to a subdir; anything else falls back to
    # explicit (a stored metadata.json with a junk type can't escape the tree).
    sub = ANOMALY if rec_type == ANOMALY else EXPLICIT
    return self._recordings_root / sub

  def _safe_path(self, recording_id: str, file_name: str, rec_type: str) -> Path:
    """Build <root>/recordings/<type>/<id>/<name>, rejecting traversal.

    Both `recording_id` and `file_name` must be single plain components, and
    the fully-resolved path must remain under the recordings root. Raises
    ValueError on any violation (the handler maps that to 400)."""
    if not _is_plain_component(recording_id) or not _is_plain_component(file_name):
      raise ValueError("invalid recording id or file name")
    candidate = (self._type_dir(rec_type) / recording_id / file_name).resolve()
    root = str(self._recordings_root)
    if os.path.commonpath([str(candidate), root]) != root:
      raise ValueError("path escapes recordings root")
    return candidate

  def _find_existing(self, recording_id: str, file_name: str) -> Path | None:
    """Locate an already-stored file, searching both type subdirs (a reader
    doesn't know the type up front). Returns the path or None."""
    for rec_type in (EXPLICIT, ANOMALY):
      try:
        p = self._safe_path(recording_id, file_name, rec_type)
      except ValueError:
        return None
      if p.is_file():
        return p
    return None

  def _rec_type_of(self, recording_id: str) -> str:
    """Which subdir holds this recording (explicit unless an anomaly dir
    exists). Used by store() so a file arriving before metadata.json still
    lands in the right tree once the dir is known."""
    for rec_type in (EXPLICIT, ANOMALY):
      d = self._type_dir(rec_type) / recording_id
      if d.is_dir():
        return rec_type
    return DEFAULT_TYPE

  # --- store -----------------------------------------------------------------

  def store(self, recording_id: str, file_name: str, chunks: Iterable[bytes]) -> None:
    # Determine the type subdir. metadata.json is the authoritative source of
    # `type`, so when THIS file is the metadata, buffer it (it's tiny — KB) and
    # read the type out of it. For any other file, use the type from an
    # already-stored metadata.json, else where the recording dir already lives,
    # else explicit/ (the uploader sends metadata.json last, so segments arrive
    # before the type is known — that's fine, and once metadata lands the
    # recording is discoverable under the right subdir).
    buffered: bytes | None = None
    if file_name == METADATA_FILENAME:
      buffered = b"".join(c for c in chunks if c)
      rec_type = _type_from_metadata_bytes(buffered)
      if rec_type is None:
        rec_type = self._rec_type_of(recording_id)
    else:
      rec_type = self._read_type(recording_id)

    dest = self._safe_path(recording_id, file_name, rec_type)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp sibling then os.replace -> per-file atomic + idempotent:
    # a retried PUT overwrites cleanly, and a reader never sees a half file.
    tmp = dest.with_name(f".{dest.name}.tmp")
    try:
      with open(tmp, "wb") as fh:
        if buffered is not None:
          fh.write(buffered)
        else:
          for chunk in chunks:
            if chunk:
              fh.write(chunk)
      os.replace(tmp, dest)
    finally:
      tmp.unlink(missing_ok=True)

  def _read_type(self, recording_id: str) -> str:
    """Type for a *store* call: prefer the type recorded in an already-stored
    metadata.json, else where the recording dir already lives, else default."""
    meta_path = self._find_existing(recording_id, METADATA_FILENAME)
    if meta_path is not None:
      meta = _load_json(meta_path)
      if isinstance(meta, dict):
        t = meta.get("type")
        if t in (EXPLICIT, ANOMALY):
          return t
    return self._rec_type_of(recording_id)

  # --- exists / list ---------------------------------------------------------

  def exists(self, recording_id: str) -> bool:
    if not _is_plain_component(recording_id):
      return False
    for rec_type in (EXPLICIT, ANOMALY):
      d = self._type_dir(rec_type) / recording_id
      if d.is_dir() and any(d.iterdir()):
        return True
    return False

  def list(self) -> list[RecordingMetadataView]:
    views: list[RecordingMetadataView] = []
    for rec_type in (EXPLICIT, ANOMALY):
      base = self._type_dir(rec_type)
      if not base.is_dir():
        continue
      for child in sorted(base.iterdir()):
        if not child.is_dir():
          continue
        files = tuple(sorted(f.name for f in child.iterdir() if f.is_file()))
        meta = _load_json(child / METADATA_FILENAME)
        views.append(
          RecordingMetadataView(
            recording_id=child.name,
            rec_type=rec_type,
            files=files,
            metadata=meta if isinstance(meta, dict) else None,
          )
        )
    # Newest first; recordings without metadata.started_at sort last.
    views.sort(key=lambda v: v.started_at, reverse=True)
    return views

  # --- playback (served by the backend) --------------------------------------

  def playback_url(self, recording_id: str, file_name: str) -> ServeLocally | None:
    if self._find_existing(recording_id, file_name) is None:
      return None
    return ServeLocally(recording_id=recording_id, file_name=file_name)

  def read(
    self, recording_id: str, file_name: str, http_range: str | None = None
  ) -> ReadResult | None:
    path = self._find_existing(recording_id, file_name)
    if path is None:
      return None
    total = path.stat().st_size
    byte_range = parse_byte_range(http_range, total)  # may raise ValueError -> 416
    if byte_range is None:
      return ReadResult(
        chunks=_stream_file(path, 0, total),
        content_type=content_type_for(file_name),
        content_length=total,
      )
    return ReadResult(
      chunks=_stream_file(path, byte_range.start, byte_range.length),
      content_type=content_type_for(file_name),
      content_length=byte_range.length,
      byte_range=byte_range,
    )


def _stream_file(path: Path, start: int, length: int) -> Iterator[bytes]:
  """Yield up to `length` bytes from `path` starting at `start`, in CHUNK_SIZE
  pieces. Used for both full and ranged reads (full = start 0, length = size)."""
  remaining = length
  with open(path, "rb") as fh:
    if start:
      fh.seek(start)
    while remaining > 0:
      chunk = fh.read(min(CHUNK_SIZE, remaining))
      if not chunk:
        break
      remaining -= len(chunk)
      yield chunk


def _type_from_metadata_bytes(raw: bytes) -> str | None:
  """Parse the `type` out of a metadata.json byte buffer, or None if it is
  absent / unparseable / not one of the known types."""
  import json

  try:
    meta = json.loads(raw)
  except ValueError:
    return None
  if isinstance(meta, dict):
    t = meta.get("type")
    if t in (EXPLICIT, ANOMALY):
      return t
  return None


def _load_json(path: Path) -> object | None:
  import json

  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except (OSError, ValueError):
    return None
