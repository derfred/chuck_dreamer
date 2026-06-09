"""In-memory storage backend for tests.

Behaves like the disk backend for the parts tests assert on (store streams +
overwrites, list parses metadata, read serves byte ranges, playback returns
ServeLocally), but holds everything in a dict so no tmp dir or real MinIO is
needed. `presigned` flips playback_url to a PresignedURL (to exercise the
backend's MinIO 302 branch without a real MinIO).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .base import (
  ANOMALY,
  DEFAULT_TYPE,
  EXPLICIT,
  METADATA_FILENAME,
  PlaybackTarget,
  PresignedURL,
  ReadResult,
  RecordingMetadataView,
  ServeLocally,
  StorageBackend,
  content_type_for,
  parse_byte_range,
)


@dataclass
class FakeStorageBackend(StorageBackend):
  # (recording_id, file_name) -> bytes. type tracked per recording.
  blobs: dict[tuple[str, str], bytes] = field(default_factory=dict)
  rec_types: dict[str, str] = field(default_factory=dict)
  # When set, playback_url returns a PresignedURL (the MinIO 302 path) instead
  # of ServeLocally; the backend handler then 302-redirects rather than reading.
  presigned: bool = False

  def store(self, recording_id: str, file_name: str, chunks: Iterable[bytes]) -> None:
    data = b"".join(chunks)
    self.blobs[(recording_id, file_name)] = data
    self.rec_types.setdefault(recording_id, DEFAULT_TYPE)
    if file_name == METADATA_FILENAME:
      import json

      try:
        meta = json.loads(data)
        t = meta.get("type") if isinstance(meta, dict) else None
        if t in (EXPLICIT, ANOMALY):
          self.rec_types[recording_id] = t
      except ValueError:
        pass

  def exists(self, recording_id: str) -> bool:
    return any(rid == recording_id for rid, _ in self.blobs)

  def list(self) -> list[RecordingMetadataView]:
    import json

    by_id: dict[str, list[str]] = {}
    for rid, name in self.blobs:
      by_id.setdefault(rid, []).append(name)
    views: list[RecordingMetadataView] = []
    for rid, files in by_id.items():
      meta = None
      if METADATA_FILENAME in files:
        try:
          parsed = json.loads(self.blobs[(rid, METADATA_FILENAME)])
          meta = parsed if isinstance(parsed, dict) else None
        except ValueError:
          meta = None
      views.append(
        RecordingMetadataView(
          recording_id=rid,
          rec_type=self.rec_types.get(rid, DEFAULT_TYPE),
          files=tuple(sorted(files)),
          metadata=meta,
        )
      )
    views.sort(key=lambda v: v.started_at, reverse=True)
    return views

  def playback_url(self, recording_id: str, file_name: str) -> PlaybackTarget | None:
    if (recording_id, file_name) not in self.blobs:
      return None
    if self.presigned:
      return PresignedURL(
        url=f"https://minio.example/recordings/{recording_id}/{file_name}?X-Amz-Signature=fake"
      )
    return ServeLocally(recording_id=recording_id, file_name=file_name)

  def read(
    self, recording_id: str, file_name: str, http_range: str | None = None
  ) -> ReadResult | None:
    if self.presigned:
      raise NotImplementedError("presigned (MinIO) backend does not serve bytes")
    data = self.blobs.get((recording_id, file_name))
    if data is None:
      return None
    total = len(data)
    byte_range = parse_byte_range(http_range, total)
    if byte_range is None:
      return ReadResult(iter([data]), content_type_for(file_name), total)
    sliced = data[byte_range.start : byte_range.end + 1]
    return ReadResult(iter([sliced]), content_type_for(file_name), len(sliced), byte_range)
