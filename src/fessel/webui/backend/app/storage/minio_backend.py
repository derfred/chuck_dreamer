"""MinIO storage backend (B5.5.2).

Repurposes the MinIO client code that previously lived on the Pi (the uploader's
objectstore.py): now the *backend* holds the S3 credentials and the Pi holds
none. Writes to an S3-compatible bucket; playback is a presigned-GET redirect
so the browser fetches segments straight from MinIO and the backend never
touches the bytes.

On-store key layout (identical to disk, architecture §4.3):
  recordings/<type>/<id>/<name>
`<type>` is read from a stored metadata.json when available, else `explicit`.

The minio SDK is imported lazily so tests (which use FakeStorageBackend) never
need it, mirroring the original Pi-side seam.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable, Iterator

from .base import (
  ANOMALY,
  DEFAULT_TYPE,
  EXPLICIT,
  METADATA_FILENAME,
  PresignedURL,
  ReadResult,
  RecordingMetadataView,
  StorageBackend,
  content_type_for,
)

KEY_ROOT = "recordings"
PRESIGN_TTL_S = 300  # 5 minutes: enough for an HLS player to walk the playlist + segments


def _key(rec_type: str, recording_id: str, file_name: str) -> str:
  sub = ANOMALY if rec_type == ANOMALY else EXPLICIT
  return f"{KEY_ROOT}/{sub}/{recording_id}/{file_name}"


class _ChunkReader(io.RawIOBase):
  """Adapts an iterable of byte chunks to a read()-able stream so the MinIO SDK
  can consume an upload incrementally (no full-body buffering). Length-unknown
  streams use put_object(length=-1, part_size=...) which reads to EOF."""

  def __init__(self, chunks: Iterable[bytes]) -> None:
    self._it: Iterator[bytes] = iter(chunks)
    self._buf = bytearray()
    self._eof = False

  def readable(self) -> bool:
    return True

  def readinto(self, b) -> int:  # noqa: ANN001 - matches RawIOBase signature
    while not self._buf and not self._eof:
      try:
        nxt = next(self._it)
      except StopIteration:
        self._eof = True
        break
      if nxt:
        self._buf.extend(nxt)
    n = min(len(b), len(self._buf))
    b[:n] = self._buf[:n]
    del self._buf[:n]
    return n


class MinioStorageBackend(StorageBackend):
  # 10 MiB part size for length-unknown streaming uploads (the SDK's minimum
  # multipart part is 5 MiB; 10 keeps part counts low for multi-MB segments).
  _PART_SIZE = 10 * 1024 * 1024

  def __init__(
    self, *, endpoint: str, access_key: str, secret_key: str, bucket: str, secure: bool = True
  ) -> None:
    from datetime import timedelta

    from minio import Minio

    self._timedelta = timedelta
    self._client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    self._bucket = bucket
    # Tiny TTL cache for list() to soften the dashboard's poll rate (B5.5.2).
    self._list_cache: tuple[float, list[RecordingMetadataView]] | None = None

  def store(self, recording_id: str, file_name: str, chunks: Iterable[bytes]) -> None:
    rec_type = self._rec_type(recording_id)
    key = _key(rec_type, recording_id, file_name)
    # put_object with length=-1 streams to EOF in part_size chunks (idempotent:
    # a repeated PUT overwrites the object; atomic: the SDK only commits on
    # completion).
    self._client.put_object(
      self._bucket,
      key,
      _ChunkReader(chunks),
      length=-1,
      part_size=self._PART_SIZE,
      content_type=content_type_for(file_name),
    )
    self._list_cache = None  # invalidate: a new file changes the listing

  def _rec_type(self, recording_id: str) -> str:
    """Read the recording's type from a stored metadata.json if present, else
    default to explicit. Checked under both type prefixes."""
    for rec_type in (EXPLICIT, ANOMALY):
      key = _key(rec_type, recording_id, METADATA_FILENAME)
      raw = self._get_object_bytes(key)
      if raw is not None:
        try:
          meta = json.loads(raw)
        except ValueError:
          continue
        t = meta.get("type") if isinstance(meta, dict) else None
        if t in (EXPLICIT, ANOMALY):
          return t
        return rec_type  # the dir exists under this prefix -> use it
    return DEFAULT_TYPE

  def exists(self, recording_id: str) -> bool:
    for rec_type in (EXPLICIT, ANOMALY):
      prefix = f"{KEY_ROOT}/{EXPLICIT if rec_type == EXPLICIT else ANOMALY}/{recording_id}/"
      if any(True for _ in self._client.list_objects(self._bucket, prefix=prefix, recursive=True)):
        return True
    return False

  def list(self) -> list[RecordingMetadataView]:
    import time

    if self._list_cache is not None:
      ts, cached = self._list_cache
      if time.monotonic() - ts < 1.5:
        return cached
    by_id: dict[str, dict] = {}
    for obj in self._client.list_objects(self._bucket, prefix=f"{KEY_ROOT}/", recursive=True):
      # key = recordings/<type>/<id>/<file>
      parts = obj.object_name.split("/")
      if len(parts) < 4 or parts[0] != KEY_ROOT:
        continue
      rec_type, rid, fname = parts[1], parts[2], "/".join(parts[3:])
      entry = by_id.setdefault(rid, {"type": rec_type, "files": []})
      entry["files"].append(fname)
    views: list[RecordingMetadataView] = []
    for rid, entry in by_id.items():
      meta = None
      if METADATA_FILENAME in entry["files"]:
        raw = self._get_object_bytes(_key(entry["type"], rid, METADATA_FILENAME))
        if raw is not None:
          try:
            parsed = json.loads(raw)
            meta = parsed if isinstance(parsed, dict) else None
          except ValueError:
            meta = None
      views.append(
        RecordingMetadataView(
          recording_id=rid,
          rec_type=entry["type"] if entry["type"] in (EXPLICIT, ANOMALY) else DEFAULT_TYPE,
          files=tuple(sorted(entry["files"])),
          metadata=meta,
        )
      )
    views.sort(key=lambda v: v.started_at, reverse=True)
    self._list_cache = (time.monotonic(), views)
    return views

  def playback_url(self, recording_id: str, file_name: str) -> PresignedURL | None:
    rec_type = self._rec_type(recording_id)
    key = _key(rec_type, recording_id, file_name)
    try:
      url = self._client.presigned_get_object(
        self._bucket, key, expires=self._timedelta(seconds=PRESIGN_TTL_S)
      )
    except Exception:  # noqa: BLE001 — missing object / transient -> treat as absent
      return None
    return PresignedURL(url=url)

  def read(
    self, recording_id: str, file_name: str, http_range: str | None = None
  ) -> ReadResult | None:
    # MinIO playback is via presigned redirect, never through the backend.
    raise NotImplementedError("MinIO backend does not serve playback bytes; use playback_url")

  def _get_object_bytes(self, key: str) -> bytes | None:
    try:
      resp = self._client.get_object(self._bucket, key)
    except Exception:  # noqa: BLE001 — absent / transient
      return None
    try:
      return resp.read()
    finally:
      resp.close()
      resp.release_conn()
