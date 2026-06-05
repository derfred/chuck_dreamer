"""MinIO-backed recordings store for the backend (B4.2/B4.3/B4.4).

Two jobs:
  - **list** recordings already uploaded to MinIO, so /api/recordings can show
    a complete picture even if the Pi-side copy was removed (out of Slice 4
    scope, but the data shape must not assume Pi-only);
  - **mint presigned URLs** for an uploaded recording's playlist + segments, so
    playback of an uploaded recording is served straight from MinIO (offloads
    Pi bandwidth) rather than proxied through supervisor.

To keep the backend testable without a real MinIO (mirroring supervisor's
pywizlight seam and the uploader's object-store seam), the store is an abstract
`RecordingsStore` with a `minio` driver and an in-memory `fake`. When no MinIO
is configured, `NoRecordingsStore` makes every recording Pi-only.

Key layout matches the uploader (V4.7): recordings/explicit/<id>/<file>.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

KEY_PREFIX = "recordings/explicit"
PRESIGN_TTL_S = 3600


@dataclass(frozen=True)
class RemoteRecording:
  recording_id: str
  files: tuple[str, ...]  # file names present in the bucket for this recording


class RecordingsStore:
  def list_recordings(self) -> list[RemoteRecording]:  # pragma: no cover - interface
    raise NotImplementedError

  def has_recording(self, recording_id: str) -> bool:  # pragma: no cover - interface
    raise NotImplementedError

  def presigned_url(self, recording_id: str, filename: str) -> str | None:  # pragma: no cover
    """A time-limited GET URL for one file, or None if it does not exist."""
    raise NotImplementedError


class NoRecordingsStore(RecordingsStore):
  """Used when no MinIO is configured: every recording is Pi-only."""

  def list_recordings(self) -> list[RemoteRecording]:
    return []

  def has_recording(self, recording_id: str) -> bool:
    return False

  def presigned_url(self, recording_id: str, filename: str) -> str | None:
    return None


class MinioRecordingsStore(RecordingsStore):
  """Production driver: MinIO SDK. Imported lazily so tests never need minio."""

  def __init__(
    self, *, endpoint: str, access_key: str, secret_key: str, bucket: str, secure: bool = True
  ) -> None:
    from datetime import timedelta

    from minio import Minio

    self._timedelta = timedelta
    self._client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    self._bucket = bucket

  def list_recordings(self) -> list[RemoteRecording]:
    by_id: dict[str, list[str]] = {}
    for obj in self._client.list_objects(self._bucket, prefix=f"{KEY_PREFIX}/", recursive=True):
      # key = recordings/explicit/<id>/<file>
      rel = obj.object_name[len(KEY_PREFIX) + 1 :]
      parts = rel.split("/", 1)
      if len(parts) != 2:
        continue
      rid, fname = parts
      by_id.setdefault(rid, []).append(fname)
    return [RemoteRecording(rid, tuple(sorted(files))) for rid, files in by_id.items()]

  def has_recording(self, recording_id: str) -> bool:
    objs = self._client.list_objects(
      self._bucket, prefix=f"{KEY_PREFIX}/{recording_id}/", recursive=True
    )
    return any(True for _ in objs)

  def presigned_url(self, recording_id: str, filename: str) -> str | None:
    key = f"{KEY_PREFIX}/{recording_id}/{filename}"
    try:
      return self._client.presigned_get_object(
        self._bucket, key, expires=self._timedelta(seconds=PRESIGN_TTL_S)
      )
    except Exception:  # noqa: BLE001 — missing object / transient -> treat as absent
      return None


@dataclass
class FakeRecordingsStore(RecordingsStore):
  """In-memory store for tests: recording_id -> file names present remotely."""

  recordings: dict[str, list[str]] = field(default_factory=dict)

  def list_recordings(self) -> list[RemoteRecording]:
    return [RemoteRecording(rid, tuple(sorted(files))) for rid, files in self.recordings.items()]

  def has_recording(self, recording_id: str) -> bool:
    return recording_id in self.recordings

  def presigned_url(self, recording_id: str, filename: str) -> str | None:
    files = self.recordings.get(recording_id)
    if files is None or filename not in files:
      return None
    return f"https://minio.example/{KEY_PREFIX}/{recording_id}/{filename}?X-Amz-Signature=fake"


def build_recordings_store() -> RecordingsStore:
  """Construct from FESSEL_MINIO_* env vars; NoRecordingsStore when unset (the
  Pi-only path). The packaging slice wires the env from a k8s Secret."""
  endpoint = os.environ.get("FESSEL_MINIO_ENDPOINT")
  bucket = os.environ.get("FESSEL_MINIO_BUCKET")
  access_key = os.environ.get("FESSEL_MINIO_ACCESS_KEY")
  secret_key = os.environ.get("FESSEL_MINIO_SECRET_KEY")
  if not (endpoint and bucket and access_key and secret_key):
    return NoRecordingsStore()
  secure = os.environ.get("FESSEL_MINIO_SECURE", "true").lower() != "false"
  return MinioRecordingsStore(
    endpoint=endpoint,
    access_key=access_key,
    secret_key=secret_key,
    bucket=bucket,
    secure=secure,
  )
