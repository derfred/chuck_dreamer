"""Recording storage backend abstraction (Slice 5.5, B5.5.1).

webui-backend now fronts the recording store: the Pi uploads recordings to a
tailnet-only ingest endpoint, and the backend persists them via a pluggable
storage backend selected by config — MinIO (S3) or a mounted disk (PVC). The
rest of the backend depends only on the `StorageBackend` interface; the factory
(`build_storage_backend`) chooses the implementation from env at startup.

Re-exports the public surface so callers do `from app.storage import ...`.
"""

from __future__ import annotations

from .base import (
  PlaybackTarget,
  PresignedURL,
  RecordingMetadataView,
  ServeLocally,
  StorageBackend,
)
from .factory import build_storage_backend
from .disk_backend import DiskStorageBackend
from .fake_backend import FakeStorageBackend
from .minio_backend import MinioStorageBackend

__all__ = [
  "StorageBackend",
  "PlaybackTarget",
  "PresignedURL",
  "ServeLocally",
  "RecordingMetadataView",
  "DiskStorageBackend",
  "MinioStorageBackend",
  "FakeStorageBackend",
  "build_storage_backend",
]
