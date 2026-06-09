"""Storage backend factory (B5.5.1).

Chooses the implementation from environment at startup. The Tanka library
(I5.5.4) wires these env vars from the `recordings_storage` config block —
MinIO credentials from a k8s Secret, the disk path from the PVC mount.

  recordings_storage.backend == "minio":
    FESSEL_RECORDINGS_BACKEND=minio
    FESSEL_MINIO_ENDPOINT / _ACCESS_KEY / _SECRET_KEY / _BUCKET / _SECURE

  recordings_storage.backend == "disk":
    FESSEL_RECORDINGS_BACKEND=disk
    FESSEL_RECORDINGS_DISK_PATH=/var/lib/fessel/recordings

Back-compat: if FESSEL_RECORDINGS_BACKEND is unset but the FESSEL_MINIO_* env is
present (the Slice-4 wiring), default to the MinIO backend so an un-migrated
deploy keeps working. With neither set, the backend has no store — but the
ingest endpoint requires one, so the factory raises rather than silently
dropping uploads.
"""

from __future__ import annotations

import os

from .base import StorageBackend
from .disk_backend import DiskStorageBackend
from .minio_backend import MinioStorageBackend


def build_storage_backend() -> StorageBackend:
  backend = os.environ.get("FESSEL_RECORDINGS_BACKEND", "").strip().lower()

  if not backend:
    # Back-compat with the Slice-4 env (MinIO-only). If those vars are present,
    # behave as a MinIO backend; otherwise there is nothing to build.
    if os.environ.get("FESSEL_MINIO_ENDPOINT"):
      backend = "minio"
    else:
      raise RuntimeError(
        "no recordings storage configured: set FESSEL_RECORDINGS_BACKEND to "
        "'minio' or 'disk' (recordings_storage.backend)"
      )

  if backend == "disk":
    path = os.environ.get("FESSEL_RECORDINGS_DISK_PATH")
    if not path:
      raise RuntimeError("FESSEL_RECORDINGS_DISK_PATH must be set for the disk backend")
    return DiskStorageBackend(path)

  if backend == "minio":
    endpoint = os.environ.get("FESSEL_MINIO_ENDPOINT")
    bucket = os.environ.get("FESSEL_MINIO_BUCKET")
    access_key = os.environ.get("FESSEL_MINIO_ACCESS_KEY")
    secret_key = os.environ.get("FESSEL_MINIO_SECRET_KEY")
    if not (endpoint and bucket and access_key and secret_key):
      raise RuntimeError(
        "MinIO backend requires FESSEL_MINIO_ENDPOINT/_BUCKET/_ACCESS_KEY/_SECRET_KEY"
      )
    secure = os.environ.get("FESSEL_MINIO_SECURE", "true").lower() != "false"
    return MinioStorageBackend(
      endpoint=endpoint,
      access_key=access_key,
      secret_key=secret_key,
      bucket=bucket,
      secure=secure,
    )

  raise RuntimeError(f"unknown recordings storage backend: {backend!r}")
