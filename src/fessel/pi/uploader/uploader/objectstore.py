"""Object-store seam for the uploader (V4.7).

The uploader ships recording files to MinIO. To keep the upload loop testable
without a real MinIO (mirroring supervisor's pywizlight seam), the store is an
abstract `ObjectStore` with two drivers:

  - `minio` (production): the official MinIO Python SDK over S3.
  - `fake`  (tests): an in-memory dict, plus knobs to simulate transient
    (retryable) and permanent (non-retryable) failures.

`put_object` raises `RetryableError` for transient failures (network down,
5xx) — the uploader keeps the marker and backs off — and `PermanentError` for
non-retryable ones (bad credentials, 4xx) — the uploader fails the marker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class RetryableError(Exception):
  """A transient failure: keep the marker, back off, retry (V4.7)."""


class PermanentError(Exception):
  """A non-retryable failure: rename the marker to .failed (V4.7)."""


@dataclass(frozen=True)
class MinioConfig:
  endpoint: str
  access_key: str
  secret_key: str
  bucket: str
  secure: bool = True


class ObjectStore:
  def put_object(self, key: str, path: Path, content_type: str) -> None:  # pragma: no cover
    """Upload the file at `path` to `key`. Raises RetryableError /
    PermanentError to classify failures for the uploader's retry policy."""
    raise NotImplementedError


class MinioObjectStore(ObjectStore):
  """Production driver: MinIO SDK. Imported lazily so tests never need minio."""

  def __init__(self, cfg: MinioConfig) -> None:
    from minio import Minio  # lazy: only the Pi runtime imports the SDK

    self._cfg = cfg
    self._client = Minio(
      cfg.endpoint,
      access_key=cfg.access_key,
      secret_key=cfg.secret_key,
      secure=cfg.secure,
    )
    self._bucket = cfg.bucket

  def put_object(self, key: str, path: Path, content_type: str) -> None:
    from minio.error import MinioException, S3Error

    try:
      self._client.fput_object(self._bucket, key, str(path), content_type=content_type)
    except S3Error as e:
      # 4xx (auth, no-such-bucket, invalid) are permanent; 5xx are retryable.
      code = getattr(e, "code", "")
      status = getattr(e, "_http_status", None) or getattr(e, "response", None)
      if _is_permanent_s3(code, status):
        raise PermanentError(f"S3Error {code}: {e}") from e
      raise RetryableError(f"S3Error {code}: {e}") from e
    except (MinioException, OSError, ConnectionError) as e:
      # Network/transport problems are retryable (cellular blip, MinIO down).
      raise RetryableError(f"{type(e).__name__}: {e}") from e


def _is_permanent_s3(code: str, status: object) -> bool:
  permanent_codes = {
    "AccessDenied",
    "InvalidAccessKeyId",
    "SignatureDoesNotMatch",
    "NoSuchBucket",
    "InvalidBucketName",
    "InvalidRequest",
  }
  if code in permanent_codes:
    return True
  if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
    return True
  return False


@dataclass
class FakeObjectStore(ObjectStore):
  """In-memory store for tests. `fail_n_then_ok` makes the first N put_object
  calls raise RetryableError (to exercise backoff); `permanent` makes every
  call raise PermanentError."""

  fail_n_then_ok: int = 0
  permanent: bool = False
  objects: dict[str, bytes] = field(default_factory=dict)
  calls: int = 0

  def put_object(self, key: str, path: Path, content_type: str) -> None:  # noqa: ARG002
    self.calls += 1
    if self.permanent:
      raise PermanentError("fake permanent failure")
    if self.fail_n_then_ok > 0:
      self.fail_n_then_ok -= 1
      raise RetryableError("fake transient failure")
    self.objects[key] = path.read_bytes()


def build_object_store(cfg: dict) -> ObjectStore:
  """Construct an ObjectStore from the uploader's `minio` config block.

    minio:
      driver: minio          # 'minio' (prod) | 'fake' (tests)
      endpoint: minio.example.com:9000
      access_key: ...
      secret_key: ...
      bucket: fessel
      secure: true

  The `fake` driver needs no credentials; production requires endpoint +
  bucket + keys.
  """
  driver = cfg.get("driver", "minio")
  if driver == "fake":
    return FakeObjectStore()
  return MinioObjectStore(
    MinioConfig(
      endpoint=cfg["endpoint"],
      access_key=cfg["access_key"],
      secret_key=cfg["secret_key"],
      bucket=cfg["bucket"],
      secure=bool(cfg.get("secure", True)),
    )
  )
