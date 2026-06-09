"""Fessel Pi-side uploader (V4.7 / V5.5.1): ships flagged recordings to
webui-backend's tailnet-only recording-ingest endpoint. The Pi holds no
cluster-store credentials — auth is by Tailscale identity."""

from .core import Uploader, UploadOutcome
from .ingest import (
  FakeIngestClient,
  HttpIngestClient,
  HttpIngestConfig,
  IngestClient,
  PermanentError,
  RetryableError,
  build_ingest_client,
)

__all__ = [
  "FakeIngestClient",
  "HttpIngestClient",
  "HttpIngestConfig",
  "IngestClient",
  "PermanentError",
  "RetryableError",
  "UploadOutcome",
  "Uploader",
  "build_ingest_client",
]
