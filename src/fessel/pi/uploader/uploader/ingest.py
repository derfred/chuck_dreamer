"""Recording-ingest client seam for the uploader (V5.5.1).

Slice 5.5 repoints the uploader: instead of S3 `PutObject` straight to MinIO
(with cluster-store credentials on the Pi), it does an HTTPS PUT of each file to
webui-backend's tailnet-only recording-ingest endpoint. The Pi holds NO
cluster-store credentials — auth is by Tailscale identity at the network layer.

  PUT <ingest_url_base>/recording-ingest/<recording-id>/<file-name>
      body: the raw file content (streamed)
      201 -> stored; 4xx -> permanent (malformed); 5xx / network -> retryable.

To keep the upload loop testable without a real backend (mirroring the old
object-store seam), the client is an abstract `IngestClient` with two drivers:
  - `http` (production): streams the file body over HTTPS PUT.
  - `fake`  (tests): an in-memory dict, plus knobs to simulate transient
    (retryable) and permanent (non-retryable) failures.

`put_file` raises `RetryableError` for transient failures (backend/store down,
5xx, network) — the uploader keeps the marker and backs off — and
`PermanentError` for non-retryable ones (4xx) — the uploader fails the marker.
The error taxonomy is identical to the old objectstore so core.py is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class RetryableError(Exception):
  """A transient failure: keep the marker, back off, retry (V4.7/V5.5.1)."""


class PermanentError(Exception):
  """A non-retryable failure: rename the marker to .failed (V4.7/V5.5.1)."""


class IngestClient:
  def put_file(self, recording_id: str, file_name: str, path: Path) -> None:  # pragma: no cover
    """Upload the file at `path` as recording `recording_id`'s `file_name`.
    Raises RetryableError / PermanentError to classify failures for the
    uploader's retry policy."""
    raise NotImplementedError


@dataclass(frozen=True)
class HttpIngestConfig:
  # e.g. "https://<ingest-name>.<tailnet>.ts.net:8443" (no trailing slash).
  url_base: str
  timeout_s: float = 60.0
  # Verify TLS. Production uses a real cert on the Tailscale ingress; the
  # integration env reaches an in-cluster plain-HTTP Service so this is moot
  # there (scheme is http://).
  verify_tls: bool = True


class HttpIngestClient(IngestClient):
  """Production driver: HTTPS PUT with the file body streamed as the request
  body. Uses httpx (already a transitive dep via the shared stack); imported
  lazily so the `fake` driver needs nothing."""

  def __init__(self, cfg: HttpIngestConfig) -> None:
    import httpx

    self._cfg = cfg
    self._base = cfg.url_base.rstrip("/")
    # A descriptive User-Agent so the backend's ingest audit log can attribute
    # the PUT to the Pi uploader (same convention supervisor uses).
    self._client = httpx.Client(
      timeout=cfg.timeout_s,
      verify=cfg.verify_tls,
      headers={"User-Agent": "fessel-uploader"},
    )

  def put_file(self, recording_id: str, file_name: str, path: Path) -> None:
    import httpx

    url = f"{self._base}/recording-ingest/{recording_id}/{file_name}"
    try:
      with open(path, "rb") as fh:
        # Stream the file body (httpx reads the file object incrementally) so a
        # multi-MB segment never fully buffers in memory.
        resp = self._client.put(
          url, content=fh, headers={"Content-Type": _content_type(file_name)}
        )
    except (httpx.HTTPError, OSError) as e:
      # Network/transport/file problems are retryable (cellular blip, backend
      # down, Tailscale tunnel re-establishing).
      raise RetryableError(f"{type(e).__name__}: {e}") from e
    if resp.status_code in (200, 201, 204):
      return
    if 400 <= resp.status_code < 500 and resp.status_code not in (408, 429):
      # 4xx (malformed request, bad path) is non-retryable. 408/429 are retryable.
      raise PermanentError(f"PUT {url} -> {resp.status_code}: {resp.text[:200]}")
    # 5xx (backend store failure) and 408/429 are retryable.
    raise RetryableError(f"PUT {url} -> {resp.status_code}")


def _content_type(name: str) -> str:
  if name.endswith(".m3u8"):
    return "application/vnd.apple.mpegurl"
  if name.endswith(".ts"):
    return "video/mp2t"
  if name.endswith(".json"):
    return "application/json"
  return "application/octet-stream"


@dataclass
class FakeIngestClient(IngestClient):
  """In-memory client for tests. `fail_n_then_ok` makes the first N put_file
  calls raise RetryableError (to exercise backoff); `permanent` makes every
  call raise PermanentError. Stored objects are keyed by (recording_id,
  file_name) so tests can assert what arrived."""

  fail_n_then_ok: int = 0
  permanent: bool = False
  objects: dict[tuple[str, str], bytes] = field(default_factory=dict)
  calls: int = 0
  # Order of (recording_id, file_name) PUTs, so the completion-ordering
  # contract (segments -> playlist -> metadata.json last, V5.5.1) is testable.
  put_order: list[tuple[str, str]] = field(default_factory=list)

  def put_file(self, recording_id: str, file_name: str, path: Path) -> None:
    self.calls += 1
    if self.permanent:
      raise PermanentError("fake permanent failure")
    if self.fail_n_then_ok > 0:
      self.fail_n_then_ok -= 1
      raise RetryableError("fake transient failure")
    self.objects[(recording_id, file_name)] = path.read_bytes()
    self.put_order.append((recording_id, file_name))


def build_ingest_client(cfg: dict) -> IngestClient:
  """Construct an IngestClient from the uploader's `uploader` config block.

    uploader:
      driver: http               # 'http' (prod) | 'fake' (tests)
      ingest_url_base: "https://<ingest-name>.<tailnet>.ts.net:8443"
      timeout_s: 60
      verify_tls: true

  The `fake` driver needs no URL; production requires ingest_url_base.
  """
  driver = cfg.get("driver", "http")
  if driver == "fake":
    return FakeIngestClient()
  url_base = cfg.get("ingest_url_base")
  if not url_base:
    raise ValueError("uploader.ingest_url_base is required for the http driver")
  return HttpIngestClient(
    HttpIngestConfig(
      url_base=url_base,
      timeout_s=float(cfg.get("timeout_s", 60.0)),
      verify_tls=bool(cfg.get("verify_tls", True)),
    )
  )
