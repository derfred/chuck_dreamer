"""HttpIngestClient status -> error taxonomy tests (V5.5.1).

The uploader's retry policy depends on the client classifying failures: 201 ->
ok; 4xx (except 408/429) -> PermanentError (fail the marker); 5xx / 408 / 429 /
network -> RetryableError (keep the marker, back off). These tests drive the
real HttpIngestClient against an httpx MockTransport, so the URL construction
(/recording-ingest/<id>/<file>) and the body streaming are exercised too."""

import httpx
import pytest

from uploader.ingest import (
  HttpIngestClient,
  HttpIngestConfig,
  PermanentError,
  RetryableError,
)


def _client(handler) -> HttpIngestClient:
  c = HttpIngestClient(HttpIngestConfig(url_base="https://ingest.example:8443"))
  # Swap the transport for a mock; keep the same headers/timeouts.
  c._client = httpx.Client(transport=httpx.MockTransport(handler))
  return c


def test_put_url_and_body(tmp_path):
  seen = {}

  def handler(req: httpx.Request) -> httpx.Response:
    seen["url"] = str(req.url)
    seen["body"] = req.content
    seen["ct"] = req.headers.get("content-type")
    return httpx.Response(201)

  f = tmp_path / "seg-00000.ts"
  f.write_bytes(b"TSDATA")
  _client(handler).put_file("rec-1", "seg-00000.ts", f)
  assert seen["url"] == "https://ingest.example:8443/recording-ingest/rec-1/seg-00000.ts"
  assert seen["body"] == b"TSDATA"
  assert seen["ct"] == "video/mp2t"


@pytest.mark.parametrize("status", [200, 201, 204])
def test_success_statuses(tmp_path, status):
  f = tmp_path / "index.m3u8"
  f.write_text("#EXTM3U")
  _client(lambda r: httpx.Response(status)).put_file("r", "index.m3u8", f)  # no raise


@pytest.mark.parametrize("status", [400, 403, 404, 422])
def test_4xx_is_permanent(tmp_path, status):
  f = tmp_path / "x.ts"
  f.write_bytes(b"x")
  with pytest.raises(PermanentError):
    _client(lambda r: httpx.Response(status)).put_file("r", "x.ts", f)


@pytest.mark.parametrize("status", [500, 502, 503, 408, 429])
def test_5xx_and_throttle_are_retryable(tmp_path, status):
  f = tmp_path / "x.ts"
  f.write_bytes(b"x")
  with pytest.raises(RetryableError):
    _client(lambda r: httpx.Response(status)).put_file("r", "x.ts", f)


def test_network_error_is_retryable(tmp_path):
  f = tmp_path / "x.ts"
  f.write_bytes(b"x")

  def handler(req: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("tunnel down")

  with pytest.raises(RetryableError):
    _client(handler).put_file("r", "x.ts", f)
