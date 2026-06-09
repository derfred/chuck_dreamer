"""Recording-ingest listener tests (B5.5.6/B5.5.7).

The ingest app is a SEPARATE ASGI app from the public one. These tests assert:
  - a PUT streams the body into the configured storage backend (201);
  - a repeated PUT overwrites cleanly (idempotent);
  - a path-unsafe id/name is rejected (400) without touching the store;
  - a backend store failure surfaces as 5xx (so the Pi retries);
  - the ingest app does NOT carry the public API surface (the two-listener
    model means /api/... and the browser routes live only on the public app),
    and the public app does NOT carry /recording-ingest (so it's unreachable
    via the public ingress).
"""

from fastapi.testclient import TestClient

import app.main as appmod
from app.storage import FakeStorageBackend
from app.storage.base import StorageBackend


def ingest_client(storage=None):
  return TestClient(appmod.create_ingest_app(storage=storage or FakeStorageBackend()))


def test_put_streams_into_store_201():
  storage = FakeStorageBackend()
  c = ingest_client(storage)
  r = c.put("/recording-ingest/rec-1/seg-00000.ts", content=b"TSDATA")
  assert r.status_code == 201
  assert storage.blobs[("rec-1", "seg-00000.ts")] == b"TSDATA"


def test_put_is_idempotent():
  storage = FakeStorageBackend()
  c = ingest_client(storage)
  c.put("/recording-ingest/rec-1/index.m3u8", content=b"first")
  r = c.put("/recording-ingest/rec-1/index.m3u8", content=b"second")
  assert r.status_code == 201
  assert storage.blobs[("rec-1", "index.m3u8")] == b"second"


def test_path_unsafe_name_is_400():
  # A backend that rejects an id/name with ValueError -> 400, body not stored.
  class _Strict(StorageBackend):
    def __init__(self):
      self.stored = []

    def store(self, recording_id, file_name, chunks):
      if ".." in recording_id or "/" in file_name:
        raise ValueError("invalid path")
      self.stored.append((recording_id, file_name))

  storage = _Strict()
  c = ingest_client(storage)
  r = c.put("/recording-ingest/..%2Fetc/passwd", content=b"x")
  # Starlette may resolve the encoded slash; either way the path is not stored.
  assert r.status_code in (400, 404)
  assert storage.stored == []


def test_store_failure_is_5xx():
  class _Boom(StorageBackend):
    def store(self, recording_id, file_name, chunks):
      # Consume the stream then fail, like a disk-full / PVC-unhealthy write.
      for _ in chunks:
        pass
      raise OSError("disk full")

  c = ingest_client(_Boom())
  r = c.put("/recording-ingest/rec-1/seg-00000.ts", content=b"TSDATA")
  assert r.status_code >= 500


def test_ingest_app_has_no_public_api():
  # The two-listener model: /api/... and the WHEP mint live ONLY on the public
  # app, never on the ingest listener.
  c = ingest_client()
  assert c.get("/api/recordings").status_code == 404
  assert c.get("/api/auth/whep-url?path=p&mode=640x480@30@1000000").status_code == 404
  assert c.get("/jwks").status_code == 404


def test_public_app_has_no_ingest_route():
  # /recording-ingest exists ONLY on the ingest listener, so a PUT to it on the
  # public app 404s — it can't be reached through the public ingress (B5.5.7).
  pub = TestClient(appmod.create_app(storage=FakeStorageBackend()))
  r = pub.put("/recording-ingest/rec-1/index.m3u8", content=b"#EXTM3U")
  assert r.status_code in (404, 405)


def test_ingest_healthz():
  c = ingest_client()
  body = c.get("/healthz").json()
  assert body["status"] == "ok" and body["listener"] == "ingest"


def test_ingest_to_disk_then_playback_roundtrip(tmp_path):
  # The unit-level proxy for the integration round-trip (T5.5.2): PUT files to
  # the REAL ingest app backed by the REAL disk backend, then serve them back
  # through the public app (302/200 + Range -> 206). Proves the async->sync
  # streaming bridge in the ingest handler feeds the disk backend correctly.
  from app.storage import DiskStorageBackend

  storage = DiskStorageBackend(str(tmp_path))
  ic = TestClient(appmod.create_ingest_app(storage=storage))
  # Upload a segment in a way that forces multiple stream chunks (a large body).
  seg = bytes(range(256)) * 4096  # 1 MiB
  assert ic.put("/recording-ingest/rec-9/seg-00000.ts", content=seg).status_code == 201
  assert ic.put("/recording-ingest/rec-9/index.m3u8", content=b"#EXTM3U\nseg-00000.ts\n").status_code == 201

  pub = TestClient(appmod.create_app(supervisor=_unreachable_sup(), storage=storage))
  AUTH = {"X-Auth-Request-User": "octocat"}
  pl = pub.get("/api/recordings/rec-9/playlist", headers=AUTH)
  assert pl.status_code == 200 and pl.content.startswith(b"#EXTM3U")
  rng = pub.get("/api/recordings/rec-9/segment/seg-00000.ts", headers={**AUTH, "Range": "bytes=0-1023"})
  assert rng.status_code == 206
  assert len(rng.content) == 1024
  assert rng.content == seg[:1024]


def _unreachable_sup():
  # A SupervisorClient whose backend always errors — the disk recording is
  # served by the storage backend, never proxied, so supervisor isn't consulted.
  import httpx

  from app.supervisor_client import SupervisorClient

  def handler(req):
    raise httpx.ConnectError("no supervisor in this test")

  http = httpx.Client(base_url="http://supervisor:8443", transport=httpx.MockTransport(handler))
  return SupervisorClient(base_url="http://supervisor:8443", client=http)
