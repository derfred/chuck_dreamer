"""Backend Slice-4 recordings API tests (B4.1–B4.6).

Load-bearing behaviours:
  - recording control forwarders require auth and fold the operator identity
    into the start request body (so V4.4 records who started it);
  - /api/recordings merges supervisor (Pi-local) + MinIO (uploaded) and tags
    available_local / available_remote;
  - playlist/segment serve a MinIO presigned redirect when uploaded, else proxy
    from supervisor (with Range passthrough for HLS scrubbing);
  - the ring is always proxied (no MinIO path);
  - upload progress is a read-only proxy of supervisor's cache.
"""

import httpx
from fastapi.testclient import TestClient

from app.recordings_store import FakeRecordingsStore
from app.supervisor_client import SupervisorClient

USER_HEADER = "X-Auth-Request-User"
AUTHED = {USER_HEADER: "octocat"}


def make_client(handler, *, store=None):
  http = httpx.Client(base_url="http://supervisor:8443", transport=httpx.MockTransport(handler))
  sup = SupervisorClient(base_url="http://supervisor:8443", client=http)
  import app.main as appmod

  return TestClient(appmod.create_app(supervisor=sup, recordings_store=store or FakeRecordingsStore()))


# --- B4.1 control forwarders -------------------------------------------------


def test_recording_start_requires_auth():
  client = make_client(lambda r: httpx.Response(200, json={}))
  assert client.post("/api/recording/start").status_code == 401


def test_recording_start_folds_operator_into_body():
  calls = []

  def handler(req: httpx.Request) -> httpx.Response:
    calls.append((req.url.path, req.content))
    return httpx.Response(200, json={"recording_id": "sup-id"})

  client = make_client(handler)
  r = client.post("/api/recording/start", headers=AUTHED, json={"mode": "1920x1080@30@8000000"})
  assert r.status_code == 200 and r.json()["recording_id"] == "sup-id"
  import json

  path, body = calls[-1]
  assert path == "/recording/start"
  assert json.loads(body)["operator"] == "octocat"
  assert json.loads(body)["mode"] == "1920x1080@30@8000000"


def test_recording_stop_and_flag_forward():
  paths = []

  def handler(req: httpx.Request) -> httpx.Response:
    paths.append(req.url.path)
    return httpx.Response(200, json={"ok": True})

  client = make_client(handler)
  assert client.post("/api/recording/stop", headers=AUTHED).status_code == 200
  assert client.post(
    "/api/recording/flag-upload", headers=AUTHED, json={"recording_id": "r1"}
  ).status_code == 200
  assert paths == ["/recording/stop", "/recording/flag-upload"]


def test_flag_upload_passes_through_404():
  def handler(req: httpx.Request) -> httpx.Response:
    return httpx.Response(404, json={"detail": "unknown recording"})

  client = make_client(handler)
  r = client.post("/api/recording/flag-upload", headers=AUTHED, json={"recording_id": "ghost"})
  assert r.status_code == 404


# --- B4.2 merged recordings list ---------------------------------------------


def test_recordings_list_merges_local_and_remote():
  def handler(req: httpx.Request) -> httpx.Response:
    assert req.url.path == "/recordings"
    return httpx.Response(
      200,
      json=[
        {
          "recording_id": "r-local",
          "started_at": "2026-06-04T01:00:00+00:00",
          "ended_at": None,
          "duration_seconds": 60,
          "operator": "octocat",
          "flagged_for_upload": False,
          "upload_state": "none",
        },
        {
          "recording_id": "r-both",
          "started_at": "2026-06-04T02:00:00+00:00",
          "flagged_for_upload": True,
          "upload_state": "uploaded",
        },
      ],
    )

  store = FakeRecordingsStore(
    recordings={"r-both": ["index.m3u8"], "r-remote-only": ["index.m3u8"]}
  )
  client = make_client(handler, store=store)
  recs = client.get("/api/recordings", headers=AUTHED).json()
  by = {r["recording_id"]: r for r in recs}
  assert by["r-local"]["available_local"] and not by["r-local"]["available_remote"]
  assert by["r-both"]["available_local"] and by["r-both"]["available_remote"]
  assert by["r-remote-only"]["available_remote"] and not by["r-remote-only"]["available_local"]
  # Sorted newest-first by started_at; the remote-only (no started_at) sorts last.
  assert [r["recording_id"] for r in recs][:2] == ["r-both", "r-local"]


def test_recordings_list_requires_auth():
  client = make_client(lambda r: httpx.Response(200, json=[]))
  assert client.get("/api/recordings").status_code == 401


# --- B4.3/B4.4 playback: presigned redirect vs proxy -------------------------


def test_uploaded_playlist_redirects_to_minio():
  store = FakeRecordingsStore(recordings={"r1": ["index.m3u8", "seg-00000.ts"]})
  client = make_client(lambda r: httpx.Response(404), store=store)
  r = client.get("/api/recordings/r1/playlist", headers=AUTHED, follow_redirects=False)
  assert r.status_code == 302
  assert "X-Amz-Signature" in r.headers["location"]
  seg = client.get("/api/recordings/r1/segment/seg-00000.ts", headers=AUTHED, follow_redirects=False)
  assert seg.status_code == 302


def test_local_only_playlist_proxied_from_supervisor():
  def handler(req: httpx.Request) -> httpx.Response:
    assert req.url.path == "/recordings/r-local/index.m3u8"
    return httpx.Response(
      200, content=b"#EXTM3U-local", headers={"content-type": "application/vnd.apple.mpegurl"}
    )

  client = make_client(handler)  # empty store -> no remote -> proxy
  r = client.get("/api/recordings/r-local/playlist", headers=AUTHED)
  assert r.status_code == 200 and r.content == b"#EXTM3U-local"


# --- B4.5 ring pass-through (always local) ------------------------------------


def test_ring_proxied_with_range_passthrough():
  def handler(req: httpx.Request) -> httpx.Response:
    if req.url.path == "/ring/index.m3u8":
      return httpx.Response(200, content=b"#EXTM3U", headers={"content-type": "application/vnd.apple.mpegurl"})
    assert req.headers.get("range") == "bytes=0-3"
    return httpx.Response(
      206, content=b"PART", headers={"content-type": "video/mp2t", "content-range": "bytes 0-3/100"}
    )

  client = make_client(handler)
  assert client.get("/api/ring/playlist", headers=AUTHED).status_code == 200
  rng = client.get("/api/ring/segment/seg-00000.ts", headers={**AUTHED, "Range": "bytes=0-3"})
  assert rng.status_code == 206 and rng.headers.get("content-range") == "bytes 0-3/100"


def test_ring_requires_auth():
  client = make_client(lambda r: httpx.Response(200))
  assert client.get("/api/ring/playlist").status_code == 401


def test_ring_supervisor_unreachable_is_502():
  def handler(req: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("egress down")

  client = make_client(handler)
  r = client.get("/api/ring/playlist", headers=AUTHED)
  assert r.status_code == 502
  assert r.json()["error"] == "supervisor_unreachable"


# --- B4.6 upload progress proxy ----------------------------------------------


def test_upload_progress_proxied():
  def handler(req: httpx.Request) -> httpx.Response:
    assert req.url.path == "/recordings/r1/upload"
    return httpx.Response(200, json={"recording_id": "r1", "state": "uploaded", "attempts": 2})

  client = make_client(handler)
  body = client.get("/api/recordings/r1/upload", headers=AUTHED).json()
  assert body["state"] == "uploaded" and body["attempts"] == 2
