"""webui backend (FastAPI) — Slice 2 surface.

Slice 2 gates the WHEP-URL mint behind operator identity forwarded by
oauth2-proxy (GitHub OIDC). The architecture's headline property — "no
Pi-side encoding without a valid auth check upstream" — becomes true for
real operators, not just under a shared-secret bring-up.

  GET /healthz
  GET /jwks                  -> JWKS mediamtx uses to validate WHEP JWTs locally.
                                Bypasses oauth2-proxy; in-cluster only; rejects
                                any request that carries identity headers (B2.3).
  GET /api/me                -> the trusted operator identity (requires auth)
  GET /api/capabilities      -> camera capability triplets (requires auth)
  GET /api/auth/whep-url?path=<path>&mode=<W>x<H>@<fps>@<bitrate>
      -> { "url": "https://<media-host>/<path>/whep?mode=<...>&jwt=<jwt>" }
         (requires auth; identity is folded into the JWT as an audit aid)

  No mediamtx auth callback: mediamtx validates the JWT itself against /jwks
  (authMethod: jwt), so webui is NOT in the per-request WHEP auth path.

Slice 3 adds the control-plane forwarders (all require auth):

  GET  /api/state                -> supervisor /state (pass-through)
  POST /api/control/pause        -> supervisor /control/pause
  POST /api/control/stop         -> supervisor /control/stop
  POST /api/control/resume       -> supervisor /control/resume
  POST /api/control/shutdown/{arm,jetson}  -> supervisor /control/shutdown/...
  POST /api/control/poweron/{arm,jetson}   -> supervisor /control/poweron/...

The backend is a pass-through: supervisor's status code and structured body
flow to the frontend verbatim (a 5xx actuator failure stays a 5xx). Every
action emits a B3.3 audit log line (operator, action, outcome, latency).

Slice 5.5 fronts the recording store: the Pi-side uploader PUTs each recording
file to a tailnet-only recording-ingest endpoint, and the backend persists it
via a configured storage backend (MinIO or disk). Playback is served the same
way to the frontend regardless of backend — a 302 redirect to a MinIO presigned
URL, or a backend-served byte range for disk.

Two listeners (B5.5.7, settled by the deploy decision):
  - PUBLIC app  (this module's `app`)        -> browser surface behind
    oauth2-proxy + the in-cluster /jwks fetch. ALL Pi->webui traffic is kept
    OFF this listener.
  - INGEST app  (`ingest_app`, create_ingest_app) -> tailnet-only. Serves
    `/recording-ingest/...` (and any future Pi->webui endpoint). The public
    Ingress never routes to the ingest port, so it is structurally impossible
    to reach the ingest endpoint through the public ingress.

Endpoint classes (see webui/deploy/README.md):
  - Behind oauth2-proxy (interactive): /, /live, /api/...  -> require identity.
  - Bypass oauth2-proxy, in-cluster (machine-to-machine): /jwks -> reject identity.
  - Tailnet-only, separate listener (Pi -> backend): /recording-ingest/...

Env config:
  FESSEL_WHEP_SECRET       shared HMAC/JWT secret (also held by mediamtx)
  FESSEL_MEDIA_BASE        public media host base, e.g. https://media-dev.example.com
  FESSEL_WHEP_TTL_S        token TTL seconds (default 30)
  FESSEL_AUTH_*_HEADER     identity header names (see app.auth)
  FESSEL_SUPERVISOR_BASE   supervisor base URL via Tailscale egress (default http://supervisor:8443)
  FESSEL_SUPERVISOR_TIMEOUT_S  forward timeout seconds (default 10)
  FESSEL_PUBLIC_WEBUI_HOST public ingress host (e.g. fessel.example.com); when
                           set, /jwks 404s for requests arriving on it (the JWK
                           is the signing secret and must stay in-cluster only).
                           Unset -> no host gate (the in-cluster default).
  FESSEL_RECORDINGS_BACKEND  "minio" | "disk" (recordings_storage.backend)
  FESSEL_RECORDINGS_DISK_PATH  mounted PVC path (disk backend)
  FESSEL_MINIO_*           endpoint/bucket/keys (minio backend)
  FESSEL_INGEST_PORT       ingest listener port (default 8001; bound by serve_both)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from tempfile import SpooledTemporaryFile
from urllib.parse import quote, urlencode

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from fessel_schemas import fessel_version, mode_from_canonical

from .auth import AuthHeaders, Identity
from .storage import PresignedURL, ServeLocally, StorageBackend, build_storage_backend
from .supervisor_client import SupervisorClient
from .token import WHEP_KID, mint_whep_token

log = logging.getLogger(__name__)

# Operator control actions -> the supervisor path each forwards to (B3.1).
# One source of truth so the endpoints and the audit log agree on names.
CONTROL_ACTIONS: dict[str, str] = {
  "pause": "/control/pause",
  "stop": "/control/stop",
  "resume": "/control/resume",
  "shutdown/arm": "/control/shutdown/arm",
  "shutdown/jetson": "/control/shutdown/jetson",
  "poweron/arm": "/control/poweron/arm",
  "poweron/jetson": "/control/poweron/jetson",
}


def _b64url(data: bytes) -> str:
  return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _require_secret() -> str:
  secret = os.environ.get("FESSEL_WHEP_SECRET")
  if not secret:
    raise RuntimeError("FESSEL_WHEP_SECRET must be set (shared with mediamtx)")
  return secret


def _audit_outcome(status_code: int) -> str:
  # Map a forwarded status to an audit outcome tag (shared by the control and
  # recording forwarders): 2xx -> success; our 502 -> unreachable; else the
  # supervisor status (a 503 actuator failure, a 404/409 recording guard, …).
  if 200 <= status_code < 300:
    return "success"
  if status_code == 502:
    return "supervisor_unreachable"
  return f"supervisor_{status_code}"


async def _safe_json_body(request: Request) -> dict:
  # Recording POSTs may carry a JSON body (flag-upload: recording_id) or none
  # (stop). Tolerate an empty/non-JSON body as {}.
  try:
    body = await request.json()
  except Exception:  # noqa: BLE001 — empty or non-JSON body
    return {}
  return body if isinstance(body, dict) else {}


def create_app(
  supervisor: SupervisorClient | None = None,
  storage: StorageBackend | None = None,
) -> FastAPI:
  app        = FastAPI(title="fessel-webui-backend")
  media_base = os.environ.get("FESSEL_MEDIA_BASE", "http://localhost:8889").rstrip("/")
  ttl_s      = int(os.environ.get("FESSEL_WHEP_TTL_S", "30"))
  # Public ingress host. /jwks must never be reachable on it (the `oct` JWK *is*
  # the signing secret); mediamtx fetches /jwks via the in-cluster Service
  # (Host: webui:8000), not this host. Empty -> no host gate (in-cluster dev).
  public_webui_host = os.environ.get("FESSEL_PUBLIC_WEBUI_HOST", "").split(":")[0].lower()
  headers    = AuthHeaders()
  supervisor = supervisor if supervisor is not None else SupervisorClient()
  # The storage backend (Slice 5.5): listing + playback for uploaded recordings
  # come from here. Built lazily so a test that only exercises control/live
  # endpoints needn't configure a backend; the recordings endpoints build one
  # on first use via _get_storage().
  _storage = storage

  def _get_storage() -> StorageBackend:
    nonlocal _storage
    if _storage is None:
      _storage = build_storage_backend()
    return _storage

  def require_identity(request: Request) -> Identity:
    """Dependency for proxied endpoints: 401 unless oauth2-proxy forwarded identity."""
    identity = headers.read(request)
    if identity is None:
      raise HTTPException(status_code=401, detail="authentication required")
    return identity

  def forbid_identity_headers(request: Request) -> None:
    """Dependency for proxy-bypass endpoints (B2.3).

    These endpoints must never be reached through oauth2-proxy, so any
    identity header on them is anomalous — a direct caller forging an
    identity. Reject the request rather than trust the headers.
    """
    for name in headers.names():
      if request.headers.get(name):
        raise HTTPException(
          status_code=400,
          detail="identity headers not permitted on bypass endpoint",
        )

  def forbid_public_host(request: Request) -> None:
    """Dependency for /jwks: 404 if the request arrived on the public host.

    The JWK is an `oct` key — it *is* the WHEP signing secret — so /jwks must
    stay reachable only in-cluster. mediamtx fetches it via the in-cluster
    Service (Host: webui:8000); a public request carries the ingress host
    (FESSEL_PUBLIC_WEBUI_HOST). This is the network-level half of the I2.1
    proxy-bypass split that the ingress used to enforce with a server-snippet
    (`location = /jwks { return 404; }`); doing it in the app keeps the block
    when the ingress controller disallows snippet annotations. forbid_identity
    _headers remains the spoofed-identity half. 404 (not 403) so the public
    surface doesn't even admit the path exists.
    """
    if not public_webui_host:
      return
    host = request.headers.get("host", "").split(":")[0].lower()
    if host == public_webui_host:
      raise HTTPException(status_code=404, detail="not found")

  @app.get("/healthz")
  def healthz() -> dict:
    # `version` is the release stamp (image tag == dpkg version). A deployed
    # cluster webui and Pi supervisor should report the same string; a
    # mismatch means the two halves of a release drifted.
    return {"status": "ok", "version": fessel_version()}

  @app.get("/jwks", dependencies=[Depends(forbid_public_host), Depends(forbid_identity_headers)])
  def jwks() -> dict:
    # mediamtx (authMethod: jwt) fetches this JWKS to validate the HS256
    # signature locally — no per-request callback to the backend. The
    # symmetric key is the shared WHEP secret exposed as an `oct` JWK.
    #
    # This endpoint bypasses oauth2-proxy (a login redirect would break
    # mediamtx's fetch) and is in-cluster only (an `oct` JWK *is* the
    # signing secret). forbid_identity_headers guards against a spoofed
    # identity arriving here directly.
    secret = _require_secret()
    return {
      "keys": [
        {
          "kty": "oct",
          "alg": "HS256",
          "use": "sig",
          "kid": WHEP_KID,
          "k": _b64url(secret.encode("utf-8")),
        }
      ]
    }

  @app.get("/api/me")
  def me(identity: Identity = Depends(require_identity)) -> dict:
    return {"user": identity.user, "email": identity.email, "groups": list(identity.groups)}

  @app.get("/api/capabilities")
  def capabilities(identity: Identity = Depends(require_identity)) -> dict:
    # Slice-2 passthrough: a static dev set. In later slices the backend
    # reads the retained arm/video/capabilities through supervisor.
    from fessel_schemas import Capabilities, ModeTriplet

    caps = Capabilities(
      modes=[
        ModeTriplet(resolution="640x480", fps=30, bitrate_bps=1_000_000),
        ModeTriplet(resolution="1280x720", fps=30, bitrate_bps=2_500_000),
        ModeTriplet(resolution="1280x720", fps=15, bitrate_bps=1_500_000),
      ]
    )
    return caps.model_dump()

  @app.get("/api/auth/whep-url")
  def whep_url(
    path: str = Query(..., min_length=1),
    mode: str = Query(..., description="canonical <W>x<H>@<fps>@<bitrate_bps>"),
    identity: Identity = Depends(require_identity),
  ) -> dict:
    # The endpoint sits behind oauth2-proxy; reaching it with identity
    # headers means the proxy already validated the GitHub session. The
    # backend's job is just to require the proxy did its job.
    #
    # Validate the mode is a well-formed canonical triplet (and reuse the
    # parsed form for signing) so signer and verifier agree on the string.
    try:
      mode_triplet = mode_from_canonical(mode)
    except ValueError as e:
      raise HTTPException(status_code=400, detail=str(e)) from e

    token = mint_whep_token(
      secret=_require_secret(),
      path=path,
      mode=mode_triplet,
      ttl_seconds=ttl_s,
      operator=identity.user,
    )
    # mediamtx (authMethod: jwt) reads the JWT from the `jwt` query
    # parameter (or an Authorization: Bearer header). `mode` is carried
    # alongside for the runOnDemand pass-through.
    query = urlencode({"mode": mode, "jwt": token})
    url = f"{media_base}/{quote(path)}/whep?{query}"
    return {"url": url}

  # --- Slice 3 control plane (B3.1–B3.3) ------------------------------------
  # All endpoints sit behind oauth2-proxy: they require a forwarded operator
  # identity (401 without it). The backend forwards to supervisor and passes
  # its status + structured body through verbatim — no retries, no reshaping.
  # Every action emits an audit log line keyed on the operator identity.

  def _audit(action: str, operator: str, outcome: str, t0: float) -> None:
    log.info(
      json.dumps(
        {
          "event": "control",
          "action": action,
          "operator": operator,
          "outcome": outcome,
          "latency_ms": int((time.monotonic() - t0) * 1000),
          "timestamp": datetime.now(timezone.utc).isoformat(),
        }
      )
    )

  def _reject_unauthenticated(action: str, request: Request) -> None:
    """B3.3: a control/recording attempt with no operator identity is REJECTED
    (401) AND audited as `auth_missing`, so the trail records who-tried-what even
    when the attempt is refused. Read endpoints keep the plain `require_identity`
    dependency (no audit); only state-changing actions log a rejection."""
    if headers.read(request) is None:
      _audit(action, "unknown", "auth_missing", time.monotonic())
      raise HTTPException(status_code=401, detail="authentication required")

  def _forward_control(action: str, identity: Identity) -> JSONResponse:
    t0     = time.monotonic()
    result = supervisor.post(CONTROL_ACTIONS[action])
    # Map the forwarded status to an audit outcome: 2xx -> success, supervisor
    # 5xx (actuator/Jetson failure) and our 502 (unreachable) -> distinct tags.
    outcome = _audit_outcome(result.status_code)
    _audit(action, identity.user, outcome, t0)
    # Pass supervisor's status + body straight through to the frontend.
    return JSONResponse(status_code=result.status_code, content=result.body)

  def _make_control_endpoint(action: str):
    # Bind `action` per-route (closure over the loop variable would alias).
    # Read identity in-handler (not via Depends) so a missing-auth rejection is
    # audited as `auth_missing` (B3.3) before the 401.
    def endpoint(request: Request) -> JSONResponse:
      _reject_unauthenticated(action, request)
      return _forward_control(action, headers.read(request))

    return endpoint

  for action in CONTROL_ACTIONS:
    app.add_api_route(
      f"/api/control/{action}",
      _make_control_endpoint(action),
      methods=["POST"],
      name=f"control_{action.replace('/', '_')}",
    )

  @app.get("/api/state")
  def api_state(identity: Identity = Depends(require_identity)) -> JSONResponse:
    # Pass-through of supervisor's /state. The backend does not reshape or add
    # fields; the dashboard consumes supervisor's StateResponse directly. Slice 5
    # adds vision/audio/recent_anomalies, which propagate automatically (B5.1).
    result = supervisor.get("/state")
    return JSONResponse(status_code=result.status_code, content=result.body)

  @app.get("/api/anomalies")
  def api_anomalies(identity: Identity = Depends(require_identity)) -> JSONResponse:
    # B5.2: pass-through of supervisor's /anomalies (the full in-memory anomaly
    # log). Behind oauth2-proxy; read-only, so no audit logging.
    result = supervisor.get("/anomalies")
    return JSONResponse(status_code=result.status_code, content=result.body)

  # --- Slice 4: recording control + recordings/ring API (B4.1–B4.6) ----------
  # All sit behind oauth2-proxy (auth required). Control forwarders fold the
  # operator identity into the supervisor request body so metadata.json (V4.4)
  # records who started the recording, and emit a B3.3-style audit line.

  @app.post("/api/recording/start")
  async def recording_start(request: Request) -> JSONResponse:
    return await _forward_recording(request, "recording/start", "/recording/start")

  @app.post("/api/recording/stop")
  async def recording_stop(request: Request) -> JSONResponse:
    return await _forward_recording(request, "recording/stop", "/recording/stop")

  @app.post("/api/recording/flag-upload")
  async def recording_flag_upload(request: Request) -> JSONResponse:
    return await _forward_recording(request, "recording/flag-upload", "/recording/flag-upload")

  def _audit_recording(action: str, operator: str, outcome: str, rec_id, t0: float) -> None:
    log.info(
      json.dumps(
        {
          "event": "recording",
          "action": action,
          "operator": operator,
          "recording_id": rec_id,
          "outcome": outcome,
          "latency_ms": int((time.monotonic() - t0) * 1000),
          "timestamp": datetime.now(timezone.utc).isoformat(),
        }
      )
    )

  async def _forward_recording(request: Request, action: str, sup_path: str) -> JSONResponse:
    t0 = time.monotonic()
    # B3.3: a recording control attempt with no operator identity is rejected
    # AND audited as `auth_missing` (read identity in-handler, not via Depends).
    identity = headers.read(request)
    if identity is None:
      _audit_recording(action, "unknown", "auth_missing", None, t0)
      raise HTTPException(status_code=401, detail="authentication required")
    body = await _safe_json_body(request)
    # Fold the trusted operator identity into start (so V4.4 records it). stop
    # carries no body; flag-upload carries the recording_id from the client.
    if action == "recording/start":
      body = {**body, "operator": identity.user}
    result = supervisor.post(sup_path, json_body=body or None)
    outcome = _audit_outcome(result.status_code)
    rec_id = (
      result.body.get("recording_id") if isinstance(result.body, dict) else None
    ) or body.get("recording_id")
    _audit_recording(action, identity.user, outcome, rec_id, t0)
    return JSONResponse(status_code=result.status_code, content=result.body)

  @app.get("/api/recordings")
  def api_recordings(identity: Identity = Depends(require_identity)) -> JSONResponse:
    # B5.5.5: unified list from supervisor (Pi-local) + the storage backend
    # (uploaded, via the new ingest path), deduplicated by recording_id, each
    # tagged available_local/available_remote. The "remote" half now comes from
    # the configured backend's list() (MinIO listing or disk walk) rather than a
    # hardcoded MinIO listing — the backend choice is invisible here.
    sup = supervisor.get("/recordings")
    local: list[dict] = sup.body if isinstance(sup.body, list) else []
    remote = {v.recording_id: v for v in _get_storage().list()}

    merged: dict[str, dict] = {}
    for item in local:
      rid = item.get("recording_id")
      if not rid:
        continue
      merged[rid] = {
        "recording_id": rid,
        # Slice 5 (S5.4/B5.3): explicit vs anomaly, from supervisor's listing.
        "type": item.get("type", "explicit"),
        "started_at": item.get("started_at"),
        "ended_at": item.get("ended_at"),
        "duration_seconds": item.get("duration_seconds"),
        "operator": item.get("operator"),
        "flagged_for_upload": item.get("flagged_for_upload", False),
        "upload_state": item.get("upload_state", "none"),
        "available_local": True,
        "available_remote": rid in remote,
      }
    # Recordings present only in the backend (Pi-side copy removed — out of
    # scope, but the shape must not assume Pi-only). metadata.json is the
    # uploader's last file, so a recording missing it is still "in progress":
    # surface upload_state accordingly so the frontend can show that.
    for rid, view in remote.items():
      if rid in merged:
        continue
      meta = view.metadata or {}
      complete = view.metadata is not None
      merged[rid] = {
        "recording_id": rid,
        "type": view.rec_type,
        "started_at": meta.get("started_at"),
        "ended_at": meta.get("ended_at"),
        "duration_seconds": meta.get("duration_seconds"),
        "operator": meta.get("operator"),
        "flagged_for_upload": True,
        "upload_state": "uploaded" if complete else "uploading",
        "available_local": False,
        "available_remote": True,
      }
    out = sorted(merged.values(), key=lambda r: r.get("started_at") or "", reverse=True)
    return JSONResponse(content=out)

  @app.get("/api/recordings/{recording_id}/playlist")
  def api_recording_playlist(
    recording_id: str, request: Request, identity: Identity = Depends(require_identity)
  ) -> Response:
    return _serve_recording_file(recording_id, "index.m3u8", request)

  @app.get("/api/recordings/{recording_id}/segment/{name}")
  def api_recording_segment(
    recording_id: str, name: str, request: Request, identity: Identity = Depends(require_identity)
  ) -> Response:
    return _serve_recording_file(recording_id, name, request)

  def _serve_recording_file(recording_id: str, filename: str, request: Request) -> Response:
    # B5.5.4 uniform front, backend-specific behaviour. Prefer the storage
    # backend (an uploaded recording):
    #   - MinIO backend -> PresignedURL -> 302 redirect (browser fetches the
    #     bytes straight from MinIO; the backend never touches them).
    #   - disk backend  -> ServeLocally -> 200/206 byte stream from read()
    #     (Range honoured end-to-end; hls.js issues ranged GETs).
    # A recording present only on the Pi (not yet uploaded) is proxied from
    # supervisor, as before.
    try:
      target = _get_storage().playback_url(recording_id, filename)
    except ValueError:
      # Backend rejected the id/name (path-safety on the disk backend).
      raise HTTPException(status_code=400, detail="invalid recording path") from None
    if isinstance(target, PresignedURL):
      return RedirectResponse(url=target.url, status_code=302)
    if isinstance(target, ServeLocally):
      return _serve_from_disk_backend(recording_id, filename, request)
    sup_path = (
      f"/recordings/{recording_id}/{filename}"
      if filename != "index.m3u8"
      else f"/recordings/{recording_id}/index.m3u8"
    )
    return _proxy_binary(sup_path, request)

  def _serve_from_disk_backend(recording_id: str, filename: str, request: Request) -> Response:
    # Stream the file from the disk backend, honouring an HTTP Range header so
    # hls.js can walk segments (an explicit test asserts Range -> 206).
    rng = request.headers.get("range")
    try:
      result = _get_storage().read(recording_id, filename, rng)
    except ValueError:
      # Malformed / unsatisfiable Range -> 416 (parse_byte_range raised).
      return Response(status_code=416, headers={"Accept-Ranges": "bytes"})
    if result is None:
      raise HTTPException(status_code=404, detail="not found")
    base_headers = {
      "Content-Type": result.content_type,
      "Content-Length": str(result.content_length),
      "Accept-Ranges": "bytes",
    }
    if result.byte_range is not None:
      r = result.byte_range
      base_headers["Content-Range"] = f"bytes {r.start}-{r.end}/{r.total}"
      status = 206
    else:
      status = 200
    return StreamingResponse(result.chunks, status_code=status, headers=base_headers)

  @app.get("/api/ring/playlist")
  def api_ring_playlist(
    request: Request, identity: Identity = Depends(require_identity)
  ) -> Response:
    # B4.5: the ring is always local — no MinIO path. Range support is
    # end-to-end (frontend -> here -> supervisor -> file).
    return _proxy_binary("/ring/index.m3u8", request)

  @app.get("/api/ring/segment/{name}")
  def api_ring_segment(
    name: str, request: Request, identity: Identity = Depends(require_identity)
  ) -> Response:
    return _proxy_binary(f"/ring/{name}", request)

  def _proxy_binary(sup_path: str, request: Request) -> Response:
    # Pass the Range header through so HLS scrubbing works; forward supervisor's
    # status (200/206/404) + the HLS-relevant response headers verbatim.
    fwd_headers = {}
    rng = request.headers.get("range")
    if rng:
      fwd_headers["Range"] = rng
    result = supervisor.get_bytes(sup_path, headers=fwd_headers)
    if result.error is not None:
      return JSONResponse(
        status_code=502, content={"error": "supervisor_unreachable", "message": result.error}
      )
    return Response(
      content=result.body, status_code=result.status_code, headers=result.headers
    )

  @app.get("/api/recordings/{recording_id}/upload")
  def api_recording_upload(
    recording_id: str, identity: Identity = Depends(require_identity)
  ) -> JSONResponse:
    # B4.6: read-only proxy of supervisor's cached per-recording upload progress.
    result = supervisor.get(f"/recordings/{recording_id}/upload")
    return JSONResponse(status_code=result.status_code, content=result.body)

  # Serve the built React app at / when present (single-image deploy). The
  # API routes above are registered first, so they take precedence.
  static_dir = os.environ.get("FESSEL_STATIC_DIR", "/app/static")
  if os.path.isdir(static_dir):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

  return app


# --- ingest listener (B5.5.6/B5.5.7) -----------------------------------------
# A SEPARATE ASGI app bound to a SEPARATE port (FESSEL_INGEST_PORT). The
# `webui-recording-ingest` Tailscale Service targets this port; the public
# Ingress never does. Because `/recording-ingest/...` exists ONLY on this app,
# it is structurally impossible to reach it through the public listener — the
# two-listener model the deploy chose. ALL Pi->backend traffic lives here.
#
# Auth is at the network layer (Tailscale identity); there is no oauth2-proxy
# in front, no application-layer token. The backend trusts what arrives on the
# tailnet (B5.5.6): the body is HLS bytes written by `video`, persisted as-is.


def create_ingest_app(storage: StorageBackend | None = None) -> FastAPI:
  ingest = FastAPI(title="fessel-webui-backend-ingest")
  _storage = storage

  def _get_storage() -> StorageBackend:
    nonlocal _storage
    if _storage is None:
      _storage = build_storage_backend()
    return _storage

  @ingest.get("/healthz")
  def healthz() -> dict:
    return {"status": "ok", "version": fessel_version(), "listener": "ingest"}

  @ingest.put("/recording-ingest/{recording_id}/{file_name}")
  async def recording_ingest(recording_id: str, file_name: str, request: Request) -> Response:
    # Two impedance mismatches to bridge: the body is async (Starlette) while the
    # storage backends are sync, and the only S3 client we have (the `minio` SDK)
    # is blocking with no async variant — so a write must run in a worker thread
    # regardless of backend.
    #
    # Rather than hand-roll a producer/consumer queue, spool the body to a
    # SpooledTemporaryFile: it keeps the bytes in memory up to a threshold and
    # transparently rolls over to a real temp file beyond it, so memory stays
    # bounded even for a large segment. Then hand the store a sync generator over
    # that file in ONE `to_thread.run_sync` call. Per-file PUT matches the
    # uploader's per-file retry: a repeated PUT overwrites cleanly (store() is
    # idempotent).
    import anyio

    t0 = time.monotonic()
    storage = _get_storage()

    with SpooledTemporaryFile(max_size=_INGEST_SPOOL_MAX) as spool:
      size = 0
      async for chunk in request.stream():
        if chunk:
          spool.write(chunk)
          size += len(chunk)
      spool.seek(0)
      try:
        await anyio.to_thread.run_sync(
          storage.store, recording_id, file_name, _file_chunks(spool)
        )
      except ValueError as e:
        # Path-safety / bad id or name -> 400 (malformed request).
        _ingest_log(recording_id, file_name, size, "rejected", t0)
        raise HTTPException(status_code=400, detail=str(e)) from e
      except Exception as e:  # noqa: BLE001 — backend store failure -> 5xx, Pi retries
        _ingest_log(recording_id, file_name, size, "store_error", t0)
        raise HTTPException(status_code=502, detail=f"store failed: {type(e).__name__}") from e

    _ingest_log(recording_id, file_name, size, "stored", t0)
    return Response(status_code=201)

  return ingest


# Spool body in memory up to this size, then roll over to a temp file on disk.
# Sized above a typical HLS segment so the common case stays in memory, while a
# pathologically large upload still can't exhaust it.
_INGEST_SPOOL_MAX = 16 * 1024 * 1024


def _file_chunks(fh, chunk_size: int = 1024 * 1024):
  """Yield `fh` in chunk_size pieces — the sync iterable the storage backends'
  store() consume. Runs inside the worker thread (the file read is blocking)."""
  while True:
    data = fh.read(chunk_size)
    if not data:
      return
    yield data


def _ingest_log(recording_id: str, file_name: str, size: int, outcome: str, t0: float) -> None:
  # B5.5.6: one structured line per PUT — the Loki audit trail for uploads.
  log.info(
    json.dumps(
      {
        "event": "recording_ingest",
        "recording_id": recording_id,
        "file": file_name,
        "size_bytes": size,
        "outcome": outcome,
        "latency_ms": int((time.monotonic() - t0) * 1000),
        "timestamp": datetime.now(timezone.utc).isoformat(),
      }
    )
  )


# Public listener (browser surface, behind oauth2-proxy) and the tailnet-only
# ingest listener share one storage backend instance so a uploaded-then-played
# round trip is consistent within the process. Build it once here (lazily, via
# the factory) and inject into both apps.
def _shared_storage() -> StorageBackend | None:
  # None when no backend is configured (e.g. a control-plane-only dev run): the
  # apps then build one on first recordings/ingest use and surface the config
  # error there rather than failing the whole process at import.
  try:
    return build_storage_backend()
  except RuntimeError:
    return None


_shared = _shared_storage()
app = create_app(storage=_shared)
ingest_app = create_ingest_app(storage=_shared)


def serve_both() -> None:
  """Run BOTH listeners in one process (the container entrypoint): the public
  app on :8000 and the ingest app on FESSEL_INGEST_PORT (default 8001).

  uvicorn.Server doesn't multiplex two apps on one process out of the box, so
  run each server in its own asyncio task. Both share the in-process storage
  backend built above.
  """
  import asyncio

  import uvicorn

  ingest_port = int(os.environ.get("FESSEL_INGEST_PORT", "8001"))
  public_port = int(os.environ.get("FESSEL_PORT", "8000"))

  async def _run() -> None:
    public = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=public_port, log_level="info"))
    ingest = uvicorn.Server(
      uvicorn.Config(ingest_app, host="0.0.0.0", port=ingest_port, log_level="info")
    )
    await asyncio.gather(public.serve(), ingest.serve())

  asyncio.run(_run())
