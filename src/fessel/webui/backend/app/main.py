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

Endpoint classes (see webui/deploy/README.md):
  - Behind oauth2-proxy (interactive): /, /live, /api/...  -> require identity.
  - Bypass oauth2-proxy (machine-to-machine): /jwks         -> reject identity.

Env config:
  FESSEL_WHEP_SECRET       shared HMAC/JWT secret (also held by mediamtx)
  FESSEL_MEDIA_BASE        public media host base, e.g. https://media-dev.example.com
  FESSEL_WHEP_TTL_S        token TTL seconds (default 30)
  FESSEL_AUTH_*_HEADER     identity header names (see app.auth)
  FESSEL_SUPERVISOR_BASE   supervisor base URL via Tailscale egress (default http://supervisor:8443)
  FESSEL_SUPERVISOR_TIMEOUT_S  forward timeout seconds (default 10)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from urllib.parse import quote, urlencode

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fessel_schemas import fessel_version, mode_from_canonical

from .auth import AuthHeaders, Identity
from .recordings_store import RecordingsStore, build_recordings_store
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
  recordings_store: RecordingsStore | None = None,
) -> FastAPI:
  app        = FastAPI(title="fessel-webui-backend")
  media_base = os.environ.get("FESSEL_MEDIA_BASE", "http://localhost:8889").rstrip("/")
  ttl_s      = int(os.environ.get("FESSEL_WHEP_TTL_S", "30"))
  headers    = AuthHeaders()
  supervisor = supervisor if supervisor is not None else SupervisorClient()
  store      = recordings_store if recordings_store is not None else build_recordings_store()

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

  @app.get("/healthz")
  def healthz() -> dict:
    # `version` is the release stamp (image tag == dpkg version). A deployed
    # cluster webui and Pi supervisor should report the same string; a
    # mismatch means the two halves of a release drifted.
    return {"status": "ok", "version": fessel_version()}

  @app.get("/jwks", dependencies=[Depends(forbid_identity_headers)])
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
    # B4.2: unified list from supervisor (Pi-local) + MinIO (uploaded),
    # deduplicated by recording_id, each tagged available_local/available_remote.
    sup = supervisor.get("/recordings")
    local: list[dict] = sup.body if isinstance(sup.body, list) else []
    remote = {r.recording_id: r for r in store.list_recordings()}

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
    # Recordings present only in MinIO (Pi-side copy removed — out of Slice 4
    # scope, but the shape must not assume Pi-only).
    for rid in remote:
      if rid not in merged:
        merged[rid] = {
          "recording_id": rid,
          "type": "explicit",
          "started_at": None,
          "ended_at": None,
          "duration_seconds": None,
          "operator": None,
          "flagged_for_upload": True,
          "upload_state": "uploaded",
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
    # B4.3/B4.4 two-path model: an uploaded recording is served straight from
    # MinIO via a presigned redirect (offloads Pi bandwidth); a local-only one
    # is proxied from supervisor. Prefer remote when available.
    url = store.presigned_url(recording_id, filename)
    if url is not None:
      return RedirectResponse(url=url, status_code=302)
    sup_path = (
      f"/recordings/{recording_id}/{filename}"
      if filename != "index.m3u8"
      else f"/recordings/{recording_id}/index.m3u8"
    )
    return _proxy_binary(sup_path, request)

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


app = create_app()
