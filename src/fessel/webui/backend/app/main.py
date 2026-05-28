"""webui backend (FastAPI) — Slice 1 surface.

Slice 1 is an internal end-to-end bring-up; OIDC gating arrives in Slice
2 (this endpoint will then refuse to sign without a valid session). For
now minting is open.

  GET /healthz
  GET /api/auth/whep-url?path=<path>&mode=<W>x<H>@<fps>@<bitrate>
      -> { "url": "https://<media-host>/<path>/whep?mode=<...>&token=<jwt>" }

Env config:
  FESSEL_WHEP_SECRET   shared HMAC/JWT secret (also held by mediamtx)
  FESSEL_MEDIA_BASE    public media host base, e.g. https://media-dev.example.com
  FESSEL_WHEP_TTL_S    token TTL seconds (default 30)
"""

from __future__ import annotations

import base64
import os
from urllib.parse import parse_qs, quote, urlencode

import jwt as pyjwt
from fastapi import FastAPI, HTTPException, Query, Response
from fessel_schemas import mode_from_canonical
from pydantic import BaseModel

from .token import mint_whep_token


def _b64url(data: bytes) -> str:
  return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _require_secret() -> str:
  secret = os.environ.get("FESSEL_WHEP_SECRET")
  if not secret:
    raise RuntimeError("FESSEL_WHEP_SECRET must be set (shared with mediamtx)")
  return secret


class MediamtxAuthRequest(BaseModel):
  """Body mediamtx POSTs to authHTTPAddress. Fields are all optional so a
  partial payload (or future field additions) doesn't 422."""

  user: str | None = None
  password: str | None = None
  ip: str | None = None
  action: str | None = None
  path: str | None = None
  protocol: str | None = None
  id: str | None = None
  query: str | None = None


def create_app() -> FastAPI:
  app = FastAPI(title="fessel-webui-backend")
  media_base = os.environ.get("FESSEL_MEDIA_BASE", "http://localhost:8889").rstrip("/")
  ttl_s = int(os.environ.get("FESSEL_WHEP_TTL_S", "30"))

  @app.get("/healthz")
  def healthz() -> dict:
    return {"status": "ok"}

  @app.get("/jwks")
  def jwks() -> dict:
    # mediamtx (authMethod: jwt) fetches this JWKS to validate the HS256
    # signature locally — no per-request callback to the backend. The
    # symmetric key is the shared WHEP secret exposed as an `oct` JWK.
    secret = _require_secret()
    from .token import WHEP_KID

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

  @app.post("/auth")
  def mediamtx_auth(req: MediamtxAuthRequest, response: Response):
    # mediamtx authMethod:http callback. Publish is excluded at mediamtx
    # (authHTTPExclude), so we only ever see read/playback here — gate those
    # on a valid WHEP JWT carried in the query (jwt=...). Returns 200 to
    # allow, 401 to deny (mediamtx treats non-2xx as deny).
    if req.action in ("api", "metrics", "pprof", "publish"):
      return {"ok": True}
    token = None
    if req.query:
      vals = parse_qs(req.query).get("jwt")
      if vals:
        token = vals[0]
    if not token:
      response.status_code = 401
      return {"error": "JWT not provided"}
    try:
      claims = pyjwt.decode(token, _require_secret(), algorithms=["HS256"])
    except pyjwt.PyJWTError as e:
      response.status_code = 401
      return {"error": f"invalid token: {e}"}
    # Path must match the token's path claim.
    if req.path and claims.get("path") not in (None, req.path):
      response.status_code = 403
      return {"error": "path mismatch"}
    return {"ok": True}

  @app.get("/api/capabilities")
  def capabilities() -> dict:
    # Slice-1 passthrough: a static dev set. In later slices the backend
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
  ) -> dict:
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
    )
    # mediamtx (authMethod: jwt) reads the JWT from the `jwt` query
    # parameter (or an Authorization: Bearer header). `mode` is carried
    # alongside for the runOnDemand pass-through.
    query = urlencode({"mode": mode, "jwt": token})
    url = f"{media_base}/{quote(path)}/whep?{query}"
    return {"url": url}

  # Serve the built React app at / when present (single-image deploy). The
  # API routes above are registered first, so they take precedence.
  static_dir = os.environ.get("FESSEL_STATIC_DIR", "/app/static")
  if os.path.isdir(static_dir):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

  return app


app = create_app()
