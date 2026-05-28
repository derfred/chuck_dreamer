"""webui backend (FastAPI) — Slice 1 surface.

Slice 1 is an internal end-to-end bring-up; OIDC gating arrives in Slice
2 (this endpoint will then refuse to sign without a valid session). For
now minting is open.

  GET /healthz
  GET /jwks                  -> JWKS mediamtx uses to validate WHEP JWTs locally
  GET /api/auth/whep-url?path=<path>&mode=<W>x<H>@<fps>@<bitrate>
      -> { "url": "https://<media-host>/<path>/whep?mode=<...>&jwt=<jwt>" }

  No mediamtx auth callback: mediamtx validates the JWT itself against /jwks
  (authMethod: jwt), so webui is NOT in the per-request WHEP auth path.

Env config:
  FESSEL_WHEP_SECRET   shared HMAC/JWT secret (also held by mediamtx)
  FESSEL_MEDIA_BASE    public media host base, e.g. https://media-dev.example.com
  FESSEL_WHEP_TTL_S    token TTL seconds (default 30)
"""

from __future__ import annotations

import base64
import os
from urllib.parse import quote, urlencode

from fastapi import FastAPI, HTTPException, Query
from fessel_schemas import mode_from_canonical

from .token import mint_whep_token


def _b64url(data: bytes) -> str:
  return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _require_secret() -> str:
  secret = os.environ.get("FESSEL_WHEP_SECRET")
  if not secret:
    raise RuntimeError("FESSEL_WHEP_SECRET must be set (shared with mediamtx)")
  return secret


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
