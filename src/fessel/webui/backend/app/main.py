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

Endpoint classes (see webui/deploy/README.md):
  - Behind oauth2-proxy (interactive): /, /live, /api/...  -> require identity.
  - Bypass oauth2-proxy (machine-to-machine): /jwks         -> reject identity.

Env config:
  FESSEL_WHEP_SECRET   shared HMAC/JWT secret (also held by mediamtx)
  FESSEL_MEDIA_BASE    public media host base, e.g. https://media-dev.example.com
  FESSEL_WHEP_TTL_S    token TTL seconds (default 30)
  FESSEL_AUTH_*_HEADER identity header names (see app.auth)
"""

from __future__ import annotations

import base64
import os
from urllib.parse import quote, urlencode

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fessel_schemas import fessel_version, mode_from_canonical

from .auth import AuthHeaders, Identity
from .token import WHEP_KID, mint_whep_token


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
  headers = AuthHeaders()

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

  # Serve the built React app at / when present (single-image deploy). The
  # API routes above are registered first, so they take precedence.
  static_dir = os.environ.get("FESSEL_STATIC_DIR", "/app/static")
  if os.path.isdir(static_dir):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

  return app


app = create_app()
