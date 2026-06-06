"""Backend WHEP-URL minting tests, incl. the token contract and auth gating."""

from urllib.parse import parse_qs, urlparse

import jwt
import pytest
from fastapi.testclient import TestClient

SECRET = "test-secret"

# Default oauth2-proxy identity header names (matches app.auth defaults).
USER_HEADER = "X-Auth-Request-User"
EMAIL_HEADER = "X-Auth-Request-Email"
# An authenticated operator, as oauth2-proxy would forward it.
AUTHED = {USER_HEADER: "octocat", EMAIL_HEADER: "octocat@example.com"}


@pytest.fixture()
def client(monkeypatch):
  monkeypatch.setenv("FESSEL_WHEP_SECRET", SECRET)
  monkeypatch.setenv("FESSEL_MEDIA_BASE", "https://media-test.example.com")
  monkeypatch.setenv("FESSEL_WHEP_TTL_S", "30")
  import app.main as appmod

  return TestClient(appmod.create_app())


def test_whep_url_mints_valid_token(client):
  r = client.get(
    "/api/auth/whep-url",
    params={"path": "pi", "mode": "1280x720@30@2500000"},
    headers=AUTHED,
  )
  assert r.status_code == 200
  url = r.json()["url"]
  parsed = urlparse(url)
  assert parsed.path == "/pi/whep"
  q = parse_qs(parsed.query)
  assert q["mode"] == ["1280x720@30@2500000"]

  # The token validates against the shared secret (the HMAC-over-claims
  # contract mediamtx will verify), and carries path + mode + exp.
  token = q["jwt"][0]
  claims = jwt.decode(token, SECRET, algorithms=["HS256"])
  assert claims["path"] == "pi"
  assert claims["mode"] == "1280x720@30@2500000"
  assert "exp" in claims
  assert claims["mediamtx_permissions"] == [{"action": "read", "path": "pi"}]
  # Operator identity is folded in as an audit aid (B2.2), and must NOT
  # leak into the mediamtx authorization claim.
  assert claims["operator"] == "octocat"


def test_whep_url_requires_identity(client):
  # No oauth2-proxy identity headers -> unauthenticated -> 401, no token.
  r = client.get("/api/auth/whep-url", params={"path": "pi", "mode": "1280x720@30@2500000"})
  assert r.status_code == 401


def test_capabilities_requires_identity(client):
  assert client.get("/api/capabilities").status_code == 401
  r = client.get("/api/capabilities", headers=AUTHED)
  assert r.status_code == 200
  assert r.json()["modes"]


def test_me_returns_identity(client):
  assert client.get("/api/me").status_code == 401
  r = client.get("/api/me", headers=AUTHED)
  assert r.status_code == 200
  assert r.json() == {"user": "octocat", "email": "octocat@example.com", "groups": []}


def test_bad_mode_rejected(client):
  r = client.get("/api/auth/whep-url", params={"path": "pi", "mode": "nope"}, headers=AUTHED)
  assert r.status_code == 400


def test_jwks_exposes_shared_key(client):
  r = client.get("/jwks")
  assert r.status_code == 200
  keys = r.json()["keys"]
  assert keys[0]["kty"] == "oct"
  assert keys[0]["alg"] == "HS256"
  # kid must be present and match the token header so mediamtx can resolve
  # the key (mediamtx errors "could not find kid in JWT header" otherwise).
  assert keys[0]["kid"]


def test_jwks_rejects_spoofed_identity_headers(client):
  # /jwks bypasses oauth2-proxy and is in-cluster only (B2.3). Identity
  # headers must never arrive here — their presence means a direct caller
  # is forging an identity, so the request is rejected.
  r = client.get("/jwks", headers=AUTHED)
  assert r.status_code == 400


PUBLIC_HOST = "fessel.example.com"


@pytest.fixture()
def public_client(monkeypatch):
  # Same app, but with the public-host gate armed (FESSEL_PUBLIC_WEBUI_HOST),
  # as the nodeport-mode Deployment sets it. Replaces the former ingress
  # server-snippet that 404'd /jwks on the public host (I2.1).
  monkeypatch.setenv("FESSEL_WHEP_SECRET", SECRET)
  monkeypatch.setenv("FESSEL_MEDIA_BASE", "https://media-test.example.com")
  monkeypatch.setenv("FESSEL_WHEP_TTL_S", "30")
  monkeypatch.setenv("FESSEL_PUBLIC_WEBUI_HOST", PUBLIC_HOST)
  import app.main as appmod

  return TestClient(appmod.create_app())


def test_jwks_404_on_public_host(public_client):
  # A request arriving on the public ingress host must not see the JWK
  # (the `oct` key is the signing secret). 404, not 403 — the public
  # surface doesn't admit the path exists.
  r = public_client.get("/jwks", headers={"host": PUBLIC_HOST})
  assert r.status_code == 404
  # Host with a port still matches (host:port is split before compare).
  r = public_client.get("/jwks", headers={"host": f"{PUBLIC_HOST}:443"})
  assert r.status_code == 404


def test_jwks_served_in_cluster_when_gate_armed(public_client):
  # mediamtx fetches via the in-cluster Service (Host: webui:8000); the gate
  # only fires for the public host, so the in-cluster fetch still works.
  r = public_client.get("/jwks", headers={"host": "webui:8000"})
  assert r.status_code == 200
  assert r.json()["keys"][0]["kty"] == "oct"


def test_jwt_header_has_matching_kid(client):
  r = client.get("/api/auth/whep-url", params={"path": "pi", "mode": "640x480@30@1000000"}, headers=AUTHED)
  token = parse_qs(urlparse(r.json()["url"]).query)["jwt"][0]
  jwks = client.get("/jwks").json()["keys"][0]
  header = jwt.get_unverified_header(token)
  assert header.get("kid") == jwks["kid"]


def test_token_rejected_with_wrong_secret(client):
  r = client.get("/api/auth/whep-url", params={"path": "pi", "mode": "640x480@30@1000000"}, headers=AUTHED)
  token = parse_qs(urlparse(r.json()["url"]).query)["jwt"][0]
  with pytest.raises(jwt.InvalidSignatureError):
    jwt.decode(token, "wrong", algorithms=["HS256"])
