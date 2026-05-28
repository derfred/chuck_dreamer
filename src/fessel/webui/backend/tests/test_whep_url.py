"""Backend WHEP-URL minting tests, incl. the token contract."""

import os
from urllib.parse import parse_qs, urlparse

import jwt
import pytest
from fastapi.testclient import TestClient

SECRET = "test-secret"


@pytest.fixture()
def client(monkeypatch):
  monkeypatch.setenv("FESSEL_WHEP_SECRET", SECRET)
  monkeypatch.setenv("FESSEL_MEDIA_BASE", "https://media-test.example.com")
  monkeypatch.setenv("FESSEL_WHEP_TTL_S", "30")
  import app.main as appmod

  return TestClient(appmod.create_app())


def test_whep_url_mints_valid_token(client):
  r = client.get("/api/auth/whep-url", params={"path": "pi", "mode": "1280x720@30@2500000"})
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


def test_bad_mode_rejected(client):
  r = client.get("/api/auth/whep-url", params={"path": "pi", "mode": "nope"})
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


def test_jwt_header_has_matching_kid(client):
  r = client.get("/api/auth/whep-url", params={"path": "pi", "mode": "640x480@30@1000000"})
  token = parse_qs(urlparse(r.json()["url"]).query)["jwt"][0]
  jwks = client.get("/jwks").json()["keys"][0]
  header = jwt.get_unverified_header(token)
  assert header.get("kid") == jwks["kid"]


def _mint_jwt(client, path="pi", mode="640x480@30@1000000"):
  r = client.get("/api/auth/whep-url", params={"path": path, "mode": mode})
  return parse_qs(urlparse(r.json()["url"]).query)["jwt"][0]


def test_auth_allows_read_with_valid_jwt(client):
  tok = _mint_jwt(client)
  r = client.post("/auth", json={"action": "read", "path": "pi", "protocol": "webrtc", "query": f"mode=640x480@30@1000000&jwt={tok}"})
  assert r.status_code == 200


def test_auth_denies_read_without_jwt(client):
  r = client.post("/auth", json={"action": "read", "path": "pi", "query": "mode=640x480@30@1000000"})
  assert r.status_code == 401


def test_auth_denies_tampered_jwt(client):
  tok = _mint_jwt(client)
  bad = tok[:-3] + ("aaa" if not tok.endswith("aaa") else "bbb")
  r = client.post("/auth", json={"action": "read", "path": "pi", "query": f"jwt={bad}"})
  assert r.status_code == 401


def test_auth_allows_publish_without_jwt(client):
  # publish is excluded at mediamtx, but if it ever reaches /auth it must
  # be allowed (the Pi's trusted SRT ingest carries no token).
  r = client.post("/auth", json={"action": "publish", "path": "pi", "query": ""})
  assert r.status_code == 200


def test_token_rejected_with_wrong_secret(client):
  r = client.get("/api/auth/whep-url", params={"path": "pi", "mode": "640x480@30@1000000"})
  token = parse_qs(urlparse(r.json()["url"]).query)["jwt"][0]
  with pytest.raises(jwt.InvalidSignatureError):
    jwt.decode(token, "wrong", algorithms=["HS256"])
