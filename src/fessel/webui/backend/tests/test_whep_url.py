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
  token = q["token"][0]
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


def test_token_rejected_with_wrong_secret(client):
  r = client.get("/api/auth/whep-url", params={"path": "pi", "mode": "640x480@30@1000000"})
  token = parse_qs(urlparse(r.json()["url"]).query)["token"][0]
  with pytest.raises(jwt.InvalidSignatureError):
    jwt.decode(token, "wrong", algorithms=["HS256"])
