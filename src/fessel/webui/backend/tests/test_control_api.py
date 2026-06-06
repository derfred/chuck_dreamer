"""Backend control-plane forwarder tests (B3.1–B3.3).

Load-bearing behaviours:
  - every /api/control/* and /api/state requires the oauth2-proxy identity
    (401 without it) — the same gate the WHEP mint uses;
  - the backend forwards 1:1 to the matching supervisor path;
  - supervisor's status + structured body pass through verbatim — a 5xx
    actuator failure reaches the frontend as a 5xx with the diagnostic;
  - supervisor unreachable surfaces as a 502 (distinct from supervisor's own
    5xx), so the operator can tell "couldn't reach the Pi" from "actuator
    failed";
  - the backend does NOT retry on a 5xx (one operator intent -> one forward).
"""

import json
import logging

import httpx
from fastapi.testclient import TestClient

from app.supervisor_client import SupervisorClient

USER_HEADER = "X-Auth-Request-User"
AUTHED = {USER_HEADER: "octocat"}


def _control_audit_lines(caplog) -> list[dict]:
  """The structured control-audit JSON lines captured from app.main's logger."""
  out = []
  for rec in caplog.records:
    try:
      obj = json.loads(rec.getMessage())
    except (ValueError, TypeError):
      continue
    if isinstance(obj, dict) and obj.get("event") == "control":
      out.append(obj)
  return out


def make_client(handler, *, raise_exc=None):
  """Build a backend TestClient whose SupervisorClient is wired to a
  MockTransport `handler` (no real Pi). `raise_exc` makes every forward raise
  to simulate an unreachable supervisor."""

  def _handler(request: httpx.Request) -> httpx.Response:
    if raise_exc is not None:
      raise raise_exc
    return handler(request)

  http = httpx.Client(base_url="http://supervisor:8443", transport=httpx.MockTransport(_handler))
  sup = SupervisorClient(base_url="http://supervisor:8443", client=http)
  import app.main as appmod

  return TestClient(appmod.create_app(supervisor=sup))


def _ok(body):
  def handler(request: httpx.Request) -> httpx.Response:
    handler.calls.append((request.method, request.url.path))
    return httpx.Response(200, json=body)

  handler.calls = []
  return handler


def test_pause_forwards_and_passes_through():
  handler = _ok({"jetson": {"state": "paused"}})
  client = make_client(handler)
  r = client.post("/api/control/pause", headers=AUTHED)
  assert r.status_code == 200
  assert r.json() == {"jetson": {"state": "paused"}}
  assert handler.calls == [("POST", "/control/pause")]


def test_all_actions_map_to_supervisor_paths():
  expected = {
    "pause": "/control/pause",
    "stop": "/control/stop",
    "resume": "/control/resume",
    "shutdown/arm": "/control/shutdown/arm",
    "shutdown/jetson": "/control/shutdown/jetson",
    "poweron/arm": "/control/poweron/arm",
    "poweron/jetson": "/control/poweron/jetson",
  }
  for action, sup_path in expected.items():
    handler = _ok({"ok": True})
    client = make_client(handler)
    r = client.post(f"/api/control/{action}", headers=AUTHED)
    assert r.status_code == 200, action
    assert handler.calls == [("POST", sup_path)], action


def test_control_requires_identity():
  client = make_client(_ok({}))
  assert client.post("/api/control/pause").status_code == 401


def test_control_auth_missing_is_audited(caplog):
  # B3.3: a rejected (unauthenticated) control attempt must STILL leave an audit
  # record — outcome `auth_missing` — so the trail shows who-tried-what even when
  # refused. The supervisor must NOT be called.
  handler = _ok({})
  client = make_client(handler)
  with caplog.at_level(logging.INFO, logger="app.main"):
    assert client.post("/api/control/shutdown/arm").status_code == 401
  lines = _control_audit_lines(caplog)
  assert any(
    line["action"] == "shutdown/arm" and line["outcome"] == "auth_missing"
    for line in lines
  ), f"no auth_missing audit line: {lines}"
  # The Pi was never reached.
  assert handler.calls == []


def test_control_success_is_audited(caplog):
  handler = _ok({"jetson": {"state": "paused"}})
  client = make_client(handler)
  with caplog.at_level(logging.INFO, logger="app.main"):
    assert client.post("/api/control/pause", headers=AUTHED).status_code == 200
  lines = _control_audit_lines(caplog)
  assert any(
    line["action"] == "pause" and line["outcome"] == "success" and line["operator"] == "octocat"
    for line in lines
  ), f"no success audit line: {lines}"


def test_state_requires_identity():
  client = make_client(_ok({}))
  assert client.get("/api/state").status_code == 401


def test_state_passes_through_supervisor_body():
  body = {
    "safety_state": "SHUTDOWN_ARM",
    "jetson": {"state": "stopped"},
    "plugs": {"arm": {"on": False, "verified": True, "verified_at": "t"}},
    "camera": {"up": True},
  }

  def handler(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/state"
    return httpx.Response(200, json=body)

  client = make_client(handler)
  r = client.get("/api/state", headers=AUTHED)
  assert r.status_code == 200
  assert r.json() == body  # no reshaping, no added fields


def test_actuator_5xx_passes_through_with_diagnostic():
  # supervisor's plug-verify failure (503 + structured body) must reach the
  # frontend unchanged so it can render the diagnostic.
  detail = {
    "detail": {
      "error": "plug_verify_failed",
      "plug": "arm",
      "message": "plug reported on after issuing off",
    }
  }

  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, json=detail)

  client = make_client(handler)
  r = client.post("/api/control/shutdown/arm", headers=AUTHED)
  assert r.status_code == 503
  assert r.json()["detail"]["error"] == "plug_verify_failed"


def test_supervisor_unreachable_is_502():
  client = make_client(_ok({}), raise_exc=httpx.ConnectError("egress down"))
  r = client.post("/api/control/stop", headers=AUTHED)
  assert r.status_code == 502


# --- Slice 5: /api/anomalies pass-through (B5.2) -----------------------------


def test_anomalies_requires_identity():
  client = make_client(_ok([]))
  assert client.get("/api/anomalies").status_code == 401


def test_anomalies_passes_through_supervisor_body():
  log = [
    {
      "ts": "2026-06-05T00:00:00Z",
      "type": "safe_box_violation",
      "anomaly_recording_id": "rec-1",
      "cleared": False,
      "payload": {"outside_fraction": 0.18},
    }
  ]

  def handler(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/anomalies"
    return httpx.Response(200, json=log)

  client = make_client(handler)
  r = client.get("/api/anomalies", headers=AUTHED)
  assert r.status_code == 200
  assert r.json() == log  # pass-through, no reshaping


def test_no_retry_on_5xx():
  # Exactly one forward per operator intent (clean audit trail, no hidden
  # side effects). A 503 must not trigger a backend retry.
  calls = []

  def handler(request: httpx.Request) -> httpx.Response:
    calls.append(request.url.path)
    return httpx.Response(503, json={"detail": "actuator failed"})

  client = make_client(handler)
  client.post("/api/control/shutdown/jetson", headers=AUTHED)
  assert calls == ["/control/shutdown/jetson"]
