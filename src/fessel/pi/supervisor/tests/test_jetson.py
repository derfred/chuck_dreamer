"""Jetson HTTP control client tests (S3.2).

Load-bearing behaviours:
  - every call carries the descriptive User-Agent (so the Jetson side and
    logs can attribute the traffic to supervisor);
  - an optional pre-shared token is sent as a Bearer header when configured;
  - a non-2xx response is raised as JetsonError carrying the status (not
    swallowed);
  - a transport error (timeout / connection refused) is raised as JetsonError.
"""

import httpx
import pytest

from supervisor.jetson import USER_AGENT, JetsonClient, JetsonConfig, JetsonError


def client_with(handler, *, auth_token=None):
  transport = httpx.MockTransport(handler)
  http = httpx.Client(
    base_url="http://jetson:8080",
    transport=transport,
    headers=(
      {"User-Agent": USER_AGENT} | ({"Authorization": f"Bearer {auth_token}"} if auth_token else {})
    ),
  )
  return JetsonClient(
    JetsonConfig(base_url="http://jetson:8080", auth_token=auth_token), client=http
  )


def test_pause_sends_user_agent_and_returns_body():
  seen = {}

  def handler(request: httpx.Request) -> httpx.Response:
    seen["ua"] = request.headers.get("user-agent")
    seen["path"] = request.url.path
    return httpx.Response(200, json={"state": "paused"})

  c = client_with(handler)
  assert c.pause() == {"state": "paused"}
  assert seen["ua"] == USER_AGENT
  assert seen["path"] == "/control/pause"


def test_auth_token_sent_as_bearer():
  seen = {}

  def handler(request: httpx.Request) -> httpx.Response:
    seen["auth"] = request.headers.get("authorization")
    return httpx.Response(200, json={})

  c = client_with(handler, auth_token="s3cr3t")
  c.resume()
  assert seen["auth"] == "Bearer s3cr3t"


def test_non_2xx_raises_with_status():
  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, text="boom")

  c = client_with(handler)
  with pytest.raises(JetsonError) as ei:
    c.stop()
  assert ei.value.status == 500
  assert ei.value.action == "stop"


def test_transport_error_raises():
  def handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused")

  c = client_with(handler)
  with pytest.raises(JetsonError) as ei:
    c.state()
  assert ei.value.status is None


def test_state_returns_jetson_view():
  def handler(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/state"
    return httpx.Response(200, json={"state": "active", "home": True})

  c = client_with(handler)
  assert c.state() == {"state": "active", "home": True}
