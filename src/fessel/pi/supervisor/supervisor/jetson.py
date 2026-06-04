"""Jetson HTTP control client (S3.2).

supervisor calls the Jetson's control API over the **local LAN** (the Jetson
is not on the tailnet — all Jetson<->Pi traffic is local). The client is
intentionally thin: it relays pause/stop/resume and reads /state, and it
surfaces failures honestly rather than swallowing them. supervisor does NOT
implement business logic on top of it; if the Jetson returns 200 the command
is considered relayed.

A non-2xx response or a timeout is raised as JetsonError, which the control
endpoints translate into a 5xx with a structured body. Long timeouts hide
failures on a local LAN, so the timeout is short and configurable.

Auth between supervisor and the Jetson is an optional pre-shared token in a
header, agreed with the Jetson team (J3.1). When unset, no auth header is
sent (the choice is documented Jetson-side and tracked as a config item here).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

# Descriptive User-Agent so the Jetson side and logs can attribute the traffic
# to supervisor (Slice 1 convention for supervisor's outbound HTTP).
USER_AGENT = "fessel-supervisor"


class JetsonError(RuntimeError):
  """Raised when the Jetson call failed (timeout, connection error, or non-2xx).

  `status` is the Jetson's HTTP status when one was received, else None.
  """

  def __init__(self, action: str, detail: str, status: int | None = None) -> None:
    super().__init__(detail)
    self.action = action
    self.detail = detail
    self.status = status


@dataclass(frozen=True)
class JetsonConfig:
  base_url: str
  timeout_s: float = 3.0
  auth_token: str | None = None


class JetsonClient:
  """Thin HTTP client over the Jetson control API.

  The httpx client factory is injected so tests can supply a transport/mock;
  production builds a default httpx.Client with the short timeout.
  """

  def __init__(self, cfg: JetsonConfig, client: httpx.Client | None = None) -> None:
    self._cfg     = cfg
    self._headers = {"User-Agent": USER_AGENT}
    if cfg.auth_token:
      self._headers["Authorization"] = f"Bearer {cfg.auth_token}"
    self._client = client or httpx.Client(
      base_url=cfg.base_url.rstrip("/"),
      timeout=cfg.timeout_s,
      headers=self._headers,
    )

  def _post(self, action: str, path: str) -> dict:
    try:
      resp = self._client.post(path)
    except httpx.HTTPError as e:
      raise JetsonError(action, f"{type(e).__name__}: {e}") from e
    if resp.status_code >= 400:
      raise JetsonError(action, f"jetson returned {resp.status_code}", status=resp.status_code)
    return _json_or_empty(resp)

  def pause(self) -> dict:
    return self._post("pause", "/control/pause")

  def stop(self) -> dict:
    return self._post("stop", "/control/stop")

  def resume(self) -> dict:
    return self._post("resume", "/control/resume")

  def state(self) -> dict:
    """Read the Jetson's current state. Used by the background refresh and the
    /state aggregation. Raises JetsonError on failure (caller decides whether
    a stale cached value is preferable to surfacing the error)."""
    try:
      resp = self._client.get("/state")
    except httpx.HTTPError as e:
      raise JetsonError("state", f"{type(e).__name__}: {e}") from e
    if resp.status_code >= 400:
      raise JetsonError("state", f"jetson returned {resp.status_code}", status=resp.status_code)
    return _json_or_empty(resp)

  def close(self) -> None:
    self._client.close()


def _json_or_empty(resp: httpx.Response) -> dict:
  try:
    body = resp.json()
  except ValueError:
    return {}
  return body if isinstance(body, dict) else {"value": body}
