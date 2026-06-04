"""Backend -> supervisor client over Tailscale egress (B3.1/B3.2).

The backend forwards operator control commands to supervisor and reads its
/state. supervisor is reached via the cluster->Pi Tailscale egress Service
(architecture §5.1); from the backend's point of view that is just a base URL
(e.g. http://supervisor:8443) that the egress proxy tunnels to the Pi.

The backend is a *pass-through*: it does NOT add retries or business logic on
top of supervisor. supervisor's status code and structured body flow through
to the frontend verbatim — a 5xx (actuator failure) reaches the frontend as a
5xx so the operator sees the real diagnostic and decides whether to retry.
This keeps the audit trail clean (one log line per operator intent).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

# Descriptive User-Agent so supervisor's logs can attribute the call to the
# backend (the same convention supervisor uses for its own outbound HTTP).
USER_AGENT = "fessel-webui-backend"


@dataclass(frozen=True)
class ForwardResult:
  """The raw outcome of a forwarded call: supervisor's status + parsed JSON
  body (or a synthesised body when supervisor was unreachable)."""

  status_code: int
  body: dict


class SupervisorClient:
  def __init__(self, base_url: str | None = None, client: httpx.Client | None = None) -> None:
    self._base = (
      base_url or os.environ.get("FESSEL_SUPERVISOR_BASE", "http://supervisor:8443")
    ).rstrip("/")
    timeout = float(os.environ.get("FESSEL_SUPERVISOR_TIMEOUT_S", "10"))
    self._client = client or httpx.Client(
      base_url=self._base, timeout=timeout, headers={"User-Agent": USER_AGENT}
    )

  def _forward(self, method: str, path: str) -> ForwardResult:
    try:
      resp = self._client.request(method, path)
    except httpx.HTTPError as e:
      # supervisor unreachable (egress down, Pi offline). Surface as 502 with
      # a structured body — the operator should see "couldn't reach the Pi",
      # distinct from supervisor's own 503 actuator failures.
      return ForwardResult(
        502, {"error": "supervisor_unreachable", "message": f"{type(e).__name__}: {e}"}
      )
    return ForwardResult(resp.status_code, _json_or_text(resp))

  def post(self, path: str) -> ForwardResult:
    return self._forward("POST", path)

  def get(self, path: str) -> ForwardResult:
    return self._forward("GET", path)


def _json_or_text(resp: httpx.Response) -> dict:
  try:
    body = resp.json()
  except ValueError:
    return {"detail": resp.text}
  return body if isinstance(body, dict) else {"value": body}
