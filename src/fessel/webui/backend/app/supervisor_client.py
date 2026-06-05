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
  body (or a synthesised body when supervisor was unreachable). The body is a
  dict for object responses and a list for array responses (e.g. /recordings),
  passed through to the frontend verbatim."""

  status_code: int
  body: dict | list


class SupervisorClient:
  def __init__(self, base_url: str | None = None, client: httpx.Client | None = None) -> None:
    self._base = (
      base_url or os.environ.get("FESSEL_SUPERVISOR_BASE", "http://supervisor:8443")
    ).rstrip("/")
    timeout = float(os.environ.get("FESSEL_SUPERVISOR_TIMEOUT_S", "10"))
    self._client = client or httpx.Client(
      base_url=self._base, timeout=timeout, headers={"User-Agent": USER_AGENT}
    )

  def _forward(self, method: str, path: str, json_body: dict | None = None) -> ForwardResult:
    try:
      resp = self._client.request(method, path, json=json_body)
    except httpx.HTTPError as e:
      # supervisor unreachable (egress down, Pi offline). Surface as 502 with
      # a structured body — the operator should see "couldn't reach the Pi",
      # distinct from supervisor's own 503 actuator failures.
      return ForwardResult(
        502, {"error": "supervisor_unreachable", "message": f"{type(e).__name__}: {e}"}
      )
    return ForwardResult(resp.status_code, _json_or_text(resp))

  def post(self, path: str, json_body: dict | None = None) -> ForwardResult:
    return self._forward("POST", path, json_body)

  def get(self, path: str) -> ForwardResult:
    return self._forward("GET", path)

  def get_bytes(self, path: str, headers: dict[str, str] | None = None) -> "ProxyResult":
    """Fetch a binary resource (ring/recording HLS file) from supervisor,
    passing through request headers (e.g. Range) and returning the body bytes +
    relevant response headers so the backend can proxy HLS playback (B4.5,
    B4.3/4 local path). Range support is end-to-end: supervisor's FileResponse
    honours Range and returns 206, which flows through here verbatim."""
    try:
      resp = self._client.request("GET", path, headers=headers or {})
    except httpx.HTTPError as e:
      return ProxyResult(502, b"", {}, error=f"{type(e).__name__}: {e}")
    # Forward the headers an HLS player needs for ranged playback + caching.
    passthrough = {
      k: v
      for k, v in resp.headers.items()
      if k.lower()
      in ("content-type", "content-length", "content-range", "accept-ranges", "cache-control")
    }
    return ProxyResult(resp.status_code, resp.content, passthrough)


@dataclass(frozen=True)
class ProxyResult:
  """A proxied binary response: status + body bytes + the subset of headers an
  HLS player needs (content-type, ranges, caching). `error` is set only when
  supervisor was unreachable (status 502)."""

  status_code: int
  body: bytes
  headers: dict
  error: str | None = None


def _json_or_text(resp: httpx.Response) -> dict | list:
  try:
    body = resp.json()
  except ValueError:
    return {"detail": resp.text}
  # Preserve both object and array bodies (the latter for /recordings); only a
  # bare scalar (rare) gets wrapped so the result is always a JSON container.
  if isinstance(body, (dict, list)):
    return body
  return {"value": body}
