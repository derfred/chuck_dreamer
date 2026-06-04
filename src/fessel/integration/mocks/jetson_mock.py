#!/usr/bin/env python3
"""Mock Jetson control API for the integration harness (sketch §2.2).

Stands in for the real Jetson HTTP control server (J3.1) so the per-PR test can
exercise supervisor's REAL JetsonClient (httpx, timeouts, User-Agent, error
mapping) end-to-end without a Jetson. It is the leaf behaviour that is mocked;
everything talking to it is real.

Stdlib only (no fessel deps), so it runs in a bare python:3-slim with the script
mounted from a ConfigMap — no new image to build.

Contract (mirrors J3.1 exactly, including the resume-when-not-paused 409):

  POST /control/pause   -> 200 {"state": "paused"}
  POST /control/stop    -> 200 {"state": "stopped"}
  POST /control/resume  -> 200 {"state": "active"}   | 409 if not currently paused
  GET  /state           -> 200 {"state": <current>, "home": true}

In-memory state; a single replica (the mock is stateful per the test run, like
the real device). Listens on :8080 by default (JETSON_MOCK_PORT to override).
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("JETSON_MOCK_PORT", "8080"))

# Module-level state: the mock is a single replica, so one in-memory value is
# the whole "device". Starts active (the real Jetson is running when the test
# begins).
_state = {"value": "active"}


class Handler(BaseHTTPRequestHandler):
  def _send(self, code: int, body: dict) -> None:
    payload = json.dumps(body).encode()
    self.send_response(code)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(payload)))
    self.end_headers()
    self.wfile.write(payload)

  def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler dispatch name
    if self.path == "/state":
      self._send(200, {"state": _state["value"], "home": True})
    elif self.path == "/healthz":
      self._send(200, {"status": "ok"})
    else:
      self._send(404, {"error": "not found", "path": self.path})

  def do_POST(self) -> None:  # noqa: N802
    if self.path == "/control/pause":
      _state["value"] = "paused"
      self._send(200, {"state": "paused"})
    elif self.path == "/control/stop":
      _state["value"] = "stopped"
      self._send(200, {"state": "stopped"})
    elif self.path == "/control/resume":
      # The real contract: resume only from paused, else 409 (J3.1).
      if _state["value"] != "paused":
        self._send(409, {"error": "not paused", "state": _state["value"]})
      else:
        _state["value"] = "active"
        self._send(200, {"state": "active"})
    else:
      self._send(404, {"error": "not found", "path": self.path})

  def log_message(self, fmt: str, *args) -> None:  # noqa: ANN002
    # One structured-ish line per request so the CI log shows the mock's view.
    print(f"[jetson-mock] {fmt % args}", flush=True)


def main() -> None:
  server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)  # noqa: S104 — test-only, in-cluster
  print(f"[jetson-mock] listening on :{PORT}", flush=True)
  server.serve_forever()


if __name__ == "__main__":
  main()
