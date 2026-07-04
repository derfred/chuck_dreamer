"""Integration test driver (Go webui + in-process Pion relay).

Runs in-cluster as a Job. Acts as the WHEP viewer and asserts the full
live chain against the deployed Fessel system, emitting JUnit XML:

  POST /whep (identity header) -> health gate -> supervisor
  /control/live/activate -> MQTT -> video attaches whipclientsink ->
  WHIP ingest at webui:8001 -> RTP into the shared track -> viewer.

The WHEP client is aiortc (a real Python ICE/DTLS/SRTP peer) — no browser.
The integration env runs the relay in `podip` mode, so both Pion (webui pod)
and aiortc (this pod) gather plain host candidates on routable pod IPs and
ICE completes in-cluster. That gives this tier a genuine DATA-PLANE
assertion (decoded frames in the viewer), which the old headless-Chrome
setup could never do; the relay's Prometheus metrics on
/metrics (fessel_relay_ingest_live, fessel_relay_ingest_packets_total, ...)
independently prove the Pi -> relay ingest leg.

Service DNS (in-namespace):
  WEBUI=http://webui:8000  WEBUI_INGEST=http://webui:8001
  SUPERVISOR=http://supervisor:8443
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

WEBUI = os.environ.get("WEBUI", "http://webui:8000")
SUPERVISOR = os.environ.get("SUPERVISOR", "http://supervisor:8443")
# The backend's tailnet-only ingest listener (B5.5.7). In the integration env
# it's exposed on the in-cluster webui Service's `ingest` port (8001). The
# auth/bypass tests (T5.5.3) PUT here vs. the public listener.
WEBUI_INGEST = os.environ.get("WEBUI_INGEST", "http://webui:8001")
MODE = os.environ.get("FESSEL_MODE", "640x480@30@1000000")
JUNIT_OUT = os.environ.get("JUNIT_OUT", "/results/junit.xml")

RECONNECT_CYCLES = int(os.environ.get("RECONNECT_CYCLES", "20"))
# The relay's idle timeout (FESSEL_LIVE_IDLE_TIMEOUT_S, default 10s) plus the
# Pi-side detach; teardown asserts within this window + margin.
IDLE_TIMEOUT_S = float(os.environ.get("FESSEL_LIVE_IDLE_TIMEOUT_S", "10"))

# oauth2-proxy is out of the integration loop (Slice 2 §6): the driver reaches
# webui directly via the in-cluster Service, which is the same network position
# oauth2-proxy forwards from. /whep and the /api endpoints are gated on the
# identity headers the proxy would inject, so the driver supplies them itself —
# a faithful stand-in for the proxy. Header name + value are overridable to
# match the backend's FESSEL_AUTH_USER_HEADER config.
AUTH_USER_HEADER = os.environ.get("FESSEL_AUTH_USER_HEADER", "X-Auth-Request-User")
AUTH_USER = os.environ.get("FESSEL_TEST_OPERATOR", "integration-driver")


# ---------- tiny JUnit emitter ----------


@dataclass
class Case:
  name: str
  ok: bool = False
  err: str | None = None
  duration: float = 0.0
  # known_gap: a documented, non-blocking expected failure. Recorded as
  # skipped in JUnit and excluded from the suite's pass/fail decision.
  known_gap: str | None = None


@dataclass
class Suite:
  name: str = "fessel-integration"
  cases: list[Case] = field(default_factory=list)

  def run(self, name: str, fn, known_gap: str | None = None) -> bool:
    c = Case(name=name, known_gap=known_gap)
    t0 = time.time()
    try:
      fn()
      c.ok = True
      print(f"[PASS] {name}", flush=True)
    except Exception as e:  # noqa: BLE001
      c.err = f"{type(e).__name__}: {e}"
      if known_gap:
        print(f"[XFAIL] {name} (known gap: {known_gap}): {c.err}", flush=True)
      else:
        print(f"[FAIL] {name}: {c.err}", flush=True)
    c.duration = time.time() - t0
    self.cases.append(c)
    return c.ok

  def blocking_failures(self) -> int:
    # known-gap cases never block, pass or fail.
    return sum(1 for c in self.cases if not c.ok and not c.known_gap)

  def to_junit(self) -> str:
    failures = self.blocking_failures()
    skipped = sum(1 for c in self.cases if not c.ok and c.known_gap)
    out = [
      '<?xml version="1.0" encoding="UTF-8"?>',
      f'<testsuite name="{self.name}" tests="{len(self.cases)}" '
      f'failures="{failures}" skipped="{skipped}">',
    ]
    for c in self.cases:
      out.append(f'  <testcase name="{_xml(c.name)}" time="{c.duration:.2f}">')
      if not c.ok and c.known_gap:
        # Emit as skipped so the gap is visible but non-blocking.
        out.append(f'    <skipped message="{_xml(c.known_gap)}: {_xml(c.err or "")}"></skipped>')
      elif not c.ok:
        out.append(f'    <failure message="{_xml(c.err or "")}"></failure>')
      out.append("  </testcase>")
    out.append("</testsuite>")
    return "\n".join(out)


def _xml(s: str) -> str:
  return (
    s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
  )


# ---------- helpers ----------


def http_get_json(url: str, timeout: float = 5.0, headers: dict | None = None) -> dict:
  req = urllib.request.Request(url, headers=headers or {})  # noqa: S310
  with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
    return json.loads(r.read())


def supervisor_live() -> dict:
  return http_get_json(f"{SUPERVISOR}/state/live")


def wait_state(target: str, timeout: float) -> list[str]:
  """Wait until supervisor's live-state history contains `target`."""
  deadline = time.time() + timeout
  hist: list[str] = []
  while time.time() < deadline:
    hist = supervisor_live().get("history", [])
    if target in hist:
      return hist
    time.sleep(0.5)
  raise AssertionError(f"state {target!r} not reached; history={hist}")


def current_state() -> str | None:
  cur = supervisor_live().get("current")
  return cur.get("state") if cur else None


def wait_off(timeout: float, what: str) -> None:
  deadline = time.time() + timeout
  while time.time() < deadline:
    if current_state() in (None, "off"):
      return
    time.sleep(1)
  raise AssertionError(f"{what}: state did not return to off; current={current_state()}")


# ---------- relay metrics (the data-plane witness for the ingest leg) ----------

_METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9eE+.\-]+)$")


def metrics_samples() -> dict[str, float]:
  """Fetch WEBUI/metrics and return {name{labels}: value} for every sample."""
  req = urllib.request.Request(f"{WEBUI}/metrics")  # noqa: S310
  with urllib.request.urlopen(req, timeout=5) as r:  # noqa: S310
    text = r.read().decode()
  out: dict[str, float] = {}
  for line in text.splitlines():
    m = _METRIC_RE.match(line.strip())
    if m:
      out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
  return out


def metric_value(name: str) -> float:
  """Sum a metric across all its label sets (0.0 if absent — counters with
  labels don't exist until first incremented)."""
  return sum(v for k, v in metrics_samples().items() if k == name or k.startswith(name + "{"))


def wait_metric(name: str, predicate, what: str, timeout: float = 20.0) -> float:
  deadline = time.time() + timeout
  last = 0.0
  while time.time() < deadline:
    last = metric_value(name)
    if predicate(last):
      return last
    time.sleep(0.5)
  raise AssertionError(f"timed out waiting for {what}; last {name}={last}")


# ---------- Slice 3 control-plane helpers ----------
# These reach WEBUI (not supervisor directly), so the cluster-side auth gate +
# /api/control forwarder + /api/state forwarder are all in the loop. The Pi
# side is real (control endpoints, send-and-verify, /state cache, real
# JetsonClient); only the leaf actuators are mocked (jetson-mock pod + the
# in-process fake WiZ bulb). See integration/README.md "Slice 3 control plane".

AUTH = {AUTH_USER_HEADER: AUTH_USER}


def api_state() -> dict:
  return http_get_json(f"{WEBUI}/api/state", headers=AUTH)


def control_post(action: str, headers: dict | None = None) -> tuple[int, dict]:
  """POST /api/control/<action>. Returns (status, body). Identity header
  defaults to the authenticated operator; pass headers={} to test the gate."""
  req = urllib.request.Request(  # noqa: S310
    f"{WEBUI}/api/control/{action}",
    method="POST",
    headers=AUTH if headers is None else headers,
  )
  try:
    with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310
      return r.status, json.loads(r.read() or b"{}")
  except urllib.error.HTTPError as e:
    body = e.read()
    try:
      return e.code, json.loads(body)
    except ValueError:
      return e.code, {"detail": body.decode(errors="replace")}


def wait_api_state(predicate, what: str, timeout: float = 15.0) -> dict:
  """Poll /api/state until predicate(state) holds (the dashboard's view)."""
  deadline = time.time() + timeout
  last: dict = {}
  while time.time() < deadline:
    last = api_state()
    try:
      if predicate(last):
        return last
    except (KeyError, TypeError):
      pass
    time.sleep(0.5)
  raise AssertionError(f"timed out waiting for {what}; last /api/state={last}")


# ---------- Slice 4 recording + upload (X4.3, recording E2E via ingest) ----------


def webui_post_json(path: str, body: dict | None = None) -> tuple[int, dict]:
  data = json.dumps(body).encode() if body is not None else None
  req = urllib.request.Request(  # noqa: S310
    f"{WEBUI}{path}",
    data=data,
    method="POST",
    headers={**AUTH, "Content-Type": "application/json"},
  )
  try:
    with urllib.request.urlopen(req, timeout=15) as r:  # noqa: S310
      return r.status, json.loads(r.read() or b"{}")
  except urllib.error.HTTPError as e:
    body = e.read()
    try:
      return e.code, json.loads(body)
    except ValueError:
      return e.code, {"detail": body.decode(errors="replace")}


def webui_get_bytes(
  path: str, extra_headers: dict | None = None
) -> tuple[int, bytes, dict]:
  """GET a binary resource through the backend (playlist/segment). Returns
  (status, body bytes, response headers). Follows the 302 the MinIO backend
  would emit (urlopen does) — for the disk backend it's a direct 200/206.

  Headers are the http.client.HTTPMessage (NOT dict(...)): header names may be
  emitted lowercase (e.g. `content-range`), and HTTPMessage.get() is
  case-insensitive, whereas dict(headers).get("Content-Range") would miss it."""
  headers = {**AUTH, **(extra_headers or {})}
  req = urllib.request.Request(f"{WEBUI}{path}", headers=headers, method="GET")  # noqa: S310
  try:
    with urllib.request.urlopen(req, timeout=15) as r:  # noqa: S310
      return r.status, r.read(), r.headers
  except urllib.error.HTTPError as e:
    return e.code, e.read(), e.headers


def http_put(url: str, body: bytes, headers: dict | None = None) -> tuple[int, bytes]:
  """Raw PUT (used by the ingest bypass/positive tests). No auth header by
  default — the ingest endpoint is network-authed, and the bypass test wants to
  see what the public listener does with an unauthenticated PUT."""
  req = urllib.request.Request(url, data=body, method="PUT", headers=headers or {})  # noqa: S310
  try:
    with urllib.request.urlopen(req, timeout=15) as r:  # noqa: S310
      return r.status, r.read()
  except urllib.error.HTTPError as e:
    return e.code, e.read()


def wait_upload_state(rid: str, targets: tuple[str, ...], timeout: float = 90.0) -> None:
  """Poll the backend's per-recording upload progress until `state` is one of
  `targets` (T5.5.2 step 3)."""
  deadline = time.time() + timeout
  last = None
  while time.time() < deadline:
    try:
      last = http_get_json(f"{WEBUI}/api/recordings/{rid}/upload", headers=AUTH)
      if last.get("state") in targets:
        return
    except (urllib.error.HTTPError, KeyError, TypeError):
      pass
    time.sleep(1.0)
  raise AssertionError(f"upload state {targets} not reached for {rid}; last={last}")


def _first_segment_name(playlist: bytes) -> str | None:
  """Return the first .ts segment file name referenced by an HLS playlist."""
  for line in playlist.decode("utf-8", "replace").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and line.endswith(".ts"):
      # The playlist may carry a relative path; we only need the file name.
      return line.rsplit("/", 1)[-1]
  return None


def list_recordings() -> list[dict]:
  return http_get_json(f"{WEBUI}/api/recordings", headers=AUTH)


def wait_recordings(predicate, what: str, timeout: float = 60.0) -> list[dict]:
  deadline = time.time() + timeout
  last: list[dict] = []
  while time.time() < deadline:
    last = list_recordings()
    try:
      if predicate(last):
        return last
    except (KeyError, TypeError):
      pass
    time.sleep(1.0)
  raise AssertionError(f"timed out waiting for {what}; last /api/recordings={last}")


# ---------- the WHEP client (aiortc) ----------

# aiortc is asyncio; the suite is synchronous. Run one event loop on a
# background thread and submit coroutines to it, so a PeerConnection can stay
# alive across test cases (happy_path connects, teardown disconnects).

_LOOP = asyncio.new_event_loop()
threading.Thread(target=_LOOP.run_forever, daemon=True).start()


def run_async(coro, timeout: float = 60.0):
  return asyncio.run_coroutine_threadsafe(coro, _LOOP).result(timeout)


def whep_post(offer_sdp: str, headers: dict | None = None) -> tuple[int, str, str | None]:
  """POST the SDP offer to WEBUI/whep. Returns (status, body, location).
  The first viewer blocks server-side until the WHIP ingest is live or
  FESSEL_LIVE_ACTIVATION_TIMEOUT_S (default 15s) — hence the long timeout."""
  req = urllib.request.Request(  # noqa: S310
    f"{WEBUI}/whep",
    data=offer_sdp.encode(),
    method="POST",
    headers={"Content-Type": "application/sdp", **(headers or {})},
  )
  try:
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
      return r.status, r.read().decode(), r.headers.get("Location")
  except urllib.error.HTTPError as e:
    return e.code, e.read().decode(errors="replace"), None


def whep_delete(location: str) -> None:
  req = urllib.request.Request(f"{WEBUI}{location}", method="DELETE", headers=AUTH)  # noqa: S310
  with urllib.request.urlopen(req, timeout=10):  # noqa: S310
    pass


def _log_candidates(label: str, sdp: str) -> None:
  cands = [ln for ln in sdp.splitlines() if "candidate:" in ln]
  print(f"[ice] {label} candidates ({len(cands)}):", flush=True)
  for c in cands:
    print(f"[ice]   {c.strip()}", flush=True)


class WhepViewer:
  """One WHEP viewer session backed by a real aiortc peer (ICE/DTLS/SRTP)."""

  def __init__(self) -> None:
    self.pc: RTCPeerConnection | None = None
    self.location: str | None = None
    self._track_ready: asyncio.Event | None = None
    self._track = None

  async def _make_offer(self) -> str:
    # No STUN: in podip mode both sides are on routable pod IPs, and the
    # cluster may not have UDP egress to public STUN anyway.
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    self.pc = pc
    self._track_ready = asyncio.Event()

    @pc.on("track")
    def _on_track(track):  # noqa: ANN001
      if track.kind == "video":
        self._track = track
        self._track_ready.set()

    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("audio", direction="recvonly")
    offer = await pc.createOffer()
    # aiortc gathers ICE during setLocalDescription (non-trickle): the
    # resulting SDP carries the pod-IP host candidates.
    await pc.setLocalDescription(offer)
    return pc.localDescription.sdp

  async def _apply_answer(self, answer_sdp: str) -> None:
    await self.pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))

  def connect(self, headers: dict | None = None, debug: bool = False) -> dict:
    """Full WHEP handshake. Returns {"ok", "status", ...}; on ok the
    PeerConnection is live and self.location holds the session resource."""
    offer_sdp = run_async(self._make_offer(), timeout=30)
    if debug:
      _log_candidates("offer(aiortc)", offer_sdp)
    status, body, location = whep_post(offer_sdp, headers=AUTH if headers is None else headers)
    if status != 201:
      run_async(self.pc.close(), timeout=10)
      return {"ok": False, "status": status, "body": body[:300]}
    if debug:
      _log_candidates("answer(relay)", body)
    assert location, "WHEP 201 without a Location header"
    assert body.startswith("v="), f"WHEP answer is not SDP: {body[:80]!r}"
    self.location = location
    run_async(self._apply_answer(body), timeout=30)
    return {"ok": True, "status": status, "location": location}

  def wait_media(self, timeout: float = 25.0) -> None:
    """The data-plane assertion: ICE/DTLS connects and a video frame decodes."""

    async def _wait() -> None:
      deadline = _LOOP.time() + timeout
      while self.pc.connectionState not in ("connected",):
        if self.pc.connectionState in ("failed", "closed"):
          raise AssertionError(f"peer connection {self.pc.connectionState}")
        if _LOOP.time() > deadline:
          raise AssertionError(
            f"ICE/DTLS never connected (state={self.pc.connectionState})"
          )
        await asyncio.sleep(0.2)
      await asyncio.wait_for(self._track_ready.wait(), timeout=10)
      frame = await asyncio.wait_for(self._track.recv(), timeout=15)
      print(f"[media] decoded first video frame: {frame.width}x{frame.height}", flush=True)

    run_async(_wait(), timeout=timeout + 30)

  def close(self, delete: bool = True) -> None:
    """Well-behaved client teardown: WHEP DELETE (immediate viewer removal on
    the relay), then close the local peer."""
    if delete and self.location:
      whep_delete(self.location)
      self.location = None
    if self.pc:
      run_async(self.pc.close(), timeout=10)
      self.pc = None


def main() -> int:
  os.makedirs(os.path.dirname(JUNIT_OUT), exist_ok=True)
  suite = Suite()

  # The viewer opened by happy_path, torn down by teardown_returns_off.
  session = WhepViewer()

  # --- happy path: WHEP -> activation -> WHIP ingest -> data plane ---
  def t_happy_path():
    assert current_state() in (None, "off"), f"expected off at start, got {current_state()}"
    res = session.connect(debug=True)
    assert res.get("ok"), f"WHEP failed: {res}"
    # Control plane: supervisor's retained live state reaches running.
    hist = wait_state("running", timeout=40)
    assert hist[:3] == ["off", "starting", "running"] or "running" in hist, f"bad history {hist}"
    # Ingest leg (Pi -> relay): the WHIP session is live and RTP is flowing —
    # packets_total must INCREASE across two samples.
    wait_metric("fessel_relay_ingest_live", lambda v: v == 1, "ingest_live=1", timeout=15)
    p1 = metric_value("fessel_relay_ingest_packets_total")
    time.sleep(2)
    p2 = metric_value("fessel_relay_ingest_packets_total")
    assert p2 > p1, f"ingest RTP not flowing: packets_total {p1} -> {p2}"
    # Viewer leg (relay -> this pod): ICE/DTLS connects and a frame decodes.
    # podip mode makes in-cluster ICE work; this is the data-plane assertion
    # the old Chrome-based suite could never make.
    session.wait_media(timeout=25)
    assert metric_value("fessel_relay_viewers") >= 1, "viewer gauge not incremented"

  # --- teardown: DELETE -> idle timeout -> deactivate -> off ---
  def t_teardown():
    session.close(delete=True)
    # Relay idle timeout (10s default) then supervisor deactivate then the
    # Pi detaches its sender; allow margin on top of the idle window.
    wait_off(IDLE_TIMEOUT_S + 30, "teardown")
    wait_metric("fessel_relay_ingest_live", lambda v: v == 0, "ingest_live=0", timeout=30)
    assert metric_value("fessel_relay_viewers") == 0, "viewer gauge leaked"

  # --- auth gate: /whep without the oauth2-proxy identity header ---
  def t_whep_requires_auth():
    assert current_state() in (None, "off")
    activations_before = metric_value("fessel_relay_activations_total")
    sessions_before = metric_value("fessel_relay_viewer_sessions_total")
    status, body, _ = whep_post("v=0\r\n", headers={})
    assert status == 401, f"expected 401 without identity, got {status}: {body[:200]}"
    assert json.loads(body) == {"detail": "authentication required"}, f"bad 401 body: {body[:200]}"
    # And no side effects: no activation attempted, no viewer answered.
    time.sleep(3)
    assert current_state() in (None, "off"), f"unauthenticated WHEP caused activation: {current_state()}"
    assert metric_value("fessel_relay_activations_total") == activations_before, "activation attempted"
    assert metric_value("fessel_relay_viewer_sessions_total") == sessions_before, "viewer answered"

  # --- rapid reconnect no-leak (highest value) ---
  def t_rapid_reconnect():
    for i in range(RECONNECT_CYCLES):
      v = WhepViewer()
      res = v.connect()
      if not res.get("ok"):
        raise AssertionError(f"cycle {i}: WHEP failed {res}")
      v.close(delete=True)
    # After churn: viewer gauge back to 0, then idle timeout -> off with no
    # leaked encoder/ingest session (the regression the live state machine +
    # permanent-shared-track design exist to prevent).
    wait_metric("fessel_relay_viewers", lambda v: v == 0, "viewers=0 after churn", timeout=20)
    wait_off(IDLE_TIMEOUT_S + 50, "post-churn settle")
    wait_metric("fessel_relay_ingest_live", lambda v: v == 0, "ingest_live=0 after churn", timeout=30)

  # --- Slice 3 control plane (mocked Jetson + WiZ) ---
  # Pure HTTP (no WebRTC): exercises the full chain backend -> supervisor ->
  # mock actuators. The Jetson is a mock HTTP server; the WiZ is the in-
  # process fake bulb. Everything between webui and the leaf actuators is
  # real. See integration/README.md "Slice 3 control plane".

  def t_control_pause_resume():
    # cluster auth gate + forwarder, Pi relay, REAL JetsonClient -> mock, /state cache.
    st, _ = control_post("pause")
    assert st == 200, f"pause -> {st}"
    wait_api_state(lambda s: s["jetson"]["state"] == "paused", "jetson paused")
    st, _ = control_post("resume")
    assert st == 200, f"resume -> {st}"
    wait_api_state(
      lambda s: s["jetson"]["state"] in ("active", "running"), "jetson resumed"
    )

  def t_power_cycle_verified():
    # send-and-verify -> cached verified state -> /api/state. No real power.
    st, _ = control_post("shutdown/jetson")
    assert st == 200, f"shutdown/jetson -> {st}"
    wait_api_state(
      lambda s: s["plugs"]["jetson"]["on"] is False and s["plugs"]["jetson"]["verified"],
      "jetson plug verified off",
    )
    st, _ = control_post("poweron/jetson")
    assert st == 200, f"poweron/jetson -> {st}"
    wait_api_state(
      lambda s: s["plugs"]["jetson"]["on"] is True and s["plugs"]["jetson"]["verified"],
      "jetson plug verified on",
    )

  def t_verify_failure_surfaces():
    # The `arm` plug is wired (test config) to always DROP commands, so
    # send-and-verify exhausts retries -> 503 with the structured body, and
    # /state shows verified=False — NOT a false success. (arm is a real
    # /api/control name; the backend only forwards the seven known actions.)
    st, body = control_post("shutdown/arm")
    assert st == 503, f"expected 503 from a failing plug, got {st}: {body}"
    detail = body.get("detail", body)
    assert detail.get("error") == "plug_verify_failed", f"bad 503 body: {body}"
    st2 = api_state()
    assert st2["plugs"]["arm"]["verified"] is False, (
      f"failed verify not visible in /state: {st2['plugs'].get('arm')}"
    )

  def t_control_requires_auth():
    # No identity header -> backend 401 -> the Pi is never reached.
    st, _ = control_post("pause", headers={})
    assert st == 401, f"expected 401 without identity, got {st}"

  # --- T5.5.2 recording round-trip through the new ingest path (disk backend) ---
  # Drives the whole Slice-5.5 path against the deployed system: the test-Pi's
  # uploader (real `http` driver, FESSEL_INGEST_URL_BASE=http://webui:8001)
  # PUTs each file to the backend's recording-ingest listener; the backend's
  # DISK backend persists them to the PVC and serves playback as byte ranges.
  # start -> ring is always on so segments exist -> stop -> flag-for-upload ->
  # uploader PUTs -> /api/recordings shows available_remote -> playback works,
  # including a Range request returning 206 (B5.5.4, the disk-backend gate).
  def t_recording_roundtrip_via_ingest():
    st, body = webui_post_json("/api/recording/start", {"mode": MODE})
    assert st == 200, f"recording/start -> {st}: {body}"
    rid = body.get("recording_id")
    assert rid, f"no recording_id from start: {body}"
    wait_api_state(
      lambda s: (s.get("recording") or {}).get("state") == "recording",
      "recording active",
      timeout=30,
    )
    time.sleep(5)  # capture a few segments
    st, _ = webui_post_json("/api/recording/stop")
    assert st == 200, f"recording/stop -> {st}"
    wait_recordings(
      lambda recs: any(r["recording_id"] == rid for r in recs),
      f"recording {rid} listed",
      timeout=30,
    )
    # Flag for upload; the uploader PUTs each file to the ingest endpoint.
    st, body = webui_post_json("/api/recording/flag-upload", {"recording_id": rid})
    assert st == 200, f"flag-upload -> {st}: {body}"
    # Poll the per-recording upload progress until it reaches `uploaded`
    # (T5.5.2 step 3), and the merged listing shows available_remote.
    wait_upload_state(rid, ("uploaded",), timeout=90)
    recs = wait_recordings(
      lambda rs: any(
        r["recording_id"] == rid and r.get("available_remote") for r in rs
      ),
      f"recording {rid} available_remote (uploaded via ingest)",
      timeout=90,
    )
    row = next(r for r in recs if r["recording_id"] == rid)
    assert row.get("available_remote") is True, row
    # Playback: the disk backend serves a 200 with a valid HLS playlist body.
    st, pl_body, _ = webui_get_bytes(f"/api/recordings/{rid}/playlist")
    assert st == 200, f"playlist status {st}"
    assert pl_body.startswith(b"#EXTM3U"), f"not an HLS playlist: {pl_body[:40]!r}"
    # Pick a segment name from the playlist and fetch it with a Range header;
    # the disk backend must return 206 Partial Content with the right count.
    seg = _first_segment_name(pl_body)
    assert seg, f"no .ts segment in playlist: {pl_body[:200]!r}"
    st, seg_body, hdrs = webui_get_bytes(
      f"/api/recordings/{rid}/segment/{seg}", extra_headers={"Range": "bytes=0-1023"}
    )
    assert st == 206, f"expected 206 for Range, got {st}"
    assert len(seg_body) == 1024, f"expected 1024 bytes, got {len(seg_body)}"
    assert hdrs.get("Content-Range", "").startswith("bytes 0-1023/"), hdrs.get("Content-Range")

  # --- T5.5.3 ingest auth + bypass ---
  def t_ingest_not_reachable_via_public_listener():
    # A PUT to /recording-ingest/... on the PUBLIC listener (:8000) must not
    # succeed — the route exists only on the ingest listener (:8001). The Go
    # mux answers 404 (route absent) or 405 (method not allowed on the static
    # fallback); 401/403 would also count as "not reachable".
    st, _ = http_put(f"{WEBUI}/recording-ingest/bypass-test/index.m3u8", b"#EXTM3U")
    assert st in (401, 403, 404, 405), f"ingest reachable via public listener: {st}"

  def t_ingest_listener_accepts_put():
    # Sanity: the ingest LISTENER (:8001) does accept the PUT (201). This is
    # the positive control for the bypass test above — proves the negative is
    # about the listener, not a broken endpoint. Uses a throwaway id.
    st, _ = http_put(
      f"{WEBUI_INGEST}/recording-ingest/ingest-probe/index.m3u8", b"#EXTM3U\n"
    )
    assert st == 201, f"ingest PUT -> {st}"

  def t_recordings_require_auth():
    # /api/recordings + playback all 401 without the oauth2-proxy identity.
    for path in (
      "/api/recordings",
      "/api/recordings/x/playlist",
      "/api/recordings/x/segment/seg-00000.ts",
    ):
      try:
        http_get_json(f"{WEBUI}{path}")
      except urllib.error.HTTPError as e:
        assert e.code == 401, f"{path} -> {e.code}, expected 401"
        assert json.loads(e.read()) == {"detail": "authentication required"}, f"bad 401 body for {path}"
      else:
        raise AssertionError(f"{path} did not require auth")

  # Live chain (WHEP -> activation -> WHIP -> metrics -> decoded frames).
  suite.run("happy_path_activation", t_happy_path)
  suite.run("teardown_returns_off", t_teardown)
  suite.run("whep_requires_auth", t_whep_requires_auth)
  suite.run("rapid_reconnect_no_leak", t_rapid_reconnect)
  # Slice 3 control plane (mocked actuators).
  suite.run("control_pause_resume", t_control_pause_resume)
  suite.run("control_power_cycle_verified", t_power_cycle_verified)
  suite.run("control_verify_failure_surfaces", t_verify_failure_surfaces)
  suite.run("control_requires_auth", t_control_requires_auth)
  # Slice 5.5 recording round-trip through the new ingest path (disk backend,
  # T5.5.2) + the ingest auth/bypass assertions (T5.5.3).
  suite.run("recording_roundtrip_via_ingest", t_recording_roundtrip_via_ingest)
  suite.run("ingest_listener_accepts_put", t_ingest_listener_accepts_put)
  suite.run("ingest_not_reachable_via_public_listener", t_ingest_not_reachable_via_public_listener)
  suite.run("recordings_require_auth", t_recordings_require_auth)

  with open(JUNIT_OUT, "w") as f:
    f.write(suite.to_junit())
  print(suite.to_junit(), flush=True)

  # Known-gap cases never block the suite.
  return 0 if suite.blocking_failures() == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
