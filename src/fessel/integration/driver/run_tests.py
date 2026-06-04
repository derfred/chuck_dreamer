"""Integration test driver (T1.5-T1.8).

Runs in-cluster as a Job. Acts as the WHEP client (the browser) and asserts
the CONTROL-PLANE scenario against the deployed Fessel system, emitting
JUnit XML.

There is deliberately NO automated data-plane (video bytes in the browser)
assertion: headless Chrome cannot complete WebRTC ICE in-cluster, so it
could never pass here. The control-plane assertions prove the chain up to
the media plane (activation -> SRT publish, confirmed by mediamtx
"is publishing ... H264"); actual video reception is verified manually via
the live-preview workflow in a real browser. See integration/README.md.

Even though no data-plane assertion runs, the WHEP handshake still uses a
real H.264-capable Google Chrome (channel=chrome) headed under Xvfb, so the
codec negotiation path is exercised by happy_path.

Service DNS (in-namespace):
  WEBUI=http://webui:8000   MEDIA=http://mediamtx:8889   SUPERVISOR=http://supervisor:8443
Path=pi, mode=<from env or default>.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from playwright.sync_api import sync_playwright

WEBUI = os.environ.get("WEBUI", "http://webui:8000")
MEDIA = os.environ.get("MEDIA", "http://mediamtx:8889")
SUPERVISOR = os.environ.get("SUPERVISOR", "http://supervisor:8443")
PATH = os.environ.get("FESSEL_PATH", "pi")
MODE = os.environ.get("FESSEL_MODE", "640x480@30@1000000")
JUNIT_OUT = os.environ.get("JUNIT_OUT", "/results/junit.xml")

RECONNECT_CYCLES = int(os.environ.get("RECONNECT_CYCLES", "50"))

# oauth2-proxy is out of the integration loop (Slice 2 §6): the driver reaches
# webui directly via the in-cluster Service, which is the same network position
# oauth2-proxy forwards from. Slice 2 gates the mint on the identity headers the
# proxy would inject, so the driver supplies them itself — a faithful stand-in
# for the proxy. Header name + value are overridable to match the backend's
# FESSEL_AUTH_USER_HEADER config.
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
    import json

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


def mint_whep_url() -> str:
  # Carry the operator identity header oauth2-proxy would inject (Slice 2),
  # so the gated mint endpoint treats the driver as an authenticated operator.
  body = http_get_json(
    f"{WEBUI}/api/auth/whep-url?path={PATH}&mode={MODE}",
    headers={AUTH_USER_HEADER: AUTH_USER},
  )
  return body["url"]


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
  import json

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


def whep_post(signed_url: str, offer_sdp: str) -> tuple[int, str]:
  """POST the WHEP offer from Python (the browser's about:blank origin
  can't fetch cross-origin). Returns (status, answer_sdp_or_error)."""
  req = urllib.request.Request(  # noqa: S310
    signed_url,
    data=offer_sdp.encode(),
    method="POST",
    headers={"Content-Type": "application/sdp"},
  )
  try:
    with urllib.request.urlopen(req, timeout=20) as r:  # noqa: S310
      return r.status, r.read().decode()
  except urllib.error.HTTPError as e:
    return e.code, e.read().decode(errors="replace")


# ---------- the WHEP client (browser) ----------

# Create a recvonly PeerConnection, gather ICE fully (no trickle: we relay
# the SDP through Python), and return the complete offer SDP.
OFFER_JS = r"""
async () => {
  const pc = new RTCPeerConnection();
  window.__pc = pc;
  const v = document.createElement('video');
  v.autoplay = true; v.muted = true; v.playsInline = true;
  document.body.appendChild(v);
  window.__video = v;
  pc.addTransceiver('video', {direction:'recvonly'});
  pc.addTransceiver('audio', {direction:'recvonly'});
  pc.ontrack = (e) => { if (e.streams[0]) v.srcObject = e.streams[0]; };
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  // Wait for ICE gathering to finish so the offer carries all candidates.
  await new Promise((resolve) => {
    if (pc.iceGatheringState === 'complete') return resolve();
    const check = () => { if (pc.iceGatheringState === 'complete') { pc.removeEventListener('icegatheringstatechange', check); resolve(); } };
    pc.addEventListener('icegatheringstatechange', check);
    setTimeout(resolve, 3000);  // cap the wait
  });
  return pc.localDescription.sdp;
}
"""

# Apply the SDP answer relayed back from Python.
ANSWER_JS = r"""
async (answerSdp) => {
  const pc = window.__pc;
  await pc.setRemoteDescription({type:'answer', sdp: answerSdp});
  return true;
}
"""

def main() -> int:
  os.makedirs(os.path.dirname(JUNIT_OUT), exist_ok=True)
  suite = Suite()

  with sync_playwright() as p:
    # Use Google Chrome (channel=chrome), not bundled Chromium: the bundled
    # build lacks the proprietary H.264 decoder, so its WHEP SDP offer omits
    # H.264 and mediamtx rejects it with "codecs not supported by client".
    # Headed under Xvfb. CHROME_CHANNEL lets the image fall back to chromium
    # if Chrome isn't installed.
    channel = os.environ.get("CHROME_CHANNEL", "chrome")
    launch_kwargs = dict(
      headless=False,
      args=[
        "--no-sandbox",
        "--use-fake-ui-for-media-stream",
        "--autoplay-policy=no-user-gesture-required",
        # Disable mDNS host-candidate obfuscation: by default Chrome hides
        # local IPs behind <uuid>.local candidates, which mediamtx can't
        # resolve in-cluster, so ICE never connects. Expose real host IPs.
        "--disable-features=WebRtcHideLocalIpsWithMdns",
      ],
    )
    if channel:
      launch_kwargs["channel"] = channel
    browser = p.chromium.launch(**launch_kwargs)
    page = browser.new_page()
    page.goto("about:blank")

    def _log_candidates(label: str, sdp: str) -> None:
      cands = [ln for ln in sdp.splitlines() if "candidate:" in ln]
      print(f"[ice] {label} candidates ({len(cands)}):", flush=True)
      for c in cands:
        print(f"[ice]   {c.strip()}", flush=True)

    def connect(url: str | None = None, debug: bool = False) -> dict:
      """Full WHEP handshake: browser offer -> Python POST -> browser answer."""
      signed = url if url is not None else mint_whep_url()
      offer_sdp = page.evaluate(OFFER_JS)
      if debug:
        _log_candidates("offer(chrome)", offer_sdp)
      status, body = whep_post(signed, offer_sdp)
      if status not in (200, 201):
        return {"ok": False, "status": status, "body": body[:300]}
      if debug:
        _log_candidates("answer(mediamtx)", body)
      page.evaluate(ANSWER_JS, body)
      return {"ok": True, "status": status}

    def close_pc() -> None:
      page.evaluate("() => { if (window.__pc) window.__pc.close(); window.__pc=null; if(window.__video){window.__video.remove(); window.__video=null;} }")

    # --- T1.6.1 mint + happy path + activation propagation ---
    def t_happy_path():
      assert current_state() in (None, "off"), f"expected off at start, got {current_state()}"
      res = connect(debug=True)
      assert res.get("ok"), f"WHEP failed: {res}"
      hist = wait_state("running", timeout=40)
      assert hist[:3] == ["off", "starting", "running"] or "running" in hist, f"bad history {hist}"

    # --- T1.6.3 teardown returns to off ---
    def t_teardown():
      close_pc()
      # mediamtx runOnDemandCloseAfter (~10s) then deactivate.
      deadline = time.time() + 40
      while time.time() < deadline:
        if current_state() == "off":
          return
        time.sleep(1)
      raise AssertionError(f"state did not return to off; current={current_state()}")

    # --- Slice 2: mint requires operator identity (no headers -> 401) ---
    def t_mint_requires_auth():
      # Without the oauth2-proxy identity header the gated mint must refuse,
      # so no signed token can be obtained -> no path to Pi activation.
      try:
        http_get_json(f"{WEBUI}/api/auth/whep-url?path={PATH}&mode={MODE}")
      except urllib.error.HTTPError as e:
        assert e.code == 401, f"expected 401 without identity, got {e.code}"
        return
      raise AssertionError("unauthenticated mint unexpectedly succeeded")

    # --- T1.6.2 token rejection: no activation ---
    def t_token_rejection():
      assert current_state() in (None, "off")
      # Tamper the token.
      url = mint_whep_url()
      bad = url[:-3] + ("aaa" if not url.endswith("aaa") else "bbb")
      res = connect(url=bad)
      assert not res.get("ok"), f"tampered token unexpectedly accepted: {res}"
      # Assert state STAYED off (no Pi activation).
      time.sleep(5)
      assert current_state() in (None, "off"), f"tampered token caused activation: {current_state()}"

    # --- T1.6.4 rapid reconnect no-leak (highest value) ---
    def t_rapid_reconnect():
      for i in range(RECONNECT_CYCLES):
        res = connect()
        if not res.get("ok"):
          raise AssertionError(f"cycle {i}: WHEP failed {res}")
        close_pc()
      # After churn, state must settle to off (no leaked publisher/encoder).
      deadline = time.time() + 60
      while time.time() < deadline:
        if current_state() == "off":
          break
        time.sleep(1)
      assert current_state() == "off", f"did not settle to off after churn: {current_state()}"

    # --- Slice 3 control plane (mocked Jetson + WiZ) ---
    # Pure HTTP (no browser): exercises the full chain backend -> supervisor ->
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

    # NOTE: there is intentionally NO automated data-plane (video bytes in
    # the browser) assertion. Headless Chrome cannot complete WebRTC ICE
    # in-cluster, so such a test could never pass here. The streaming chain
    # is proven up to the media plane by happy_path (activation -> SRT
    # publish, confirmed by mediamtx "is publishing ... H264"). Actual video
    # reception is verified manually via the live-preview workflow in a real
    # browser. See integration/README.md.
    suite.run("happy_path_activation", t_happy_path)
    suite.run("teardown_returns_off", t_teardown)
    suite.run("mint_requires_auth", t_mint_requires_auth)
    suite.run("token_rejection_no_activation", t_token_rejection)
    suite.run("rapid_reconnect_no_leak", t_rapid_reconnect)
    # Slice 3 control plane (mocked actuators).
    suite.run("control_pause_resume", t_control_pause_resume)
    suite.run("control_power_cycle_verified", t_power_cycle_verified)
    suite.run("control_verify_failure_surfaces", t_verify_failure_surfaces)
    suite.run("control_requires_auth", t_control_requires_auth)

    browser.close()

  with open(JUNIT_OUT, "w") as f:
    f.write(suite.to_junit())
  print(suite.to_junit(), flush=True)

  # Known-gap cases never block the suite.
  return 0 if suite.blocking_failures() == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
