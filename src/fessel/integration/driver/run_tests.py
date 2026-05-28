"""Integration test driver (T1.5-T1.8).

Runs in-cluster as a Job. Acts as the WHEP client (the browser) and drives
the full control-plane + data-plane scenario against the deployed Fessel
system, emitting JUnit XML.

Hard requirements baked in (learned from the streaming work):
- H.264-capable Chromium: Playwright Chromium run HEADED under Xvfb (the
  bundled headless build lacks an H.264 decoder; mediamtx then rejects the
  client with "codecs not supported"). The Job image starts Xvfb and runs
  headless=False.
- Media-liveness, not ICE state: the data-plane assertion polls
  getStats().inboundRtp.bytesReceived AND <video>.currentTime over a
  sustained window.

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

DATA_PLANE_WINDOW_S = float(os.environ.get("DATA_PLANE_WINDOW_S", "30"))
RECONNECT_CYCLES = int(os.environ.get("RECONNECT_CYCLES", "50"))


# ---------- tiny JUnit emitter ----------


@dataclass
class Case:
  name: str
  ok: bool = False
  err: str | None = None
  duration: float = 0.0


@dataclass
class Suite:
  name: str = "fessel-integration"
  cases: list[Case] = field(default_factory=list)

  def run(self, name: str, fn) -> bool:
    c = Case(name=name)
    t0 = time.time()
    try:
      fn()
      c.ok = True
      print(f"[PASS] {name}", flush=True)
    except Exception as e:  # noqa: BLE001
      c.err = f"{type(e).__name__}: {e}"
      print(f"[FAIL] {name}: {c.err}", flush=True)
    c.duration = time.time() - t0
    self.cases.append(c)
    return c.ok

  def to_junit(self) -> str:
    failures = sum(0 if c.ok else 1 for c in self.cases)
    out = [
      '<?xml version="1.0" encoding="UTF-8"?>',
      f'<testsuite name="{self.name}" tests="{len(self.cases)}" failures="{failures}">',
    ]
    for c in self.cases:
      out.append(f'  <testcase name="{_xml(c.name)}" time="{c.duration:.2f}">')
      if not c.ok:
        out.append(f'    <failure message="{_xml(c.err or "")}"></failure>')
      out.append("  </testcase>")
    out.append("</testsuite>")
    return "\n".join(out)


def _xml(s: str) -> str:
  return (
    s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
  )


# ---------- helpers ----------


def http_get_json(url: str, timeout: float = 5.0) -> dict:
  with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
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
  body = http_get_json(f"{WEBUI}/api/auth/whep-url?path={PATH}&mode={MODE}")
  return body["url"]


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

STATS_JS = r"""
async () => {
  const pc = window.__pc, v = window.__video;
  if (!pc) return {bytes: 0, currentTime: 0, ice: 'none'};
  let bytes = 0;
  const stats = await pc.getStats();
  stats.forEach(r => { if (r.type === 'inbound-rtp' && r.kind === 'video') bytes = r.bytesReceived || 0; });
  return {bytes, currentTime: v ? v.currentTime : 0, ice: pc.iceConnectionState};
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

    # --- T1.7.1 data-plane: media actually flows ---
    def t_data_plane():
      # Poll bytesReceived + currentTime over the window; both must advance.
      samples = []
      t_end = time.time() + DATA_PLANE_WINDOW_S
      last = {"bytes": 0, "currentTime": 0.0}
      advanced_bytes = 0
      advanced_time = 0
      while time.time() < t_end:
        s = page.evaluate(STATS_JS)
        if s["bytes"] > last["bytes"]:
          advanced_bytes += 1
        if s["currentTime"] > last["currentTime"]:
          advanced_time += 1
        last = s
        samples.append(s)
        time.sleep(2)
      assert advanced_bytes >= 5, f"bytesReceived did not climb enough: {samples}"
      assert advanced_time >= 3, f"video.currentTime did not advance enough: {samples}"

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

    suite.run("happy_path_activation", t_happy_path)
    suite.run("data_plane_media_flows", t_data_plane)
    suite.run("teardown_returns_off", t_teardown)
    suite.run("token_rejection_no_activation", t_token_rejection)
    suite.run("rapid_reconnect_no_leak", t_rapid_reconnect)

    browser.close()

  with open(JUNIT_OUT, "w") as f:
    f.write(suite.to_junit())
  print(suite.to_junit(), flush=True)

  return 0 if all(c.ok for c in suite.cases) else 1


if __name__ == "__main__":
  sys.exit(main())
