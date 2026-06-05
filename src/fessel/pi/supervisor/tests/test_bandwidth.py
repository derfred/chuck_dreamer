"""BandwidthCoordinator tests (§2.12).

The decision logic is exercised directly (no thread, no MQTT) via an injected
monotonic clock and a fake publish sink, so ring-window expiry is deterministic.
`set_live_running` triggers an immediate re-evaluate; ring reads are picked up on
the next evaluate (the background loop's tick) — both modelled here by calling
the coordinator's public API and a manual `_evaluate()` for the tick.
"""

from fessel_schemas import UploadGate
from fessel_shared import topics

from supervisor.bandwidth import BandwidthCoordinator


class Clock:
  def __init__(self):
    self.t = 1000.0

  def __call__(self):
    return self.t

  def advance(self, dt):
    self.t += dt


def make_coordinator(ring_window_s=10.0):
  pubs = []
  clock = Clock()

  def publish(topic, payload, qos, retain):
    pubs.append((topic, payload, qos, retain))

  bw = BandwidthCoordinator(
    publish=publish, ring_active_window_s=ring_window_s, monotonic=clock
  )
  return bw, pubs, clock


def _gates(pubs):
  return [UploadGate.model_validate(p) for (t, p, _q, _r) in pubs if t == topics.CMD_UPLOAD_GATE]


def test_initial_publish_allows_uploads_when_no_viewer():
  bw, pubs, _ = make_coordinator()
  # The background loop publishes the initial gate; emulate its first evaluate.
  bw._evaluate()
  gates = _gates(pubs)
  assert len(gates) == 1
  assert gates[0].uploads_allowed is True
  assert gates[0].reason == "no viewer"
  # Retained + QoS 1 so a (re)connecting uploader sees it.
  assert pubs[-1][2] == topics.QOS_CMD and pubs[-1][3] is topics.RETAIN_UPLOAD_GATE


def test_live_viewer_suspends_uploads():
  bw, pubs, _ = make_coordinator()
  bw._evaluate()  # initial: allowed
  bw.set_live_running(True)  # fires an immediate re-evaluate
  gates = _gates(pubs)
  assert gates[-1].uploads_allowed is False
  assert gates[-1].reason == "live viewer"


def test_live_viewer_leaving_resumes_uploads():
  bw, pubs, _ = make_coordinator()
  bw.set_live_running(True)
  bw.set_live_running(False)
  gates = _gates(pubs)
  assert gates[-1].uploads_allowed is True


def test_ring_viewer_suspends_then_window_expiry_resumes():
  bw, pubs, clock = make_coordinator(ring_window_s=10.0)
  bw._evaluate()  # initial: allowed
  # A ring read makes a viewer present; the loop picks it up on the next tick.
  bw.note_ring_read()
  bw._evaluate()
  assert _gates(pubs)[-1].uploads_allowed is False
  assert _gates(pubs)[-1].reason == "ring viewer"
  # Within the window, still a viewer.
  clock.advance(5.0)
  bw._evaluate()
  assert _gates(pubs)[-1].uploads_allowed is False
  # Past the window with no new read -> viewer gone -> uploads resume.
  clock.advance(6.0)  # now 11s since the read, > 10s window
  bw._evaluate()
  assert _gates(pubs)[-1].uploads_allowed is True


def test_live_overrides_ring_window_expiry():
  bw, pubs, clock = make_coordinator(ring_window_s=10.0)
  bw.note_ring_read()
  bw.set_live_running(True)
  # Even after the ring window expires, the live viewer keeps uploads suspended.
  clock.advance(100.0)
  bw._evaluate()
  assert _gates(pubs)[-1].uploads_allowed is False
  assert _gates(pubs)[-1].reason == "live viewer"


def test_no_churn_on_unchanged_decision():
  bw, pubs, clock = make_coordinator()
  bw._evaluate()  # 1 publish (initial allowed)
  bw._evaluate()  # no change -> no publish
  bw._evaluate()
  assert len(_gates(pubs)) == 1
  # A change publishes once; repeats of the same state do not.
  bw.set_live_running(True)  # change -> publish
  bw._evaluate()  # same -> no publish
  assert len(_gates(pubs)) == 2


def test_ring_read_refreshes_window():
  bw, pubs, clock = make_coordinator(ring_window_s=10.0)
  bw.note_ring_read()
  bw._evaluate()
  assert _gates(pubs)[-1].uploads_allowed is False
  # A fresh read 8s later (within window) keeps the viewer present past the
  # original deadline.
  clock.advance(8.0)
  bw.note_ring_read()
  clock.advance(8.0)  # 8s since the 2nd read, < 10s window
  bw._evaluate()
  assert _gates(pubs)[-1].uploads_allowed is False


# --- wiring seams ------------------------------------------------------------


def test_ring_proxy_fires_read_hook_only_for_ring(tmp_path):
  # RingProxy calls on_ring_read for ring playlist/segment, NOT for explicit
  # recording files (those are out of the §2.12 start-simple scope).
  from supervisor.recordings import RingProxy

  ring = tmp_path / "ring"
  ring.mkdir()
  (ring / "index.m3u8").write_text("#EXTM3U")
  (ring / "seg-00000.ts").write_bytes(b"X")
  explicit = tmp_path / "explicit"
  rec = explicit / "r1"
  rec.mkdir(parents=True)
  (rec / "index.m3u8").write_text("#EXTM3U")

  calls = []
  proxy = RingProxy(ring, explicit, on_ring_read=lambda: calls.append(1))

  proxy.ring_playlist()
  proxy.ring_segment("seg-00000.ts")
  assert len(calls) == 2
  # An explicit-recording read does not count as a ring viewer.
  proxy.recording_playlist("r1")
  assert len(calls) == 2


def test_relay_live_state_drives_live_running_hook(monkeypatch):
  # Relay._on_live_state fires on_live_running(True) only in `running`.
  import supervisor.app as appmod
  from fessel_schemas import LiveState

  relay = appmod.Relay({"mqtt": {"host": "x"}})
  seen = []
  relay.on_live_running = seen.append
  relay._on_live_state("t", LiveState(state="starting", path="pi"))
  relay._on_live_state("t", LiveState(state="running", path="pi"))
  relay._on_live_state("t", LiveState(state="off"))
  assert seen == [False, True, False]
