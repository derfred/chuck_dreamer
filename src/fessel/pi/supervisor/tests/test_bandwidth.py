"""BandwidthCoordinator tests (§2.12).

The decision logic is exercised directly (no thread, no MQTT) via a fake publish
sink. Uploads suspend iff a live viewer is present; `set_live_running` triggers
an immediate re-evaluate (the ring is never viewed, so it is not a signal here).
"""

from fessel_schemas import UploadGate
from fessel_shared import topics

from supervisor.bandwidth import BandwidthCoordinator


def make_coordinator():
  pubs = []

  def publish(topic, payload, qos, retain):
    pubs.append((topic, payload, qos, retain))

  bw = BandwidthCoordinator(publish=publish)
  return bw, pubs


def _gates(pubs):
  return [UploadGate.model_validate(p) for (t, p, _q, _r) in pubs if t == topics.CMD_UPLOAD_GATE]


def test_initial_publish_allows_uploads_when_no_viewer():
  bw, pubs = make_coordinator()
  # The background loop publishes the initial gate; emulate its first evaluate.
  bw._evaluate()
  gates = _gates(pubs)
  assert len(gates) == 1
  assert gates[0].uploads_allowed is True
  assert gates[0].reason == "no viewer"
  # Retained + QoS 1 so a (re)connecting uploader sees it.
  assert pubs[-1][2] == topics.QOS_CMD and pubs[-1][3] is topics.RETAIN_UPLOAD_GATE


def test_live_viewer_suspends_uploads():
  bw, pubs = make_coordinator()
  bw._evaluate()  # initial: allowed
  bw.set_live_running(True)  # fires an immediate re-evaluate
  gates = _gates(pubs)
  assert gates[-1].uploads_allowed is False
  assert gates[-1].reason == "live viewer"


def test_live_viewer_leaving_resumes_uploads():
  bw, pubs = make_coordinator()
  bw.set_live_running(True)
  bw.set_live_running(False)
  gates = _gates(pubs)
  assert gates[-1].uploads_allowed is True


def test_no_churn_on_unchanged_decision():
  bw, pubs = make_coordinator()
  bw._evaluate()  # 1 publish (initial allowed)
  bw._evaluate()  # no change -> no publish
  bw._evaluate()
  assert len(_gates(pubs)) == 1
  # A change publishes once; repeats of the same state do not.
  bw.set_live_running(True)  # change -> publish
  bw._evaluate()  # same -> no publish
  assert len(_gates(pubs)) == 2


# --- wiring seam -------------------------------------------------------------


def test_relay_live_state_drives_live_running_hook(monkeypatch):
  # Relay._on_live_state fires on_live_running(True) only in `running`.
  from fessel_schemas import LiveState

  import supervisor.app as appmod

  relay = appmod.Relay({"mqtt": {"host": "x"}})
  seen = []
  relay.on_live_running = seen.append
  relay._on_live_state("t", LiveState(state="starting", path="pi"))
  relay._on_live_state("t", LiveState(state="running", path="pi"))
  relay._on_live_state("t", LiveState(state="off"))
  assert seen == [False, True, False]
