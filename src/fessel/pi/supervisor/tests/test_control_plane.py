"""supervisor control-plane tests with a fake MQTT relay."""

from fastapi.testclient import TestClient

from supervisor.control import ControlPlane
from supervisor.jetson import JetsonError
from supervisor.wiz import PlugConfig, PlugController


def make_client(monkeypatch, control=None):
  # Avoid touching a real broker but keep the REAL subscribe/subscribe_model
  # logic (so the validating-dispatch path is exercised, not re-stubbed): a
  # FakeMqtt subclasses MqttClient, skips paho, and routes deliver() through
  # the registered handlers exactly as the real on_message would.
  import supervisor.app as appmod
  from fessel_shared import MqttClient

  published: list = []

  class FakeMqtt(MqttClient):
    def __init__(self, *a, **k):
      self._handlers = {}
      self._client = None  # no paho client; subscribe() doesn't need a live one

    def connect(self):
      pass

    def subscribe(self, topic, handler, qos=0):
      # Real subscribe touches self._client.subscribe; here we only record the
      # handler so deliver()/subscribe_model behave like the real dispatch.
      self._handlers[topic] = (handler, qos)

    def loop_start(self):
      pass

    def loop_stop(self):
      pass

    def disconnect(self):
      pass

    def publish(self, topic, payload, qos=0, retain=False):
      published.append((topic, payload, qos, retain))

    def deliver(self, topic, payload):
      """Simulate a broker message: dispatch to the registered handler (which,
      for subscribe_model topics, is the validating wrapper)."""
      handler, _ = self._handlers[topic]
      handler(topic, payload)

  monkeypatch.setattr(appmod, "MqttClient", FakeMqtt)
  app = appmod.create_app({"mqtt": {"host": "127.0.0.1"}}, control=control)
  return TestClient(app), published


# --- Slice 3 fakes -----------------------------------------------------------


class FakeBulb:
  """In-memory bulb (mirrors test_wiz). `fail` drops every command so verify
  never succeeds — the actuator-failure path."""

  def __init__(self, on=True, fail=False):
    self._on = on
    self._fail = fail
    self.commands = []

  async def turn_on(self):
    self.commands.append(True)
    if not self._fail:
      self._on = True

  async def turn_off(self):
    self.commands.append(False)
    if not self._fail:
      self._on = False

  async def is_on(self):
    return self._on


class FakeJetson:
  """Records relayed actions; `state()` returns a placeholder Jetson view."""

  def __init__(self, fail=False):
    self.fail = fail
    self.calls = []
    self._state = {"state": "active"}

  def _act(self, action, new_state):
    self.calls.append(action)
    if self.fail:
      raise JetsonError(action, "jetson unreachable")
    self._state = {"state": new_state}
    return self._state

  def pause(self):
    return self._act("pause", "paused")

  def stop(self):
    return self._act("stop", "stopped")

  def resume(self):
    return self._act("resume", "active")

  def state(self):
    if self.fail:
      raise JetsonError("state", "jetson unreachable")
    return self._state

  def close(self):
    pass


def make_control(*, arm_fail=False, jetson_fail=False, published=None):
  """Build a ControlPlane wired to fake actuators. The publish callback
  records (topic, payload, qos, retain) into `published` if provided."""
  pub_sink = published if published is not None else []

  def publish(topic, payload, qos, retain):
    pub_sink.append((topic, payload, qos, retain))

  plugs = {
    "arm": PlugController(
      PlugConfig(name="arm", address="x", retries=2, retry_delay_s=0.0),
      FakeBulb(on=True, fail=arm_fail),
    ),
    "jetson": PlugController(
      PlugConfig(name="jetson", address="y", retries=2, retry_delay_s=0.0),
      FakeBulb(on=True),
    ),
  }
  jetson = FakeJetson(fail=jetson_fail)
  # Long refresh intervals so the background thread does not race the test.
  ctl = ControlPlane(
    plugs=plugs,
    jetson=jetson,
    publish=publish,
    jetson_refresh_s=3600,
    plug_refresh_s=3600,
  )
  return ctl, jetson, pub_sink


def test_healthz(monkeypatch):
  client, _ = make_client(monkeypatch)
  with client:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    # Release-version stamp (env/file or "unknown" in a source checkout).
    assert "version" in body


def test_activate_relays_to_mqtt(monkeypatch):
  client, published = make_client(monkeypatch)
  with client:
    r = client.post("/control/live/activate", json={"path": "pi", "mode": "1280x720@30@2500000"})
    assert r.status_code == 200
  topics = [p[0] for p in published]
  assert "arm/video/cmd/live/activate" in topics
  activate = next(p for p in published if p[0] == "arm/video/cmd/live/activate")
  # mode canonical string was parsed into the triplet object on the bus.
  assert activate[1]["mode"]["resolution"] == "1280x720"
  assert activate[1]["mode"]["bitrate_bps"] == 2_500_000
  assert activate[3] is False  # commands not retained


def test_activate_rejects_bad_mode(monkeypatch):
  client, _ = make_client(monkeypatch)
  with client:
    r = client.post("/control/live/activate", json={"path": "pi", "mode": "garbage"})
    assert r.status_code == 400


def test_activate_extracts_mode_from_raw_query(monkeypatch):
  # mediamtx runOnDemand posts the raw WHEP query; supervisor extracts mode.
  client, published = make_client(monkeypatch)
  with client:
    r = client.post(
      "/control/live/activate",
      json={"path": "pi", "query": "mode=640x480@30@1000000&token=abc.def.ghi"},
    )
    assert r.status_code == 200
  activate = next(p for p in published if p[0] == "arm/video/cmd/live/activate")
  assert activate[1]["mode"]["resolution"] == "640x480"


def test_activate_extracts_mode_from_url_encoded_query(monkeypatch):
  # mediamtx 1.18.2 passes $MTX_QUERY url-encoded, and the browser's WHEP
  # query is already encoded -> '@' arrives double-encoded as %2540.
  # supervisor must recover the canonical mode regardless.
  client, published = make_client(monkeypatch)
  with client:
    for q in (
      "mode=640x480%4030%401000000&jwt=abc",  # value single-encoded
      "mode=640x480%254030%25401000000&jwt=abc",  # value double-encoded
      "mode%3D640x480%254030%25401000000%26jwt%3Dabc",  # whole query encoded (1.18.2)
    ):
      r = client.post("/control/live/activate", json={"path": "pi", "query": q})
      assert r.status_code == 200, (q, r.json())
  activate = [p for p in published if p[0] == "arm/video/cmd/live/activate"][-1]
  assert activate[1]["mode"]["resolution"] == "640x480"
  assert activate[1]["mode"]["bitrate_bps"] == 1_000_000


def test_deactivate_relays(monkeypatch):
  client, published = make_client(monkeypatch)
  with client:
    r = client.post("/control/live/deactivate", json={"path": "pi"})
    assert r.status_code == 200
  assert any(p[0] == "arm/video/cmd/live/deactivate" for p in published)


def test_state_live_tracks_history(monkeypatch):
  from fessel_schemas import LiveState

  client, _ = make_client(monkeypatch)
  with client:
    relay = client.app.state.relay
    # Simulate retained live-state messages arriving from video. The handler
    # now receives a validated LiveState (subscribe_model), so feed it one.
    for st in ["off", "starting", "running", "starting", "off"]:
      relay._on_live_state("arm/video/state/live", LiveState(state=st, path="pi"))
    r = client.get("/state/live")
    assert r.status_code == 200
    body = r.json()
    assert body["current"]["state"] == "off"
    # Consecutive duplicates collapse; transitions preserved in order.
    assert body["history"] == ["off", "starting", "running", "starting", "off"]


def test_state_live_drops_invalid_payload(monkeypatch):
  # A malformed STATE_LIVE payload is validated-and-dropped at the transport
  # (subscribe_model) before it reaches the handler — so a bad message can't
  # poison the cached history. Deliver one through the real dispatch path.
  client, _ = make_client(monkeypatch)
  with client:
    relay = client.app.state.relay
    relay._mqtt.deliver("arm/video/state/live", {"state": "bogus-state"})
    assert relay.live_state()["history"] == []
    # A valid one still lands, proving the topic is wired (not just inert).
    relay._mqtt.deliver("arm/video/state/live", {"state": "running", "path": "pi"})
    assert relay.live_state()["history"] == ["running"]


# --- Slice 3: /state aggregation (S3.3) --------------------------------------


def test_state_default_placeholder(monkeypatch):
  ctl, _, _ = make_control()
  client, _ = make_client(monkeypatch, control=ctl)
  with client:
    body = client.get("/state").json()
  assert body["safety_state"] == "IDLE"  # placeholder default
  assert body["jetson"] is None  # not reached yet
  assert body["camera"]["up"] is None
  # Plugs exist but are unverified until an action or refresh confirms them.
  assert body["plugs"]["arm"]["verified"] is False


def test_state_reflects_camera_from_mqtt(monkeypatch):
  ctl, _, _ = make_control()
  client, _ = make_client(monkeypatch, control=ctl)
  with client:
    relay = client.app.state.relay
    # video publishes retained arm/video/state/camera = {"up": false}.
    relay._on_camera_state("arm/video/state/camera", {"up": False})
    body = client.get("/state").json()
  assert body["camera"]["up"] is False


# --- Slice 3: Jetson control relay (S3.2/S3.4) -------------------------------


def test_pause_relays_to_jetson_and_updates_state(monkeypatch):
  ctl, jetson, _ = make_control()
  client, _ = make_client(monkeypatch, control=ctl)
  with client:
    r = client.post("/control/pause")
    assert r.status_code == 200
    assert jetson.calls == ["pause"]
    assert client.get("/state").json()["safety_state"] == "PAUSED"


def test_resume_sets_running(monkeypatch):
  ctl, jetson, _ = make_control()
  client, _ = make_client(monkeypatch, control=ctl)
  with client:
    client.post("/control/pause")
    r = client.post("/control/resume")
    assert r.status_code == 200
    assert client.get("/state").json()["safety_state"] == "RUNNING"


def test_jetson_failure_is_5xx_not_swallowed(monkeypatch):
  ctl, _, _ = make_control(jetson_fail=True)
  client, _ = make_client(monkeypatch, control=ctl)
  with client:
    r = client.post("/control/stop")
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["error"] == "jetson_call_failed"
    # Safety state must NOT advance on a failed relay.
    assert client.get("/state").json()["safety_state"] == "IDLE"


# --- Slice 3: WiZ plug control, send-and-verify (S3.1/S3.4) ------------------


def test_shutdown_arm_verifies_and_publishes(monkeypatch):
  published = []
  ctl, _, pub = make_control(published=published)
  client, _ = make_client(monkeypatch, control=ctl)
  with client:
    r = client.post("/control/shutdown/arm")
    assert r.status_code == 200
    body = r.json()
    assert body["plug"] == "arm"
    assert body["state"]["on"] is False
    assert body["state"]["verified"] is True
    assert body["state"]["verified_at"]
    # /state reflects the verified off state and the shutdown placeholder.
    st = client.get("/state").json()
    assert st["plugs"]["arm"]["on"] is False
    assert st["safety_state"] == "SHUTDOWN_ARM"
  # Verified plug state was published retained on the plug topic.
  plug_pub = [p for p in pub if p[0] == "arm/supervisor/plug/arm"]
  assert plug_pub, "expected a retained plug-state publish"
  topic, payload, qos, retain = plug_pub[-1]
  assert payload["on"] is False and payload["verified"] is True
  assert retain is True and qos == 1


def test_poweron_arm_returns_to_idle(monkeypatch):
  ctl, _, _ = make_control()
  client, _ = make_client(monkeypatch, control=ctl)
  with client:
    client.post("/control/shutdown/arm")
    r = client.post("/control/poweron/arm")
    assert r.status_code == 200
    assert r.json()["state"]["on"] is True
    assert client.get("/state").json()["safety_state"] == "IDLE"


def test_plug_verify_failure_is_5xx_and_visible_in_state(monkeypatch):
  # The arm plug drops every command: send-and-verify never confirms off.
  ctl, _, _ = make_control(arm_fail=True)
  client, _ = make_client(monkeypatch, control=ctl)
  with client:
    r = client.post("/control/shutdown/arm")
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["error"] == "plug_verify_failed"
    assert detail["plug"] == "arm"
    assert detail["intended_on"] is False
    assert detail["observed_on"] is True  # never went off
    # The failed verify is visible in /state (verified=False), not hidden.
    arm = client.get("/state").json()["plugs"]["arm"]
    assert arm["verified"] is False


def test_unknown_plug_is_404(monkeypatch):
  ctl, _, _ = make_control()
  client, _ = make_client(monkeypatch, control=ctl)
  with client:
    r = client.post("/control/shutdown/toaster")
    assert r.status_code == 404
