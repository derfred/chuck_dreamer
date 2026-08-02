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
  from fessel_shared import MqttClient

  import supervisor.app as appmod

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


def make_control(*, arm_fail=False, jetson_fail=False, published=None, now=None):
  """Build a ControlPlane wired to fake actuators. The publish callback
  records (topic, payload, qos, retain) into `published` if provided. `now`
  overrides the clock used for heartbeat-freshness checks (defaults to the
  real time.monotonic)."""
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
  kwargs = {}
  if now is not None:
    kwargs["now"] = now
  ctl = ControlPlane(
    plugs=plugs,
    jetson=jetson,
    publish=publish,
    jetson_refresh_s=3600,
    plug_refresh_s=3600,
    **kwargs,
  )
  return ctl, jetson, pub_sink


def test_healthz(monkeypatch):
  client, _ = make_client(monkeypatch)
  with client:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    # Release-version stamp (env/file or "unknown" in a source checkout).
    assert "version" in body


def test_diag_payload_returns_requested_size(monkeypatch):
  client, _ = make_client(monkeypatch)
  with client:
    r = client.get("/diag/payload", params={"size": 1000})
    assert r.status_code == 200
    assert len(r.content) == 1000


def test_diag_payload_clamps_oversized_request(monkeypatch):
  client, _ = make_client(monkeypatch)
  with client:
    r = client.get("/diag/payload", params={"size": 999_000_000})
    assert r.status_code == 200
    assert len(r.content) == 8 * 1024 * 1024


def test_diag_payload_default_size(monkeypatch):
  client, _ = make_client(monkeypatch)
  with client:
    r = client.get("/diag/payload")
    assert r.status_code == 200
    assert len(r.content) == 1_000_000


def test_activate_relays_to_mqtt_parameterless(monkeypatch):
  # Parameterless activation (§2.2/§2.9): no body needed, no mode anywhere.
  client, published = make_client(monkeypatch)
  with client:
    r = client.post("/control/live/activate")
    assert r.status_code == 200
    assert r.json()["relayed"] == "activate"
  topics = [p[0] for p in published]
  assert "arm/video/cmd/live/activate" in topics
  activate = next(p for p in published if p[0] == "arm/video/cmd/live/activate")
  # The slimmed LiveActivate carries no mode.
  assert "mode" not in activate[1]
  assert activate[3] is False  # commands not retained


def test_activate_ignores_legacy_bodies(monkeypatch):
  # Legacy callers must still get a 200 — the JSON body is accepted and
  # IGNORED (the Go relay currently sends {"path": "pi"}; older callers sent
  # mode / raw-query forms, including garbage).
  client, published = make_client(monkeypatch)
  with client:
    for body in (
      {"path": "pi"},
      {"path": "pi", "mode": "1280x720@30@2500000"},
      {"path": "pi", "query": "mode=640x480%4030%401000000&jwt=abc"},
      {"path": "pi", "mode": "garbage"},
    ):
      r = client.post("/control/live/activate", json=body)
      assert r.status_code == 200, (body, r.json())
  activates = [p for p in published if p[0] == "arm/video/cmd/live/activate"]
  assert len(activates) == 4
  for a in activates:
    assert "mode" not in a[1]


def test_deactivate_relays(monkeypatch):
  client, published = make_client(monkeypatch)
  with client:
    # Parameterless; a legacy {"path": ...} body is tolerated too.
    r = client.post("/control/live/deactivate")
    assert r.status_code == 200
    r = client.post("/control/live/deactivate", json={"path": "pi"})
    assert r.status_code == 200
  assert sum(p[0] == "arm/video/cmd/live/deactivate" for p in published) == 2


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


# --- Slice 5.5: uploader liveness heartbeat ----------------------------------
# healthy is None (never reported) until the first heartbeat arrives, then
# tracks freshness live at read time — the same pattern as vision/audio
# heartbeats in AnomalyTracker (see test_anomalies.py).


class _Clock:
  def __init__(self):
    self.t = 100.0

  def __call__(self):
    return self.t


def test_uploader_healthy_none_before_first_heartbeat():
  ctl, _, _ = make_control()
  assert ctl.state().upload_backlog.healthy is None


def test_uploader_healthy_true_then_stale_after_timeout():
  clock = _Clock()
  ctl, _, _ = make_control(now=clock)
  ctl.note_uploader_heartbeat()
  assert ctl.state().upload_backlog.healthy is True
  clock.t += 200.0  # past UPLOADER_HEARTBEAT_STALENESS_S (90s)
  assert ctl.state().upload_backlog.healthy is False


def test_uploader_healthy_reflected_via_mqtt_relay(monkeypatch):
  ctl, _, _ = make_control()
  client, _ = make_client(monkeypatch, control=ctl)
  with client:
    relay = client.app.state.relay
    body = client.get("/state").json()
    assert body["upload_backlog"]["healthy"] is None
    relay._on_uploader_heartbeat("arm/video/upload/heartbeat", {})
    body = client.get("/state").json()
    assert body["upload_backlog"]["healthy"] is True


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
  ctl, _jetson, _ = make_control()
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
  _topic, payload, qos, retain = plug_pub[-1]
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


# --- §2.13 SIGHUP reload: hot-reloadable control tunables --------------------


def test_apply_config_updates_refresh_and_retry_policy():
  ctl, _, _ = make_control()
  ctl.apply_config(
    {
      "control": {
        "refresh": {"jetson_s": 11.0, "plug_s": 22.0},
        "wiz": {"retries": 5, "retry_delay_s": 0.5},
      }
    }
  )
  assert ctl._jetson_refresh_s == 11.0
  assert ctl._plug_refresh_s == 22.0
  # Each plug controller's retry policy was updated in place.
  for plug in ctl._plugs.values():
    assert plug._cfg.retries == 5
    assert plug._cfg.retry_delay_s == 0.5


def test_apply_config_does_not_rebind_plug_addresses():
  # Restart-only: the physical device a plug actuates must NOT change on reload.
  ctl, _, _ = make_control()
  before = {name: p._cfg.address for name, p in ctl._plugs.items()}
  ctl.apply_config(
    {"control": {"plugs": {"arm": {"address": "999.999.999.999"}}, "wiz": {"retries": 4}}}
  )
  after = {name: p._cfg.address for name, p in ctl._plugs.items()}
  assert after == before  # addresses unchanged (reload ignores plug bindings)


def test_validate_config_rejects_bad_tunables():
  import pytest

  with pytest.raises(ValueError):
    ControlPlane.validate_config({"control": {"refresh": {"jetson_s": 0}}})
  with pytest.raises(ValueError):
    ControlPlane.validate_config({"control": {"wiz": {"retries": 0}}})
  with pytest.raises(ValueError):
    ControlPlane.validate_config({"control": {"wiz": {"retry_delay_s": -1}}})
  # A good config validates clean.
  ControlPlane.validate_config(
    {"control": {"refresh": {"jetson_s": 5, "plug_s": 10}, "wiz": {"retries": 3}}}
  )
