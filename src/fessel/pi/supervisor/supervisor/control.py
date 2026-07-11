"""supervisor control plane: cached state + actuator dispatch (S3.3, S3.4).

This is the *action surface* of the interventions the architecture assigns to
supervisor (§2.4) — pause/stop/resume the Jetson, cut/restore power via the
WiZ plugs — plus the read-only `/state` aggregation that drives the dashboard.

The full safety **state machine** (transitions, anomaly fusion, tier timers)
is Slice 6. Slice 3 ships a **placeholder** safety state that holds whatever
the operator's last command implies: pause -> PAUSED, stop -> STOPPED,
shutdown/arm -> SHUTDOWN_ARM, power-on -> IDLE, etc. It is enough to drive the
dashboard and explicitly not a real safety machine.

`/state` is read-only and idempotent: it returns supervisor's locally cached
view, NOT a fresh re-query of the Jetson and plugs on every call (that would
amplify load and add latency). The cache is updated on every action and on a
slow background refresh (Jetson /state, plug getPilot), both configurable.

Plug state in the cache is the last *verified* read from send-and-verify
(S3.1), not "the last command we issued". A failed verify is visible.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone

from fessel_schemas import (
  CameraState,
  PlugState,
  RecordingState,
  SafetyState,
  StateResponse,
  UploadBacklog,
  VersionState,
  fessel_version,
)
from fessel_shared import topics

from .anomalies import AnomalyTracker
from .jetson import JetsonClient, JetsonError
from .wiz import PlugController, PlugError

log = logging.getLogger(__name__)

# Plug names. Stable strings used in URLs (/control/shutdown/<name>), the
# cache keys, and the MQTT plug-state topic.
PLUG_ARM = "arm"
PLUG_JETSON = "jetson"

UPLOADER_HEARTBEAT_STALENESS_S = 90.0

# A publish callback: (topic, payload-dict, qos, retain) -> None. The control
# plane uses it to publish verified plug state; injecting it keeps this module
# free of an MQTT-client dependency and trivially testable.
PublishFn = Callable[[str, dict, int, bool], None]


def _now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


class ControlPlane:
  """Owns the actuator clients and the cached control-plane state.

  Thread-safe: actions and the background refresh both mutate the cache under
  a lock. Action methods raise (PlugError / JetsonError) on failure so the
  HTTP layer can map them to a 5xx; on success they update the placeholder
  safety state and the cached plug/Jetson view.
  """

  def __init__(
    self,
    *,
    plugs: dict[str, PlugController],
    jetson: JetsonClient | None,
    publish: PublishFn,
    jetson_refresh_s: float = 5.0,
    plug_refresh_s: float = 10.0,
    anomalies: AnomalyTracker | None = None,
    now: Callable[[], float] = time.monotonic,
  ) -> None:
    self._plugs = plugs
    self._jetson = jetson
    self._publish = publish
    self._jetson_refresh_s = jetson_refresh_s
    self._plug_refresh_s = plug_refresh_s
    self._now = now
    # Slice 5: live vision/audio values + the in-memory anomaly log, fed from
    # the new MQTT topics via the relay (S5.1). Surfaced in /state + /anomalies.
    self._anomalies = anomalies if anomalies is not None else AnomalyTracker()
    # Uploader liveness (distinct from _upload_backlog): last heartbeat time,
    self._uploader_hb_at: float | None = None

    self._lock = threading.Lock()
    self._safety = SafetyState.IDLE
    self._plug_state: dict[str, PlugState] = {name: PlugState() for name in plugs}
    self._jetson_state: dict | None = None
    self._camera_up: bool | None = None
    # Slice 4: recording state + upload backlog, fed from the retained MQTT
    # topics via the relay (S4.2). Cached so /state does not re-query per call.
    self._recording = RecordingState()
    self._upload_backlog = UploadBacklog()

    self._stop = threading.Event()
    self._refresh_thread = threading.Thread(
      target=self._refresh_loop, name="control-refresh", daemon=True
    )

  # --- lifecycle -------------------------------------------------------------

  def start(self) -> None:
    # Publish the initial (IDLE) safety state retained so video's rest detection
    # (V5.4) has a value immediately. Done here, not in __init__, so it runs
    # after the relay has connected MQTT (create_app builds the control plane
    # before relay.start()).
    self._publish_safety_state(self._safety)
    self._refresh_thread.start()

  def close(self) -> None:
    # Lifecycle teardown — deliberately NOT named `stop` to avoid colliding
    # with the Jetson `stop` action method below.
    self._stop.set()
    if self._jetson is not None:
      self._jetson.close()

  # --- inputs from the MQTT relay -------------------------------------------

  def set_camera_up(self, up: bool | None) -> None:
    with self._lock:
      self._camera_up = up

  def set_recording(self, recording: RecordingState) -> None:
    with self._lock:
      self._recording = recording

  def set_upload_backlog(self, backlog: UploadBacklog) -> None:
    with self._lock:
      self._upload_backlog = backlog

  def note_uploader_heartbeat(self) -> None:
    with self._lock:
      self._uploader_hb_at = self._now()

  @property
  def anomalies(self) -> AnomalyTracker:
    """The Slice-5 sensing tracker; the relay feeds it from MQTT, the app reads
    it for /anomalies."""
    return self._anomalies

  def _publish_safety_state(self, state: SafetyState) -> None:
    """Publish the safety state retained on arm/supervisor/state so video's
    rest detection (V5.4) can learn the operator-driven state. Publishing the
    full StateResponse-shaped {safety_state} keeps the topic forward-compatible
    with the Slice-6 machine."""
    self._publish(
      topics.STATE_SUPERVISOR,
      {"safety_state": state.value},
      topics.QOS_STATE,
      topics.RETAIN_STATE,
    )

  # --- SIGHUP reload (§2.13): hot-reloadable control tunables ---------------

  @staticmethod
  def validate_config(cfg: dict) -> None:
    """Reject a control config whose reloadable tunables are malformed. Raises
    so the reloader keeps the running config. Restart-only keys (plug
    addresses, wiz.driver, jetson.base_url) are NOT validated here — they are
    not applied on reload, so a change to them is inert until restart."""
    control = cfg.get("control", {})
    refresh = control.get("refresh", {})
    for key in ("jetson_s", "plug_s"):
      if key in refresh and not (
        isinstance(refresh[key], (int, float)) and refresh[key] > 0
      ):
        raise ValueError(f"control.refresh.{key} must be a positive number")
    wiz = control.get("wiz", {})
    if "retries" in wiz and not (isinstance(wiz["retries"], int) and wiz["retries"] >= 1):
      raise ValueError("control.wiz.retries must be an int >= 1")
    if "retry_delay_s" in wiz and not (
      isinstance(wiz["retry_delay_s"], (int, float)) and wiz["retry_delay_s"] >= 0
    ):
      raise ValueError("control.wiz.retry_delay_s must be a non-negative number")

  def apply_config(self, cfg: dict) -> None:
    """Apply the hot-reloadable control tunables (§2.13 conservative scope):
    the background-refresh intervals and the WiZ send-and-verify retry policy.

    NOT reloaded (restart-only, deliberately): which physical device each plug
    actuates (`control.plugs.*.address`), the WiZ driver, and the Jetson
    base_url — re-binding a safety actuator or a control endpoint at runtime is
    a footgun, so a change to those is inert until a restart. The Jetson client
    is also left as-is (rebuilding it is in the next scope tier, not this one)."""
    control = cfg.get("control", {})
    refresh = control.get("refresh", {})
    with self._lock:
      if "jetson_s" in refresh:
        self._jetson_refresh_s = float(refresh["jetson_s"])
      if "plug_s" in refresh:
        self._plug_refresh_s = float(refresh["plug_s"])
    wiz = control.get("wiz", {})
    if "retries" in wiz or "retry_delay_s" in wiz:
      retries = int(wiz.get("retries", 3))
      delay = float(wiz.get("retry_delay_s", 0.2))
      for plug in self._plugs.values():
        plug.set_retry_policy(retries, delay)

  # --- read path -------------------------------------------------------------

  def state(self) -> StateResponse:
    # The Slice-5 sensing fields come from the tracker (its own lock); read them
    # outside self._lock to avoid nesting the two locks.
    vision = self._anomalies.vision_state()
    audio = self._anomalies.audio_state()
    recent = self._anomalies.recent()
    with self._lock:
      hb_at = self._uploader_hb_at
      healthy = None if hb_at is None else (self._now() - hb_at) < UPLOADER_HEARTBEAT_STALENESS_S
      backlog = self._upload_backlog.model_copy(update={"healthy": healthy})
      return StateResponse(
        safety_state=self._safety,
        jetson=self._jetson_state,
        plugs={name: ps.model_copy() for name, ps in self._plug_state.items()},
        camera=CameraState(up=self._camera_up),
        recording=self._recording.model_copy(),
        upload_backlog=backlog,
        vision=vision,
        audio=audio,
        recent_anomalies=recent,
        version=VersionState(component=fessel_version()),
      )

  # --- Jetson actions --------------------------------------------------------

  def pause(self) -> dict:
    return self._jetson_action("pause", SafetyState.PAUSED)

  def stop(self) -> dict:
    return self._jetson_action("stop", SafetyState.STOPPED)

  def resume(self) -> dict:
    return self._jetson_action("resume", SafetyState.RUNNING)

  def _jetson_action(self, action: str, new_state: SafetyState) -> dict:
    if self._jetson is None:
      raise JetsonError(action, "jetson client not configured")
    result = getattr(self._jetson, action)()
    changed = False
    with self._lock:
      changed = self._safety != new_state
      self._safety = new_state
      if isinstance(result, dict) and result:
        self._jetson_state = result
    if changed:
      self._publish_safety_state(new_state)
    return result

  # --- plug actions ----------------------------------------------------------

  def shutdown(self, name: str) -> PlugState:
    return self._set_plug(name, on=False)

  def poweron(self, name: str) -> PlugState:
    return self._set_plug(name, on=True)

  def _set_plug(self, name: str, *, on: bool) -> PlugState:
    plug = self._plugs.get(name)
    if plug is None:
      raise KeyError(name)
    try:
      verified_on = plug.set_state(on)
    except PlugError as e:
      # Record the failed verify in the cache so /state reflects it, then
      # re-raise so the HTTP layer returns a 5xx. `verified=False` is the
      # signal that the cached `on` is NOT trustworthy.
      ps = PlugState(on=e.observed, verified=False, verified_at=_now_iso())
      with self._lock:
        self._plug_state[name] = ps
      raise
    ps = PlugState(on=verified_on, verified=True, verified_at=_now_iso())
    new_safety = self._safety_after_plug(name, verified_on)
    with self._lock:
      self._plug_state[name] = ps
      changed = self._safety != new_safety
      self._safety = new_safety
    self._publish_plug(name, ps)
    if changed:
      self._publish_safety_state(new_safety)
    return ps

  @staticmethod
  def _safety_after_plug(name: str, on: bool) -> SafetyState:
    # Placeholder semantics: a verified power-off implies the matching
    # shutdown state; a verified power-on returns to IDLE.
    if not on:
      return SafetyState.SHUTDOWN_ARM if name == PLUG_ARM else SafetyState.SHUTDOWN_JETSON
    return SafetyState.IDLE

  def _publish_plug(self, name: str, ps: PlugState) -> None:
    self._publish(topics.plug_state(name), ps.model_dump(), topics.QOS_PLUG, topics.RETAIN_PLUG)

  # --- background refresh ----------------------------------------------------

  def _refresh_loop(self) -> None:
    # Wait one interval before the first refresh: the cache already starts
    # empty/known and an immediate startup refresh would (a) race operator
    # actions issued right after boot and (b) make tests nondeterministic.
    next_jetson = self._jetson_refresh_s
    next_plugs = self._plug_refresh_s
    start = time.monotonic()
    while not self._stop.is_set():
      # Re-read the intervals each iteration so a SIGHUP reload (§2.13) that
      # changes them is honoured promptly, including a shortened tick.
      tick = min(self._jetson_refresh_s, self._plug_refresh_s)
      self._stop.wait(tick)
      if self._stop.is_set():
        return
      elapsed = time.monotonic() - start
      if elapsed >= next_jetson:
        self._refresh_jetson()
        next_jetson = elapsed + self._jetson_refresh_s
      if elapsed >= next_plugs:
        self._refresh_plugs()
        next_plugs = elapsed + self._plug_refresh_s

  def _refresh_jetson(self) -> None:
    if self._jetson is None:
      return
    try:
      st = self._jetson.state()
    except JetsonError as e:
      log.warning('{"event":"jetson_refresh","outcome":"error","detail":"%s"}', e.detail)
      return
    with self._lock:
      self._jetson_state = st

  def _refresh_plugs(self) -> None:
    for name, plug in self._plugs.items():
      try:
        on = plug.read_state()
      except Exception as e:  # noqa: BLE001 — refresh is best-effort
        log.warning(
          '{"event":"plug_refresh","plug":"%s","outcome":"error","detail":"%s"}',
          name,
          f"{type(e).__name__}: {e}",
        )
        with self._lock:
          # Mark the cached state unverified so a refresh failure is visible,
          # but keep the last-known `on` (more useful than nulling it).
          prev = self._plug_state.get(name, PlugState())
          self._plug_state[name] = PlugState(on=prev.on, verified=False, verified_at=_now_iso())
        continue
      ps = PlugState(on=on, verified=True, verified_at=_now_iso())
      with self._lock:
        self._plug_state[name] = ps
      self._publish_plug(name, ps)


def build_control_plane(config: dict, publish: PublishFn) -> ControlPlane:
  """Construct a ControlPlane from the supervisor component's `control` config
  (the `supervisor.control` subtree of fessel.yaml, flattened to `control:` at
  the top of the dict this receives).

  Expected shape (all under `control:`):

    control:
      plugs:
        arm:    {address: 192.168.1.50}
        jetson: {address: 192.168.1.51}
      wiz:
        driver: pywizlight          # 'pywizlight' (prod) | 'fake' (integration)
        retries: 3
        retry_delay_s: 0.2
      jetson:
        base_url: http://192.168.1.40:8080
        timeout_s: 3.0
        auth_token: null            # optional pre-shared token
      refresh:
        jetson_s: 5.0
        plug_s: 10.0

  Plug controllers are built via the pywizlight driver (real device) by
  default. The integration harness sets `wiz.driver: fake` for an in-memory
  bulb, and a plug may carry `mode: drop` to make it always fail verify (the
  safety-path assertion). When a plug has no address AND the driver is
  pywizlight, it is omitted (the matching endpoints then 404 rather than
  pretending an actuator exists); the fake driver needs no address.
  """
  from .wiz import PlugConfig, build_controller

  cc = config.get("control", {})
  wiz_cfg = cc.get("wiz", {})
  driver = wiz_cfg.get("driver", "pywizlight")
  retries = int(wiz_cfg.get("retries", 3))
  retry_delay = float(wiz_cfg.get("retry_delay_s", 0.2))

  plugs: dict[str, PlugController] = {}
  for name, pcfg in (cc.get("plugs") or {}).items():
    pcfg = pcfg or {}
    address = pcfg.get("address")
    mode = pcfg.get("mode", "ok")
    # Real driver needs an address; the fake driver synthesises the bulb.
    if driver != "fake" and not address:
      continue
    plugs[name] = build_controller(
      PlugConfig(name=name, address=address or "fake", retries=retries, retry_delay_s=retry_delay),
      driver=driver,
      mode=mode,
    )

  jetson_client: JetsonClient | None = None
  jcfg = cc.get("jetson") or {}
  if jcfg.get("base_url"):
    from .jetson import JetsonConfig

    jetson_client = JetsonClient(
      JetsonConfig(
        base_url=jcfg["base_url"],
        timeout_s=float(jcfg.get("timeout_s", 3.0)),
        auth_token=jcfg.get("auth_token"),
      )
    )

  refresh = cc.get("refresh", {})
  # Slice 5: the sensing tracker. `anomalies.{log_cap, recent_cap, staleness_s}`
  # live under the supervisor subtree (X5.2); defaults match the plan.
  acfg = config.get("anomalies") or {}
  anomalies = AnomalyTracker(
    log_cap=int(acfg.get("log_cap", 100)),
    recent_cap=int(acfg.get("recent_cap", 10)),
    staleness_s=float(acfg.get("heartbeat_staleness_s", 6.0)),
  )
  return ControlPlane(
    plugs=plugs,
    jetson=jetson_client,
    publish=publish,
    jetson_refresh_s=float(refresh.get("jetson_s", 5.0)),
    plug_refresh_s=float(refresh.get("plug_s", 10.0)),
    anomalies=anomalies,
  )
