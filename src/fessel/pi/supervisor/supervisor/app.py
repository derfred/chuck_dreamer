"""supervisor HTTP control plane + MQTT relay + heartbeat.

Slice 1 gave supervisor a single control-plane job: relay live
activate/deactivate from mediamtx (over Tailscale) onto the Pi-internal MQTT
bus. It is NOT an authorization gate — authorization was settled upstream at
mediamtx's token check.

Slice 3 adds supervisor's *action surface* (§2.4): the operator control
commands that take real-world action — pause/stop/resume the Jetson, cut and
restore power to the arm and Jetson via WiZ plugs — plus a read-only `/state`
aggregation that drives the dashboard. The full safety state machine is still
Slice 6; the safety_state field is a placeholder (see control.py).

Endpoints (authenticated at the network layer by Tailscale identity):

  GET  /healthz
  GET  /state                    -> aggregated control-plane state (S3.3)
  POST /control/live/activate    body {path, mode}  -> arm/video/cmd/live/activate
  POST /control/live/deactivate  body {path}        -> arm/video/cmd/live/deactivate
  POST /control/pause            -> Jetson POST /control/pause
  POST /control/stop             -> Jetson POST /control/stop
  POST /control/resume           -> Jetson POST /control/resume
  POST /control/shutdown/{arm,jetson}   -> WiZ turn_off (send-and-verify)
  POST /control/poweron/{arm,jetson}    -> WiZ turn_on  (send-and-verify)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fessel_schemas import (
  LiveActivate,
  LiveDeactivate,
  LiveState,
  fessel_version,
  mode_from_canonical,
)
from fessel_shared import MqttClient, load_yaml_config
from fessel_shared import topics
from pydantic import BaseModel

from .control import ControlPlane, build_control_plane
from .jetson import JetsonError
from .wiz import PlugError

log = logging.getLogger(__name__)

CONFIG_PATH = os.environ.get("SUPERVISOR_CONFIG", "/etc/fessel/supervisor.yaml")
HEARTBEAT_INTERVAL_S = 5.0


class ActivateRequest(BaseModel):
  """mediamtx runOnDemand provides path + either the canonical mode string
  or the raw WHEP query (from which we extract mode). Accepting the raw
  query avoids fragile shell quote-parsing in the mediamtx runOnDemand
  command."""

  path: str
  mode: str | None = None
  query: str | None = None

  def resolved_mode(self) -> str:
    if self.mode:
      return self.mode
    if self.query:
      from urllib.parse import parse_qs, unquote

      # mediamtx 1.18.2 url-encodes the ENTIRE $MTX_QUERY string (the '='
      # and '&' separators become %3D/%26), sometimes on top of the
      # browser's already-encoded WHEP query. Unquote the whole string until
      # it has literal separators (or stops changing), THEN parse, THEN
      # unquote the value's remaining layers. Recovers the canonical mode
      # ('<W>x<H>@<fps>@<bps>') regardless of how many times it was encoded.
      q = self.query.lstrip("?")
      for _ in range(4):
        if "%3D" not in q.upper() and "%26" not in q.upper():
          break
        q = unquote(q)
      vals = parse_qs(q).get("mode")
      if vals:
        v = vals[0]
        for _ in range(4):
          if "%" not in v:
            break
          v = unquote(v)
        return v
    raise ValueError("no mode in request (neither mode nor query[mode])")


class DeactivateRequest(BaseModel):
  path: str


class Relay:
  """Owns the MQTT client and the heartbeat thread.

  `on_camera` (set by create_app) is invoked with the camera up/down boolean
  whenever the retained arm/video/state/camera topic changes, so the control
  plane's /state aggregation has a current camera view (S3.3).
  """

  def __init__(self, config: dict) -> None:
    mqtt_cfg = config.get("mqtt", {})
    self._mqtt = MqttClient(
      client_id="supervisor",
      host=mqtt_cfg.get("host", "127.0.0.1"),
      port=mqtt_cfg.get("port", 1883),
    )
    self._stop = threading.Event()
    self._hb_thread = threading.Thread(target=self._heartbeat_loop, name="hb", daemon=True)
    # Cache of the latest retained live state + an ordered history of state
    # values, so the integration test can assert the off->starting->running
    # ->off progression via an HTTP read path (no MQTT exposed off-Pi).
    self._live_state: dict | None = None
    self._live_history: list[str] = []
    self._lock = threading.Lock()
    self.on_camera: Callable[[bool | None], None] | None = None

  def start(self) -> None:
    self._mqtt.connect()
    # STATE_LIVE has a model (LiveState) -> validate at the transport.
    self._mqtt.subscribe_model(
      topics.STATE_LIVE, LiveState.model_validate, self._on_live_state, qos=topics.QOS_STATE
    )
    # STATE_CAMERA has no dedicated payload model yet (video may publish a
    # richer body than {"up": bool}); stay on raw subscribe and read defensively.
    self._mqtt.subscribe(topics.STATE_CAMERA, self._on_camera_state, qos=topics.QOS_STATE)
    self._mqtt.loop_start()
    self._hb_thread.start()

  def publish(self, topic: str, payload: dict, qos: int, retain: bool) -> None:
    """Publish helper the control plane uses for verified plug state."""
    self._mqtt.publish(topic, payload, qos=qos, retain=retain)

  def _on_live_state(self, _topic: str, live: LiveState) -> None:
    # Validated by subscribe_model. Cache the dumped dict (JSON-serialisable
    # for the /state/live read path) and track the ordered state history.
    with self._lock:
      self._live_state = live.model_dump(mode="json")
      state = live.state.value
      if not self._live_history or self._live_history[-1] != state:
        self._live_history.append(state)

  def _on_camera_state(self, _topic: str, payload: dict) -> None:
    # video publishes {"up": true|false} (or a richer body with an `up` field)
    # retained on arm/video/state/camera. Forward the boolean to the control
    # plane; tolerate a missing field (treated as unknown).
    if self.on_camera is not None:
      up = payload.get("up")
      self.on_camera(up if isinstance(up, bool) else None)

  def live_state(self) -> dict:
    with self._lock:
      return {
        "current": self._live_state,
        "history": list(self._live_history),
      }

  def stop(self) -> None:
    self._stop.set()
    self._mqtt.loop_stop()
    self._mqtt.disconnect()

  def _heartbeat_loop(self) -> None:
    while not self._stop.is_set():
      self._mqtt.publish(
        topics.HEARTBEAT,
        {"from": "supervisor", "ts": time.time()},
        qos=topics.QOS_HEARTBEAT,
      )
      self._stop.wait(HEARTBEAT_INTERVAL_S)

  def publish_activate(self, cmd: LiveActivate) -> None:
    self._mqtt.publish(topics.CMD_LIVE_ACTIVATE, cmd.model_dump(), qos=topics.QOS_CMD, retain=False)

  def publish_deactivate(self, cmd: LiveDeactivate) -> None:
    self._mqtt.publish(
      topics.CMD_LIVE_DEACTIVATE, cmd.model_dump(), qos=topics.QOS_CMD, retain=False
    )


def create_app(config: dict | None = None, control: ControlPlane | None = None) -> FastAPI:
  cfg = config if config is not None else load_yaml_config(CONFIG_PATH)
  relay = Relay(cfg)
  # The control plane publishes verified plug state through the relay's MQTT
  # client. Tests inject a ControlPlane with a fake publish + fake actuators.
  ctl = control if control is not None else build_control_plane(cfg, relay.publish)
  relay.on_camera = ctl.set_camera_up

  @asynccontextmanager
  async def lifespan(app: FastAPI):  # noqa: ANN001
    relay.start()
    ctl.start()
    yield
    ctl.close()
    relay.stop()

  app = FastAPI(title="fessel-supervisor", lifespan=lifespan)
  app.state.relay = relay
  app.state.control = ctl

  @app.get("/healthz")
  def healthz() -> dict:
    # `version` is the release stamp (dpkg version == webui image tag),
    # read from /opt/fessel/VERSION on the Pi. Compare against the cluster
    # webui's /healthz version to detect a drifted release.
    return {"status": "ok", "version": fessel_version()}

  @app.get("/state/live")
  def state_live() -> dict:
    # Observability read path for the integration harness: the cached
    # retained arm/video/state/live plus the ordered state history.
    return relay.live_state()

  @app.post("/control/live/activate")
  def live_activate(req: ActivateRequest) -> dict:
    try:
      mode = mode_from_canonical(req.resolved_mode())
    except ValueError as e:
      raise HTTPException(status_code=400, detail=str(e)) from e
    cmd = LiveActivate(path=req.path, mode=mode)
    log.info("relay activate path=%s mode=%s", cmd.path, cmd.mode)
    relay.publish_activate(cmd)
    return {"relayed": "activate", "path": cmd.path}

  @app.post("/control/live/deactivate")
  def live_deactivate(req: DeactivateRequest) -> dict:
    cmd = LiveDeactivate(path=req.path)
    log.info("relay deactivate path=%s", cmd.path)
    relay.publish_deactivate(cmd)
    return {"relayed": "deactivate", "path": cmd.path}

  # --- Slice 3 control surface (S3.3, S3.4) ---------------------------------
  # Authenticated at the network layer (Tailscale identity); no app-level auth,
  # matching the rest of supervisor's control plane. Each call emits one
  # structured log line (action, caller, outcome, latency) and updates the
  # cached state on success. Actuator/Jetson failures surface as 503 with a
  # structured body — a 5xx, not 4xx, because the caller did nothing wrong.

  def _caller(request: Request) -> str:
    # Best-effort caller attribution from the Tailscale egress proxy. Falls
    # back to the peer IP. Used only for the audit log line.
    return request.headers.get("Tailscale-User-Login") or (
      request.client.host if request.client else "unknown"
    )

  def _run_control(action: str, request: Request, fn: Callable[[], object]):
    t0     = time.monotonic()
    caller = _caller(request)
    try:
      result = fn()
    except PlugError as e:
      _log_control(action, caller, "actuator_failed", t0)
      raise HTTPException(
        status_code=503,
        detail={
          "error": "plug_verify_failed",
          "plug": e.name,
          "intended_on": e.intent,
          "observed_on": e.observed,
          "message": e.detail,
        },
      ) from e
    except JetsonError as e:
      _log_control(action, caller, "jetson_failed", t0)
      raise HTTPException(
        status_code=503,
        detail={
          "error": "jetson_call_failed",
          "action": e.action,
          "jetson_status": e.status,
          "message": e.detail,
        },
      ) from e
    except KeyError as e:
      # Unknown plug name -> 404 (a malformed request, not an actuator fault).
      _log_control(action, caller, "unknown_target", t0)
      raise HTTPException(status_code=404, detail=f"unknown plug: {e}") from e
    _log_control(action, caller, "success", t0)
    return result

  def _log_control(action: str, caller: str, outcome: str, t0: float) -> None:
    log.info(
      '{"event":"control","action":"%s","caller":"%s","outcome":"%s","latency_ms":%d}',
      action,
      caller,
      outcome,
      int((time.monotonic() - t0) * 1000),
    )

  @app.get("/state")
  def state() -> dict:
    # Read-only, idempotent: returns the locally cached view (updated on every
    # action + a slow background refresh). It does NOT re-query the Jetson or
    # the plugs per call (that would amplify load and add latency, S3.3).
    return ctl.state().model_dump()

  @app.post("/control/pause")
  def control_pause(request: Request) -> dict:
    return {"jetson": _run_control("pause", request, ctl.pause)}

  @app.post("/control/stop")
  def control_stop(request: Request) -> dict:
    return {"jetson": _run_control("stop", request, ctl.stop)}

  @app.post("/control/resume")
  def control_resume(request: Request) -> dict:
    return {"jetson": _run_control("resume", request, ctl.resume)}

  @app.post("/control/shutdown/{name}")
  def control_shutdown(name: str, request: Request) -> dict:
    ps = _run_control(f"shutdown/{name}", request, lambda: ctl.shutdown(name))
    return {"plug": name, "state": ps.model_dump()}

  @app.post("/control/poweron/{name}")
  def control_poweron(name: str, request: Request) -> dict:
    ps = _run_control(f"poweron/{name}", request, lambda: ctl.poweron(name))
    return {"plug": name, "state": ps.model_dump()}

  return app


# Module-level app for `uvicorn supervisor.app:app`.
app = create_app() if os.environ.get("SUPERVISOR_EAGER_APP") else None
