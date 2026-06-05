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

import uuid

from fastapi import FastAPI, HTTPException, Request
from fessel_schemas import (
  LiveActivate,
  LiveDeactivate,
  LiveState,
  LiveStateValue,
  RecordingFlagUploadCmd,
  RecordingListItem,
  RecordingMetadata,
  RecordingStartCmd,
  RecordingState,
  UploadBacklog,
  UploadProgress,
  fessel_version,
  mode_from_canonical,
)
from fessel_shared import ConfigReloader, MqttClient, Storage, load_component_config
from fessel_shared import topics
from pydantic import BaseModel

from .bandwidth import BandwidthCoordinator
from .control import ControlPlane, build_control_plane
from .jetson import JetsonError
from .recordings import RingProxy
from .wiz import PlugError

log = logging.getLogger(__name__)

# Single Pi config (Requirements §6); supervisor reads its `supervisor:` subtree
# plus the shared `mqtt`/`storage` sections. FESSEL_CONFIG overrides the path.
CONFIG_PATH = os.environ.get("FESSEL_CONFIG", "/etc/fessel/fessel.yaml")
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


class RecordingStartBody(BaseModel):
  """Body of supervisor POST /recording/start (S4.1). webui-backend is the only
  caller; the canonical mode string + operator pass-through are enough (no
  mediamtx-style raw-query flexibility needed)."""

  mode: str
  operator: str | None = None


class FlagUploadBody(BaseModel):
  """Body of supervisor POST /recording/flag-upload (S4.1)."""

  recording_id: str


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
    # Slice 4: recording state + upload backlog pushed into the control plane,
    # plus a cache of per-recording upload progress for /recordings/<id>/upload.
    self.on_recording: Callable[[RecordingState], None] | None = None
    self.on_backlog: Callable[[UploadBacklog], None] | None = None
    # §2.12: the bandwidth coordinator needs live-viewer presence; fired with
    # True when the live state machine is `running`, else False.
    self.on_live_running: Callable[[bool], None] | None = None
    self._backlog_count = 0
    self._backlog_oldest: int | None = None
    self._upload_progress: dict[str, dict] = {}

  def start(self) -> None:
    self._mqtt.connect()
    # STATE_LIVE has a model (LiveState) -> validate at the transport.
    self._mqtt.subscribe_model(
      topics.STATE_LIVE, LiveState.model_validate, self._on_live_state, qos=topics.QOS_STATE
    )
    # STATE_CAMERA has no dedicated payload model yet (video may publish a
    # richer body than {"up": bool}); stay on raw subscribe and read defensively.
    self._mqtt.subscribe(topics.STATE_CAMERA, self._on_camera_state, qos=topics.QOS_STATE)
    # Slice 4: recording state (retained, validated), upload backlog gauges, and
    # per-recording upload progress (wildcard, validated). All Pi-internal.
    self._mqtt.subscribe_model(
      topics.STATE_RECORDING,
      RecordingState.model_validate,
      self._on_recording_state,
      qos=topics.QOS_STATE,
    )
    self._mqtt.subscribe(topics.UPLOAD_BACKLOG_COUNT, self._on_backlog_count, qos=topics.QOS_UPLOAD)
    self._mqtt.subscribe(
      topics.UPLOAD_BACKLOG_OLDEST, self._on_backlog_oldest, qos=topics.QOS_UPLOAD
    )
    self._mqtt.subscribe_model(
      f"{topics.UPLOAD_PREFIX}/+",
      UploadProgress.model_validate,
      self._on_upload_progress,
      qos=topics.QOS_UPLOAD,
    )
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
    # §2.12: a live viewer is present exactly while the live branch is running.
    if self.on_live_running is not None:
      self.on_live_running(live.state is LiveStateValue.running)

  def _on_camera_state(self, _topic: str, payload: dict) -> None:
    # video publishes {"up": true|false} (or a richer body with an `up` field)
    # retained on arm/video/state/camera. Forward the boolean to the control
    # plane; tolerate a missing field (treated as unknown).
    if self.on_camera is not None:
      up = payload.get("up")
      self.on_camera(up if isinstance(up, bool) else None)

  # --- Slice 4: recording state + upload progress/backlog caching -----------

  def _on_recording_state(self, _topic: str, recording: RecordingState) -> None:
    # Validated by subscribe_model; forward to the control plane so /state's
    # `recording` field reflects video's retained state machine value (S4.2).
    if self.on_recording is not None:
      self.on_recording(recording)

  def _on_backlog_count(self, _topic: str, payload: dict) -> None:
    with self._lock:
      self._backlog_count = int(payload.get("value") or 0)
    self._emit_backlog()

  def _on_backlog_oldest(self, _topic: str, payload: dict) -> None:
    with self._lock:
      val = payload.get("value")
      self._backlog_oldest = int(val) if isinstance(val, int) else None
    self._emit_backlog()

  def _emit_backlog(self) -> None:
    if self.on_backlog is None:
      return
    with self._lock:
      backlog = UploadBacklog(
        count=self._backlog_count, oldest_pending_seconds=self._backlog_oldest
      )
    self.on_backlog(backlog)

  def _on_upload_progress(self, _topic: str, progress: UploadProgress) -> None:
    # Cache the latest per-recording progress for /recordings/<id>/upload (B4.6).
    with self._lock:
      self._upload_progress[progress.recording_id] = progress.model_dump(mode="json")

  def upload_progress(self, recording_id: str) -> dict | None:
    with self._lock:
      got = self._upload_progress.get(recording_id)
      return dict(got) if got is not None else None

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

  # --- Slice 4: recording command publishers (S4.1) -------------------------

  def publish_recording_start(self, cmd: RecordingStartCmd) -> None:
    self._mqtt.publish(
      topics.CMD_RECORDING_START, cmd.model_dump(), qos=topics.QOS_CMD, retain=False
    )

  def publish_recording_stop(self) -> None:
    self._mqtt.publish(topics.CMD_RECORDING_STOP, {}, qos=topics.QOS_CMD, retain=False)

  def publish_recording_flag_upload(self, cmd: RecordingFlagUploadCmd) -> None:
    self._mqtt.publish(
      topics.CMD_RECORDING_FLAG_UPLOAD, cmd.model_dump(), qos=topics.QOS_CMD, retain=False
    )


def create_app(
  config: dict | None = None,
  control: ControlPlane | None = None,
  coordinator: BandwidthCoordinator | None = None,
) -> FastAPI:
  cfg = config if config is not None else load_component_config(CONFIG_PATH, "supervisor")
  relay = Relay(cfg)
  # The control plane publishes verified plug state through the relay's MQTT
  # client. Tests inject a ControlPlane with a fake publish + fake actuators.
  ctl = control if control is not None else build_control_plane(cfg, relay.publish)
  relay.on_camera = ctl.set_camera_up
  relay.on_recording = ctl.set_recording
  relay.on_backlog = ctl.set_upload_backlog

  # §2.12: the bandwidth coordinator gates recording uploads on viewer presence.
  # It publishes the gate through the relay's MQTT client; tests may inject one
  # with a fake publish. Live-viewer presence comes from the relay's live-state
  # callback; ring-viewer presence from the ring proxy's read hook below.
  bw = coordinator if coordinator is not None else BandwidthCoordinator(publish=relay.publish)
  relay.on_live_running = bw.set_live_running

  # Slice 4: SSD-backed recording storage. supervisor reads the ring + explicit
  # recordings from the same mount video writes to (S4.3/S4.4); the proxy
  # enforces traversal protection. ssd_path is the shared `storage` section of
  # fessel.yaml (same value video + uploader read).
  storage = Storage((cfg.get("storage") or {}).get("ssd_path", "/mnt/ssd"))
  ring_proxy = RingProxy(storage.ring_dir, storage.explicit_dir, on_ring_read=bw.note_ring_read)

  # SIGHUP reload (§2.13): re-read the supervisor config and apply the
  # hot-reloadable control tunables (refresh intervals, WiZ retry policy).
  # uvicorn handles only SIGINT/SIGTERM, so SIGHUP is ours to claim. Installed
  # in the lifespan startup (main thread); the worker thread does the reload.
  reloader = ConfigReloader(
    path=CONFIG_PATH,
    component="supervisor",
    validate=ControlPlane.validate_config,
    apply=ctl.apply_config,
  )

  @asynccontextmanager
  async def lifespan(app: FastAPI):  # noqa: ANN001
    relay.start()
    ctl.start()
    bw.start()
    reloader.install()
    yield
    reloader.close()
    bw.close()
    ctl.close()
    relay.stop()

  app = FastAPI(title="fessel-supervisor", lifespan=lifespan)
  app.state.relay = relay
  app.state.control = ctl
  app.state.bandwidth = bw
  app.state.reloader = reloader

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

  # --- Slice 4: recording control + ring/recordings proxy --------------------
  # Same network-layer auth model (Tailscale identity) as the rest of
  # supervisor's control plane. Recording commands are async: they return once
  # the MQTT command is published (S4.1), not once video has transitioned —
  # the caller polls /state (S4.2) or the retained recording topic.

  @app.post("/recording/start")
  def recording_start(body: RecordingStartBody, request: Request) -> dict:
    # Mint a fresh recording-id here (S4.1) and return it; video records under
    # it. 200 means "command published", not "recording active".
    try:
      mode = mode_from_canonical(body.mode)
    except ValueError as e:
      raise HTTPException(status_code=400, detail=str(e)) from e
    recording_id = str(uuid.uuid4())
    cmd = RecordingStartCmd(recording_id=recording_id, mode=mode, operator=body.operator)
    _log_control("recording/start", _caller(request), "published", time.monotonic())
    relay.publish_recording_start(cmd)
    return {"recording_id": recording_id}

  @app.post("/recording/stop")
  def recording_stop(request: Request) -> dict:
    _log_control("recording/stop", _caller(request), "published", time.monotonic())
    relay.publish_recording_stop()
    return {"relayed": "recording/stop"}

  @app.post("/recording/flag-upload")
  def recording_flag_upload(body: FlagUploadBody, request: Request) -> dict:
    # 404 if the recording does not exist locally; 409 if it is still the active
    # recording (cannot upload a recording that has not finalised, S4.1). The
    # actual flag + marker write happens in video (V4.5); supervisor only
    # publishes the command after these guards pass.
    rid = body.recording_id
    meta = storage.read_metadata(rid)
    if meta is None:
      raise HTTPException(status_code=404, detail=f"unknown recording: {rid}")
    active = ctl.state().recording.active_recording_id
    if active == rid:
      raise HTTPException(status_code=409, detail="recording is still active")
    _log_control("recording/flag-upload", _caller(request), "published", time.monotonic())
    relay.publish_recording_flag_upload(RecordingFlagUploadCmd(recording_id=rid))
    return {"relayed": "recording/flag-upload", "recording_id": rid}

  @app.get("/recordings")
  def list_recordings() -> list[dict]:
    # Enumerate finalised explicit recordings on the Pi, newest first (S4.4).
    items: list[RecordingListItem] = []
    for meta in storage.list_recordings():
      items.append(_meta_to_list_item(meta))
    return [i.model_dump(mode="json") for i in items]

  @app.get("/recordings/{recording_id}/upload")
  def recording_upload(recording_id: str) -> dict:
    # B4.6: the cached per-recording upload progress (from the retained MQTT
    # topic). 404 if no progress has been seen (never flagged).
    progress = relay.upload_progress(recording_id)
    if progress is None:
      raise HTTPException(status_code=404, detail="no upload progress for recording")
    return progress

  @app.get("/recordings/{recording_id}/index.m3u8")
  def recording_playlist(recording_id: str):  # noqa: ANN202
    # Local playback proxy for an unflagged/not-yet-uploaded recording (B4.3
    # open seam: symmetric with the ring proxy).
    return ring_proxy.recording_playlist(recording_id)

  @app.get("/recordings/{recording_id}/{segment}")
  def recording_segment(recording_id: str, segment: str):  # noqa: ANN202
    return ring_proxy.recording_segment(recording_id, segment)

  @app.get("/ring/index.m3u8")
  def ring_playlist():  # noqa: ANN202
    # Serve the ring playlist as-is (S4.3). Range requests + traversal guard are
    # handled in RingProxy.
    return ring_proxy.ring_playlist()

  @app.get("/ring/{segment}")
  def ring_segment(segment: str):  # noqa: ANN202
    return ring_proxy.ring_segment(segment)

  return app


def _meta_to_list_item(meta: RecordingMetadata) -> RecordingListItem:
  return RecordingListItem(
    recording_id=meta.id,
    started_at=meta.started_at,
    ended_at=meta.ended_at,
    duration_seconds=meta.duration_seconds,
    operator=meta.operator,
    flagged_for_upload=meta.flagged_for_upload,
    upload_state=meta.upload_state,
  )


# Module-level app for `uvicorn supervisor.app:app`.
app = create_app() if os.environ.get("SUPERVISOR_EAGER_APP") else None
