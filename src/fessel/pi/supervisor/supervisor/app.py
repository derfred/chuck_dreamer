"""supervisor HTTP control plane + MQTT relay + heartbeat.

Slice 1 gave supervisor a single control-plane job: relay live
activate/deactivate from the cluster-side webui relay (over Tailscale) onto
the Pi-internal MQTT bus. It is NOT an authorization gate — authorization was
settled upstream (oauth2-proxy in front of webui); supervisor is a relay, not
a gate (architecture §2.9).

Slice 3 adds supervisor's *action surface* (§2.4): the operator control
commands that take real-world action — pause/stop/resume the Jetson, cut and
restore power to the arm and Jetson via WiZ plugs — plus a read-only `/state`
aggregation that drives the dashboard. The full safety state machine is still
Slice 6; the safety_state field is a placeholder (see control.py).

Endpoints (authenticated at the network layer by Tailscale identity):

  GET  /healthz
  GET  /diag/payload?size=N      -> N bytes (clamped), for the cluster-side
                                     connection bandwidth probe
  GET  /state                    -> aggregated control-plane state (S3.3)
  POST /control/live/activate    parameterless      -> arm/video/cmd/live/activate
  POST /control/live/deactivate  parameterless      -> arm/video/cmd/live/deactivate
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
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fessel_schemas import (
  AnomalyRecordingMetadata,
  AnomalyType,
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
)
from fessel_shared import ConfigReloader, MqttClient, Storage, load_component_config, topics
from pydantic import BaseModel, Field

from .bandwidth import BandwidthCoordinator
from .control import ControlPlane, build_control_plane
from .jetson import JetsonError
from .recordings import RecordingProxy
from .wiz import PlugError

log = logging.getLogger(__name__)

# Single Pi config (Requirements §6); supervisor reads its `supervisor:` subtree
# plus the shared `mqtt`/`storage` sections. FESSEL_CONFIG overrides the path.
CONFIG_PATH = os.environ.get("FESSEL_CONFIG", "/etc/fessel/fessel.yaml")
HEARTBEAT_INTERVAL_S = 5.0


class RecordingStartBody(BaseModel):
  """Body of supervisor POST /recording/start (S4.1). webui-backend is the only
  caller. Recording resolution is a deploy setting (recordings reuse the ring
  encode), so there is no mode: `lookback_seconds` reaches the recording's start
  back into the ring (0 = now), `upload_when_done` flags it for upload on
  finalise, and `operator` is passed through for metadata."""

  lookback_seconds: float = Field(default=0.0, ge=0)
  upload_when_done: bool = False
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
    self.on_uploader_heartbeat: Callable[[], None] | None = None
    # §2.12: the bandwidth coordinator needs live-viewer presence; fired with
    # True when the live state machine is `running`, else False.
    self.on_live_running: Callable[[bool], None] | None = None
    self._backlog_count = 0
    self._backlog_oldest: int | None = None
    self._upload_progress: dict[str, dict] = {}
    # Slice 5: the sensing tracker (live vision/audio + anomaly log), set by
    # create_app to the control plane's tracker. The relay feeds it from MQTT.
    self.anomalies = None  # type: AnomalyTracker | None

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
    self._mqtt.subscribe(
      topics.UPLOAD_HEARTBEAT, self._on_uploader_heartbeat, qos=topics.QOS_UPLOAD_HEARTBEAT
    )
    self._mqtt.subscribe_model(
      f"{topics.UPLOAD_PREFIX}/+",
      UploadProgress.model_validate,
      self._on_upload_progress,
      qos=topics.QOS_UPLOAD,
    )
    # Slice 5: vision/audio sensing ingest (S5.1). Summary + level feed the live
    # /state values; heartbeat + level feed the health flags; the event topics
    # and the anomaly-recording event feed the in-memory anomaly log. Raw
    # subscribes (read defensively into the tracker); the tracker validates shape.
    self._mqtt.subscribe(
      topics.VISION_SUMMARY, self._on_vision_summary, qos=topics.QOS_VISION_SUMMARY
    )
    self._mqtt.subscribe(
      topics.VISION_HEARTBEAT, self._on_vision_heartbeat, qos=topics.QOS_VISION_HEARTBEAT
    )
    self._mqtt.subscribe(topics.AUDIO_LEVEL, self._on_audio_level, qos=topics.QOS_AUDIO_LEVEL)
    for topic, etype in (
      (topics.VISION_EVENT_SAFE_BOX, AnomalyType.safe_box_violation),
      (topics.VISION_EVENT_SAFE_BOX_CLEARED, AnomalyType.safe_box_cleared),
      (topics.VISION_EVENT_REST, AnomalyType.rest_violation),
      (topics.VISION_EVENT_REST_CLEARED, AnomalyType.rest_violation_cleared),
      (topics.AUDIO_EVENT_SPIKE, AnomalyType.audio_spike),
      (topics.AUDIO_EVENT_SPIKE_CLEARED, AnomalyType.audio_spike_cleared),
    ):
      self._mqtt.subscribe(
        topic,
        lambda _t, payload, _et=etype: self._on_anomaly_event(_et, payload),
        qos=topics.QOS_VISION_EVENT,
      )
    self._mqtt.subscribe(
      topics.EVENT_ANOMALY_RECORDING,
      self._on_anomaly_recording,
      qos=topics.QOS_ANOMALY_RECORDING,
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

  def _on_uploader_heartbeat(self, _topic: str, _payload: dict) -> None:
    if self.on_uploader_heartbeat is not None:
      self.on_uploader_heartbeat()

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

  # --- Slice 5: vision/audio sensing ingest (S5.1) ---------------------------

  def _on_vision_summary(self, _topic: str, payload: dict) -> None:
    if self.anomalies is not None:
      self.anomalies.note_vision_summary(payload)

  def _on_vision_heartbeat(self, _topic: str, payload: dict) -> None:
    if self.anomalies is not None:
      self.anomalies.note_vision_heartbeat(payload)

  def _on_audio_level(self, _topic: str, payload: dict) -> None:
    if self.anomalies is not None:
      self.anomalies.note_audio_level(payload)

  def _on_anomaly_event(self, event_type: AnomalyType, payload: dict) -> None:
    if self.anomalies is not None:
      self.anomalies.note_event(event_type, payload)

  def _on_anomaly_recording(self, _topic: str, payload: dict) -> None:
    if self.anomalies is not None:
      self.anomalies.note_anomaly_recording(payload)

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
  relay.on_uploader_heartbeat = ctl.note_uploader_heartbeat
  # Slice 5: the relay feeds the control plane's sensing tracker from MQTT; the
  # app reads it for /state (via ctl.state()) and /anomalies.
  relay.anomalies = ctl.anomalies

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
  rec_proxy = RecordingProxy(
    storage.explicit_dir,
    anomaly_dir=storage.anomaly_dir,
  )

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

  # Diagnostic payload for the cluster-side connection-check (bandwidth probe,
  # architecture §4.3-adjacent operator tooling): the webui backend GETs this
  # with a chosen size and times the download to estimate throughput over the
  # tailnet control-plane path. Bounded well below the cellular link's
  # plausible per-request budget so the probe itself cannot be turned into an
  # amplification vector; size is clamped, never trusted verbatim.
  DIAG_PAYLOAD_MAX_BYTES = 8 * 1024 * 1024

  @app.get("/diag/payload")
  def diag_payload(size: int = 1_000_000) -> Response:
    size = max(0, min(size, DIAG_PAYLOAD_MAX_BYTES))
    return Response(content=b"\x00" * size, media_type="application/octet-stream")

  @app.get("/state/live")
  def state_live() -> dict:
    # Observability read path for the integration harness: the cached
    # retained arm/video/state/live plus the ordered state history.
    return relay.live_state()

  @app.post("/control/live/activate")
  def live_activate() -> dict:
    # Parameterless (architecture §2.2/§2.9): no mode, no path. Any legacy
    # JSON body a caller still sends ({"path": ...}, {"mode": ...}, the old
    # raw-query form) is accepted and IGNORED — no body parameter is declared,
    # so FastAPI never parses or validates it.
    cmd = LiveActivate()
    log.info("relay activate")
    relay.publish_activate(cmd)
    return {"relayed": "activate"}

  @app.post("/control/live/deactivate")
  def live_deactivate() -> dict:
    # Parameterless; legacy bodies accepted and ignored (see activate).
    cmd = LiveDeactivate()
    log.info("relay deactivate")
    relay.publish_deactivate(cmd)
    return {"relayed": "deactivate"}

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
    # it. 200 means "command published", not "recording active". Recording
    # resolution is a deploy setting resolved on video (recordings reuse the ring
    # encode), so there is no mode here — just look-back + upload-on-done.
    recording_id = str(uuid.uuid4())
    cmd = RecordingStartCmd(
      recording_id=recording_id,
      lookback_seconds=body.lookback_seconds,
      upload_when_done=body.upload_when_done,
      operator=body.operator,
    )
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
    # The recording may be explicit or anomaly (S5.4); accept either.
    meta = storage.read_metadata(rid)
    if meta is None and storage.read_anomaly_metadata(rid) is None:
      raise HTTPException(status_code=404, detail=f"unknown recording: {rid}")
    active = ctl.state().recording.active_recording_id
    if active == rid:
      raise HTTPException(status_code=409, detail="recording is still active")
    _log_control("recording/flag-upload", _caller(request), "published", time.monotonic())
    relay.publish_recording_flag_upload(RecordingFlagUploadCmd(recording_id=rid))
    return {"relayed": "recording/flag-upload", "recording_id": rid}

  @app.get("/recordings")
  def list_recordings() -> list[dict]:
    # Enumerate finalised explicit AND anomaly recordings on the Pi, newest
    # first, merged with a `type` discriminator (S4.4 + S5.4). The metadata
    # schemas are compatible on the listed fields; `type` distinguishes them.
    rows: list[dict] = []
    for meta in storage.list_recordings():
      rows.append({**_meta_to_list_item(meta).model_dump(mode="json"), "type": "explicit"})
    for ameta in storage.list_anomaly_recordings():
      rows.append({**_anomaly_meta_to_list_item(ameta).model_dump(mode="json"), "type": "anomaly"})
    rows.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return rows

  @app.get("/anomalies")
  def list_anomalies() -> list[dict]:
    # S5.3: the full in-memory anomaly log, newest first, each entry carrying its
    # original event payload + the linked anomaly-recording id (if any). Bounded
    # by the tracker's log cap; in-memory (a supervisor restart loses it).
    return [e.model_dump(mode="json") for e in ctl.anomalies.full_log()]

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
    # Local playback proxy for an unflagged/not-yet-uploaded recording (B4.3).
    return rec_proxy.recording_playlist(recording_id)

  @app.get("/recordings/{recording_id}/{segment}")
  def recording_segment(recording_id: str, segment: str):  # noqa: ANN202
    return rec_proxy.recording_segment(recording_id, segment)

  # No /ring/* routes: the ring buffer is never served for viewing (it lives on
  # the Pi behind the cellular link). It only backs recording look-back.

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


def _anomaly_meta_to_list_item(meta: AnomalyRecordingMetadata) -> RecordingListItem:
  # S5.4: an anomaly recording projected onto the same list-item shape as an
  # explicit one (the `type` field, added at the call site, discriminates).
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
