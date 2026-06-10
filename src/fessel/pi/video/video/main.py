"""video process entrypoint.

Wires together:
  - capability probing -> retained publish on arm/video/capabilities (V1.3)
  - MQTT command handler for activate/deactivate (V1.5)
  - the live state machine (V1.4), which owns the live GStreamer pipeline
  - the always-on ring buffer (V4.1) and the explicit-recording state
    machine (V4.3), which own the Slice-4 HLS capture branches
  - recording command handlers (V4.5) + flag-for-upload marker (V4.6)
  - heartbeat (Slice 0)

Config — the `video:` subtree of the single fessel.yaml (Requirements §6),
plus the shared `mqtt`/`storage` sections, flattened by load_component_config
into the shape below:
  mqtt: {host, port}        # shared
  storage: {ssd_path}       # shared; USB SSD mount, default /mnt/ssd
  srt:  {host, port}        # mediamtx Tailscale magicDNS name + 8890
  camera: {device, use_test_source}
  live: {connect_timeout_s}
  ring:      {bitrate_bps, segment_seconds, playlist_length, max_files,
              resolution, fps}
  recording: {bitrate_bps, segment_seconds, start_timeout_s}
"""

from __future__ import annotations

import logging
import os
import re
import signal
import threading
import time

from fessel_schemas import (
  AnomalyRecordingEvent,
  AnomalyType,
  Capabilities,
  LiveActivate,
  LiveDeactivate,
  LiveStateValue,
  ModeTriplet,
  RecordingFlagUploadCmd,
  RecordingStartCmd,
  RecordingState,
  RecordingStateValue,
  UploadStateValue,
  VisionHeartbeat,
)
from fessel_shared import ConfigReloader, MqttClient, Storage, load_component_config
from fessel_shared import topics

from .analysis import AnalysisCoordinator
from .anomaly_recording import AnomalyRecorder, AnomalyWindow
from .audio import AudioAnalyzer, audio_config_from_dict
from .capabilities import probe
from .gst_capture import GstCapturePipeline
from .gst_pipeline_handle import GstLivePipeline
from .gst_recording_handle import GstRecordingPipeline
from .recording_state_machine import RecordingStateMachine
from .state_machine import LiveStateMachine
from .vision import VisionAnalyzer, vision_config_from_dict

log = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 5.0
# Vision thread heartbeat cadence (V5.1): more frequent than the process
# heartbeat so supervisor notices a wedged vision thread within a couple seconds.
VISION_HEARTBEAT_INTERVAL_S = 2.0
# Single Pi config (Requirements §6); this process reads its `video:` subtree
# plus the shared `mqtt`/`storage` sections. FESSEL_CONFIG overrides the path.
CONFIG_PATH = os.environ.get("FESSEL_CONFIG", "/etc/fessel/fessel.yaml")

# Ring-buffer defaults (V4.1) — review-quality, short window. Real values come
# from the `video.ring` config; these are documented starting points, not
# frozen-in-code.
DEFAULT_RING = {
  "bitrate_bps": 2_000_000,
  "segment_seconds": 2,
  "playlist_length": 150,  # ~5 min at 2s segments
  "max_files": 160,  # slightly > playlist_length so rotated segments are deleted
  "resolution": "1280x720",
  "fps": 30,
}
# Explicit-recording defaults (V4.2) — retention-quality, finite.
DEFAULT_RECORDING = {
  "bitrate_bps": 8_000_000,
  "segment_seconds": 2,
  "start_timeout_s": 15.0,
}
# Anomaly-recording window defaults (V5.7) — bracket the moment.
DEFAULT_ANOMALY = {
  "before_seconds": 30.0,
  "after_seconds": 60.0,
  "auto_upload": False,
}


class VideoApp:
  def __init__(self, config: dict) -> None:
    self._cfg = config
    mqtt_cfg = config.get("mqtt", {})
    self._mqtt = MqttClient(
      client_id="video",
      host=mqtt_cfg.get("host", "127.0.0.1"),
      port=mqtt_cfg.get("port", 1883),
    )

    srt_cfg = config.get("srt", {})
    cam_cfg = config.get("camera", {})
    live_cfg = config.get("live", {})
    self._srt_host = srt_cfg.get("host", "mediamtx-srt")
    self._srt_port = srt_cfg.get("port", 8890)
    self._device = cam_cfg.get("device", "/dev/video0")
    self._use_test_source = bool(cam_cfg.get("use_test_source", False))
    # Test-only seam: x264enc software encoder. Off by default; only the
    # integration-test container sets this. Production stays fail-loud
    # hardware-only.
    self._allow_software_encoder = bool(cam_cfg.get("allow_software_encoder", False))

    self._sm = LiveStateMachine(
      pipeline_factory=self._make_pipeline,
      publish_state=self._publish_live_state,
      connect_timeout_s=float(live_cfg.get("connect_timeout_s", 15.0)),
    )
    self._stop = threading.Event()

    # --- Slice 4: storage, ring buffer, recording state machine --------------
    storage_cfg = config.get("storage", {})
    self._storage = Storage(storage_cfg.get("ssd_path", "/mnt/ssd"))
    self._ring_cfg = {**DEFAULT_RING, **(config.get("ring") or {})}
    self._rec_cfg = {**DEFAULT_RECORDING, **(config.get("recording") or {})}
    # The single always-on capture pipeline (§2.2): one camera/mic open, fanned
    # out via tee to the ring + vision + audio-level branches, with live +
    # recording added on demand. Built + started in run().
    self._capture: GstCapturePipeline | None = None
    self._audio_device = (config.get("audio") or {}).get("device", "default")
    # Boot snapshots of the restart-only sections, so a SIGHUP reload (§2.13)
    # can warn about a change to something it cannot hot-apply.
    self._boot_srt = srt_cfg or None
    self._boot_camera = cam_cfg or None
    self._boot_mqtt = mqtt_cfg or None
    self._boot_storage = storage_cfg or None

    self._rec_sm = RecordingStateMachine(
      pipeline_factory=self._make_recording_pipeline,
      publish_state=self._publish_recording_state,
      finalise=self._storage.write_metadata,
      start_timeout_s=float(self._rec_cfg["start_timeout_s"]),
      count_segments=self._storage.count_segments,
    )

    # --- Slice 5: vision + audio analysis ------------------------------------
    self._vision_cfg = config.get("vision") or {}
    self._audio_cfg = config.get("audio") or {}
    self._anomaly_cfg = {**DEFAULT_ANOMALY, **(config.get("anomaly") or {})}
    self._vision_analyzer = VisionAnalyzer(vision_config_from_dict(self._vision_cfg))
    self._audio_analyzer = AudioAnalyzer(audio_config_from_dict(self._audio_cfg))
    self._recorder = AnomalyRecorder(
      ring_dir=self._storage.ring_dir,
      anomaly_dir=self._storage.anomaly_dir,
      forward_capture_factory=self._make_forward_capture,
      publish_event=self._publish_anomaly_recording,
      window=AnomalyWindow(
        before_seconds=float(self._anomaly_cfg["before_seconds"]),
        after_seconds=float(self._anomaly_cfg["after_seconds"]),
      ),
      mode=self._ring_mode(),
      count_segments=self._storage.count_anomaly_segments,
      auto_upload=bool(self._anomaly_cfg["auto_upload"]),
      flag_for_upload=self._flag_anomaly_for_upload,
    )
    self._analysis = AnalysisCoordinator(
      vision=self._vision_analyzer,
      audio=self._audio_analyzer,
      recorder=self._recorder,
      publish=self._publish,
      summary_hz=float(self._vision_cfg.get("publish_summary_hz", 1.0)),
      level_hz=float(self._audio_cfg.get("publish_level_hz", 2.0)),
    )
    self._gst_vision = None  # GstVisionPipeline (best-effort spawn in run())
    self._gst_audio = None  # GstAudioPipeline

  def _make_pipeline(self, mode: ModeTriplet, path: str) -> GstLivePipeline:
    # The live branch attaches to the shared capture pipeline's tee_v (§2.2),
    # not a standalone pipeline. The capture pipeline must be up.
    assert self._capture is not None, "capture pipeline not started"
    return GstLivePipeline(
      sm=self._sm,
      capture=self._capture,
      mode=mode,
      path=path,
      srt_host=self._srt_host,
      srt_port=self._srt_port,
      allow_software_encoder=self._allow_software_encoder,
    )

  def _publish_live_state(
    self, state: LiveStateValue, path: str | None, mode: ModeTriplet | None
  ) -> None:
    from fessel_schemas import LiveState

    payload = LiveState(state=state, path=path, mode=mode)
    self._mqtt.publish(
      topics.STATE_LIVE,
      payload.model_dump(),
      qos=topics.QOS_STATE,
      retain=topics.RETAIN_STATE,
    )

  def _on_activate(self, _topic: str, cmd: LiveActivate) -> None:
    # Validated by subscribe_model; a malformed payload was already dropped.
    log.info("activate path=%s mode=%s", cmd.path, cmd.mode)
    self._sm.request_activate(cmd.mode, cmd.path)

  def _on_deactivate(self, _topic: str, cmd: LiveDeactivate) -> None:
    log.info("deactivate path=%s", cmd.path)
    self._sm.request_deactivate(cmd.path)

  # --- Slice 4: ring buffer (V4.1) -------------------------------------------

  # --- SIGHUP reload (§2.13): ring/recording *values*, next-spawn semantics --

  @staticmethod
  def validate_config(cfg: dict) -> None:
    """Reject a video config whose reloadable encoder values are malformed.
    Raises so the reloader keeps the running config. Resolution must stay
    "<W>x<H>"; numeric tunables must be positive."""
    ring = cfg.get("ring", {})
    if "resolution" in ring:
      res = ring["resolution"]
      if not (isinstance(res, str) and re.fullmatch(r"\d+x\d+", res)):
        raise ValueError('ring.resolution must be "<W>x<H>"')
    for section in ("ring", "recording"):
      sec = cfg.get(section, {})
      for key in ("fps", "bitrate_bps", "segment_seconds", "playlist_length", "max_files"):
        if key in sec and not (isinstance(sec[key], int) and sec[key] > 0):
          raise ValueError(f"{section}.{key} must be a positive int")
      if "start_timeout_s" in sec and not (
        isinstance(sec["start_timeout_s"], (int, float)) and sec["start_timeout_s"] > 0
      ):
        raise ValueError(f"{section}.start_timeout_s must be a positive number")
    # Slice 5: vision/audio tunables. A malformed safe_box must be rejected
    # (it gates the safe-box detector); vision_config_from_dict raises on a bad
    # rectangle via the SafeBox pydantic model.
    vis = cfg.get("vision") or {}
    if isinstance(vis, dict) and isinstance(vis.get("safe_box"), dict):
      vision_config_from_dict(vis)  # raises on a malformed safe_box

  def apply_config(self, cfg: dict) -> None:
    """Apply the hot-reloadable subset (§2.13 conservative scope): the ring and
    recording encoder *values*. Next-spawn semantics, NOT a live encoder
    hot-swap — a new `recording.*` takes effect on the next recording, a new
    `ring.*` on the next ring (re)spawn (e.g. after a camera blip). The running
    ring is deliberately NOT torn down to apply new params (that would punch a
    gap in the always-on ring and churn the reserved-core pipeline). srt /
    camera / mqtt / storage are restart-only; a change is logged."""
    self._ring_cfg = {**DEFAULT_RING, **(cfg.get("ring") or {})}
    self._rec_cfg = {**DEFAULT_RECORDING, **(cfg.get("recording") or {})}
    self._rec_sm.set_start_timeout(float(self._rec_cfg["start_timeout_s"]))
    # Slice 5: vision/audio detector tunables hot-apply (thresholds, debounce,
    # safe_box, EMA, spike params) — the running analysis threads pick them up on
    # the next frame/level. The background-subtractor algorithm + the appsink
    # caps are restart-only (they're baked into the spawned pipeline), like the
    # encoder; a change there is logged, not applied.
    self._vision_cfg = cfg.get("vision") or {}
    self._audio_cfg = cfg.get("audio") or {}
    self._anomaly_cfg = {**DEFAULT_ANOMALY, **(cfg.get("anomaly") or {})}
    self._vision_analyzer.update_config(vision_config_from_dict(self._vision_cfg))
    self._audio_analyzer.update_config(audio_config_from_dict(self._audio_cfg))
    self._analysis.update_rates(
      float(self._vision_cfg.get("publish_summary_hz", 1.0)),
      float(self._audio_cfg.get("publish_level_hz", 2.0)),
    )
    self._recorder.update_window(
      AnomalyWindow(
        before_seconds=float(self._anomaly_cfg["before_seconds"]),
        after_seconds=float(self._anomaly_cfg["after_seconds"]),
      )
    )
    self._recorder.update_auto_upload(bool(self._anomaly_cfg["auto_upload"]))
    if cfg.get("srt") != self._boot_srt or cfg.get("camera") != self._boot_camera:
      log.warning(
        '{"event":"config_reload","note":"srt/camera changed; restart required to apply"}'
      )
    if cfg.get("mqtt") != self._boot_mqtt or cfg.get("storage") != self._boot_storage:
      log.warning(
        '{"event":"config_reload","note":"mqtt/storage changed; restart required to apply"}'
      )
    log.info('{"event":"config_reload","note":"ring/recording values applied on next spawn"}')

  def _ring_mode(self) -> ModeTriplet:
    """The ring's camera operating point — its own resolution/fps/bitrate
    profile, independent of any live mode (the ring is always-on)."""
    return ModeTriplet(
      resolution=str(self._ring_cfg["resolution"]),
      fps=int(self._ring_cfg["fps"]),
      bitrate_bps=int(self._ring_cfg["bitrate_bps"]),
    )

  def _start_capture(self) -> None:
    """Build + start the single always-on capture pipeline (§2.2): one camera
    open fanned out via tee to the ring (V4.1), vision (V5.1), and audio-level
    (V5.6) branches. The live + recording branches attach to its tee on demand.

    NOT best-effort like the old standalone ring: this pipeline IS the camera,
    so if it fails the whole video plane is down — fail loud so the operator
    notices (a crashlooping video process is a clear signal), rather than
    silently running with no capture."""
    audio_level_hz = float(self._audio_cfg.get("publish_level_hz", 2.0))
    self._capture = GstCapturePipeline(
      mode=self._ring_mode(),
      ring_dir=str(self._storage.ring_dir),
      ring_bitrate_bps=int(self._ring_cfg["bitrate_bps"]),
      ring_segment_seconds=int(self._ring_cfg["segment_seconds"]),
      ring_playlist_length=int(self._ring_cfg["playlist_length"]),
      ring_max_files=int(self._ring_cfg["max_files"]),
      rec_staging_dir=str(self._storage.rec_staging_dir),
      device=self._device,
      audio_device=str(self._audio_device or "default"),
      use_test_source=self._use_test_source,
      allow_software_encoder=self._allow_software_encoder,
      audio_level_interval_ns=int(1e9 / max(0.1, audio_level_hz)),
    )
    self._capture.start()
    log.info("capture pipeline started (ring at %s)", self._storage.ring_dir)

  # --- Slice 5: vision + audio analysis (V5.1–V5.7) --------------------------

  def _publish(self, topic: str, payload: dict, qos: int, retain: bool) -> None:
    """Generic MQTT publish the analysis coordinator + recorder share."""
    self._mqtt.publish(topic, payload, qos=qos, retain=retain)

  def _make_forward_capture(self, rec_dir, after_seconds, on_closed):  # noqa: ANN001, ANN202
    """Factory for the anomaly forward-capture branch (V5.7). The forward
    capture is a recording branch on the shared capture tee (§2.2) — it needs
    camera frames, which only the shared pipeline has. Lazily imports the
    GStreamer handle so the analysis core stays testable without gi."""
    from .gst_anomaly_handle import GstAnomalyForwardCapture

    return GstAnomalyForwardCapture(
      capture=self._capture,
      rec_dir=rec_dir,
      after_seconds=after_seconds,
      on_closed=on_closed,
      mode=self._ring_mode(),
      bitrate_bps=int(self._rec_cfg["bitrate_bps"]),
      segment_seconds=int(self._rec_cfg["segment_seconds"]),
      allow_software_encoder=self._allow_software_encoder,
    )

  def _publish_anomaly_recording(
    self, anomaly_recording_id: str, anomaly_event_type: AnomalyType, trigger_ts: str
  ) -> None:
    """Announce a finalised anomaly recording (V5.7) so supervisor + dashboard
    can link the anomaly to its evidence."""
    ev = AnomalyRecordingEvent(
      anomaly_recording_id=anomaly_recording_id,
      anomaly_event_type=anomaly_event_type,
      trigger_ts=trigger_ts,
    )
    self._mqtt.publish(
      topics.EVENT_ANOMALY_RECORDING,
      ev.model_dump(),
      qos=topics.QOS_ANOMALY_RECORDING,
      retain=False,
    )

  def _flag_anomaly_for_upload(self, anomaly_id: str) -> None:
    """Auto-flag an anomaly recording for upload (V5.7 optional knob). Updates
    its metadata.json in place and drops the upload-queue marker the uploader
    watches — same mechanism as explicit flag-for-upload (V4.5/V4.6)."""
    meta = self._storage.read_anomaly_metadata(anomaly_id)
    if meta is None:
      log.warning("auto-flag for unknown anomaly recording %s — ignoring", anomaly_id)
      return
    meta.flagged_for_upload = True
    meta.upload_state = UploadStateValue.queued
    self._storage.write_anomaly_metadata(meta)
    self._storage.create_upload_marker(anomaly_id)
    log.info("auto-flagged anomaly recording %s for upload", anomaly_id)

  def _start_vision_audio(self) -> None:
    """Spawn the always-on vision + audio analysis branches (V5.1/V5.6).
    Best-effort, like the ring: an analysis failure is degraded, not fatal —
    supervision (live, recording, ring) still works, and the missing heartbeat
    tells supervisor vision is down."""
    try:
      from .gst_vision_handle import GstVisionPipeline

      bg = self._vision_cfg.get("background_subtractor") or {}
      self._gst_vision = GstVisionPipeline(
        analyzer=self._vision_analyzer,
        capture=self._capture,
        ring_dir=str(self._storage.ring_dir),
        on_summary=self._analysis.on_vision_summary,
        on_events=self._analysis.on_vision_events,
        bg_algorithm=str(bg.get("algorithm", "MOG2")),
        bg_history=int(bg.get("history", 500)),
        bg_var_threshold=float(bg.get("var_threshold", 16)),
        morph_kernel=int((self._vision_cfg.get("morphology") or {}).get("kernel", 3)),
      )
      self._gst_vision.start()
      log.info("vision analysis started")
    except Exception:  # noqa: BLE001
      log.exception("vision analysis failed to start (continuing without it)")
      self._gst_vision = None

    try:
      from .gst_audio_handle import GstAudioPipeline

      self._gst_audio = GstAudioPipeline(
        analyzer=self._audio_analyzer,
        capture=self._capture,
        on_level=self._analysis.on_audio_level,
        on_events=self._analysis.on_audio_events,
      )
      self._gst_audio.start()
      log.info("audio analysis started")
    except Exception:  # noqa: BLE001
      log.exception("audio analysis failed to start (continuing without it)")
      self._gst_audio = None

  def _publish_vision_heartbeat(self) -> None:
    """Publish the vision thread heartbeat (V5.1). Absence -> supervisor marks
    vision degraded. Built from the GstVision handle's live counters; if vision
    never started, publish a heartbeat with zero counters and no last_frame so
    supervisor still sees the thread is not delivering frames."""
    if self._gst_vision is not None:
      hb = self._gst_vision.heartbeat()
    else:
      hb = {"last_frame_at": None, "frames_processed": 0, "frames_dropped": 0}
    payload = VisionHeartbeat(ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **hb)
    self._mqtt.publish(
      topics.VISION_HEARTBEAT, payload.model_dump(), qos=topics.QOS_VISION_HEARTBEAT
    )

  # --- Slice 4: explicit recording (V4.2/V4.3/V4.5) --------------------------

  def _make_recording_pipeline(self, recording_id: str, mode: ModeTriplet) -> GstRecordingPipeline:
    # An explicit recording copies the relevant window out of the always-on ring
    # (§2.2) — it is NOT a pipeline branch (hlssink2 can't be added live, and an
    # idle cold-built hlssink2 would block preroll). The capture (hence the ring)
    # must be up. segment_seconds matches the ring's so the assembled playlist's
    # EXTINF is right (recording reuses the ring's encode).
    assert self._capture is not None, "capture pipeline not started"
    recording_dir = str(self._storage.recording_dir(recording_id))
    self._storage.recording_dir(recording_id).mkdir(parents=True, exist_ok=True)
    return GstRecordingPipeline(
      sm=self._rec_sm,
      capture=self._capture,
      ring_dir=str(self._storage.ring_dir),
      mode=mode,
      recording_dir=recording_dir,
      segment_seconds=int(self._ring_cfg["segment_seconds"]),
    )

  def _publish_recording_state(
    self, state: RecordingStateValue, recording_id: str | None, started_at: str | None
  ) -> None:
    payload = RecordingState(state=state, active_recording_id=recording_id, started_at=started_at)
    self._mqtt.publish(
      topics.STATE_RECORDING,
      payload.model_dump(),
      qos=topics.QOS_STATE,
      retain=topics.RETAIN_RECORDING,
    )

  def _on_recording_start(self, _topic: str, cmd: RecordingStartCmd) -> None:
    log.info("recording start id=%s mode=%s op=%s", cmd.recording_id, cmd.mode, cmd.operator)
    self._rec_sm.request_start(cmd.recording_id, cmd.mode, cmd.operator)

  def _on_recording_stop(self, _topic: str, _payload: dict) -> None:
    log.info("recording stop")
    self._rec_sm.request_stop()

  def _on_recording_flag_upload(self, _topic: str, cmd: RecordingFlagUploadCmd) -> None:
    # Update metadata.json in place (flagged_for_upload=true, upload_state=queued)
    # AND drop the upload-queue marker the uploader watches (V4.5/V4.6). If the
    # recording has no metadata (never finalised / unknown id), do nothing but
    # log — there is nothing meaningful to upload.
    rid = cmd.recording_id
    meta = self._storage.read_metadata(rid)
    if meta is not None:
      meta.flagged_for_upload = True
      meta.upload_state = UploadStateValue.queued
      self._storage.write_metadata(meta)
      self._storage.create_upload_marker(rid)
      log.info("flagged recording %s for upload (marker written)", rid)
      return
    # Slice 5: the recording may be an anomaly recording (same flag mechanism).
    ameta = self._storage.read_anomaly_metadata(rid)
    if ameta is not None:
      ameta.flagged_for_upload = True
      ameta.upload_state = UploadStateValue.queued
      self._storage.write_anomaly_metadata(ameta)
      self._storage.create_upload_marker(rid)
      log.info("flagged anomaly recording %s for upload (marker written)", rid)
      return
    log.warning("flag-upload for unknown/partial recording %s — ignoring", rid)

  def _on_jetson_event(self, topic: str, _payload: dict) -> None:
    # arm/jetson/events/<type>; the trailing path segment is the event type
    # (the Jetson publishes per-type topics, §2.7). Forward to the analyzer's
    # rest logic (V5.4). Unknown types are ignored there.
    event_type = topic.rsplit("/", 1)[-1]
    self._analysis.note_jetson_event(event_type)

  def _on_supervisor_state(self, _topic: str, payload: dict) -> None:
    # arm/supervisor/state carries {"safety_state": "..."} (supervisor publishes
    # the StateResponse-shaped value; we read the field defensively). Drives the
    # operator-driven rest period (V5.4).
    state = payload.get("safety_state") or payload.get("state")
    self._analysis.note_supervisor_state(state if isinstance(state, str) else None)

  def _publish_capabilities(self) -> None:
    modes = probe(self._device if not self._use_test_source else "")
    caps = Capabilities(modes=modes)
    self._mqtt.publish(
      topics.CAPABILITIES,
      caps.model_dump(),
      qos=topics.QOS_CAPABILITIES,
      retain=topics.RETAIN_CAPABILITIES,
    )
    log.info("published %d capability modes", len(modes))

  def run(self) -> None:
    # Make sure the SSD layout exists before any branch writes to it (X4.2).
    self._storage.ensure_layout()

    self._mqtt.connect()
    # Commands are schema-validated at the transport: a malformed activate /
    # deactivate is logged and dropped before it reaches the handler.
    self._mqtt.subscribe_model(
      topics.CMD_LIVE_ACTIVATE, LiveActivate.model_validate, self._on_activate, qos=topics.QOS_CMD
    )
    self._mqtt.subscribe_model(
      topics.CMD_LIVE_DEACTIVATE,
      LiveDeactivate.model_validate,
      self._on_deactivate,
      qos=topics.QOS_CMD,
    )
    # Slice 4 recording command surface (V4.5). start/flag-upload carry a
    # validated model; stop is an empty payload (raw subscribe).
    self._mqtt.subscribe_model(
      topics.CMD_RECORDING_START,
      RecordingStartCmd.model_validate,
      self._on_recording_start,
      qos=topics.QOS_CMD,
    )
    self._mqtt.subscribe(topics.CMD_RECORDING_STOP, self._on_recording_stop, qos=topics.QOS_CMD)
    self._mqtt.subscribe_model(
      topics.CMD_RECORDING_FLAG_UPLOAD,
      RecordingFlagUploadCmd.model_validate,
      self._on_recording_flag_upload,
      qos=topics.QOS_CMD,
    )
    # Slice 5 rest-period inputs (V5.4): Jetson episode boundaries and supervisor
    # safety state both feed the vision analyzer's in_rest flag. Raw subscribes —
    # we read defensively (the Jetson's event types and the supervisor state
    # shape may evolve; the subscription is stable).
    self._mqtt.subscribe(
      f"{topics.SITE_PREFIX}/jetson/events/+", self._on_jetson_event, qos=topics.QOS_STATE
    )
    self._mqtt.subscribe(topics.STATE_SUPERVISOR, self._on_supervisor_state, qos=topics.QOS_STATE)
    self._mqtt.loop_start()
    self._sm.start()
    self._rec_sm.start()
    # The capture pipeline (camera open + tee + ring) must be up BEFORE vision/
    # audio attach to its appsink/level elements and before live/recording can
    # add branches onto its tee.
    self._start_capture()
    self._start_vision_audio()
    self._publish_capabilities()

    signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
    signal.signal(signal.SIGINT, lambda *_: self._stop.set())
    # SIGHUP -> reload ring/recording values (§2.13; next-spawn semantics).
    reloader = ConfigReloader(
      path=CONFIG_PATH,
      component="video",
      validate=self.validate_config,
      apply=self.apply_config,
    )
    reloader.install()

    # The vision heartbeat (V5.1) ticks faster than the process heartbeat, so
    # the loop waits on the vision cadence and publishes the process heartbeat
    # every Nth tick.
    last_proc_hb = 0.0
    while not self._stop.is_set():
      now = time.monotonic()
      if now - last_proc_hb >= HEARTBEAT_INTERVAL_S:
        self._mqtt.publish(
          topics.HEARTBEAT, {"from": "video", "ts": time.time()}, qos=topics.QOS_HEARTBEAT
        )
        last_proc_hb = now
      self._publish_vision_heartbeat()
      self._stop.wait(VISION_HEARTBEAT_INTERVAL_S)

    log.info("video shutting down")
    reloader.close()
    self._sm.shutdown()
    self._rec_sm.shutdown()
    self._recorder.shutdown()
    # Detach the analysis consumers, then tear down the one capture pipeline
    # (camera + tee + ring + appsink/level) last.
    if self._gst_vision is not None:
      self._gst_vision.stop()
    if self._gst_audio is not None:
      self._gst_audio.stop()
    if self._capture is not None:
      self._capture.stop()
    self._mqtt.loop_stop()
    self._mqtt.disconnect()


def main() -> None:
  logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
  )
  config = load_component_config(CONFIG_PATH, "video")
  VideoApp(config).run()


if __name__ == "__main__":
  main()
