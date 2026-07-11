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
  whip: {endpoint}          # webui relay's WHIP ingest URL (…/whip/ingest)
  camera: {device, use_test_source}
  live: {resolution, fps, bitrate_bps,   # fixed deploy-time live profile (§2.2)
         connect_timeout_s}
  recording: {resolution, fps, bitrate_bps, segment_seconds, start_timeout_s,
              max_lookback_seconds}   # the always-on ring IS the recording
                                      # buffer (copy-from-ring), so one section
                                      # describes both; window from max_lookback
  snapshot: {ingest_url_base, interval_s, timeout_s, verify_tls}
      # Monitor freeze-frame push (best-effort, like vision/audio): PUTs the
      # latest JPEG to webui's tailnet ingest listener at most once per
      # interval_s. ingest_url_base defaults to the shared webui_base (below)
      # unless snapshot: sets its own (including "" to disable the push).
  webui_base: "http://..."  # shared; also whip.endpoint's base + the
                            # uploader's default ingest_url_base
"""

from __future__ import annotations

import logging
import math
import os
import re
import signal
import threading
import time

from fessel_schemas import (
  AnomalyRecordingEvent,
  AnomalyType,
  CameraState,
  Capabilities,
  LiveActivate,
  LiveDeactivate,
  LiveStateValue,
  ModeTriplet,
  RecordingFlagUploadCmd,
  RecordingMetadata,
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

# Recording defaults (V4.1/V4.2). There is no separate `ring` config: the
# always-on ring IS the recording buffer (a recording is copy-from-ring, §2.2),
# so one section describes both. `resolution`/`fps`/`bitrate_bps` are the camera
# operating point the ring encodes at — a deploy setting, not a per-request
# choice. `segment_seconds` is the ring's segment length (reused as the recording
# playlist's EXTINF). `max_lookback_seconds` is how far a recording may reach
# back into the ring; it sizes the ring window (max_files) and is the bound
# published to the UI. Real values come from `video.recording`; these are
# documented starting points.
DEFAULT_RECORDING = {
  "resolution": "1280x720",
  "fps": 30,
  "bitrate_bps": 2_000_000,
  "segment_seconds": 2,
  "start_timeout_s": 15.0,
  "max_lookback_seconds": 120.0,
}
# Warm live-encoder profile defaults (§2.2) — fixed at deploy time, NOT a
# per-session parameter. Deliberately lower quality than the recording
# encoders: the live view is for situational awareness, not the archived
# artifact. Real values come from `video.live`.
DEFAULT_LIVE = {
  "resolution": "1280x720",
  "fps": 15,
  "bitrate_bps": 1_500_000,
  "connect_timeout_s": 15.0,
}
# Anomaly-recording window defaults (V5.7) — bracket the moment.
DEFAULT_ANOMALY = {
  "before_seconds": 30.0,
  "after_seconds": 60.0,
  "auto_upload": False,
}
# Freeze-frame push defaults (Monitor UX): a low-rate JPEG PUT to webui's
# tailnet ingest listener, so the Monitor page has something to show before
# Stream On without ever costing WebRTC bandwidth. `interval_s` bounds how
# stale the shown frame can be (up to ~30s is the documented tolerance).
# `ingest_url_base` is NOT defaulted here — it falls back to the shared
# `webui_base` (the same tailnet endpoint the uploader PUTs recordings to)
# unless `snapshot:` sets its own (including an explicit "" to disable the
# push entirely).
DEFAULT_SNAPSHOT = {
  "interval_s": 25.0,
  "timeout_s": 10.0,
  "verify_tls": True,
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

    whip_cfg = config.get("whip", {})
    cam_cfg = config.get("camera", {})
    live_cfg = {**DEFAULT_LIVE, **(config.get("live") or {})}
    self._live_cfg = live_cfg
    # The webui relay's WHIP ingest endpoint (full URL ending in /whip/ingest).
    # Deploy-time config; without it the live path cannot work — warn loudly
    # at startup (an activate would then fail into the SM's connect-failure
    # path rather than crashing the whole video plane).
    self._whip_endpoint = str(whip_cfg.get("endpoint", "") or "")
    if not self._whip_endpoint:
      log.warning(
        '{"event":"config","note":"video.whip.endpoint is not set; '
        'live activation will fail until it is configured"}'
      )
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
    self._rec_cfg = {**DEFAULT_RECORDING, **(config.get("recording") or {})}
    # The single always-on capture pipeline (§2.2): one camera/mic open, fanned
    # out via tee to the ring + vision + audio-level branches and the WARM live
    # encoder (tee_live), with the WHIP sender attached on demand. Built +
    # started in run().
    self._capture: GstCapturePipeline | None = None
    self._audio_device = (config.get("audio") or {}).get("device", "default")
    # audio.enabled=false skips the mic chain entirely (deploy-time knob for
    # rigs without a microphone); the capture pipeline ALSO auto-degrades at
    # startup when the device can't be opened.
    self._audio_enabled = bool((config.get("audio") or {}).get("enabled", True))
    # Boot snapshots of the restart-only sections, so a SIGHUP reload (§2.13)
    # can warn about a change to something it cannot hot-apply. The whip
    # endpoint + the live profile are restart-only like srt/camera were: the
    # warm live encoder is baked into the capture pipeline at startup.
    self._boot_whip = whip_cfg or None
    self._boot_live = config.get("live") or None
    self._boot_camera = cam_cfg or None
    self._boot_mqtt = mqtt_cfg or None
    self._boot_storage = storage_cfg or None

    self._rec_sm = RecordingStateMachine(
      pipeline_factory=self._make_recording_pipeline,
      publish_state=self._publish_recording_state,
      finalise=self._finalise_recording,
      mode_provider=self._recording_mode,
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
      mode=self._recording_mode(),
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

    # --- Monitor freeze-frame (best-effort, like vision/audio) ---------------
    snapshot_cfg = config.get("snapshot") or {}
    self._snapshot_cfg = {**DEFAULT_SNAPSHOT, **snapshot_cfg}
    if "ingest_url_base" not in snapshot_cfg:
      self._snapshot_cfg["ingest_url_base"] = config.get("webui_base") or ""
    self._gst_snapshot = None  # GstSnapshotPipeline
    self._snapshot_pusher = None  # SnapshotPusher
    self._last_snapshot_push = 0.0

  def _make_pipeline(self) -> GstLivePipeline:
    # The WHIP sender attaches to the shared capture pipeline's tee_live
    # (§2.2: the live encoder is warm; only the sender is attached on demand).
    # The capture pipeline must be up. Parameterless: the live profile is a
    # deploy-time config setting baked into the capture pipeline at startup.
    assert self._capture is not None, "capture pipeline not started"
    return GstLivePipeline(
      sm=self._sm,
      capture=self._capture,
      whip_endpoint=self._whip_endpoint,
    )

  def _publish_live_state(self, state: LiveStateValue) -> None:
    from fessel_schemas import LiveState

    payload = LiveState(state=state)
    self._mqtt.publish(
      topics.STATE_LIVE,
      payload.model_dump(),
      qos=topics.QOS_STATE,
      retain=topics.RETAIN_STATE,
    )

  def _on_activate(self, _topic: str, _cmd: LiveActivate) -> None:
    # Validated by subscribe_model; a malformed payload was already dropped.
    # Parameterless (§2.2): any legacy fields on the wire are ignored.
    log.info("activate")
    self._sm.request_activate()

  def _on_deactivate(self, _topic: str, _cmd: LiveDeactivate) -> None:
    log.info("deactivate")
    self._sm.request_deactivate()

  # --- Slice 4: ring buffer (V4.1) -------------------------------------------

  # --- SIGHUP reload (§2.13): ring/recording *values*, next-spawn semantics --

  @staticmethod
  def validate_config(cfg: dict) -> None:
    """Reject a video config whose reloadable encoder values are malformed.
    Raises so the reloader keeps the running config. Resolution must stay
    "<W>x<H>"; numeric tunables must be positive."""
    # One `recording` section (the always-on ring IS the recording buffer):
    # resolution is the camera operating point the ring encodes at.
    recording = cfg.get("recording", {})
    if "resolution" in recording:
      res = recording["resolution"]
      if not (isinstance(res, str) and re.fullmatch(r"\d+x\d+", res)):
        raise ValueError('recording.resolution must be "<W>x<H>"')
    for key in ("fps", "bitrate_bps", "segment_seconds"):
      if key in recording and not (isinstance(recording[key], int) and recording[key] > 0):
        raise ValueError(f"recording.{key} must be a positive int")
    for key in ("start_timeout_s", "max_lookback_seconds"):
      if key in recording and not (
        isinstance(recording[key], (int, float)) and recording[key] > 0
      ):
        raise ValueError(f"recording.{key} must be a positive number")
    # Slice 5: vision/audio tunables. A malformed safe_box must be rejected
    # (it gates the safe-box detector); vision_config_from_dict raises on a bad
    # rectangle via the SafeBox pydantic model.
    vis = cfg.get("vision") or {}
    if isinstance(vis, dict) and isinstance(vis.get("safe_box"), dict):
      vision_config_from_dict(vis)  # raises on a malformed safe_box

  def apply_config(self, cfg: dict) -> None:
    """Apply the hot-reloadable subset (§2.13 conservative scope): the
    `recording.*` encoder values. Next-spawn semantics, NOT a live hot-swap — a
    new resolution/bitrate/max_lookback takes effect on the next capture
    (re)spawn (e.g. after a camera blip); the start-timeout applies to the next
    recording. The running ring is deliberately NOT torn down to apply new params
    (that would punch a gap in the always-on ring and churn the reserved-core
    pipeline). whip / live / camera / mqtt / storage are restart-only; a change
    is logged."""
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
    if (
      cfg.get("whip") != self._boot_whip
      or cfg.get("live") != self._boot_live
      or cfg.get("camera") != self._boot_camera
    ):
      log.warning(
        '{"event":"config_reload","note":"whip/live/camera changed; restart required to apply"}'
      )
    if cfg.get("mqtt") != self._boot_mqtt or cfg.get("storage") != self._boot_storage:
      log.warning(
        '{"event":"config_reload","note":"mqtt/storage changed; restart required to apply"}'
      )
    log.info('{"event":"config_reload","note":"ring/recording values applied on next spawn"}')

  def _recording_mode(self) -> ModeTriplet:
    """The camera operating point the always-on capture (hence the ring, hence
    recordings) encodes at. Resolution/fps/bitrate all come from
    `video.recording` (a deploy setting — recordings are copy-from-ring, so there
    is no per-recording mode)."""
    return ModeTriplet(
      resolution=str(self._rec_cfg["resolution"]),
      fps=int(self._rec_cfg["fps"]),
      bitrate_bps=int(self._rec_cfg["bitrate_bps"]),
    )

  def _max_lookback_seconds(self) -> float:
    """How far a recording may reach back into the ring (= the ring window).
    Published to the UI in capabilities and used to size the ring's max_files."""
    return float(self._rec_cfg["max_lookback_seconds"])

  def _ring_window_files(self) -> tuple[int, int]:
    """(playlist_length, max_files) sizing the ring to hold max_lookback_seconds
    of footage. max_files keeps a little slack beyond the playlist so a segment
    is only deleted once it has rotated fully out of the look-back window."""
    seg = max(1, int(self._rec_cfg["segment_seconds"]))
    playlist_length = max(1, math.ceil(self._max_lookback_seconds() / seg))
    max_files = playlist_length + 10  # slack: don't delete a segment still in-window
    return playlist_length, max_files

  def _start_capture(self) -> None:
    """Build + start the single always-on capture pipeline (§2.2): one camera
    open fanned out via tee to the ring (V4.1), vision (V5.1), audio-level
    (V5.6), and warm live-encoder branches. The live WHIP sender attaches to
    its tee_live on demand.

    NOT best-effort like the old standalone ring: this pipeline IS the camera,
    so if it fails the whole video plane is down — fail loud so the operator
    notices (a crashlooping video process is a clear signal), rather than
    silently running with no capture."""
    audio_level_hz = float(self._audio_cfg.get("publish_level_hz", 2.0))
    ring_playlist_length, ring_max_files = self._ring_window_files()
    self._capture = GstCapturePipeline(
      mode=self._recording_mode(),
      ring_dir=str(self._storage.ring_dir),
      ring_bitrate_bps=int(self._rec_cfg["bitrate_bps"]),
      ring_segment_seconds=int(self._rec_cfg["segment_seconds"]),
      ring_playlist_length=ring_playlist_length,
      ring_max_files=ring_max_files,
      live_resolution=str(self._live_cfg["resolution"]),
      live_fps=int(self._live_cfg["fps"]),
      live_bitrate_bps=int(self._live_cfg["bitrate_bps"]),
      device=self._device,
      audio_device=str(self._audio_device or "default"),
      use_test_source=self._use_test_source,
      allow_software_encoder=self._allow_software_encoder,
      audio_level_interval_ns=int(1e9 / max(0.1, audio_level_hz)),
      include_audio=self._audio_enabled,
    )
    self._capture.start()
    log.info("capture pipeline started (ring at %s)", self._storage.ring_dir)
    # §7.4: camera presence is a retained state topic. The capture pipeline IS
    # the camera (it owns the device and delivers frames — a videotestsrc dev
    # source counts: frames flow), so pipeline-up is the honest `up` signal.
    # supervisor caches this into /state.camera.up, which the webui health
    # gate consults before answering /whep. A failed start never reaches this
    # line (_start_capture fails loud and the process exits), and supervisor's
    # heartbeat-staleness handling covers a dead video process going forward.
    self._publish_camera_state(up=True)

  def _publish_camera_state(self, up: bool) -> None:
    self._mqtt.publish(
      topics.STATE_CAMERA,
      CameraState(up=up).model_dump(),
      qos=topics.QOS_STATE,
      retain=topics.RETAIN_STATE,
    )

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
      mode=self._recording_mode(),
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

    if self._capture is None or not self._capture.audio_enabled:
      # The capture pipeline was built without the audio chain (no usable mic
      # or audio.enabled=false): there is no level element to attach to.
      log.info("audio analysis disabled (no usable audio device / audio.enabled=false)")
      self._gst_audio = None
      return
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

  def _start_snapshot(self) -> None:
    """Attach the freeze-frame cache to the shared capture pipeline's snapshot
    appsink (Monitor UX). Best-effort, like vision/audio: a failure here must
    never take down live/recording — the Monitor page just falls back to its
    "no snapshot yet" state."""
    try:
      from .gst_snapshot_handle import GstSnapshotPipeline

      self._gst_snapshot = GstSnapshotPipeline(capture=self._capture)
      self._gst_snapshot.start()
      log.info("snapshot capture started")
    except Exception:  # noqa: BLE001
      log.exception("snapshot capture failed to start (continuing without it)")
      self._gst_snapshot = None

    url_base = str(self._snapshot_cfg.get("ingest_url_base") or "")
    if not url_base:
      log.info("video.snapshot.ingest_url_base not set; freeze-frame push disabled")
      self._snapshot_pusher = None
      return
    try:
      from .snapshot_push import SnapshotPusher

      self._snapshot_pusher = SnapshotPusher(
        ingest_url_base=url_base,
        timeout_s=float(self._snapshot_cfg.get("timeout_s", 10.0)),
        verify_tls=bool(self._snapshot_cfg.get("verify_tls", True)),
      )
    except Exception:  # noqa: BLE001
      log.exception("snapshot pusher failed to start (continuing without it)")
      self._snapshot_pusher = None

  def _maybe_push_snapshot(self, now_monotonic: float) -> None:
    """Push the cached freeze-frame at most once per `interval_s` (Monitor UX).
    Runs on the main loop tick, not a separate thread — the push itself is a
    quick best-effort HTTP PUT (SnapshotPusher swallows its own errors)."""
    if self._gst_snapshot is None or self._snapshot_pusher is None:
      return
    interval = float(self._snapshot_cfg.get("interval_s", 10.0))
    if now_monotonic - self._last_snapshot_push < interval:
      return
    jpeg = self._gst_snapshot.latest()
    if jpeg is None:
      return
    self._last_snapshot_push = now_monotonic
    self._snapshot_pusher.push(jpeg)

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

  def _make_recording_pipeline(
    self, recording_id: str, lookback_seconds: float
  ) -> GstRecordingPipeline:
    # An explicit recording copies the relevant window out of the always-on ring
    # (§2.2) — it is NOT a pipeline branch (hlssink2 can't be added live, and an
    # idle cold-built hlssink2 would block preroll). The capture (hence the ring)
    # must be up. segment_seconds matches the ring's so the assembled playlist's
    # EXTINF is right (recording reuses the ring's encode). lookback_seconds
    # reaches the copy window back into the ring (0 = start now); it is clamped
    # by the handle to what the ring actually holds.
    assert self._capture is not None, "capture pipeline not started"
    recording_dir = str(self._storage.recording_dir(recording_id))
    self._storage.recording_dir(recording_id).mkdir(parents=True, exist_ok=True)
    return GstRecordingPipeline(
      sm=self._rec_sm,
      capture=self._capture,
      ring_dir=str(self._storage.ring_dir),
      recording_dir=recording_dir,
      segment_seconds=int(self._rec_cfg["segment_seconds"]),
      lookback_seconds=lookback_seconds,
    )

  def _finalise_recording(self, meta: RecordingMetadata, upload_when_done: bool) -> None:
    # Write metadata.json, then (if the operator asked at start time) flag the
    # finished recording for upload — same mechanism as an after-the-fact
    # flag-for-upload (V4.6): mark metadata queued + drop the uploader's marker.
    if upload_when_done:
      meta.flagged_for_upload = True
      meta.upload_state = UploadStateValue.queued
    self._storage.write_metadata(meta)
    if upload_when_done:
      self._storage.create_upload_marker(meta.id)
      log.info("recording %s finalised and flagged for upload", meta.id)

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
    log.info(
      "recording start id=%s lookback=%.1fs upload=%s op=%s",
      cmd.recording_id,
      cmd.lookback_seconds,
      cmd.upload_when_done,
      cmd.operator,
    )
    self._rec_sm.request_start(
      cmd.recording_id,
      lookback_seconds=cmd.lookback_seconds,
      upload_when_done=cmd.upload_when_done,
      operator=cmd.operator,
    )

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
    # recording_mode is the deploy's configured recording operating point (the
    # UI no longer picks resolution); max_lookback_seconds bounds how far the
    # record dialog can reach back into the ring.
    caps = Capabilities(
      recording_mode=self._recording_mode(),
      max_lookback_seconds=self._max_lookback_seconds(),
    )
    self._mqtt.publish(
      topics.CAPABILITIES,
      caps.model_dump(),
      qos=topics.QOS_CAPABILITIES,
      retain=topics.RETAIN_CAPABILITIES,
    )
    log.info(
      "published capabilities (recording %s, max_lookback %.0fs)",
      self._recording_mode().resolution,
      self._max_lookback_seconds(),
    )

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
    self._start_snapshot()
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
      self._maybe_push_snapshot(now)
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
    if self._gst_snapshot is not None:
      self._gst_snapshot.stop()
    if self._capture is not None:
      self._capture.stop()
      # Clean shutdown: retract the retained camera-up so a stopped video
      # doesn't leave a stale green camera fact. (A crash skips this — then
      # supervisor's heartbeat staleness turns the video fact red instead.)
      self._publish_camera_state(up=False)
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
