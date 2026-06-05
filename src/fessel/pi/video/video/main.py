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
  Capabilities,
  LiveActivate,
  LiveDeactivate,
  LiveStateValue,
  ModeTriplet,
  RecordingFlagUploadCmd,
  RecordingStartCmd,
  RecordingState,
  RecordingStateValue,
)
from fessel_shared import ConfigReloader, MqttClient, Storage, load_component_config
from fessel_shared import topics

from .capabilities import probe
from .gst_pipeline_handle import GstLivePipeline
from .gst_recording_handle import GstRecordingPipeline
from .gst_ring_handle import GstRingPipeline
from .recording_state_machine import RecordingStateMachine
from .state_machine import LiveStateMachine

log = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 5.0
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
    self._ring: GstRingPipeline | None = None
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

  def _make_pipeline(self, mode: ModeTriplet, path: str) -> GstLivePipeline:
    return GstLivePipeline(
      sm=self._sm,
      mode=mode,
      path=path,
      srt_host=self._srt_host,
      srt_port=self._srt_port,
      device=self._device,
      use_test_source=self._use_test_source,
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
    if cfg.get("srt") != self._boot_srt or cfg.get("camera") != self._boot_camera:
      log.warning(
        '{"event":"config_reload","note":"srt/camera changed; restart required to apply"}'
      )
    if cfg.get("mqtt") != self._boot_mqtt or cfg.get("storage") != self._boot_storage:
      log.warning(
        '{"event":"config_reload","note":"mqtt/storage changed; restart required to apply"}'
      )
    log.info(
      '{"event":"config_reload","note":"ring/recording values applied on next spawn"}'
    )

  def _ring_mode(self) -> ModeTriplet:
    """The ring's camera operating point — its own resolution/fps/bitrate
    profile, independent of any live mode (the ring is always-on)."""
    return ModeTriplet(
      resolution=str(self._ring_cfg["resolution"]),
      fps=int(self._ring_cfg["fps"]),
      bitrate_bps=int(self._ring_cfg["bitrate_bps"]),
    )

  def _start_ring(self) -> None:
    """Spawn the always-on ring buffer (V4.1). Best-effort: a ring failure is
    a degraded condition, not fatal to supervision, so it is logged and the
    process continues (recordings + live still work)."""
    try:
      self._ring = GstRingPipeline(
        mode=self._ring_mode(),
        ring_dir=str(self._storage.ring_dir),
        bitrate_bps=int(self._ring_cfg["bitrate_bps"]),
        segment_seconds=int(self._ring_cfg["segment_seconds"]),
        playlist_length=int(self._ring_cfg["playlist_length"]),
        max_files=int(self._ring_cfg["max_files"]),
        device=self._device,
        use_test_source=self._use_test_source,
        allow_software_encoder=self._allow_software_encoder,
      )
      self._ring.start()
      log.info("ring buffer started at %s", self._storage.ring_dir)
    except Exception:  # noqa: BLE001
      log.exception("ring buffer failed to start (continuing without it)")
      self._ring = None

  # --- Slice 4: explicit recording (V4.2/V4.3/V4.5) --------------------------

  def _make_recording_pipeline(self, recording_id: str, mode: ModeTriplet) -> GstRecordingPipeline:
    recording_dir = str(self._storage.recording_dir(recording_id))
    self._storage.recording_dir(recording_id).mkdir(parents=True, exist_ok=True)
    return GstRecordingPipeline(
      sm=self._rec_sm,
      mode=mode,
      recording_dir=recording_dir,
      bitrate_bps=int(self._rec_cfg["bitrate_bps"]),
      segment_seconds=int(self._rec_cfg["segment_seconds"]),
      device=self._device,
      use_test_source=self._use_test_source,
      allow_software_encoder=self._allow_software_encoder,
    )

  def _publish_recording_state(
    self, state: RecordingStateValue, recording_id: str | None, started_at: str | None
  ) -> None:
    payload = RecordingState(
      state=state, active_recording_id=recording_id, started_at=started_at
    )
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
    if meta is None:
      log.warning("flag-upload for unknown/partial recording %s — ignoring", rid)
      return
    from fessel_schemas import UploadStateValue

    meta.flagged_for_upload = True
    meta.upload_state = UploadStateValue.queued
    self._storage.write_metadata(meta)
    self._storage.create_upload_marker(rid)
    log.info("flagged recording %s for upload (marker written)", rid)

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
    self._mqtt.loop_start()
    self._sm.start()
    self._rec_sm.start()
    self._start_ring()
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

    while not self._stop.is_set():
      self._mqtt.publish(
        topics.HEARTBEAT, {"from": "video", "ts": time.time()}, qos=topics.QOS_HEARTBEAT
      )
      self._stop.wait(HEARTBEAT_INTERVAL_S)

    log.info("video shutting down")
    reloader.close()
    self._sm.shutdown()
    self._rec_sm.shutdown()
    if self._ring is not None:
      self._ring.stop()
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
