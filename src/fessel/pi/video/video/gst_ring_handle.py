"""GStreamer-backed always-on ring-buffer pipeline (V4.1).

Unlike the live and recording branches, the ring has NO state machine: it is
always running while the camera is up (V4.1). This handle is a thin wrapper that
spawns the ring pipeline at startup and runs it on its own GLib MainLoop thread.
On error it logs and (best-effort) restarts after a short delay, since the ring
losing its recorder is a degraded-but-recoverable condition, not a fatal one.
"""

from __future__ import annotations

import logging
import threading

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from fessel_schemas import ModeTriplet  # noqa: E402

from .pipeline import build_pipeline, build_ring_launch  # noqa: E402

log = logging.getLogger(__name__)

RING_RESTART_DELAY_S = 5


class GstRingPipeline:
  def __init__(
    self,
    *,
    mode: ModeTriplet,
    ring_dir: str,
    bitrate_bps: int,
    segment_seconds: int,
    playlist_length: int,
    max_files: int,
    device: str,
    use_test_source: bool,
    allow_software_encoder: bool = False,
  ) -> None:
    self._launch = build_ring_launch(
      mode=mode,
      ring_dir=ring_dir,
      bitrate_bps=bitrate_bps,
      segment_seconds=segment_seconds,
      playlist_length=playlist_length,
      max_files=max_files,
      device=device,
      use_test_source=use_test_source,
      allow_software_encoder=allow_software_encoder,
    )
    self._loop = GLib.MainLoop()
    self._thread = threading.Thread(target=self._loop.run, name="gst-ring-loop", daemon=True)
    self._pipeline: Gst.Pipeline | None = None
    self._stopping = False

  def start(self) -> None:
    log.info("ring launch: %s", self._launch)
    self._pipeline = build_pipeline(self._launch)  # fails loud if encoder missing
    bus = self._pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", self._on_message)
    self._thread.start()
    self._pipeline.set_state(Gst.State.PLAYING)

  def stop(self) -> None:
    self._stopping = True
    if self._pipeline is not None:
      self._pipeline.set_state(Gst.State.NULL)
    if self._loop.is_running():
      self._loop.quit()

  def _on_message(self, bus: Gst.Bus, msg: Gst.Message) -> None:  # noqa: ARG002
    t = msg.type
    if t == Gst.MessageType.ERROR:
      err, _debug = msg.parse_error()
      log.error("gst ring error: %s", err)
      if not self._stopping:
        # Best-effort restart: the ring is always-on; a transient error
        # (e.g. SSD blip) should not leave the recorder permanently down.
        if self._pipeline is not None:
          self._pipeline.set_state(Gst.State.NULL)
        GLib.timeout_add_seconds(RING_RESTART_DELAY_S, self._restart)
    elif t == Gst.MessageType.EOS:
      # The ring should not normally EOS; treat like an error path.
      if not self._stopping:
        GLib.timeout_add_seconds(RING_RESTART_DELAY_S, self._restart)

  def _restart(self) -> bool:
    if self._stopping:
      return False
    log.info("restarting ring pipeline")
    self._pipeline = build_pipeline(self._launch)
    bus = self._pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", self._on_message)
    self._pipeline.set_state(Gst.State.PLAYING)
    return False  # one-shot
