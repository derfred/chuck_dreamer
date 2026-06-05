"""GStreamer-backed RecordingPipelineHandle for the explicit-recording branch.

Mirrors gst_pipeline_handle.py (the live branch): runs the recording pipeline
on a GLib MainLoop in a dedicated thread, watches the bus for EOS/ERROR, and
reports back to the RecordingStateMachine via callbacks.

The "first segment written" signal (starting -> recording) is inferred from the
pipeline reaching PLAYING: at that point hlssink2 is producing segments to disk.
That is the same heuristic the live handle uses for "connected" (PLAYING ->
on_connected). On EOS (graceful stop) the playlist is finalised by hlssink2 and
the SM is told EXITED; on ERROR the SM is told FAILED then EXITED.
"""

from __future__ import annotations

import logging
import threading

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from fessel_schemas import ModeTriplet  # noqa: E402

from .pipeline import build_pipeline, build_recording_launch  # noqa: E402
from .recording_state_machine import RecordingPipelineHandle, RecordingStateMachine  # noqa: E402

log = logging.getLogger(__name__)


class GstRecordingPipeline(RecordingPipelineHandle):
  def __init__(
    self,
    *,
    sm: RecordingStateMachine,
    mode: ModeTriplet,
    recording_dir: str,
    bitrate_bps: int,
    segment_seconds: int,
    device: str,
    use_test_source: bool,
    allow_software_encoder: bool = False,
  ) -> None:
    self._sm = sm
    launch = build_recording_launch(
      mode=mode,
      recording_dir=recording_dir,
      bitrate_bps=bitrate_bps,
      segment_seconds=segment_seconds,
      device=device,
      use_test_source=use_test_source,
      allow_software_encoder=allow_software_encoder,
    )
    log.info("recording launch: %s", launch)
    self._pipeline = build_pipeline(launch)  # fails loud if encoder missing
    self._loop = GLib.MainLoop()
    self._thread = threading.Thread(target=self._loop.run, name="gst-rec-loop", daemon=True)
    self._segment_reported = False
    self._errored = False
    bus = self._pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", self._on_message)

  def start(self) -> None:
    self._thread.start()
    self._pipeline.set_state(Gst.State.PLAYING)

  def stop(self) -> None:
    # Graceful: EOS the recording branch so hlssink2 finalises its playlist;
    # the bus EOS handler then tears down and reports EXITED.
    self._pipeline.send_event(Gst.Event.new_eos())
    GLib.timeout_add_seconds(5, self._force_exit)

  def _force_exit(self) -> bool:
    self._teardown_and_report()
    return False  # one-shot

  def _on_message(self, bus: Gst.Bus, msg: Gst.Message) -> None:  # noqa: ARG002
    t = msg.type
    if t == Gst.MessageType.STATE_CHANGED:
      if msg.src is self._pipeline:
        _old, new, _pending = msg.parse_state_changed()
        if new == Gst.State.PLAYING and not self._segment_reported:
          # hlssink2 is producing segments to disk -> recording is live.
          self._segment_reported = True
          self._sm.on_segment()
    elif t == Gst.MessageType.EOS:
      self._teardown_and_report()
    elif t == Gst.MessageType.ERROR:
      err, _debug = msg.parse_error()
      log.error("gst recording error: %s", err)
      if not self._errored:
        self._errored = True
        self._teardown()
        # If we never reached recording, this is a start failure (-> idle);
        # if we were recording, the EXITED path finalises whatever is on disk.
        if not self._segment_reported:
          self._sm.on_failed()
        else:
          self._sm.on_exited()

  def _teardown(self) -> None:
    self._pipeline.set_state(Gst.State.NULL)
    if self._loop.is_running():
      self._loop.quit()

  def _teardown_and_report(self) -> None:
    self._teardown()
    self._sm.on_exited()
