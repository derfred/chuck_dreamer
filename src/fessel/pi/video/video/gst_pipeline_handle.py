"""GStreamer-backed PipelineHandle bridging the bus to the state machine.

Runs the pipeline on a GLib MainLoop in a dedicated thread, watches the
bus for EOS/ERROR, and infers srtsink connect/disconnect from state
changes and errors. Reports back to the LiveStateMachine via callbacks.
"""

from __future__ import annotations

import logging
import threading

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from fessel_schemas import ModeTriplet  # noqa: E402

from .pipeline import build_live_launch, build_pipeline  # noqa: E402
from .state_machine import LiveStateMachine, PipelineHandle  # noqa: E402

log = logging.getLogger(__name__)


class GstLivePipeline(PipelineHandle):
  def __init__(
    self,
    *,
    sm: LiveStateMachine,
    mode: ModeTriplet,
    path: str,
    srt_host: str,
    srt_port: int,
    device: str,
    use_test_source: bool,
  ) -> None:
    self._sm = sm
    launch = build_live_launch(
      mode=mode,
      path=path,
      srt_host=srt_host,
      srt_port=srt_port,
      device=device,
      use_test_source=use_test_source,
    )
    log.info("live launch: %s", launch)
    self._pipeline = build_pipeline(launch)  # fails loud if encoder missing
    self._loop = GLib.MainLoop()
    self._thread = threading.Thread(target=self._loop.run, name="gst-loop", daemon=True)
    self._connected = False
    bus = self._pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", self._on_message)

  def start(self) -> None:
    self._thread.start()
    self._pipeline.set_state(Gst.State.PLAYING)

  def stop(self) -> None:
    # Graceful: send EOS, then the bus EOS handler tears down and reports.
    self._pipeline.send_event(Gst.Event.new_eos())
    # Guard against a pipeline that never delivers EOS (e.g. never linked):
    GLib.timeout_add_seconds(5, self._force_exit)

  def _force_exit(self) -> bool:
    self._teardown_and_report()
    return False  # one-shot

  def _on_message(self, bus: Gst.Bus, msg: Gst.Message) -> None:  # noqa: ARG002
    t = msg.type
    if t == Gst.MessageType.STATE_CHANGED:
      if msg.src is self._pipeline:
        _old, new, _pending = msg.parse_state_changed()
        if new == Gst.State.PLAYING and not self._connected:
          # srtsink reached PLAYING -> publisher connected to mediamtx.
          self._connected = True
          self._sm.on_connected()
    elif t == Gst.MessageType.EOS:
      self._teardown_and_report()
    elif t == Gst.MessageType.ERROR:
      err, _debug = msg.parse_error()
      log.error("gst error: %s", err)
      if self._connected:
        self._connected = False
        self._teardown()
        self._sm.on_disconnected()
      else:
        self._teardown()
        self._sm.on_connect_failed()

  def _teardown(self) -> None:
    self._pipeline.set_state(Gst.State.NULL)
    if self._loop.is_running():
      self._loop.quit()

  def _teardown_and_report(self) -> None:
    self._teardown()
    self._sm.on_exited()
