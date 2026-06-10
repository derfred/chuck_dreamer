"""The single always-on capture pipeline + dynamic tee branches (architecture §2.2).

The Pi opens its camera (and mic) ONCE. A V4L2 USB device cannot be opened by
multiple `v4l2src` concurrently, so every consumer must hang off one shared
pipeline's `tee`, not its own source. `GstCapturePipeline` owns that pipeline:

  v4l2src ! ... ! tee_v   ├─ ring  (hlssink2, always-on, linked at build)
                          ├─ vision (appsink, always-on, linked at build)
                          ├─ live   (added/removed at runtime)
                          └─ recording (added/removed at runtime)
  alsasrc ! ... ! tee_a   └─ level  (always-on)

The on-demand live and recording branches are attached/detached on the RUNNING
pipeline via tee request pads + a blocking pad probe (the standard GStreamer
dynamic-branch dance): request a `tee_v` src pad, link a parsed branch bin onto
it, `sync_state_with_parent()`; to remove, block the tee pad, unlink, send EOS
into just that branch so its sink finalises, then set the bin to NULL and
release the request pad — without disturbing the always-on branches.

This module is the gi-touching owner. The branch *strings* come from pipeline.py
(pure, testable); the state machines drive add/remove through thin handles
(gst_live_handle / gst_recording_handle) that preserve their existing callback
contracts, so state_machine.py and recording_state_machine.py are unchanged.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import GLib, Gst, GstVideo  # noqa: E402

from .pipeline import VIDEO_TEE_NAME, build_capture_launch, build_pipeline  # noqa: E402

log = logging.getLogger(__name__)


class VideoBranch:
  """A branch linked onto the shared `tee_v` at runtime.

  Holds the parsed branch bin, the tee request (src) pad feeding it, and the
  callbacks the owning handle wants on lifecycle edges. Removal is idempotent.
  """

  def __init__(
    self,
    *,
    pipeline: Gst.Pipeline,
    tee: Gst.Element,
    bin_: Gst.Bin,
    tee_pad: Gst.Pad,
    on_playing: Callable[[], None] | None,
    on_error: Callable[[bool], None] | None,
    on_removed: Callable[[], None] | None,
  ) -> None:
    self._pipeline = pipeline
    self._tee = tee
    self._bin = bin_
    self._tee_pad = tee_pad
    self._on_playing = on_playing
    self._on_error = on_error
    self._on_removed = on_removed
    self._removed = False
    self._reached_playing = False
    self._lock = threading.Lock()

  @property
  def bin(self) -> Gst.Bin:
    return self._bin

  def note_state_changed(self, new: Gst.State) -> None:
    # Called by the owner's bus watch for STATE_CHANGED whose src is this bin.
    if new == Gst.State.PLAYING and not self._reached_playing:
      self._reached_playing = True
      if self._on_playing is not None:
        self._on_playing()

  def remove(self) -> None:
    """Detach this branch from the running pipeline without disturbing others.

    Blocks the tee src pad, unlinks, sends EOS into the branch so its sink
    (srtsink/hlssink2) finalises, then NULLs + removes the bin and releases the
    tee request pad. The `on_removed` callback fires once teardown is done."""
    with self._lock:
      if self._removed:
        return
      self._removed = True
    # Block the tee pad; the probe does the actual surgery on the streaming
    # thread, which is where dynamic relinking is safe.
    self._tee_pad.add_probe(Gst.PadProbeType.IDLE | Gst.PadProbeType.BLOCK, self._on_unlink_probe)

  def _on_unlink_probe(self, pad: Gst.Pad, _info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
    # On the streaming thread, with the pad blocked: unlink tee->branch and send
    # EOS into the branch so its sink finalises its file/stream, then tear down.
    peer = pad.get_peer()
    if peer is not None:
      pad.unlink(peer)
      # EOS travels downstream from the branch's sink pad through to its sink.
      peer.send_event(Gst.Event.new_eos())
    # Defer the NULL+remove off the streaming thread (can't change state from
    # inside a pad probe safely); schedule on the GLib loop.
    GLib.idle_add(self._finish_remove)
    return Gst.PadProbeReturn.REMOVE

  def _finish_remove(self) -> bool:
    try:
      self._bin.set_state(Gst.State.NULL)
      self._pipeline.remove(self._bin)
      self._tee.release_request_pad(self._tee_pad)
    except Exception:  # noqa: BLE001 — teardown must not crash the loop
      log.exception("error finishing branch removal")
    if self._on_removed is not None:
      self._on_removed()
    return False  # one-shot


class GstCapturePipeline:
  """Owns the single capture pipeline and the dynamic-branch machinery."""

  def __init__(
    self,
    *,
    mode,  # noqa: ANN001 — ModeTriplet
    ring_dir: str,
    ring_bitrate_bps: int,
    ring_segment_seconds: int,
    ring_playlist_length: int,
    ring_max_files: int,
    device: str = "/dev/video0",
    audio_device: str = "default",
    use_test_source: bool = False,
    allow_software_encoder: bool = False,
    audio_level_interval_ns: int = 500_000_000,
  ) -> None:
    launch = build_capture_launch(
      mode=mode,
      ring_dir=ring_dir,
      ring_bitrate_bps=ring_bitrate_bps,
      ring_segment_seconds=ring_segment_seconds,
      ring_playlist_length=ring_playlist_length,
      ring_max_files=ring_max_files,
      device=device,
      audio_device=audio_device,
      use_test_source=use_test_source,
      allow_software_encoder=allow_software_encoder,
      audio_level_interval_ns=audio_level_interval_ns,
    )
    log.info("capture launch: %s", launch)
    self._pipeline = build_pipeline(launch)  # fails loud if encoder missing
    self._tee = self._pipeline.get_by_name(VIDEO_TEE_NAME)
    self._loop = GLib.MainLoop()
    self._thread = threading.Thread(target=self._loop.run, name="capture-loop", daemon=True)
    self._branches: dict[Gst.Bin, VideoBranch] = {}
    self._lock = threading.Lock()
    self._stopping = False
    bus = self._pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", self._on_message)

  # --- lifecycle -------------------------------------------------------------

  def start(self) -> None:
    self._thread.start()
    self._pipeline.set_state(Gst.State.PLAYING)
    log.info("capture pipeline started")

  def stop(self) -> None:
    self._stopping = True
    self._pipeline.set_state(Gst.State.NULL)
    if self._loop.is_running():
      self._loop.quit()

  def loop_is_running(self) -> bool:
    """Whether the capture GLib loop is live. The recording handle schedules its
    deferred finalise on this loop, and falls back to a synchronous finalise when
    it is down (capture stopped) so the state machine's EXITED is never lost."""
    return self._loop.is_running() and not self._stopping

  def get_by_name(self, name: str):  # noqa: ANN201 — Gst.Element | None
    return self._pipeline.get_by_name(name)

  def bus(self) -> Gst.Bus:
    # Shared bus (already add_signal_watch()'d). The audio handle connects a
    # second `message::element` handler here for the level element's messages.
    return self._pipeline.get_bus()

  # --- dynamic video branches (live, recording) ------------------------------

  def add_video_branch(
    self,
    chain: str,
    *,
    on_playing: Callable[[], None] | None = None,
    on_error: Callable[[bool], None] | None = None,
    on_removed: Callable[[], None] | None = None,
  ) -> VideoBranch | None:
    """Parse `chain` (a sourceless 'queue ! ... ! sink' string from pipeline.py),
    link it onto a fresh tee_v request pad, and sync it into the running
    pipeline. Returns the VideoBranch handle, or None on a build/link failure
    (which fires on_error so the caller's state machine sees a connect failure
    rather than hanging)."""
    try:
      branch_bin = Gst.parse_bin_from_description(chain, True)
    except GLib.Error:
      log.exception("failed to parse branch chain: %s", chain)
      if on_error is not None:
        on_error(False)
      return None

    self._pipeline.add(branch_bin)
    tee_pad = self._tee.get_request_pad("src_%u")
    sink_pad = branch_bin.get_static_pad("sink")
    if tee_pad is None or sink_pad is None or tee_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
      log.error("failed to link branch onto tee_v")
      branch_bin.set_state(Gst.State.NULL)
      self._pipeline.remove(branch_bin)
      if tee_pad is not None:
        self._tee.release_request_pad(tee_pad)
      if on_error is not None:
        on_error(False)
      return None

    branch = VideoBranch(
      pipeline=self._pipeline,
      tee=self._tee,
      bin_=branch_bin,
      tee_pad=tee_pad,
      on_playing=on_playing,
      on_error=on_error,
      on_removed=on_removed,
    )
    with self._lock:
      self._branches[branch_bin] = branch
    # Bring the new branch up to the pipeline's current state.
    if branch_bin.sync_state_with_parent() is False:
      log.error("branch failed to sync state with parent")
      if on_error is not None:
        on_error(False)
    return branch

  def remove_video_branch(self, branch: VideoBranch) -> None:
    with self._lock:
      self._branches.pop(branch.bin, None)
    branch.remove()

  # --- bus -------------------------------------------------------------------

  def _on_message(self, _bus: Gst.Bus, msg: Gst.Message) -> None:
    t = msg.type
    if t == Gst.MessageType.STATE_CHANGED:
      src = msg.src
      # Route a branch bin's PLAYING to its handle (live "connected" signal).
      with self._lock:
        branch = self._branches.get(src) if src is not None else None
      if branch is not None:
        _old, new, _pending = msg.parse_state_changed()
        branch.note_state_changed(new)
    elif t == Gst.MessageType.ERROR:
      err, _debug = msg.parse_error()
      log.error("capture pipeline error: %s (src=%s)", err, msg.src.get_name() if msg.src else "?")
      # A whole-pipeline error: the source/ring failed. Best-effort restart is a
      # later concern (the old ring handle self-restarted); here we log loudly.
      # Branch-local errors route to that branch's on_error.
      with self._lock:
        branch = self._branches.get(msg.src) if msg.src is not None else None
      if branch is not None and branch._on_error is not None:  # noqa: SLF001
        branch._on_error(branch._reached_playing)  # noqa: SLF001


def force_idr(element: Gst.Element | None) -> None:
  """Send an upstream force-key-unit to `element` (the live encoder), best-effort
  (V2.2). Shared by the live handle; lives here so the gi import is in one place."""
  if element is None:
    log.warning("force_idr: live encoder not found")
    return
  sink = element.get_static_pad("sink")
  if sink is None:
    return
  event = GstVideo.video_event_new_upstream_force_key_unit(
    Gst.CLOCK_TIME_NONE, all_headers=True, count=0
  )
  if not sink.send_event(event):
    log.info("force_idr: encoder did not accept force-key-unit (ignored)")
