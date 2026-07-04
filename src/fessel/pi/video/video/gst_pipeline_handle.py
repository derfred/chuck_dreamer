"""GStreamer-backed PipelineHandle for the live WHIP sender — a dynamic branch
on the shared capture pipeline's WARM live-encoder tee (architecture §2.2), not
a standalone pipeline.

The live encoder runs warm in the capture pipeline whenever it is up; this
handle governs only the sender. `start()` attaches the WHIP sender branch
(queue ! whipclientsink) onto `tee_live`; `stop()` detaches it.
whipclientsink owns its WebRTC session lifecycle, so attaching IS the WHIP
handshake and detaching tears the session down.

The state-machine callback contract is unchanged: branch-reaches-PLAYING ->
on_connected, branch-removed -> on_exited, branch error -> on_disconnected (if
it had connected) / on_connect_failed (if not). NOTE: branch-PLAYING is a
conservative "connected" signal — it fires when the sender bin is live, not
when ICE/DTLS completes; a deeper signal from whipclientsink's signaller
(e.g. its session/webrtcbin state notifications) can refine this later.
force_idr() targets the warm live encoder upstream of tee_live.
"""

from __future__ import annotations

import logging
import threading

from .gst_capture import GstCapturePipeline, VideoBranch, force_idr
from .pipeline import LIVE_ENCODER_NAME, LIVE_TEE_NAME, whip_sender_chain
from .state_machine import LiveStateMachine, PipelineHandle

log = logging.getLogger(__name__)


class GstLivePipeline(PipelineHandle):
  def __init__(
    self,
    *,
    sm: LiveStateMachine,
    capture: GstCapturePipeline,
    whip_endpoint: str,
  ) -> None:
    self._sm = sm
    self._capture = capture
    self._chain = whip_sender_chain(whip_endpoint=whip_endpoint)
    self._branch: VideoBranch | None = None
    self._connected = False
    self._lock = threading.Lock()

  def start(self) -> None:
    log.info("attaching WHIP sender branch: %s", self._chain)
    self._branch = self._capture.add_video_branch(
      self._chain,
      tee_name=LIVE_TEE_NAME,
      on_playing=self._on_playing,
      on_error=self._on_error,
      on_removed=self._on_removed,
    )
    if self._branch is not None:
      # OPTIMISTIC connected: whipclientsink is a webrtcsink bin that adds its
      # internals dynamically, so the outer branch bin never aggregates to
      # PLAYING and the bus-based on_playing never fires — the state machine's
      # connect-timeout then kills a HEALTHY session at connect_timeout_s
      # (observed in production: media flowing, EOF at exactly 15s). A
      # successful attach is the honest local signal; a failed WHIP session
      # surfaces as a branch error -> on_disconnected -> re-attach with
      # backoff. End-to-end, the relay's live_activation_timeout remains the
      # authoritative arbiter (§4.3). The bus on_playing path stays as a
      # harmless dedup'd no-op.
      self._on_playing()

  def stop(self) -> None:
    # Detach the WHIP sender branch from tee_live (tears the WebRTC session
    # down); the warm live encoder keeps running. on_removed -> on_exited.
    if self._branch is not None:
      self._capture.remove_video_branch(self._branch)
    else:
      self._sm.on_exited()

  def force_idr(self) -> None:
    # §2.2: ask the WARM live encoder (upstream of tee_live in the shared
    # capture pipeline) to emit an IDR at the next frame, so the stream starts
    # with a decodable keyframe on attach rather than waiting out the GOP.
    # Best-effort; never raises into the state machine.
    force_idr(self._capture.get_by_name(LIVE_ENCODER_NAME))

  # --- branch lifecycle callbacks (from the capture pipeline's bus) ----------

  def _on_playing(self) -> None:
    with self._lock:
      if self._connected:
        return
      self._connected = True
    self._sm.on_connected()

  def _on_error(self, had_connected: bool) -> None:
    # A branch-local error. If it had connected, the WHIP session dropped;
    # else the branch never came up (connect failure). Tear the branch down
    # either way — the encoder stays warm.
    if self._branch is not None:
      self._capture.remove_video_branch(self._branch)
    if had_connected:
      self._sm.on_disconnected()
    else:
      self._sm.on_connect_failed()

  def _on_removed(self) -> None:
    self._sm.on_exited()
