"""GStreamer-backed PipelineHandle for the live branch — a dynamic branch on the
shared capture pipeline (architecture §2.2), not a standalone pipeline.

`start()` attaches a live branch (queue ! encode ! mpegtsmux ! srtsink) onto the
capture pipeline's `tee_v`; `stop()` detaches it. The state-machine callback
contract is unchanged: branch-reaches-PLAYING -> on_connected (srtsink published
to mediamtx), branch-removed -> on_exited, branch error -> on_disconnected (if it
had connected) / on_connect_failed (if not). force_idr() targets the live
encoder living in the shared pipeline.
"""

from __future__ import annotations

import logging
import threading

from fessel_schemas import ModeTriplet

from .gst_capture import GstCapturePipeline, VideoBranch, force_idr
from .pipeline import LIVE_ENCODER_NAME, live_branch_chain
from .state_machine import LiveStateMachine, PipelineHandle

log = logging.getLogger(__name__)


class GstLivePipeline(PipelineHandle):
  def __init__(
    self,
    *,
    sm: LiveStateMachine,
    capture: GstCapturePipeline,
    mode: ModeTriplet,
    path: str,
    srt_host: str,
    srt_port: int,
    allow_software_encoder: bool = False,
  ) -> None:
    self._sm = sm
    self._capture = capture
    self._chain = live_branch_chain(
      mode=mode,
      path=path,
      srt_host=srt_host,
      srt_port=srt_port,
      allow_software_encoder=allow_software_encoder,
    )
    self._branch: VideoBranch | None = None
    self._connected = False
    self._lock = threading.Lock()

  def start(self) -> None:
    log.info("live branch chain: %s", self._chain)
    self._branch = self._capture.add_video_branch(
      self._chain,
      on_playing=self._on_playing,
      on_error=self._on_error,
      on_removed=self._on_removed,
    )

  def stop(self) -> None:
    # Detach the live branch from the shared tee; on_removed -> on_exited.
    if self._branch is not None:
      self._capture.remove_video_branch(self._branch)
    else:
      self._sm.on_exited()

  def force_idr(self) -> None:
    # V2.2: ask the live encoder (in the shared capture pipeline) to emit an IDR
    # at the next frame. Best-effort; never raises into the state machine.
    force_idr(self._capture.get_by_name(LIVE_ENCODER_NAME))

  # --- branch lifecycle callbacks (from the capture pipeline's bus) ----------

  def _on_playing(self) -> None:
    with self._lock:
      if self._connected:
        return
      self._connected = True
    self._sm.on_connected()

  def _on_error(self, had_connected: bool) -> None:
    # A branch-local error. If it had connected, the srt link dropped; else the
    # branch never came up (connect failure). Tear the branch down either way.
    if self._branch is not None:
      self._capture.remove_video_branch(self._branch)
    if had_connected:
      self._sm.on_disconnected()
    else:
      self._sm.on_connect_failed()

  def _on_removed(self) -> None:
    self._sm.on_exited()
