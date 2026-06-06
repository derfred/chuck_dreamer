"""GStreamer-backed RecordingPipelineHandle — a dynamic branch on the shared
capture pipeline (architecture §2.2), not a standalone pipeline.

`start()` attaches a recording branch (queue ! encode ! hlssink2) onto the
capture pipeline's `tee_v`; `stop()` detaches it (EOS finalises the hlssink2
playlist). The state-machine callback contract is unchanged: branch-reaches-
PLAYING -> on_segment (hlssink2 producing segments), branch-removed -> on_exited,
branch error -> on_failed (if no segment yet) / on_exited (if it was recording).
"""

from __future__ import annotations

import logging
import threading

from fessel_schemas import ModeTriplet

from .gst_capture import GstCapturePipeline, VideoBranch
from .pipeline import recording_branch_chain
from .recording_state_machine import RecordingPipelineHandle, RecordingStateMachine

log = logging.getLogger(__name__)


class GstRecordingPipeline(RecordingPipelineHandle):
  def __init__(
    self,
    *,
    sm: RecordingStateMachine,
    capture: GstCapturePipeline,
    mode: ModeTriplet,
    recording_dir: str,
    bitrate_bps: int,
    segment_seconds: int,
    allow_software_encoder: bool = False,
  ) -> None:
    self._sm = sm
    self._capture = capture
    self._chain = recording_branch_chain(
      mode=mode,
      recording_dir=recording_dir,
      bitrate_bps=bitrate_bps,
      segment_seconds=segment_seconds,
      allow_software_encoder=allow_software_encoder,
    )
    self._branch: VideoBranch | None = None
    self._segment_reported = False
    self._lock = threading.Lock()

  def start(self) -> None:
    log.info("recording branch chain: %s", self._chain)
    self._branch = self._capture.add_video_branch(
      self._chain,
      on_playing=self._on_playing,
      on_error=self._on_error,
      on_removed=self._on_removed,
    )

  def stop(self) -> None:
    if self._branch is not None:
      self._capture.remove_video_branch(self._branch)
    else:
      self._sm.on_exited()

  # --- branch lifecycle callbacks (from the capture pipeline's bus) ----------

  def _on_playing(self) -> None:
    with self._lock:
      if self._segment_reported:
        return
      self._segment_reported = True
    # hlssink2 is producing segments to disk -> recording is live.
    self._sm.on_segment()

  def _on_error(self, _had_segment: bool) -> None:
    if self._branch is not None:
      self._capture.remove_video_branch(self._branch)
    # If we never reached recording, this is a start failure (-> idle); if we
    # were recording, EXITED finalises whatever is on disk.
    if not self._segment_reported:
      self._sm.on_failed()
    else:
      self._sm.on_exited()

  def _on_removed(self) -> None:
    self._sm.on_exited()
