"""GStreamer-backed RecordingPipelineHandle — drives the always-on, valve-gated
recording sink on the shared capture pipeline (architecture §2.2).

The explicit-recording hlssink2 is built COLD into the capture pipeline (it
cannot be added live — splitmuxsink fails its muxer→sink link against a running
pipeline), sharing the ring encoder's H.264 through a `valve`. So this handle
does NOT add/remove a branch; it opens/closes the valve:

  start() -> capture.recording_open(dir): retarget the sink at this recording's
             dir + open the valve. A GLib poll watches the dir for the first
             fragment and fires on_segment (-> recording).
  stop()  -> capture.recording_close(): close the valve, wait one segment
             duration for the in-flight fragment to finalise, rebuild the real
             index.m3u8 from the raw-*.ts, then on_exited (-> finalise + idle).

The RecordingStateMachine contract is unchanged: start() eventually causes
on_segment() or on_failed(); stop() eventually causes on_exited(). The SM itself
stamps metadata.segments via count_segments() AFTER on_exited(), so the playlist
assembly (which renames raw-*.ts -> seg-*.ts) MUST run before on_exited().
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import gi
from fessel_schemas import ModeTriplet
from fessel_shared import assemble_recording_playlist

from .gst_capture import GstCapturePipeline
from .recording_state_machine import RecordingPipelineHandle, RecordingStateMachine

gi.require_version("Gst", "1.0")
from gi.repository import GLib  # noqa: E402

log = logging.getLogger(__name__)

# How often to poll the recording dir for the first fragment (ms).
_FIRST_SEGMENT_POLL_MS = 250


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
    self._recording_dir = recording_dir
    # mode/bitrate_bps/allow_software_encoder are accepted for the factory
    # signature but no longer drive a separate encode — recording shares the ring
    # encoder (see build_capture_launch). segment_seconds sets the assembled
    # playlist's TARGETDURATION/EXTINF and the post-stop flush wait.
    self._segment_seconds = int(segment_seconds)
    self._segment_reported = False
    self._stopped = False
    self._poll_id: int | None = None
    self._lock = threading.Lock()

  def start(self) -> None:
    try:
      self._capture.recording_open(self._recording_dir)
    except Exception:  # noqa: BLE001 — any open failure is a start failure
      log.exception("recording_open failed for %s", self._recording_dir)
      self._sm.on_failed()
      return
    log.info("recording opened (valve) in %s", self._recording_dir)
    # Watch the dir for the first fragment; its appearance == recording is live.
    # Poll on the capture GLib loop (the established pattern, cf. anomaly
    # forward-capture) rather than a pad probe on the shared muxer.
    self._poll_id = GLib.timeout_add(_FIRST_SEGMENT_POLL_MS, self._poll_first_segment)

  def _poll_first_segment(self) -> bool:
    if not self._first_fragment_exists():
      return True  # keep polling; the SM start-timeout is the backstop
    with self._lock:
      if self._segment_reported:
        return False
      self._segment_reported = True
    self._poll_id = None
    self._sm.on_segment()
    return False  # one-shot

  def _first_fragment_exists(self) -> bool:
    return any(Path(self._recording_dir).glob("raw-*.ts"))

  def stop(self) -> None:
    with self._lock:
      if self._stopped:
        return
      self._stopped = True
    if self._poll_id is not None:
      GLib.source_remove(self._poll_id)
      self._poll_id = None
    self._capture.recording_close()  # close the valve (idempotent)
    # Give hlssink2 one segment duration to finalise the in-flight fragment, then
    # assemble the real playlist and report EXITED. If the capture loop is down
    # (shutdown), finalise synchronously so the state machine never hangs waiting
    # for EXITED.
    if self._capture.loop_is_running():
      GLib.timeout_add_seconds(self._segment_seconds + 1, self._finalise_and_exit)
    else:
      self._finalise_and_exit()

  def _finalise_and_exit(self) -> bool:
    try:
      n = assemble_recording_playlist(self._recording_dir, self._segment_seconds)
      log.info("recording finalised (%d segments) in %s", n, self._recording_dir)
    except Exception:  # noqa: BLE001 — never block EXITED on assembly
      log.exception("failed to assemble recording playlist in %s", self._recording_dir)
    self._sm.on_exited()
    return False  # one-shot
