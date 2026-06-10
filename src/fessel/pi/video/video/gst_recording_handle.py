"""RecordingPipelineHandle for explicit recording — copy-from-ring (§2.2).

An explicit recording is NOT a GStreamer branch. hlssink2 cannot be added live
to the running capture pipeline (splitmuxsink fails its muxer→sink link), and a
cold-built idle hlssink2 sink blocks the whole pipeline from prerolling. So
instead of muxing a second time, we reuse the ALWAYS-ON ring: it already encodes
+ segments continuously. Recording just remembers when it started and, on stop,
copies the ring `.ts` segments covering [start, stop] into the recording dir and
assembles a finite VOD playlist (mirroring the anomaly forward-capture's
copy-from-ring, anomaly_recording.py).

The RecordingStateMachine contract is unchanged: start() eventually causes
on_segment() (-> recording) or on_failed(); stop() eventually causes on_exited()
(-> finalise + idle). The SM stamps metadata.segments via count_segments() AFTER
on_exited(), so the copy+assemble runs BEFORE on_exited().

Consequence: a recording inherits the ring's resolution/bitrate; the requested
`recording.bitrate_bps`/`mode` do not drive a separate encode (one encoder).
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path

import gi
from fessel_schemas import ModeTriplet
from fessel_shared import assemble_recording_playlist

from .anomaly_recording import select_ring_segments
from .gst_capture import GstCapturePipeline
from .recording_state_machine import RecordingPipelineHandle, RecordingStateMachine

gi.require_version("Gst", "1.0")
from gi.repository import GLib  # noqa: E402

log = logging.getLogger(__name__)

# Slack added to the [start, stop] window so the ring segment whose mtime lands
# just after the stop request (it was being written when stop fired) is included.
_WINDOW_END_SLACK_S = 3.0


class GstRecordingPipeline(RecordingPipelineHandle):
  def __init__(
    self,
    *,
    sm: RecordingStateMachine,
    capture: GstCapturePipeline,
    ring_dir: str,
    mode: ModeTriplet,
    recording_dir: str,
    segment_seconds: int,
    bitrate_bps: int = 0,
    allow_software_encoder: bool = False,
  ) -> None:
    self._sm = sm
    self._capture = capture
    self._ring_dir = Path(ring_dir)
    self._recording_dir = Path(recording_dir)
    # mode/bitrate_bps/allow_software_encoder are vestigial (recording reuses the
    # ring's encode); segment_seconds sets the assembled playlist's EXTINF.
    self._segment_seconds = int(segment_seconds)
    self._start_epoch: float | None = None
    self._lock = threading.Lock()
    self._stopped = False

  def start(self) -> None:
    # The ring is always recording, so "start" is just marking the window open.
    # There is no encoder to spin up, so the recording is live immediately:
    # report a segment so the state machine advances idle -> recording.
    self._start_epoch = time.time()
    log.info("recording window opened at %.3f -> %s", self._start_epoch, self._recording_dir)
    self._sm.on_segment()

  def stop(self) -> None:
    with self._lock:
      if self._stopped:
        return
      self._stopped = True
    # Copy + assemble on the capture GLib loop if it's running (keeps all GStreamer
    # adjacent work on one thread), else inline — either way on_exited() must fire.
    if self._capture.loop_is_running():
      GLib.idle_add(self._finalise_and_exit)
    else:
      self._finalise_and_exit()

  def _finalise_and_exit(self) -> bool:
    try:
      n = self._copy_window_from_ring()
      assemble_recording_playlist(self._recording_dir, self._segment_seconds)
      log.info("recording finalised (%d segments) in %s", n, self._recording_dir)
    except Exception:  # noqa: BLE001 — never block EXITED on copy/assembly
      log.exception("failed to assemble recording from ring in %s", self._recording_dir)
    self._sm.on_exited()
    return False  # one-shot

  def _copy_window_from_ring(self) -> int:
    """Copy (never move — the ring is still rolling) the ring segments covering
    [start, stop] into the recording dir as raw-NNNNN.ts in write order;
    assemble_recording_playlist renames them to seg-NNNNN.ts + writes the
    playlist."""
    start = self._start_epoch if self._start_epoch is not None else 0.0
    end = time.time() + _WINDOW_END_SLACK_S
    self._recording_dir.mkdir(parents=True, exist_ok=True)
    segs = select_ring_segments(self._ring_dir, start, end)
    for i, src in enumerate(segs):
      try:
        shutil.copy2(src, self._recording_dir / f"raw-{i:05d}.ts")
      except OSError:
        log.warning("failed to copy ring segment %s into recording", src)
    return len(segs)
