"""GStreamer forward-capture handle for anomaly recordings (V5.7).

Implements `ForwardCaptureHandle` (anomaly_recording.py) as a dynamic recording
branch on the shared capture pipeline (§2.2) — it needs camera frames, which
only the one shared pipeline (the single v4l2src) has. It attaches a finite-HLS
recording branch onto `tee_v`, runs for `after_seconds`, then detaches (EOS
finalises the hlssink2 playlist) and calls `on_closed`. `extend` pushes the
close deadline out (coalesce).

The forward-capture segments are numbered high (seg-9NNNN.ts) so they never
collide with the ring segments copied as seg-0000N into the same anomaly dir;
the AnomalyRecorder rewrites a combined playlist over both on close.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib  # noqa: E402

from .pipeline import recording_branch_chain  # noqa: E402

log = logging.getLogger(__name__)

# A distinct hlssink2 element name + high segment template so the forward branch
# does not clash with the always-on ring's hlssink2 nor the copied ring segments.
_FORWARD_HLS_NAME = "anomaly_fwd_hls"
_FORWARD_SEGMENT_TEMPLATE = "seg-9%04d.ts"


class GstAnomalyForwardCapture:
  def __init__(
    self,
    *,
    capture,  # noqa: ANN001 — GstCapturePipeline
    rec_dir: Path,
    after_seconds: float,
    on_closed: Callable[[], None],
    mode,  # noqa: ANN001 — ModeTriplet
    bitrate_bps: int,
    segment_seconds: int,
    allow_software_encoder: bool = False,
  ) -> None:
    self._capture = capture
    self._rec_dir = Path(rec_dir)
    self._on_closed = on_closed
    self._after_seconds = after_seconds
    self._chain = recording_branch_chain(
      mode=mode,
      recording_dir=str(self._rec_dir),
      bitrate_bps=bitrate_bps,
      segment_seconds=segment_seconds,
      hls_name=_FORWARD_HLS_NAME,
      segment_template=_FORWARD_SEGMENT_TEMPLATE,
      allow_software_encoder=allow_software_encoder,
    )
    self._branch = None
    self._closed = False
    self._lock = threading.Lock()
    self._deadline = 0.0
    self._timer_id: int | None = None

  def start(self) -> None:
    self._branch = self._capture.add_video_branch(
      self._chain, on_removed=self._on_removed, on_error=self._on_error
    )
    if self._branch is None:
      # Build/link failed: finalise with whatever the ring copy already wrote.
      self._fire_closed()
      return
    self._deadline = time.monotonic() + self._after_seconds
    # Poll the deadline on the capture GLib loop; polling (vs a one-shot) lets
    # extend() push the deadline out without rescheduling.
    self._timer_id = GLib.timeout_add_seconds(1, self._tick)
    log.info("anomaly forward capture started in %s", self._rec_dir)

  def extend(self, extra_seconds: float) -> None:
    with self._lock:
      self._deadline = max(self._deadline, time.monotonic() + extra_seconds)
    log.info("anomaly forward capture extended by +%.0fs", extra_seconds)

  def _tick(self) -> bool:
    with self._lock:
      reached = time.monotonic() >= self._deadline
    if reached:
      self._detach()
      return False  # stop the timer
    return True  # keep polling

  def _detach(self) -> None:
    # Detach the branch from the shared tee; EOS finalises the hlssink2 playlist
    # and on_removed -> _fire_closed.
    if self._branch is not None:
      self._capture.remove_video_branch(self._branch)
      self._branch = None
    else:
      self._fire_closed()

  def _on_removed(self) -> None:
    self._fire_closed()

  def _on_error(self, _had_playing: bool) -> None:
    log.error("anomaly forward capture branch errored; finalising")
    self._fire_closed()

  def _fire_closed(self) -> None:
    with self._lock:
      if self._closed:
        return
      self._closed = True
    try:
      self._on_closed()
    except Exception:  # noqa: BLE001
      log.exception("anomaly forward-capture on_closed callback failed")

  def stop(self) -> None:
    self._detach()
