"""Freeze-frame capture + push for the Monitor UX (no bandwidth before Stream On).

The snapshot appsink lives inside the shared capture pipeline (§2.2, one camera
open fanned via tee_v), mirroring the vision appsink's attach idiom
(gst_vision_handle.py): this handle connects to `new-sample`, caches the latest
JPEG buffer, and lets the app push that cached frame to webui on a plain timer.
It does NOT decode/analyse anything — the appsink already emits JPEG bytes
(pipeline.py's snapshot branch: videoscale ! jpegenc ! appsink), so this handle
is just a thin cache + HTTP PUT.
"""

from __future__ import annotations

import logging
import threading
import time

from .pipeline import SNAPSHOT_APPSINK_NAME

log = logging.getLogger(__name__)


class GstSnapshotPipeline:
  """Caches the latest JPEG frame from the shared capture pipeline's snapshot
  appsink. `start()`/`stop()` attach/detach the signal handler — the capture
  pipeline's own lifecycle drives PLAYING/NULL. `latest()` returns the most
  recent JPEG bytes *and when they were captured* (or None before the first
  sample).

  The capture instant is stamped here, in the appsink callback, rather than at
  push time: the two differ whenever the pipeline stalls (a wedged camera keeps
  the last cached frame around indefinitely), and it is the capture instant the
  operator's "Xs ago" label is actually about. It is a `time.monotonic()`
  reading, not wall clock — the consumer turns it into an *age* to send, so a
  Pi with a skewed clock can't misreport freshness."""

  def __init__(self, *, capture) -> None:  # noqa: ANN001 — GstCapturePipeline
    self._capture = capture
    self._lock = threading.Lock()
    self._latest: bytes | None = None
    self._latest_at: float | None = None
    self._handler_id: int | None = None
    self._appsink = self._capture.get_by_name(SNAPSHOT_APPSINK_NAME)

  def start(self) -> None:
    if self._appsink is None:
      log.error("snapshot appsink %r not found in capture pipeline", SNAPSHOT_APPSINK_NAME)
      return
    self._handler_id = self._appsink.connect("new-sample", self._on_sample)
    log.info("snapshot capture attached to capture appsink")

  def stop(self) -> None:
    if self._appsink is not None and self._handler_id is not None:
      self._appsink.disconnect(self._handler_id)
      self._handler_id = None

  def latest(self) -> tuple[bytes, float] | None:
    """The cached JPEG and the `time.monotonic()` instant it was captured at,
    or None before the first sample."""
    with self._lock:
      if self._latest is None or self._latest_at is None:
        return None
      return self._latest, self._latest_at

  def _on_sample(self, appsink):  # noqa: ANN001, ANN202
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    sample = appsink.emit("pull-sample")
    if sample is None:
      return Gst.FlowReturn.OK
    buf = sample.get_buffer()
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if ok:
      try:
        captured_at = time.monotonic()
        with self._lock:
          self._latest = bytes(mapinfo.data)
          self._latest_at = captured_at
      finally:
        buf.unmap(mapinfo)
    return Gst.FlowReturn.OK
