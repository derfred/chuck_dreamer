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

from .pipeline import SNAPSHOT_APPSINK_NAME

log = logging.getLogger(__name__)


class GstSnapshotPipeline:
  """Caches the latest JPEG frame from the shared capture pipeline's snapshot
  appsink. `start()`/`stop()` attach/detach the signal handler — the capture
  pipeline's own lifecycle drives PLAYING/NULL. `latest()` returns the most
  recent JPEG bytes (or None before the first sample)."""

  def __init__(self, *, capture) -> None:  # noqa: ANN001 — GstCapturePipeline
    self._capture = capture
    self._lock = threading.Lock()
    self._latest: bytes | None = None
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

  def latest(self) -> bytes | None:
    with self._lock:
      return self._latest

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
        with self._lock:
          self._latest = bytes(mapinfo.data)
      finally:
        buf.unmap(mapinfo)
    return Gst.FlowReturn.OK
