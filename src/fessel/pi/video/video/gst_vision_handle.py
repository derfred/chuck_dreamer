"""GStreamer + OpenCV wiring for the vision-analysis branch (V5.1).

Thin handle around the appsink pipeline (build_vision_launch): it owns the
GLib MainLoop thread, pulls downscaled frames in the appsink `new-sample`
callback, runs the OpenCV background subtractor to produce a foreground mask,
and hands the mask to the pure `VisionAnalyzer` (vision.py). The analyzer's
FrameResult events + summary are forwarded to the supplied callbacks, which the
video app turns into MQTT publishes and anomaly-recording triggers.

This is the only place cv2/numpy are imported (lazily), mirroring how the other
handles import `gi`. The pipeline.py builders and vision.py analyzer stay pure
and testable; this module is exercised on the Pi / in the integration harness.

Backpressure (plan §V5.1): the appsink is configured max-buffers=1 drop=true, so
a slow analyzer drops frames at the source rather than processing stale ones.
A heartbeat (frames_processed / frames_dropped / last_frame_at) is published by
the app on a timer; this handle exposes the counters.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from fessel_schemas import SafeBox

from .anomaly_recording import latest_ring_segment
from .pipeline import VISION_APPSINK_NAME, build_pipeline, build_vision_launch
from .vision import FrameResult, Mask, VisionAnalyzer

log = logging.getLogger(__name__)


class _CvMask:
  """A foreground mask backed by a numpy uint8 array (255 = foreground).

  Wraps the BackgroundSubtractor output so the pure analyzer can query it via
  the Mask protocol without importing numpy itself."""

  def __init__(self, mask) -> None:  # noqa: ANN001 — numpy array
    self._mask = mask  # H x W uint8, 0 or 255
    self._h, self._w = mask.shape[:2]

  @property
  def total_pixels(self) -> int:
    return int(self._h * self._w)

  def foreground_pixels(self) -> int:
    import numpy as np

    return int(np.count_nonzero(self._mask))

  def foreground_outside(self, box: SafeBox) -> int:
    import numpy as np

    total_fg = int(np.count_nonzero(self._mask))
    # Clamp the box to the frame and count the foreground inside it; outside is
    # the remainder.
    x0 = max(0, box.x)
    y0 = max(0, box.y)
    x1 = min(self._w, box.x + box.w)
    y1 = min(self._h, box.y + box.h)
    if x1 <= x0 or y1 <= y0:
      return total_fg
    inside = int(np.count_nonzero(self._mask[y0:y1, x0:x1]))
    return total_fg - inside


# on_summary(FrameResult) -> None   (called per processed frame; app rate-limits)
SummaryCallback = Callable[[FrameResult], None]
# on_events(list[(AnomalyType, dict)]) -> None
EventsCallback = Callable[[list], None]


class GstVisionPipeline:
  """Owns the vision appsink pipeline + background subtractor + analyzer."""

  def __init__(
    self,
    *,
    analyzer: VisionAnalyzer,
    mode,  # noqa: ANN001 — ModeTriplet
    ring_dir: str,
    on_summary: SummaryCallback,
    on_events: EventsCallback,
    device: str = "/dev/video0",
    use_test_source: bool = False,
    bg_algorithm: str = "MOG2",
    bg_history: int = 500,
    bg_var_threshold: float = 16.0,
    morph_kernel: int = 3,
  ) -> None:
    self._analyzer = analyzer
    self._ring_dir = ring_dir
    self._on_summary = on_summary
    self._on_events = on_events
    self._bg_algorithm = bg_algorithm
    self._bg_history = bg_history
    self._bg_var_threshold = bg_var_threshold
    self._morph_kernel = max(1, morph_kernel)

    launch = build_vision_launch(mode=mode, device=device, use_test_source=use_test_source)
    self._pipeline = build_pipeline(launch)

    import gi

    gi.require_version("GLib", "2.0")
    from gi.repository import GLib

    self._loop = GLib.MainLoop()
    self._thread = threading.Thread(target=self._loop.run, name="vision-loop", daemon=True)
    self._subtractor = self._make_subtractor()
    self._kernel = self._make_kernel()

    self._lock = threading.Lock()
    self.frames_processed = 0
    self.frames_dropped = 0
    self.last_frame_at: float | None = None
    self._stopping = False

    appsink = self._pipeline.get_by_name(VISION_APPSINK_NAME)
    appsink.connect("new-sample", self._on_sample)

  def _make_subtractor(self):  # noqa: ANN202
    import cv2

    if self._bg_algorithm.upper() == "KNN":
      return cv2.createBackgroundSubtractorKNN(history=self._bg_history, detectShadows=False)
    return cv2.createBackgroundSubtractorMOG2(
      history=self._bg_history, varThreshold=self._bg_var_threshold, detectShadows=False
    )

  def _make_kernel(self):  # noqa: ANN202
    import cv2

    k = self._morph_kernel
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

  def start(self) -> None:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    self._thread.start()
    self._pipeline.set_state(Gst.State.PLAYING)
    log.info("vision pipeline started")

  def stop(self) -> None:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    self._stopping = True
    self._pipeline.set_state(Gst.State.NULL)
    self._loop.quit()

  # --- counters for the heartbeat (V5.1) -------------------------------------

  def heartbeat(self) -> dict:
    with self._lock:
      from datetime import datetime, timezone

      last = (
        datetime.fromtimestamp(self.last_frame_at, timezone.utc).isoformat()
        if self.last_frame_at is not None
        else None
      )
      return {
        "last_frame_at": last,
        "frames_processed": self.frames_processed,
        "frames_dropped": self.frames_dropped,
      }

  # --- appsink callback ------------------------------------------------------

  def _on_sample(self, appsink):  # noqa: ANN001, ANN202
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    sample = appsink.emit("pull-sample")
    if sample is None:
      return Gst.FlowReturn.OK
    try:
      self._process_sample(sample)
    except Exception:  # noqa: BLE001 — a single bad frame must not wedge the thread
      log.exception("vision frame processing error")
    return Gst.FlowReturn.OK

  def _process_sample(self, sample) -> None:  # noqa: ANN001
    import cv2
    import numpy as np

    buf = sample.get_buffer()
    caps = sample.get_caps()
    structure = caps.get_structure(0)
    width = structure.get_value("width")
    height = structure.get_value("height")
    ok, mapinfo = buf.map(0)  # Gst.MapFlags.READ == 1; 0 works via PyGObject default
    if not ok:
      return
    try:
      # I420: Y plane is width*height bytes at the front; we only need luma for
      # motion (the subtractor works on grayscale fine, and it's cheap).
      y_plane = np.frombuffer(mapinfo.data[: width * height], dtype=np.uint8).reshape(
        (height, width)
      )
      fg = self._subtractor.apply(y_plane)
      # Morphological open rejects single-pixel noise before counting (V5.2).
      fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel)
      # MOG2 emits 0/127/255; treat anything > 0 as foreground.
      mask: Mask = _CvMask((fg > 0).astype(np.uint8) * 255)
    finally:
      buf.unmap(mapinfo)

    hint = latest_ring_segment_safe(self._ring_dir)
    result = self._analyzer.process_mask(mask, ring_segment_hint=hint)
    with self._lock:
      self.frames_processed += 1
      self.last_frame_at = time.time()
    if result.events:
      self._on_events(result.events)
    self._on_summary(result)


def latest_ring_segment_safe(ring_dir: str) -> str | None:
  from pathlib import Path

  try:
    return latest_ring_segment(Path(ring_dir))
  except Exception:  # noqa: BLE001
    return None
