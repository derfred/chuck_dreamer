"""GStreamer wiring for the audio-analysis branch (V5.6).

The `level` element lives inside the shared capture pipeline (§2.2, one mic open
fanned via tee_a) and posts a message on that pipeline's bus every `interval`;
this handle reads the per-channel RMS + peak dBFS off those messages and feeds
the pure `AudioAnalyzer` (audio.py). Level readings and spike edge events are
forwarded to callbacks the video app turns into MQTT publishes (the retained
level meter) and anomaly-recording triggers.

The pure analyzer and the pipeline builder stay testable; this module runs on
the Pi / in the integration harness.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from .audio import AudioAnalyzer
from .pipeline import AUDIO_LEVEL_NAME

log = logging.getLogger(__name__)

# on_level(rms_db, peak_db) -> None   (per level message; app rate-limits the publish)
LevelCallback = Callable[[float, float], None]
# on_events(list[(AnomalyType, dict)]) -> None
EventsCallback = Callable[[list], None]


class GstAudioPipeline:
  """Audio analysis attached to the shared capture pipeline's `level` element (§2.2).

  The `level` element lives INSIDE the always-on capture pipeline (one mic open,
  fanned via tee_a). This handle does NOT own a pipeline; it watches the capture
  pipeline's bus for the level element's `message::element` and feeds the pure
  AudioAnalyzer. `start()`/`stop()` attach/detach the bus handler.
  """

  def __init__(
    self,
    *,
    analyzer: AudioAnalyzer,
    capture,  # noqa: ANN001 — GstCapturePipeline
    on_level: LevelCallback,
    on_events: EventsCallback,
  ) -> None:
    self._analyzer = analyzer
    self._capture = capture
    self._on_level = on_level
    self._on_events = on_events
    self._stopping = False
    self._handler_id: int | None = None

  def start(self) -> None:
    # The capture pipeline already add_signal_watch()'d its bus; connect a
    # second handler for the level element's messages.
    bus = self._capture.bus()
    self._handler_id = bus.connect("message::element", self._on_element_message)
    log.info("audio analysis attached to capture level element")

  def stop(self) -> None:
    self._stopping = True
    if self._handler_id is not None:
      self._capture.bus().disconnect(self._handler_id)
      self._handler_id = None

  def _on_element_message(self, _bus, message) -> None:  # noqa: ANN001
    src = message.src
    if src is None or src.get_name() != AUDIO_LEVEL_NAME:
      return
    structure = message.get_structure()
    if structure is None or not structure.has_field("rms"):
      return
    try:
      # `rms` and `peak` are per-channel GValueArrays of dBFS; take the loudest
      # channel (mono mic is the common case, but be robust to stereo).
      rms_db = max(self._values(structure, "rms"))
      peak_db = max(self._values(structure, "peak"))
    except (ValueError, TypeError):
      return
    try:
      events = self._analyzer.process_level(rms_db, peak_db, time.monotonic())
    except Exception:  # noqa: BLE001 — a bad sample must not wedge the bus thread
      log.exception("audio level processing error")
      return
    self._on_level(rms_db, peak_db)
    if events:
      self._on_events(events)

  @staticmethod
  def _values(structure, field: str) -> list[float]:  # noqa: ANN001
    # GstStructure stores the level arrays as a GValueArray; PyGObject exposes
    # it as a list of floats via get_value.
    vals = structure.get_value(field)
    return [float(v) for v in vals]
