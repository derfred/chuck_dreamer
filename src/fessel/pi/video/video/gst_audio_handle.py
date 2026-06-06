"""GStreamer wiring for the audio-analysis branch (V5.6).

Thin handle around the audio pipeline (build_audio_launch). The `level` element
posts a message on the GStreamer bus every `interval`; this handle reads the
per-channel RMS + peak dBFS off those messages and feeds the pure
`AudioAnalyzer` (audio.py). Level readings and spike edge events are forwarded
to callbacks the video app turns into MQTT publishes (the retained level meter)
and anomaly-recording triggers.

Like the other handles, GStreamer (`gi`) is imported lazily so the pure analyzer
and the pipeline builder stay testable; this module runs on the Pi / in the
integration harness.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from .audio import AudioAnalyzer
from .pipeline import AUDIO_LEVEL_NAME, build_audio_launch, build_pipeline

log = logging.getLogger(__name__)

# on_level(rms_db, peak_db) -> None   (per level message; app rate-limits the publish)
LevelCallback = Callable[[float, float], None]
# on_events(list[(AnomalyType, dict)]) -> None
EventsCallback = Callable[[list], None]


class GstAudioPipeline:
  def __init__(
    self,
    *,
    analyzer: AudioAnalyzer,
    on_level: LevelCallback,
    on_events: EventsCallback,
    device: str = "default",
    use_test_source: bool = False,
    level_interval_ns: int = 500_000_000,
  ) -> None:
    self._analyzer = analyzer
    self._on_level = on_level
    self._on_events = on_events

    launch = build_audio_launch(
      device=device, use_test_source=use_test_source, level_interval_ns=level_interval_ns
    )
    self._pipeline = build_pipeline(launch)

    import gi

    gi.require_version("GLib", "2.0")
    from gi.repository import GLib

    self._loop = GLib.MainLoop()
    self._thread = threading.Thread(target=self._loop.run, name="audio-loop", daemon=True)
    self._stopping = False

    bus = self._pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message::element", self._on_element_message)

  def start(self) -> None:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    self._thread.start()
    self._pipeline.set_state(Gst.State.PLAYING)
    log.info("audio pipeline started")

  def stop(self) -> None:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    self._stopping = True
    self._pipeline.set_state(Gst.State.NULL)
    self._loop.quit()

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
