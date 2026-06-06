"""GStreamer forward-capture handle for anomaly recordings (V5.7).

Implements `ForwardCaptureHandle` (anomaly_recording.py). It spawns a finite HLS
capture into the anomaly dir using the same builder as explicit recordings
(build_recording_launch — keep everything, finite playlist), runs it for
`after_seconds`, then EOS-finalises and calls `on_closed`. `extend` pushes the
close deadline out (coalesce). The hlssink2 writes its own seg-NNNNN.ts files
into the anomaly dir, where they sit alongside the copied ring segments; the
AnomalyRecorder rewrites a combined playlist on close.

GStreamer is imported lazily, like the other handles; this runs on the Pi / in
the integration harness. The before-segments copied from the ring use seg-0000N;
to avoid a filename collision the forward capture starts its numbering high.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .pipeline import build_pipeline, build_recording_launch

log = logging.getLogger(__name__)

# hlssink2's `location` template — start the forward capture's numbering high so
# it never collides with the ring segments copied as seg-0000N (V5.7).
_FORWARD_SEGMENT_BASE = 90000


class GstAnomalyForwardCapture:
  def __init__(
    self,
    *,
    rec_dir: Path,
    after_seconds: float,
    on_closed: Callable[[], None],
    mode,  # noqa: ANN001 — ModeTriplet
    bitrate_bps: int,
    segment_seconds: int,
    device: str = "/dev/video0",
    use_test_source: bool = False,
    allow_software_encoder: bool = False,
  ) -> None:
    self._rec_dir = Path(rec_dir)
    self._on_closed = on_closed
    self._closed = False
    self._lock = threading.Lock()
    # The forward capture writes into a subnumbered location so its segments do
    # not overwrite the copied ring segments. We point hlssink2 at the same dir
    # but with a high segment base via the location template.
    launch = build_recording_launch(
      mode=mode,
      recording_dir=str(self._rec_dir),
      bitrate_bps=bitrate_bps,
      segment_seconds=segment_seconds,
      device=device,
      use_test_source=use_test_source,
      allow_software_encoder=allow_software_encoder,
    )
    # Renumber the forward segments high to avoid colliding with the copied
    # ring segments (seg-0000N). build_recording_launch uses seg-%05d.ts; shift
    # the start index by overriding the location with a start-index template.
    launch = launch.replace(
      f"location={self._rec_dir}/seg-%05d.ts",
      f"location={self._rec_dir}/seg-9%04d.ts",
    )
    self._pipeline = build_pipeline(launch)

    import gi

    gi.require_version("GLib", "2.0")
    from gi.repository import GLib

    self._glib = GLib
    self._loop = GLib.MainLoop()
    self._thread = threading.Thread(target=self._loop.run, name="anomaly-fwd-loop", daemon=True)
    self._deadline = time.monotonic() + after_seconds
    self._timer_id: int | None = None

    bus = self._pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", self._on_message)

  def start(self) -> None:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    self._thread.start()
    self._pipeline.set_state(Gst.State.PLAYING)
    # Arm the close timer on the GLib loop; re-checked on extend.
    self._arm_timer()
    log.info("anomaly forward capture started in %s", self._rec_dir)

  def extend(self, extra_seconds: float) -> None:
    with self._lock:
      self._deadline = max(self._deadline, time.monotonic() + extra_seconds)
    log.info("anomaly forward capture extended to +%.0fs", extra_seconds)

  def _arm_timer(self) -> None:
    # Poll the deadline each second on the GLib loop; EOS when reached. Polling
    # (rather than a single one-shot) lets `extend` push the deadline out without
    # re-scheduling machinery.
    self._timer_id = self._glib.timeout_add_seconds(1, self._tick)

  def _tick(self) -> bool:
    with self._lock:
      reached = time.monotonic() >= self._deadline
    if reached:
      self._eos()
      return False  # stop the timer
    return True  # keep polling

  def _eos(self) -> None:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    # Graceful EOS finalises the hlssink2 playlist; the EOS bus message tears down.
    self._pipeline.send_event(Gst.Event.new_eos())

  def _on_message(self, _bus, message) -> None:  # noqa: ANN001
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    if message.type == Gst.MessageType.EOS:
      self._teardown_and_report()
    elif message.type == Gst.MessageType.ERROR:
      err, _ = message.parse_error()
      log.error("anomaly forward capture error: %s", err)
      self._teardown_and_report()

  def _teardown_and_report(self) -> None:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    with self._lock:
      if self._closed:
        return
      self._closed = True
    self._pipeline.set_state(Gst.State.NULL)
    self._loop.quit()
    try:
      self._on_closed()
    except Exception:  # noqa: BLE001
      log.exception("anomaly forward-capture on_closed callback failed")

  def stop(self) -> None:
    self._eos()
