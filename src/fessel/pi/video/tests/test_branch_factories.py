"""Recording-factory wiring test — the kwargs VideoApp._make_recording_pipeline
passes must match GstRecordingPipeline's constructor.

This guards the factory<->constructor seam directly. An explicit recording is
copy-from-ring (not a pipeline branch): the handle takes `capture=` (for the
GLib loop) and `ring_dir=` (the segments to copy) and no longer takes the old
`device`/`use_test_source` source kwargs. A stale caller would only blow up at
recording-start time on the rec-sm thread (recording never leaves `idle`, seen
in integration as recording_roundtrip failing) — constructing via the REAL
factory turns that into a fast, local failure.

No GStreamer: the constructor only stores its args; nothing runs until start().
"""

from __future__ import annotations

from fessel_schemas import ModeTriplet
from fessel_shared import Storage

from video.gst_recording_handle import GstRecordingPipeline
from video.main import VideoApp

MODE = ModeTriplet(resolution="640x480", fps=30, bitrate_bps=1_000_000)


class FakeCapture:
  """Stands in for the always-on GstCapturePipeline. The recording handle only
  stores it (and calls loop_is_running on stop); nothing runs in the ctor."""


def _app_for_factory(tmp_path) -> VideoApp:
  """A VideoApp with only the attributes _make_recording_pipeline reads, built
  without __init__ so the test needs no MQTT/GStreamer. If the factory grows a
  new dependency this fails loudly with AttributeError — the intended signal."""
  app = object.__new__(VideoApp)
  app._capture = FakeCapture()
  app._storage = Storage(str(tmp_path))
  app._rec_cfg = {"segment_seconds": 2}
  app._rec_sm = object()  # the handle only stores sm; never called here
  return app


def test_make_recording_pipeline_matches_constructor(tmp_path):
  # The regression guard: _make_recording_pipeline must pass exactly the kwargs
  # GstRecordingPipeline accepts (a stale/missing kwarg raises TypeError here).
  # The factory takes (recording_id, lookback_seconds) — recording has no mode
  # (it reuses the ring encode).
  app = _app_for_factory(tmp_path)
  handle = VideoApp._make_recording_pipeline(app, "rec-abc", 30.0)
  assert isinstance(handle, GstRecordingPipeline)
  # Wired to the shared capture and pointed at the ring it copies from.
  assert handle._capture is app._capture
  assert handle._ring_dir == app._storage.ring_dir
  # Look-back reached the handle.
  assert handle._lookback_seconds == 30.0
  # And the factory created the on-disk recording dir as a side effect.
  assert app._storage.recording_dir("rec-abc").is_dir()


def test_recording_handle_rejects_legacy_source_kwargs(tmp_path):
  # Pins the contract the prior regression violated: recording is sourceless
  # (copy-from-ring), so the old source kwargs must NOT be accepted.
  import pytest

  # `mode` is also gone now — recording resolution is a deploy setting, not a
  # per-recording kwarg.
  for bad in ("device", "use_test_source", "mode"):
    with pytest.raises(TypeError):
      GstRecordingPipeline(
        sm=object(),
        capture=FakeCapture(),
        ring_dir=str(tmp_path / "ring"),
        recording_dir=str(tmp_path),
        segment_seconds=2,
        **{bad: "anything"},
      )
