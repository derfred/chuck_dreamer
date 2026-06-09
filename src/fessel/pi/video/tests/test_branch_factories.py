"""Branch-factory wiring tests — the kwargs VideoApp passes to each branch
handle must match that handle's constructor.

These guard the factory<->constructor seam directly: GstRecordingPipeline (and
its siblings) was refactored into a branch on the shared capture pipeline, so it
takes `capture=` and no longer takes `device`/`use_test_source` (the capture owns
the source). A stale caller passing the old kwargs raises TypeError only at
recording-start time, on the rec-sm thread, where it manifests as the recording
never leaving `idle` (caught in integration as recording_roundtrip_via_ingest,
never in unit tests until now). Constructing via the REAL factory turns that into
a fast, local failure.

No GStreamer: the handle constructors only build a sourceless element-chain
string; nothing is realised until start().
"""

from __future__ import annotations

from fessel_schemas import ModeTriplet
from fessel_shared import Storage

from video.gst_recording_handle import GstRecordingPipeline
from video.main import VideoApp

MODE = ModeTriplet(resolution="640x480", fps=30, bitrate_bps=1_000_000)


class FakeCapture:
  """Stands in for the always-on GstCapturePipeline. The recording-branch
  constructor only stores it; nothing is called until start()."""


def _app_for_factory(tmp_path) -> VideoApp:
  """A VideoApp with only the attributes _make_recording_pipeline reads, built
  without __init__ so the test needs no MQTT/GStreamer. If the factory grows a
  new dependency this fails loudly with AttributeError — the intended signal."""
  app = object.__new__(VideoApp)
  app._capture = FakeCapture()
  app._storage = Storage(str(tmp_path))
  app._rec_cfg = {"bitrate_bps": 1_000_000, "segment_seconds": 2}
  app._allow_software_encoder = True
  app._rec_sm = object()  # the handle only stores sm; never called here
  return app


def test_make_recording_pipeline_matches_constructor(tmp_path):
  # The regression: _make_recording_pipeline must pass exactly the kwargs
  # GstRecordingPipeline accepts. A stale `device=`/`use_test_source=` (or a
  # missing `capture=`) raises TypeError here.
  app = _app_for_factory(tmp_path)
  handle = VideoApp._make_recording_pipeline(app, "rec-abc", MODE)
  assert isinstance(handle, GstRecordingPipeline)
  # The branch is wired to the shared capture, not a standalone source.
  assert handle._capture is app._capture
  # And the factory created the on-disk recording dir as a side effect.
  assert app._storage.recording_dir("rec-abc").is_dir()


def test_recording_handle_rejects_legacy_source_kwargs(tmp_path):
  # Pins the contract the regression violated: the handle is sourceless, so the
  # source kwargs that used to be threaded through must NOT be accepted.
  import pytest

  for bad in ("device", "use_test_source"):
    with pytest.raises(TypeError):
      GstRecordingPipeline(
        sm=object(),
        capture=FakeCapture(),
        mode=MODE,
        recording_dir=str(tmp_path),
        bitrate_bps=1_000_000,
        segment_seconds=2,
        **{bad: "anything"},
      )
