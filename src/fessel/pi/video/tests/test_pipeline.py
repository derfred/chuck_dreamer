"""Launch-string builder tests. Skipped if gi/Gst is unavailable."""

import pytest
from fessel_schemas import ModeTriplet

gi = pytest.importorskip("gi")

from video.pipeline import build_live_launch  # noqa: E402

MODE = ModeTriplet(resolution="1280x720", fps=30, bitrate_bps=2_500_000)


def test_live_launch_has_loadbearing_caps():
  launch = build_live_launch(
    mode=MODE, path="pi", srt_host="mediamtx-srt", use_test_source=True
  )
  # SRT publisher streamid + MTU-safe pkt_size.
  assert "streamid=publish:pi" in launch
  assert "pkt_size=1316" in launch
  # Fail-fast srtsink (reconnect is the state machine's job).
  assert "wait-for-connection=false" in launch
  # MPEG-TS over SRT.
  assert "mpegtsmux" in launch
  assert launch.strip().endswith('wait-for-connection=false')


def test_test_source_avoids_v4l2src():
  launch = build_live_launch(mode=MODE, path="pi", srt_host="h", use_test_source=True)
  assert "videotestsrc" in launch
  assert "v4l2src" not in launch
