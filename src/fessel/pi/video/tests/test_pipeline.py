"""Launch-string + encoder-selection tests.

These no longer require gi/Gst — build_live_launch and encoder_profile are
pure string/dataclass logic; only build_pipeline() touches GStreamer.
"""

import pytest
from fessel_schemas import ModeTriplet

from video.pipeline import build_live_launch
from video.platform import encoder_profile

MODE = ModeTriplet(resolution="1280x720", fps=30, bitrate_bps=2_500_000)


def test_live_launch_has_loadbearing_caps():
  launch = build_live_launch(
    mode=MODE, path="pi", srt_host="mediamtx-srt", allow_software_encoder=True
  )
  assert "streamid=publish:pi" in launch
  assert "pkt_size=1316" in launch
  assert "wait-for-connection=false" in launch
  assert "mpegtsmux" in launch
  # Load-bearing input cap for any v4l2 encoder negotiation; the I420 cap
  # is present via the source chain.
  assert "video/x-raw,format=I420" in launch
  assert launch.strip().endswith("wait-for-connection=false")


def test_test_source_avoids_v4l2src():
  launch = build_live_launch(
    mode=MODE, path="pi", srt_host="h", use_test_source=True, allow_software_encoder=True
  )
  assert "videotestsrc" in launch
  assert "v4l2src" not in launch


def test_software_encoder_off_by_default_fails_loud_off_platform(monkeypatch):
  # On a non-Linux/non-macOS platform with software disabled, encoder
  # selection must raise rather than degrade. We simulate by forcing an
  # unknown platform.
  import video.platform as plat

  monkeypatch.setattr(plat.platform, "system", lambda: "Plan9")
  with pytest.raises(RuntimeError):
    encoder_profile()  # default allow_software=False


def test_software_encoder_seam_selects_x264enc():
  prof = encoder_profile(allow_software=True)
  assert prof.element == "x264enc"
  assert prof.is_software is True
  launch = build_live_launch(
    mode=MODE, path="pi", srt_host="h", allow_software_encoder=True
  )
  assert "x264enc" in launch
  assert "v4l2h264enc" not in launch


def test_default_profile_is_not_software_on_real_platforms():
  # On the CI/dev platform (Linux or macOS), the default must be hardware.
  import platform as _p

  if _p.system() in ("Linux", "Darwin"):
    assert encoder_profile().is_software is False
