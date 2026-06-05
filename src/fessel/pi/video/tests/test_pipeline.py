"""Launch-string + encoder-selection tests.

These no longer require gi/Gst — build_live_launch and encoder_profile are
pure string/dataclass logic; only build_pipeline() touches GStreamer.
"""

import pytest
from fessel_schemas import ModeTriplet

from video.pipeline import build_live_launch, build_recording_launch, build_ring_launch
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


# --- Slice 4: ring + explicit-recording launch builders ----------------------


def test_ring_launch_writes_hls_with_rolling_window():
  launch = build_ring_launch(
    mode=MODE,
    ring_dir="/mnt/ssd/ring",
    bitrate_bps=2_000_000,
    segment_seconds=2,
    playlist_length=150,
    max_files=160,
    use_test_source=True,
    allow_software_encoder=True,
  )
  assert "hlssink2" in launch
  assert "playlist-location=/mnt/ssd/ring/index.m3u8" in launch
  assert "location=/mnt/ssd/ring/seg-%05d.ts" in launch
  assert "target-duration=2" in launch
  # Rolling window: bounded playlist + max-files (segments are deleted past it).
  assert "playlist-length=150" in launch
  assert "max-files=160" in launch
  # GOP = fps * segment_seconds so every segment starts on a keyframe.
  assert "key-int-max=60" in launch  # x264enc software seam
  assert "video/x-raw,format=I420" in launch


def test_recording_launch_is_finite_keep_everything():
  launch = build_recording_launch(
    mode=MODE,
    recording_dir="/mnt/ssd/recordings/explicit/rec-1",
    bitrate_bps=8_000_000,
    segment_seconds=2,
    use_test_source=True,
    allow_software_encoder=True,
  )
  assert "hlssink2" in launch
  assert "playlist-location=/mnt/ssd/recordings/explicit/rec-1/index.m3u8" in launch
  # Finite recording: keep everything, do not roll (documented hlssink2 values).
  assert "playlist-length=0" in launch
  assert "max-files=0" in launch


def test_ring_and_recording_use_their_own_bitrate():
  # The ring/recording bitrates are independent of any live mode bitrate.
  ring = build_ring_launch(
    mode=MODE, ring_dir="/r", bitrate_bps=2_000_000, segment_seconds=2,
    playlist_length=10, max_files=12, use_test_source=True, allow_software_encoder=True,
  )
  rec = build_recording_launch(
    mode=MODE, recording_dir="/d", bitrate_bps=8_000_000, segment_seconds=2,
    use_test_source=True, allow_software_encoder=True,
  )
  # x264enc bitrate is kbit/s -> bitrate=2000 (ring) and bitrate=8000 (recording).
  assert "bitrate=2000 " in ring
  assert "bitrate=8000 " in rec
