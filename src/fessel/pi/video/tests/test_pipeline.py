"""Launch-string + encoder-selection tests.

These no longer require gi/Gst — build_live_launch and encoder_profile are
pure string/dataclass logic; only build_pipeline() touches GStreamer.
"""

import pytest
from fessel_schemas import ModeTriplet

from video.pipeline import (
  build_audio_launch,
  build_capture_launch,
  build_live_launch,
  build_recording_launch,
  build_ring_launch,
  build_vision_launch,
  live_branch_chain,
  recording_branch_chain,
)
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
  launch = build_live_launch(mode=MODE, path="pi", srt_host="h", allow_software_encoder=True)
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


# --- Slice 5: vision + audio analysis launch builders ------------------------


def test_vision_launch_downscales_to_640x360_appsink():
  launch = build_vision_launch(mode=MODE, use_test_source=True)
  # Analyses at 640x360 (the coord space safe_box + detectors use).
  assert "width=640,height=360" in launch
  assert "appsink name=vision_sink" in launch
  # Backpressure: drop old frames at the appsink, never queue stale ones (V5.1).
  assert "drop=true" in launch
  assert "max-buffers=1" in launch
  # No encoder in the vision branch — it analyses raw frames in-process.
  assert "h264" not in launch
  assert "v4l2h264enc" not in launch


def test_vision_launch_uses_test_source_in_dev():
  launch = build_vision_launch(mode=MODE, use_test_source=True)
  assert "videotestsrc" in launch
  assert "v4l2src" not in launch


def test_audio_launch_has_level_element_posting_messages():
  launch = build_audio_launch(use_test_source=True, level_interval_ns=500_000_000)
  assert "level name=audio_level" in launch
  assert "post-messages=true" in launch
  assert "interval=500000000" in launch
  # Dev fallback avoids alsasrc (no mic on the dev box).
  assert "audiotestsrc" in launch
  assert "alsasrc" not in launch


def test_audio_launch_real_mic_uses_alsasrc():
  launch = build_audio_launch(device="hw:1", use_test_source=False)
  assert "alsasrc device=hw:1" in launch
  assert "audiotestsrc" not in launch


def test_ring_and_recording_use_their_own_bitrate():
  # The ring/recording bitrates are independent of any live mode bitrate.
  ring = build_ring_launch(
    mode=MODE,
    ring_dir="/r",
    bitrate_bps=2_000_000,
    segment_seconds=2,
    playlist_length=10,
    max_files=12,
    use_test_source=True,
    allow_software_encoder=True,
  )
  rec = build_recording_launch(
    mode=MODE,
    recording_dir="/d",
    bitrate_bps=8_000_000,
    segment_seconds=2,
    use_test_source=True,
    allow_software_encoder=True,
  )
  # x264enc bitrate is kbit/s -> bitrate=2000 (ring) and bitrate=8000 (recording).
  assert "bitrate=2000 " in ring
  assert "bitrate=8000 " in rec


# --- shared single-capture pipeline + dynamic branch chains (architecture §2.2) --


def test_capture_launch_opens_one_camera_and_tees_to_always_on_branches():
  launch = build_capture_launch(
    mode=MODE,
    ring_dir="/mnt/ssd/ring",
    ring_bitrate_bps=2_000_000,
    ring_segment_seconds=2,
    ring_playlist_length=10,
    ring_max_files=12,
    use_test_source=True,
    allow_software_encoder=True,
  )
  # Exactly ONE video source open, fanned out via a single video tee.
  assert launch.count("videotestsrc") == 1
  assert launch.count("tee name=tee_v") == 1
  # The always-on ring + vision branches hang off tee_v (the `tee_v.` re-link).
  assert "tee_v. ! queue ! " in launch
  assert "hlssink2 name=ring_hls" in launch
  assert "appsink name=vision_sink" in launch
  assert "width=640,height=360" in launch  # vision downscale
  # Audio: one mic open, its own tee_a feeding the level branch.
  assert launch.count("tee name=tee_a") == 1
  assert "level name=audio_level" in launch
  # The on-demand live/recording branches are NOT baked in (added at runtime).
  assert "srtsink" not in launch
  assert launch.count("hlssink2") == 1  # only the ring's


def test_capture_launch_real_camera_opens_v4l2src_once():
  launch = build_capture_launch(
    mode=MODE,
    ring_dir="/r",
    ring_bitrate_bps=2_000_000,
    ring_segment_seconds=2,
    ring_playlist_length=10,
    ring_max_files=12,
    device="/dev/video0",
    use_test_source=False,
    allow_software_encoder=True,
  )
  # The whole point of the refactor: the camera is opened exactly once.
  assert launch.count("v4l2src") == 1
  assert launch.count("alsasrc") == 1


def test_live_branch_chain_is_sourceless_and_keeps_its_own_encoder():
  chain = live_branch_chain(
    mode=MODE, path="pi", srt_host="mediamtx-srt", allow_software_encoder=True
  )
  # A sourceless 'queue ! ... ! srtsink' chain to link onto a tee_v request pad.
  assert chain.startswith("queue ! ")
  assert "v4l2src" not in chain and "videotestsrc" not in chain
  assert "streamid=publish:pi" in chain
  assert "pkt_size=1316" in chain
  assert "wait-for-connection=false" in chain
  # Its own short-GOP encoder, named so the handle can force an IDR.
  assert "name=live_enc" in chain
  assert "key-int-max=30" in chain  # gop = fps


def test_recording_branch_chain_is_sourceless_finite_hls():
  chain = recording_branch_chain(
    mode=MODE,
    recording_dir="/mnt/ssd/recordings/explicit/r1",
    bitrate_bps=8_000_000,
    segment_seconds=2,
    allow_software_encoder=True,
  )
  assert chain.startswith("queue ! ")
  assert "v4l2src" not in chain and "videotestsrc" not in chain
  assert "playlist-length=0" in chain and "max-files=0" in chain
  # Overridable hls name + segment template (anomaly forward-capture uses these).
  anomaly = recording_branch_chain(
    mode=MODE,
    recording_dir="/mnt/ssd/recordings/anomaly/a1",
    bitrate_bps=1_500_000,
    segment_seconds=2,
    hls_name="anomaly_fwd_hls",
    segment_template="seg-9%04d.ts",
    allow_software_encoder=True,
  )
  assert "hlssink2 name=anomaly_fwd_hls" in anomaly
  assert "location=/mnt/ssd/recordings/anomaly/a1/seg-9%04d.ts" in anomaly
