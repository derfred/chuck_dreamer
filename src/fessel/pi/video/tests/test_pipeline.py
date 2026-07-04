"""Launch-string + encoder-selection tests.

These do not require gi/Gst — the launch builders and encoder_profile are
pure string/dataclass logic; only build_pipeline() touches GStreamer.
"""

import pytest
from fessel_schemas import ModeTriplet

from video.pipeline import (
  build_audio_launch,
  build_capture_launch,
  build_recording_launch,
  build_ring_launch,
  build_vision_launch,
  live_warm_branch,
  recording_branch_chain,
  whip_sender_chain,
)
from video.platform import encoder_profile

MODE = ModeTriplet(resolution="1280x720", fps=30, bitrate_bps=2_500_000)


def _capture_launch(**overrides) -> str:
  kwargs = dict(
    mode=MODE,
    ring_dir="/mnt/ssd/ring",
    ring_bitrate_bps=2_000_000,
    ring_segment_seconds=2,
    ring_playlist_length=10,
    ring_max_files=12,
    use_test_source=True,
    allow_software_encoder=True,
  )
  kwargs.update(overrides)
  return build_capture_launch(**kwargs)


def test_test_source_avoids_v4l2src():
  launch = _capture_launch(use_test_source=True)
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
  launch = _capture_launch(allow_software_encoder=True)
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
  launch = _capture_launch()
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
  # NO SRT anywhere in the live path any more (§2.2: WHIP replaced it), and
  # the on-demand WHIP sender is NOT baked in (attached at runtime). Explicit
  # recording is NOT a branch at all (copy-from-ring), so only the ring's sink.
  assert "srtsink" not in launch
  assert "mpegtsmux" not in launch
  assert "whipclientsink" not in launch
  assert launch.count("hlssink2") == 1  # only the ring's
  assert "valve" not in launch


def test_capture_launch_bakes_in_the_warm_live_encoder():
  # §2.2 warm-but-detached: the fixed-profile live encoder + tee_live (with a
  # permanently linked fakesink drop branch) are built into the capture
  # pipeline, so activation only attaches the WHIP sender (no cold start).
  launch = _capture_launch(live_resolution="640x360", live_fps=15, live_bitrate_bps=1_000_000)
  assert launch.count("tee name=tee_live") == 1
  assert "name=live_enc" in launch
  # Scaled/rated to the fixed deploy-time live profile off the raw tee.
  assert "video/x-raw,width=640,height=360,framerate=15/1,format=I420" in launch
  # Short ~1s GOP for a fast join (x264enc software seam: key-int-max = fps).
  assert "key-int-max=15" in launch
  # x264enc bitrate is kbit/s -> the live profile's own bitrate, not the ring's.
  assert "bitrate=1000 " in launch
  # The drop branch keeps the warm encoder flowing while unwatched.
  assert "tee_live. ! queue ! fakesink sync=false" in launch


def test_capture_launch_real_camera_opens_v4l2src_once():
  launch = _capture_launch(ring_dir="/r", device="/dev/video0", use_test_source=False)
  # The whole point of the refactor: the camera is opened exactly once.
  assert launch.count("v4l2src") == 1
  assert launch.count("alsasrc") == 1


def test_whip_sender_chain_is_sourceless_rtp_whip():
  # The on-demand WHIP sender branch (§2.2/§2.3): a sourceless chain linked
  # onto a tee_live request pad on activation. whipclientsink owns the WebRTC
  # session — attaching IS the WHIP handshake.
  chain = whip_sender_chain(whip_endpoint="http://webui:8001/whip/ingest")
  assert chain.startswith("queue ! ")
  # whipclientsink payloads internally; RTP elements must NOT appear.
  assert "rtph264pay" not in chain
  assert "whipclientsink" in chain
  assert 'signaller::whip-endpoint="http://webui:8001/whip/ingest"' in chain
  # No encoder (it rides the warm live encoder) and no SRT remnants.
  assert "enc" not in chain
  assert "srtsink" not in chain and "mpegtsmux" not in chain


def test_live_warm_branch_uses_its_own_fixed_profile():
  branch = live_warm_branch(
    live_resolution="1280x720",
    live_fps=15,
    live_bitrate_bps=1_500_000,
    allow_software_encoder=True,
  )
  assert branch.startswith("tee_v. ! queue ! ")
  assert "name=live_enc" in branch
  assert "tee name=tee_live" in branch
  assert "fakesink sync=false" in branch
  assert "key-int-max=15" in branch  # ~1s GOP at the live fps
  assert "bitrate=1500 " in branch  # kbit/s (x264enc seam)


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
