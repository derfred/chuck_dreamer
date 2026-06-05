"""GStreamer live pipeline (V1.1 capture core + V1.2 SRT branch) and the
Slice 4 capture branches (V4.1 ring buffer, V4.2 explicit recording).

Built in-process via gi/Gst (never by shelling out to gst-launch).

Every cap below is load-bearing on the Pi hardware:
  - USB webcams output MJPG -> image/jpeg ! jpegdec ! videoconvert is required.
  - v4l2h264enc requires explicit input cap video/x-raw,format=I420.
  - encoder bitrate is set via extra-controls, not a property.
  - output cap video/x-h264,level=(string)4 is required for negotiation.
  - mpegtsmux -> srtsink with streamid=publish:<path> and pkt_size=1316.
  - wait-for-connection=false: srtsink fails fast; reconnect is the state
    machine's job, not the element's.

Slice 4 branch shape (HLS to the USB SSD):
  - Both the ring and the explicit-recording branch hardware-encode H.264 and
    write HLS via hlssink2 (segment .ts files + an index.m3u8 playlist).
  - GOP must equal one segment's worth of frames so every HLS segment begins
    on a keyframe (an HLS requirement), so the GOP is derived from the segment
    duration and fps, not the short ~1s live GOP.
  - The ring rolls (playlist-length + max-files bound the window); an explicit
    recording is finite (playlist-length=0, max-files=0 -> keep everything).

Like build_live_launch, the ring/recording builders are pure string builders
(unit-testable without GStreamer); the same encoder-profile selection
(v4l2h264enc / vtenc_h264 / x264enc test seam) applies to every H.264 branch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fessel_schemas import ModeTriplet

from .platform import BitrateStyle, EncoderProfile, encoder_profile

if TYPE_CHECKING:
  from gi.repository import Gst

log = logging.getLogger(__name__)

# SRT-over-Ethernet MTU-safe payload size; not the default. Larger values
# risk IP fragmentation over cellular paths.
SRT_PKT_SIZE = 1316

# Element name for the live encoder, so the pipeline handle can look it up to
# send a force-key-unit (IDR) event on activation (V2.2). Stable name kept in
# one place so the builder and the handle agree.
LIVE_ENCODER_NAME = "live_enc"

# Stable element names for the Slice-4 HLS branches, so the recording pipeline
# handle can resolve the hlssink2 sink to detect first-segment / finalisation.
RING_HLS_NAME = "ring_hls"
RECORDING_HLS_NAME = "rec_hls"


def _resolution_wh(mode: ModeTriplet) -> tuple[int, int]:
  w, h = (int(x) for x in mode.resolution.split("x"))
  return w, h


def _source_chain(device: str, width: int, height: int, fps: int, use_test_source: bool) -> str:
  """Capture front of the pipeline.

  Real camera: v4l2src MJPG -> jpegdec -> videoconvert -> I420.
  Dev fallback: videotestsrc -> I420 (no camera needed).
  """
  if use_test_source:
    return (
      f"videotestsrc is-live=true pattern=ball "
      f"! video/x-raw,width={width},height={height},framerate={fps}/1 "
      f"! videoconvert "
      f"! video/x-raw,format=I420"
    )
  return (
    f"v4l2src device={device} "
    f"! image/jpeg,width={width},height={height},framerate={fps}/1 "
    f"! jpegdec "
    f"! videoconvert "
    f"! video/x-raw,format=I420"
  )


def _encoder_chain(
  prof: EncoderProfile,
  bitrate_bps: int,
  gop: int,
  name: str = LIVE_ENCODER_NAME,
) -> str:
  """H.264 encode parameterised by bitrate + GOP, per encoder style.

  `name` lets each branch name its encoder (the live branch uses
  `LIVE_ENCODER_NAME` so the pipeline handle can resolve it and send a
  force-key-unit (IDR) event on activation, V2.2; the ring/recording branches
  use their own names). Takes a plain bitrate rather than a ModeTriplet so the
  ring/recording branches can encode at a profile bitrate independent of the
  live mode.
  """
  if prof.bitrate_style is BitrateStyle.EXTRA_CONTROLS:
    # v4l2h264enc: bitrate + GOP via the control string.
    enc = (
      f'{prof.element} name={name} extra-controls="controls,'
      f"video_bitrate={bitrate_bps},h264_i_frame_period={gop}\""
    )
  elif prof.is_software:
    # x264enc (test-only): bitrate in kbit/s; key-int-max gives the short
    # GOP; zerolatency for low-latency streaming; baseline for broad decode.
    enc = (
      f"{prof.element} name={name} bitrate={bitrate_bps // 1000} "
      f"key-int-max={gop} tune=zerolatency speed-preset=ultrafast"
    )
  else:
    # vtenc_h264 (macOS dev): bitrate is a plain property (bits/s);
    # max-keyframe-interval gives the short GOP.
    enc = (
      f"{prof.element} name={name} bitrate={bitrate_bps // 1000} "
      f"max-keyframe-interval={gop} realtime=true allow-frame-reordering=false"
    )
  return f"{enc} ! {prof.output_caps} ! h264parse config-interval=1"


def build_live_launch(
  *,
  mode: ModeTriplet,
  path: str,
  srt_host: str,
  srt_port: int = 8890,
  device: str = "/dev/video0",
  use_test_source: bool = False,
  allow_software_encoder: bool = False,
) -> str:
  """Return the gst-parse launch string for the live SRT-push branch.

  Kept as a pure string builder so it is unit-testable without a running
  pipeline; build_pipeline() turns it into a Gst.Pipeline.

  `allow_software_encoder` is the test-only seam (x264enc); it defaults
  False so production keeps fail-loud hardware-only behaviour.
  """
  prof = encoder_profile(allow_software=allow_software_encoder)
  width, height = _resolution_wh(mode)
  gop = max(1, mode.fps)  # ~1s keyframe interval -> short GOP for fast WHEP start.

  source = _source_chain(device, width, height, mode.fps, use_test_source)
  encoder = _encoder_chain(prof, mode.bitrate_bps, gop, name=LIVE_ENCODER_NAME)
  srt_uri = (
    f"srt://{srt_host}:{srt_port}?streamid=publish:{path}&pkt_size={SRT_PKT_SIZE}"
  )
  sink = f'srtsink uri="{srt_uri}" wait-for-connection=false'

  return f"{source} ! {encoder} ! mpegtsmux ! {sink}"


def _hls_gop(fps: int, segment_seconds: int) -> int:
  """GOP = one segment's worth of frames, so every HLS segment starts on a
  keyframe (an HLS requirement). At least 1."""
  return max(1, fps * max(1, segment_seconds))


def build_ring_launch(
  *,
  mode: ModeTriplet,
  ring_dir: str,
  bitrate_bps: int,
  segment_seconds: int,
  playlist_length: int,
  max_files: int,
  device: str = "/dev/video0",
  use_test_source: bool = False,
  allow_software_encoder: bool = False,
) -> str:
  """Launch string for the always-on ring-buffer branch (V4.1).

  A standalone capture->encode->hlssink2 pipeline writing a rolling HLS
  playlist to `ring_dir`. `playlist-length` + `max-files` bound the window;
  `max-files` should be slightly larger than `playlist-length` so segments
  leaving the playlist are deleted (hlssink2 deletes past `max-files`) rather
  than retained indefinitely.

  `mode` provides resolution + fps (the camera operating point); `bitrate_bps`
  is the ring's own (review-quality) profile, independent of any live mode.
  """
  prof = encoder_profile(allow_software=allow_software_encoder)
  width, height = _resolution_wh(mode)
  gop = _hls_gop(mode.fps, segment_seconds)

  source = _source_chain(device, width, height, mode.fps, use_test_source)
  encoder = _encoder_chain(prof, bitrate_bps, gop, name=RING_HLS_NAME + "_enc")
  sink = (
    f"hlssink2 name={RING_HLS_NAME} "
    f"playlist-location={ring_dir}/index.m3u8 "
    f"location={ring_dir}/seg-%05d.ts "
    f"target-duration={segment_seconds} "
    f"playlist-length={playlist_length} "
    f"max-files={max_files}"
  )
  return f"{source} ! {encoder} ! {sink}"


def build_recording_launch(
  *,
  mode: ModeTriplet,
  recording_dir: str,
  bitrate_bps: int,
  segment_seconds: int,
  device: str = "/dev/video0",
  use_test_source: bool = False,
  allow_software_encoder: bool = False,
) -> str:
  """Launch string for an explicit (operator) recording branch (V4.2).

  A standalone capture->encode->hlssink2 pipeline writing a *finite* HLS
  recording to `recording_dir`. `playlist-length=0` + `max-files=0` are the
  documented hlssink2 values for "keep everything, write a finite playlist":
  recordings are bounded by start/stop, not by a rolling window. A typically
  higher (retention-quality) `bitrate_bps` than the ring.
  """
  prof = encoder_profile(allow_software=allow_software_encoder)
  width, height = _resolution_wh(mode)
  gop = _hls_gop(mode.fps, segment_seconds)

  source = _source_chain(device, width, height, mode.fps, use_test_source)
  encoder = _encoder_chain(prof, bitrate_bps, gop, name=RECORDING_HLS_NAME + "_enc")
  sink = (
    f"hlssink2 name={RECORDING_HLS_NAME} "
    f"playlist-location={recording_dir}/index.m3u8 "
    f"location={recording_dir}/seg-%05d.ts "
    f"target-duration={segment_seconds} "
    f"playlist-length=0 "
    f"max-files=0"
  )
  return f"{source} ! {encoder} ! {sink}"


def build_pipeline(launch: str) -> "Gst.Pipeline":
  """Parse a launch string into a Gst.Pipeline.

  Imports gi lazily so the launch-string builders above stay testable
  without GStreamer. Fails loud (raises) if the encoder element is missing
  — Gst.parse_launch raises GLib.Error.
  """
  import gi

  gi.require_version("Gst", "1.0")
  from gi.repository import Gst

  Gst.init(None)
  return Gst.parse_launch(launch)
