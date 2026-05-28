"""GStreamer live pipeline (V1.1 capture core + V1.2 SRT branch).

Built in-process via gi/Gst (never by shelling out to gst-launch). For
Slice 1 the only branch is the live SRT-push branch; ring buffer,
recording, and vision branches arrive in later slices.

Every cap below is load-bearing on the Pi hardware:
  - USB webcams output MJPG -> image/jpeg ! jpegdec ! videoconvert is required.
  - v4l2h264enc requires explicit input cap video/x-raw,format=I420.
  - encoder bitrate is set via extra-controls, not a property.
  - output cap video/x-h264,level=(string)4 is required for negotiation.
  - mpegtsmux -> srtsink with streamid=publish:<path> and pkt_size=1316.
  - wait-for-connection=false: srtsink fails fast; reconnect is the state
    machine's job, not the element's.
"""

from __future__ import annotations

import logging

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from fessel_schemas import ModeTriplet  # noqa: E402

from .platform import EncoderProfile, encoder_profile  # noqa: E402

log = logging.getLogger(__name__)

# SRT-over-Ethernet MTU-safe payload size; not the default. Larger values
# risk IP fragmentation over cellular paths.
SRT_PKT_SIZE = 1316


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


def _encoder_chain(prof: EncoderProfile, mode: ModeTriplet, gop: int) -> str:
  """Hardware H.264 encode parameterised by the mode triplet."""
  if prof.uses_extra_controls:
    # v4l2h264enc: bitrate + GOP via the control string.
    enc = (
      f'{prof.element} extra-controls="controls,'
      f"video_bitrate={mode.bitrate_bps},h264_i_frame_period={gop}\""
    )
  else:
    # vtenc_h264 (macOS dev): bitrate is a plain property (bits/s);
    # max-keyframe-interval gives the short GOP.
    enc = (
      f"{prof.element} bitrate={mode.bitrate_bps // 1000} "
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
) -> str:
  """Return the gst-parse launch string for the live SRT-push branch.

  Kept as a pure string builder so it is unit-testable without a running
  pipeline; build_pipeline() turns it into a Gst.Pipeline.
  """
  prof = encoder_profile()
  width, height = _resolution_wh(mode)
  gop = max(1, mode.fps)  # ~1s keyframe interval -> short GOP for fast WHEP start.

  source = _source_chain(device, width, height, mode.fps, use_test_source)
  encoder = _encoder_chain(prof, mode, gop)
  srt_uri = (
    f"srt://{srt_host}:{srt_port}?streamid=publish:{path}&pkt_size={SRT_PKT_SIZE}"
  )
  sink = f'srtsink uri="{srt_uri}" wait-for-connection=false'

  return f"{source} ! {encoder} ! mpegtsmux ! {sink}"


def build_pipeline(launch: str) -> Gst.Pipeline:
  """Parse a launch string into a Gst.Pipeline.

  Fails loud (raises) if the hardware encoder element is missing on the
  platform — Gst.parse_launch raises GLib.Error.
  """
  Gst.init(None)
  return Gst.parse_launch(launch)
