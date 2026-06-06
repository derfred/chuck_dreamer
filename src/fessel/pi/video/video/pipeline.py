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

# Slice 5: stable names for the vision appsink and the audio level element, so
# the analysis handles can resolve them (pull samples / read bus messages).
VISION_APPSINK_NAME = "vision_sink"
AUDIO_LEVEL_NAME = "audio_level"

# The downscaled resolution the vision branch analyses at (architecture §2.2:
# "videoscale ! video/x-raw,width=640,height=360 ! appsink"). Fixed, not a
# config value — the detectors and the safe_box config are in this coord space.
VISION_WIDTH = 640
VISION_HEIGHT = 360


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
      f'video_bitrate={bitrate_bps},h264_i_frame_period={gop}"'
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


# --- shared tee names (architecture §2.2) ------------------------------------
# The single capture pipeline opens each device ONCE and fans out via a tee:
# tee_v feeds the always-on ring + vision branches and the on-demand live +
# recording branches (added/removed at runtime); tee_a feeds the audio level
# branch. A V4L2 USB camera cannot be opened by multiple v4l2src at once, so
# everything that needs frames must hang off this one tee, not its own source.
VIDEO_TEE_NAME = "tee_v"
AUDIO_TEE_NAME = "tee_a"


def _live_sink_chain(*, path: str, srt_host: str, srt_port: int) -> str:
  srt_uri = f"srt://{srt_host}:{srt_port}?streamid=publish:{path}&pkt_size={SRT_PKT_SIZE}"
  return f'mpegtsmux ! srtsink uri="{srt_uri}" wait-for-connection=false'


def live_branch_chain(
  *,
  mode: ModeTriplet,
  path: str,
  srt_host: str,
  srt_port: int = 8890,
  allow_software_encoder: bool = False,
) -> str:
  """The live branch as a SOURCELESS element chain (queue -> encode -> srtsink),
  to be linked onto a tee_v request pad at runtime (§2.2 dynamic branch).

  The tee fans out raw I420, so each branch encodes independently — the live
  branch keeps its own short-GOP encoder (fast WHEP start) regardless of the
  ring's profile. `name=LIVE_ENCODER_NAME` so the handle can force an IDR.
  """
  prof = encoder_profile(allow_software=allow_software_encoder)
  gop = max(1, mode.fps)  # ~1s keyframe interval -> short GOP for fast WHEP start.
  encoder = _encoder_chain(prof, mode.bitrate_bps, gop, name=LIVE_ENCODER_NAME)
  sink = _live_sink_chain(path=path, srt_host=srt_host, srt_port=srt_port)
  return f"queue ! {encoder} ! {sink}"


def recording_branch_chain(
  *,
  mode: ModeTriplet,
  recording_dir: str,
  bitrate_bps: int,
  segment_seconds: int,
  hls_name: str = RECORDING_HLS_NAME,
  segment_template: str = "seg-%05d.ts",
  allow_software_encoder: bool = False,
) -> str:
  """The explicit-recording branch as a SOURCELESS element chain, to be linked
  onto a tee_v request pad at runtime. Finite HLS (playlist-length=0,
  max-files=0). `hls_name`/`segment_template` are overridable for the anomaly
  forward-capture, which writes into a shared dir with a high segment base."""
  prof = encoder_profile(allow_software=allow_software_encoder)
  gop = _hls_gop(mode.fps, segment_seconds)
  encoder = _encoder_chain(prof, bitrate_bps, gop, name=hls_name + "_enc")
  sink = (
    f"hlssink2 name={hls_name} "
    f"playlist-location={recording_dir}/index.m3u8 "
    f"location={recording_dir}/{segment_template} "
    f"target-duration={segment_seconds} "
    f"playlist-length=0 "
    f"max-files=0"
  )
  return f"queue ! {encoder} ! {sink}"


def build_capture_launch(
  *,
  mode: ModeTriplet,
  ring_dir: str,
  ring_bitrate_bps: int,
  ring_segment_seconds: int,
  ring_playlist_length: int,
  ring_max_files: int,
  device: str = "/dev/video0",
  audio_device: str = "default",
  use_test_source: bool = False,
  allow_software_encoder: bool = False,
  audio_level_interval_ns: int = 500_000_000,
) -> str:
  """The single always-on capture pipeline (architecture §2.2).

  Opens the camera (and mic) ONCE and fans frames out via a tee. Linked at
  build time: the ring-buffer HLS branch and the vision appsink branch off
  `tee_v`, and the audio `level` branch off `tee_a`. The on-demand live and
  recording branches are NOT here — they are added onto `tee_v`'s request pads
  at runtime by the state machines (so activating live never disturbs the
  always-on ring/vision).

  Pure string builder (no gi), so it is unit-testable; build_pipeline() turns
  it into a Gst.Pipeline.
  """
  prof = encoder_profile(allow_software=allow_software_encoder)
  width, height = _resolution_wh(mode)
  video_source = _source_chain(device, width, height, mode.fps, use_test_source)

  # tee_v: raw I420 fan-out. queue per consumer so a slow branch (e.g. disk on
  # the ring) can't back-pressure the others or the source.
  ring_gop = _hls_gop(mode.fps, ring_segment_seconds)
  ring_enc = _encoder_chain(prof, ring_bitrate_bps, ring_gop, name=RING_HLS_NAME + "_enc")
  ring_branch = (
    f"{VIDEO_TEE_NAME}. ! queue ! {ring_enc} ! "
    f"hlssink2 name={RING_HLS_NAME} "
    f"playlist-location={ring_dir}/index.m3u8 "
    f"location={ring_dir}/seg-%05d.ts "
    f"target-duration={ring_segment_seconds} "
    f"playlist-length={ring_playlist_length} "
    f"max-files={ring_max_files}"
  )
  vision_branch = (
    f"{VIDEO_TEE_NAME}. ! queue ! videoscale ! "
    f"video/x-raw,width={VISION_WIDTH},height={VISION_HEIGHT},format=I420 ! "
    f"appsink name={VISION_APPSINK_NAME} emit-signals=true max-buffers=1 drop=true sync=false"
  )

  if use_test_source:
    audio_source = "audiotestsrc is-live=true wave=silence"
  else:
    audio_source = f"alsasrc device={audio_device}"
  audio_branch = (
    f"{AUDIO_TEE_NAME}. ! queue ! "
    f"level name={AUDIO_LEVEL_NAME} interval={audio_level_interval_ns} post-messages=true ! "
    f"fakesink sync=false"
  )

  return (
    f"{video_source} ! tee name={VIDEO_TEE_NAME} "
    f"{ring_branch} {vision_branch} "
    f"{audio_source} ! audioconvert ! audioresample ! tee name={AUDIO_TEE_NAME} "
    f"{audio_branch}"
  )


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
  """Standalone live SRT-push pipeline (source + live branch).

  Retained for the unit tests and as a self-contained reference; the running
  system uses `live_branch_chain` linked onto the shared capture pipeline's
  tee (§2.2) rather than this standalone form.
  """
  prof = encoder_profile(allow_software=allow_software_encoder)
  width, height = _resolution_wh(mode)
  gop = max(1, mode.fps)  # ~1s keyframe interval -> short GOP for fast WHEP start.

  source = _source_chain(device, width, height, mode.fps, use_test_source)
  encoder = _encoder_chain(prof, mode.bitrate_bps, gop, name=LIVE_ENCODER_NAME)
  sink = _live_sink_chain(path=path, srt_host=srt_host, srt_port=srt_port)

  return f"{source} ! {encoder} ! {sink}"


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


def build_vision_launch(
  *,
  mode: ModeTriplet,
  device: str = "/dev/video0",
  use_test_source: bool = False,
) -> str:
  """Launch string for the always-on vision-analysis branch (V5.1).

  Capture -> videoscale to 640x360 -> appsink. No encoder: the frames are
  analysed in-process by OpenCV (the appsink callback), never written or sent,
  so this branch is hardware-encoder-independent (it works on any platform with
  the camera, including dev machines without a hardware H.264 encoder).

  The appsink is configured drop=true max-buffers=1 (plan §V5.1 backpressure: a
  vision thread that can't keep up drops frames at the appsink rather than
  queueing stale ones). `emit-signals=true` so the handle can connect to
  `new-sample`. The 640x360 cap is the coordinate space the safe_box config and
  the detectors operate in.
  """
  source = _source_chain(device, *_resolution_wh(mode), mode.fps, use_test_source)
  scale = f"videoscale ! video/x-raw,width={VISION_WIDTH},height={VISION_HEIGHT},format=I420"
  sink = f"appsink name={VISION_APPSINK_NAME} emit-signals=true max-buffers=1 drop=true sync=false"
  return f"{source} ! {scale} ! {sink}"


def build_audio_launch(
  *,
  device: str = "default",
  use_test_source: bool = False,
  level_interval_ns: int = 500_000_000,
) -> str:
  """Launch string for the always-on audio-analysis branch (V5.6).

  Mic -> audioconvert -> level -> fakesink. The `level` element emits a
  per-interval message on the GStreamer bus carrying RMS + peak dBFS; the audio
  handle reads those messages (no appsink needed for spike detection — plan
  §V5.6 notes the appsink is optional, for waveform archival only). `interval`
  is in nanoseconds (default 0.5s, matching the ~2 Hz publish cadence).

  Dev fallback (`use_test_source`) uses audiotestsrc so the branch builds and
  runs on a machine without a microphone.
  """
  if use_test_source:
    src = "audiotestsrc is-live=true wave=silence"
  else:
    src = f"alsasrc device={device}"
  return (
    f"{src} ! audioconvert ! audioresample "
    f"! level name={AUDIO_LEVEL_NAME} interval={level_interval_ns} post-messages=true "
    f"! fakesink sync=false"
  )


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
