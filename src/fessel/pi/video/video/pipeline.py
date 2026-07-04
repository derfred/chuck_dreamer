"""GStreamer capture pipeline (V1.1 capture core + warm live encoder, §2.2)
and the Slice 4 capture branches (V4.1 ring buffer, V4.2 explicit recording).

Built in-process via gi/Gst (never by shelling out to gst-launch).

Every cap below is load-bearing on the Pi hardware:
  - USB webcams output MJPG -> image/jpeg ! jpegdec ! videoconvert is required.
  - v4l2h264enc requires explicit input cap video/x-raw,format=I420.
  - encoder bitrate is set via extra-controls, not a property.
  - output cap video/x-h264,level=(string)4 is required for negotiation.

Live path (architecture §2.2, warm-but-detached / Variant B):
  - A dedicated live encoder at a FIXED deploy-time profile is built into the
    always-on capture pipeline and runs warm whenever the pipeline is up:
    `queue ! videoscale ! videorate ! <caps> ! <hw enc> ! h264parse
    config-interval=1 ! tee name=tee_live`, with a permanently linked
    `queue ! fakesink` drop branch so the encoder output flows nowhere while
    unwatched (zero uplink, no cold start on first viewer).
  - On activation, ONLY the WHIP sender sub-branch (`queue ! rtph264pay !
    whipclientsink`) is linked onto tee_live; whipclientsink owns its WebRTC
    session lifecycle — attaching IS the WHIP handshake, detaching tears the
    session down. Reconnect after an uplink blip is the state machine's job.

Slice 4 branch shape (HLS to the USB SSD):
  - Both the ring and the explicit-recording branch hardware-encode H.264 and
    write HLS via hlssink2 (segment .ts files + an index.m3u8 playlist).
  - GOP must equal one segment's worth of frames so every HLS segment begins
    on a keyframe (an HLS requirement), so the GOP is derived from the segment
    duration and fps, not the short ~1s live GOP.
  - The ring rolls (playlist-length + max-files bound the window); an explicit
    recording is finite (playlist-length=0, max-files=0 -> keep everything).

The ring/recording/live builders are pure string builders (unit-testable
without GStreamer); the same encoder-profile selection (v4l2h264enc /
vtenc_h264 / x264enc test seam) applies to every H.264 branch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fessel_schemas import ModeTriplet

from .platform import BitrateStyle, EncoderProfile, encoder_profile

if TYPE_CHECKING:
  from gi.repository import Gst

log = logging.getLogger(__name__)

# Element name for the warm live encoder, so the pipeline handle can look it
# up to send a force-key-unit (IDR) event on activation (V2.2). Stable name
# kept in one place so the builder and the handle agree.
LIVE_ENCODER_NAME = "live_enc"

# The warm live encoder's output tee (§2.2). Always present in the capture
# pipeline; the WHIP sender branch is attached to / detached from its request
# pads at runtime. A permanently linked fakesink drop branch keeps the encoder
# flowing while no sender is attached.
LIVE_TEE_NAME = "tee_live"

# Element name for the on-demand whipclientsink, for diagnostics/lookup.
WHIP_SINK_NAME = "live_whip"

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


def _wh(resolution: str) -> tuple[int, int]:
  w, h = (int(x) for x in resolution.split("x"))
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


def live_warm_branch(
  *,
  live_resolution: str,
  live_fps: int,
  live_bitrate_bps: int,
  allow_software_encoder: bool = False,
) -> str:
  """The WARM live-encoder branch built into the capture pipeline (§2.2).

  A dedicated encoder at the fixed deploy-time live profile, always running
  while the pipeline is up: tee_v -> queue -> scale/rate to the live profile
  -> encode (short ~1s GOP for a fast join) -> tee_live -> fakesink. The
  fakesink drop branch is permanently linked so the encoder output flows
  nowhere while unwatched; the WHIP sender (whip_sender_chain) is attached to
  tee_live's request pads on activation. `name=LIVE_ENCODER_NAME` so the
  handle can force an IDR on attach.
  """
  prof = encoder_profile(allow_software=allow_software_encoder)
  width, height = _wh(live_resolution)
  gop = max(1, live_fps)  # ~1s keyframe interval -> short GOP for fast join.
  encoder = _encoder_chain(prof, live_bitrate_bps, gop, name=LIVE_ENCODER_NAME)
  return (
    f"{VIDEO_TEE_NAME}. ! queue ! videoscale ! videorate ! "
    f"video/x-raw,width={width},height={height},framerate={live_fps}/1,format=I420 ! "
    f"{encoder} ! tee name={LIVE_TEE_NAME} "
    f"{LIVE_TEE_NAME}. ! queue ! fakesink sync=false"
  )


def whip_sender_chain(*, whip_endpoint: str) -> str:
  """The on-demand WHIP sender as a SOURCELESS element chain, to be linked
  onto a tee_live request pad on activation (§2.2/§2.3).

  whipclientsink (gst-plugins-rs net/webrtc) owns its WebRTC session: it POSTs
  the WHIP offer to `whip_endpoint` (the webui relay's /whip/ingest), runs
  ICE/DTLS, and streams SRTP. Attaching this branch IS the handshake;
  detaching it tears the session down. Reconnect after an uplink blip is the
  state machine's job (re-attach with backoff), not the element's.
  """
  return (
    f"queue ! rtph264pay ! "
    f'whipclientsink name={WHIP_SINK_NAME} signaller::whip-endpoint="{whip_endpoint}"'
  )


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
  """A SOURCELESS finite-HLS branch (playlist-length=0, max-files=0) to be linked
  onto a tee_v request pad at runtime. `hls_name`/`segment_template` are
  overridable for the anomaly forward-capture, which writes into a shared dir
  with a high segment base.

  TODO(anomaly): this dynamic-add path is broken the same way explicit recording
  was — hlssink2/splitmuxsink fails its muxer→sink link when added to a running
  pipeline. Explicit recording was moved off it to copy-from-ring (the always-on
  ring already segments continuously; GstRecordingPipeline copies the window).
  The anomaly forward-capture (gst_anomaly_handle.py) still uses this dynamic-add
  path and needs the same treatment in a follow-up; left as-is here to keep that
  path compiling and unchanged."""
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
  live_resolution: str = "1280x720",
  live_fps: int = 15,
  live_bitrate_bps: int = 1_500_000,
  device: str = "/dev/video0",
  audio_device: str = "default",
  use_test_source: bool = False,
  allow_software_encoder: bool = False,
  audio_level_interval_ns: int = 500_000_000,
) -> str:
  """The single always-on capture pipeline (architecture §2.2).

  Opens the camera (and mic) ONCE and fans frames out via a tee. Linked at
  build time: the ring-buffer HLS branch, the vision appsink branch, and the
  WARM live-encoder branch (fixed deploy-time profile, ending in tee_live +
  a fakesink drop branch) off `tee_v`, and the audio `level` branch off
  `tee_a`. The on-demand WHIP sender is added onto `tee_live`'s request pad
  at runtime (whip_sender_chain). An EXPLICIT recording is NOT a pipeline
  branch at all:
  the recording state machine copies the ring's segments for the recording
  window out of the always-on ring (GstRecordingPipeline, mirroring the anomaly
  forward-capture) — hlssink2 cannot be added live to a running pipeline, and a
  cold-built idle hlssink2 sink would block the pipeline from prerolling.

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
  live_branch = live_warm_branch(
    live_resolution=live_resolution,
    live_fps=live_fps,
    live_bitrate_bps=live_bitrate_bps,
    allow_software_encoder=allow_software_encoder,
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
    f"{ring_branch} {vision_branch} {live_branch} "
    f"{audio_source} ! audioconvert ! audioresample ! tee name={AUDIO_TEE_NAME} "
    f"{audio_branch}"
  )


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
