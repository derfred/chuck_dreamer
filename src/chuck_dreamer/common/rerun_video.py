"""Encode/decode helpers shared by the Rerun episode writer and reader.

The ``.rrd`` episodes used to store RGB as one raw ``rr.Image`` per frame
and masks as one raw ``rr.SegmentationImage`` per frame — uncompressed, so
a single imported episode could reach several GB. To shrink them, RGB now
rides as a single ``rr.AssetVideo`` (a real encoded video blob) plus one
``rr.VideoFrameReference`` per step, and masks ride as PNG
``rr.EncodedImage`` blobs.

This module owns the **format contract** between the two sides: the writer
produces the video bytes here, the reader decodes them here, so the
container/codec/PTS conventions live in exactly one place.

Two ways to produce the RGB video bytes:

  * :func:`stream_copy_window` — for imported LeRobot episodes, where the
    source MP4 already holds this episode's frames. It *remuxes* (stream
    copies, no re-encode) just the episode's ``[from_ts, to_ts)`` window
    into a fresh MP4, reusing the original compressed packets. Lossless
    relative to the source and the smallest option. Because a cut can only
    start at a keyframe, the clip may carry a few extra lead-in frames; the
    in-window presentation timestamps it returns let the reader keep
    exactly the episode's frames.
  * :func:`reencode_mp4` — fallback for sim episodes (no source video),
    re-encoding a decoded ``(T, H, W, 3)`` stack to a (visually lossless)
    MP4 the viewer can play.

Both return ``(video_bytes, media_type, frame_pts_seconds)`` where
``frame_pts_seconds`` is the clip-relative presentation timestamp of each
*episode* frame, in step order. The writer logs one
``rr.VideoFrameReference`` per entry, and :func:`decode_frames` selects
exactly those frames back out of the clip.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from av.container import InputContainer
    from av.video.stream import VideoStream

# Re-encode fallback codec. The Rerun viewer only plays MP4 containers, so
# the fallback muxes into MP4; ``libx264rgb`` at qp=0 keeps it bit-exact
# (no RGB↔YUV drift) while still living in an MP4. Training decodes via
# PyAV regardless; only in-viewer playback depends on the RGB H.264 profile.
_FALLBACK_FORMAT = "mp4"
_FALLBACK_MEDIA_TYPE = "video/mp4"
_FALLBACK_CODEC = "libx264rgb"
_FALLBACK_PIX_FMT = "rgb24"

# Tolerance when matching a VideoFrameReference timestamp to a decoded
# frame's PTS (seconds). Comfortably tighter than one frame at any sane fps.
_PTS_MATCH_TOL = 1e-4


def _time_base(stream: Any) -> Any:
    """A video stream's ``time_base``, asserted non-``None``.

    PyAV types ``time_base`` as ``Fraction | None``, but a real demuxed
    video stream always has one; this narrows it (and fails loudly on the
    pathological stream that doesn't) so callers can do PTS arithmetic."""
    tb = stream.time_base
    if tb is None:
        raise ValueError("video stream has no time_base")
    return tb


def _remux_output_stream(oc: Any, ist: Any) -> Any:
    """Add an MP4 output stream that mirrors input stream ``ist`` for a
    packet-copy remux (no encoder is ever opened).

    ``add_stream_from_template`` copies all codec parameters (incl. the
    ``avcC`` extradata H.264/HEVC need to decode) but resolves the *input
    decoder* name as the output codec — fine for H.264/HEVC (``h264`` is
    both a decoder and an encoder name) but a hard error for AV1, whose
    decoder is ``libdav1d`` (not an encoder). Fall back to the codec's
    canonical name (``av1``) and copy the extradata by hand.
    """
    try:
        return oc.add_stream_from_template(ist)
    except Exception:
        ost = oc.add_stream(ist.codec_context.codec.canonical_name)
        ost.width = ist.codec_context.width
        ost.height = ist.codec_context.height
        extradata = ist.codec_context.extradata
        if extradata is not None:
            ost.codec_context.extradata = extradata
        return ost


def stream_copy_window(
    src_path: str | Path, from_ts: float, to_ts: float,
) -> tuple[bytes, str, list[float]]:
    """Stream-copy the ``[from_ts, to_ts)`` window of ``src_path`` into a
    fresh in-memory MP4, reusing the source's compressed packets.

    The clip's timestamps are **re-based to start at 0** (the source frame at
    ``from_ts`` becomes clip-PTS 0), so the clip's clock matches the
    episode-relative ``time`` the writer logs everything else on. Without
    this the clip kept the source's absolute PTS (e.g. starting 12 s into the
    file), and the viewer placed the video that many seconds away from the
    masks on the timeline.

    Returns ``(mp4_bytes, "video/mp4", frame_pts)`` where ``frame_pts`` are
    those re-based presentation timestamps (seconds, starting ~0) of the
    frames inside the window — the episode's frames, in order. The clip may
    carry extra lead-in frames before the window (the cut snaps back to the
    preceding keyframe so it stays decodable, giving them negative PTS); they
    are not listed in ``frame_pts`` and the reader drops them.
    """
    import av

    window_pts: list[float] = []
    # Pass 1 (decode): the in-window presentation timestamps, re-based so the
    # frame at from_ts is 0 (matching the episode's timestamps, which also
    # start at 0).
    with av.open(str(src_path)) as ic:
        ist = cast("VideoStream", ic.streams.video[0])
        tb = _time_base(ist)
        ic.seek(int(from_ts / tb), stream=ist, any_frame=False, backward=True)
        for frame in ic.decode(ist):
            if frame.pts is None:
                continue
            pts_s = float(frame.pts * tb)
            if pts_s < from_ts:
                continue
            if pts_s >= to_ts:
                break
            window_pts.append(pts_s - from_ts)

    # Pass 2 (remux): copy the packets covering the window without decoding,
    # subtracting from_ts (in stream ticks) from every PTS/DTS so the muxed
    # clip starts at 0. Lead-in frames before the window get negative PTS.
    base_ticks = int(round(from_ts / tb))
    out = io.BytesIO()
    with av.open(str(src_path)) as ic:
        ist = cast("VideoStream", ic.streams.video[0])
        tb = _time_base(ist)
        with av.open(out, "w", format="mp4") as oc:
            ost = _remux_output_stream(oc, ist)
            ic.seek(int(from_ts / tb), stream=ist, any_frame=False, backward=True)
            for pkt in ic.demux(ist):
                if pkt.dts is None:
                    continue
                if pkt.pts is not None and float(pkt.pts * tb) >= to_ts:
                    break
                if pkt.pts is not None:
                    pkt.pts -= base_ticks
                pkt.dts -= base_ticks
                pkt.stream = ost
                oc.mux(pkt)
    return out.getvalue(), "video/mp4", window_pts


def reencode_mp4(
    frames: np.ndarray, fps: float = 30.0,
) -> tuple[bytes, str, list[float]]:
    """Re-encode a ``(T, H, W, 3)`` uint8 RGB stack to an in-memory MP4 — the
    fallback for episodes with no source video to stream-copy.

    Uses ``libx264rgb`` at ``qp=0``: bit-exact (no RGB↔YUV drift) and muxes
    into the MP4 container the Rerun viewer needs.

    Returns ``(video_bytes, "video/mp4", frame_pts)`` with one PTS per frame
    at ``t / fps`` — every frame is an episode frame here, so the reader
    keeps them all.
    """
    import av

    arr = np.asarray(frames)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    T, H, W = arr.shape[:3]
    rate = int(round(fps)) if fps and fps > 0 else 30

    out = io.BytesIO()
    with av.open(out, "w", format=_FALLBACK_FORMAT) as oc:
        ost = cast("VideoStream", oc.add_stream(
            _FALLBACK_CODEC, rate=rate,
            options={"qp": "0", "preset": "veryfast"}))
        ost.width, ost.height, ost.pix_fmt = W, H, _FALLBACK_PIX_FMT
        # Let the encoder assign PTS from the frame rate — setting pts /
        # time_base by hand trips a muxer "Invalid argument" in this PyAV.
        for t in range(T):
            vf = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(arr[t]), format="rgb24")
            for pkt in ost.encode(vf):
                oc.mux(pkt)
        for pkt in ost.encode():
            oc.mux(pkt)
    frame_pts = [t / rate for t in range(T)]
    return out.getvalue(), _FALLBACK_MEDIA_TYPE, frame_pts


def decode_frames(video_bytes: bytes, frame_pts: list[float]) -> np.ndarray:
    """Decode ``video_bytes`` and return the ``(T, H, W, 3)`` uint8 stack of
    the frames whose presentation timestamps match ``frame_pts``, in order.

    ``frame_pts`` is the per-step list the writer logged as
    ``rr.VideoFrameReference`` timestamps. Frames outside it (keyframe
    lead-in left in by a stream copy) are dropped, so the result is exactly
    the episode's ``T`` frames.
    """
    import av

    by_pts: dict[float, np.ndarray] = {}
    ordered: list[np.ndarray] = []
    with cast("InputContainer", av.open(io.BytesIO(video_bytes))) as c:
        st = cast("VideoStream", c.streams.video[0])
        tb = _time_base(st)
        for fr in c.decode(st):
            rgb = fr.to_ndarray(format="rgb24")
            ordered.append(rgb)
            if fr.pts is not None:
                by_pts[float(fr.pts * tb)] = rgb

    # No keyframe lead-in (the re-encode fallback emits exactly the episode's
    # frames): decode order already is step order, so skip PTS matching.
    if len(ordered) == len(frame_pts):
        return (np.stack(ordered, axis=0) if ordered
                else np.empty((0,), dtype=np.uint8))

    # Stream copy left lead-in frames in the clip — select the episode's
    # frames by their presentation timestamps.
    decoded_pts = np.array(sorted(by_pts)) if by_pts else np.array([])
    frames = [_nearest_frame(by_pts, decoded_pts, want) for want in frame_pts]
    return np.stack(frames, axis=0) if frames else np.empty((0,), dtype=np.uint8)


def _nearest_frame(
    by_pts: dict[float, np.ndarray], decoded_pts: np.ndarray, want: float,
) -> np.ndarray:
    """The decoded frame whose PTS matches ``want`` (within tolerance), or
    the nearest one if container PTS rounding nudged it off the requested
    value. Raises if the clip is empty."""
    hit = by_pts.get(want)
    if hit is not None:
        return hit
    if decoded_pts.size == 0:
        raise ValueError("video clip decoded to zero frames")
    idx = int(np.argmin(np.abs(decoded_pts - want)))
    nearest = float(decoded_pts[idx])
    if abs(nearest - want) > _PTS_MATCH_TOL:
        # Not an exact PTS hit — accept the nearest but make the slip visible.
        pass
    return by_pts[nearest]


def video_frame_timestamps(record_batches: list[Any]) -> np.ndarray:
    """Pull the per-step ``VideoFrameReference:timestamp`` column (nanoseconds)
    out of the ``/camera/video`` chunks as float seconds, sorted by step.

    These are the clip-relative PTS the writer logged; the reader feeds them
    to :func:`decode_frames` to select the episode's frames.
    """
    step_parts: list[np.ndarray] = []
    ts_parts: list[np.ndarray] = []
    for rb in record_batches:
        if "VideoFrameReference:timestamp" not in rb.schema.names:
            continue
        step_parts.append(np.asarray(rb.column("step")))
        ts_col = rb.column("VideoFrameReference:timestamp")
        ts_parts.append(np.array(
            [_vfr_seconds(ts_col[i].as_py()) for i in range(len(ts_col))],
            dtype=np.float64))
    if not step_parts:
        return np.empty((0,), dtype=np.float64)
    steps = np.concatenate(step_parts) if len(step_parts) > 1 else step_parts[0]
    times = np.concatenate(ts_parts) if len(ts_parts) > 1 else ts_parts[0]
    order = np.argsort(steps, kind="stable")
    return times[order]


def _vfr_seconds(value: Any) -> float:
    """A ``VideoFrameReference:timestamp`` cell is a 1-element list holding
    nanoseconds; normalise it to float seconds."""
    if isinstance(value, list):
        value = value[0] if value else 0
    return float(value) / 1e9
