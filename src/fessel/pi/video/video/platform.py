"""Platform detection and hardware-encoder selection.

Hardware H.264 only: v4l2h264enc on Pi, vtenc_h264 on macOS (dev).
Pipeline construction must fail loud if the platform-appropriate hardware
encoder is unavailable — never silently fall back to software encoding.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderProfile:
  """How to drive the hardware H.264 encoder on this platform."""

  # GStreamer element name for the hardware encoder.
  element: str
  # True when bitrate is set via extra-controls (v4l2h264enc) rather than a
  # plain element property (vtenc_h264).
  uses_extra_controls: bool
  # The output caps required after the encoder for negotiation to succeed.
  output_caps: str


def _is_macos() -> bool:
  return platform.system() == "Darwin"


def _is_linux() -> bool:
  return platform.system() == "Linux"


def encoder_profile() -> EncoderProfile:
  """Return the hardware-encoder profile for this platform, or raise.

  Raises RuntimeError if there is no known hardware encoder for the
  platform — by design, so we never degrade to software encoding silently.
  """
  if _is_linux():
    return EncoderProfile(
      element="v4l2h264enc",
      uses_extra_controls=True,
      output_caps="video/x-h264,level=(string)4",
    )
  if _is_macos():
    return EncoderProfile(
      element="vtenc_h264",
      uses_extra_controls=False,
      output_caps="video/x-h264,level=(string)4",
    )
  raise RuntimeError(
    f"no hardware H.264 encoder known for platform {platform.system()!r}; "
    "refusing to fall back to software encoding"
  )
