"""Platform detection and hardware-encoder selection.

Hardware H.264 only by default: v4l2h264enc on Pi, vtenc_h264 on macOS
(dev). Pipeline construction must fail loud if the platform-appropriate
hardware encoder is unavailable — never silently fall back to software
encoding.

The ONE exception is an explicit, config-gated software-encoder mode used
only by the integration-test container (which has no hardware encoder).
It must be off by default; a unit test asserts that. See
encoder_profile callers and the Slice 1.5 plan (T1.2).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from enum import Enum


class BitrateStyle(Enum):
  # v4l2h264enc: bitrate + GOP via extra-controls string.
  EXTRA_CONTROLS = "extra_controls"
  # vtenc_h264 / x264enc: bitrate as a plain element property.
  PROPERTY = "property"


@dataclass(frozen=True)
class EncoderProfile:
  """How to drive the chosen H.264 encoder on this platform."""

  # GStreamer element name for the encoder.
  element: str
  # How bitrate/GOP are set on the element.
  bitrate_style: BitrateStyle
  # The output caps required after the encoder for negotiation to succeed.
  output_caps: str
  # True only for the explicit test-only software encoder.
  is_software: bool = False


def _is_macos() -> bool:
  return platform.system() == "Darwin"


def _is_linux() -> bool:
  return platform.system() == "Linux"


_SOFTWARE_PROFILE = EncoderProfile(
  element="x264enc",
  bitrate_style=BitrateStyle.PROPERTY,
  output_caps="video/x-h264,level=(string)4",
  is_software=True,
)


def encoder_profile(allow_software: bool = False) -> EncoderProfile:
  """Return the encoder profile for this platform, or raise.

  By default returns the platform hardware encoder and raises
  RuntimeError if none is known — so we never degrade to software encoding
  silently in production.

  `allow_software=True` is the test-only seam: it selects x264enc. It must
  be passed explicitly (defaults False), so the fail-loud production
  behaviour is preserved everywhere the flag isn't set.
  """
  if allow_software:
    return _SOFTWARE_PROFILE
  if _is_linux():
    return EncoderProfile(
      element="v4l2h264enc",
      bitrate_style=BitrateStyle.EXTRA_CONTROLS,
      output_caps="video/x-h264,level=(string)4",
    )
  if _is_macos():
    return EncoderProfile(
      element="vtenc_h264",
      bitrate_style=BitrateStyle.PROPERTY,
      output_caps="video/x-h264,level=(string)4",
    )
  raise RuntimeError(
    f"no hardware H.264 encoder known for platform {platform.system()!r}; "
    "refusing to fall back to software encoding "
    "(set allow_software only in the test image)"
  )
