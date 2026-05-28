"""Pydantic models for the Slice-1 wire shapes.

These models are the source of truth. JSON Schema is exported from them
(see tools/generate-types.sh) and TypeScript is generated for the frontend.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field

# Canonical mode string: "<W>x<H>@<fps>@<bitrate_bps>", e.g. "1280x720@30@2500000".
# '@' separates the three fields; 'x' separates width/height. URL-safe.
_MODE_RE = re.compile(r"^(?P<w>\d+)x(?P<h>\d+)@(?P<fps>\d+)@(?P<bps>\d+)$")


class ModeTriplet(BaseModel):
  """A concrete video operating point.

  `resolution` is the canonical "<W>x<H>" string; fps and bitrate_bps are
  integers. The full canonical form (incl. fps and bitrate) is produced by
  mode_to_canonical and is what travels in the WHEP `mode` query parameter
  and the signed token.
  """

  resolution: str = Field(pattern=r"^\d+x\d+$", description='"<W>x<H>", e.g. "1280x720"')
  fps: int = Field(gt=0)
  bitrate_bps: int = Field(gt=0)


class LiveStateValue(str, Enum):
  off = "off"
  starting = "starting"
  running = "running"
  stopping = "stopping"


class Capabilities(BaseModel):
  """Retained payload on arm/video/capabilities."""

  modes: list[ModeTriplet]


class LiveActivate(BaseModel):
  """Payload of arm/video/cmd/live/activate."""

  path: str
  mode: ModeTriplet


class LiveDeactivate(BaseModel):
  """Payload of arm/video/cmd/live/deactivate."""

  path: str


class LiveState(BaseModel):
  """Retained payload on arm/video/state/live."""

  state: LiveStateValue
  path: str | None = None
  mode: ModeTriplet | None = None


def mode_to_canonical(mode: ModeTriplet) -> str:
  """Serialise a ModeTriplet to the canonical "<W>x<H>@<fps>@<bitrate_bps>" form.

  This is the ONLY place the canonical string is built. The token signer
  (backend) and verifier (mediamtx) both depend on this exact form.
  """
  return f"{mode.resolution}@{mode.fps}@{mode.bitrate_bps}"


def mode_from_canonical(s: str) -> ModeTriplet:
  """Parse the canonical "<W>x<H>@<fps>@<bitrate_bps>" form into a ModeTriplet."""
  m = _MODE_RE.match(s)
  if not m:
    raise ValueError(f"invalid canonical mode string: {s!r}")
  return ModeTriplet(
    resolution=f"{m['w']}x{m['h']}",
    fps=int(m["fps"]),
    bitrate_bps=int(m["bps"]),
  )
