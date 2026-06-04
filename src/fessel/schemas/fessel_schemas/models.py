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


# --- Slice 3 control-plane shapes --------------------------------------------
# supervisor's /state aggregation, forwarded verbatim by the backend's
# /api/state and consumed by the dashboard. The safety state is a placeholder
# in Slice 3 (Slice 6 replaces the placeholder with the real state machine).


class SafetyState(str, Enum):
  """Abbreviated supervisor safety state (architecture §2.4).

  Slice 3 only ever holds whatever the operator's last command implies; the
  real transitions land in Slice 6. The full set is declared here so the
  enum doesn't have to change later.
  """

  INIT = "INIT"
  IDLE = "IDLE"
  RUNNING = "RUNNING"
  PAUSED = "PAUSED"
  STOPPED = "STOPPED"
  SHUTDOWN_ARM = "SHUTDOWN_ARM"
  SHUTDOWN_JETSON = "SHUTDOWN_JETSON"
  DEGRADED = "DEGRADED"


class PlugState(BaseModel):
  """A WiZ plug's last *verified* state (S3.1/S3.3).

  `on` reflects the state read back via getPilot after the last action or
  background refresh — not the last command issued. `verified` is False when
  the most recent verify failed (the command may have been sent but could not
  be confirmed); `verified_at` is the ISO-8601 timestamp of that read.
  """

  on: bool | None = None
  verified: bool = False
  verified_at: str | None = None


class CameraState(BaseModel):
  """Camera up/down, derived from the retained arm/video/state/* topics."""

  up: bool | None = None


class StateResponse(BaseModel):
  """Body of supervisor GET /state and backend GET /api/state (S3.3).

  `jetson` is whatever the Jetson's GET /state returned (opaque to the
  backend and dashboard — supervisor caches it and passes it through);
  `null` when the Jetson has not been reached yet.
  """

  safety_state: SafetyState = SafetyState.IDLE
  jetson: dict | None = None
  plugs: dict[str, PlugState] = Field(default_factory=dict)
  camera: CameraState = Field(default_factory=CameraState)


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
