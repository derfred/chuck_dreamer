"""Fessel canonical data shapes — single source of truth.

Both the Pi-side processes (video, supervisor) and the cluster-side
backend consume these. The frontend consumes the generated TypeScript
emitted by tools/generate-types.sh; it must never hand-define the shapes.

The canonical mode-triplet string form is fixed here once:

    "<W>x<H>@<fps>@<bitrate_bps>"

e.g. "1280x720@30@2500000". This exact form flows through:
  - the WHEP `mode` query parameter,
  - the HMAC-signed token (over path + mode + expiry),
  - the mediamtx config and the backend token signer.

It is parsed/serialised only via mode_to_canonical / mode_from_canonical
so signer (backend) and verifier (mediamtx) never diverge.
"""

from .models import (
    Capabilities,
    LiveActivate,
    LiveDeactivate,
    LiveState,
    LiveStateValue,
    ModeTriplet,
    mode_from_canonical,
    mode_to_canonical,
)

__all__ = [
    "Capabilities",
    "LiveActivate",
    "LiveDeactivate",
    "LiveState",
    "LiveStateValue",
    "ModeTriplet",
    "mode_from_canonical",
    "mode_to_canonical",
]
