"""WHEP token contract — the authoritative signer side.

The token is an HS256 JWT signed with the shared secret. Its signature is
an HMAC over (path, mode, expiry) — exactly the contract the architecture
specifies — expressed as JWT claims so mediamtx can validate it locally
with authMethod:jwt (no callback to the backend).

Claims:
  - path : the mediamtx source path (also the JWT subject)
  - mode : the canonical "<W>x<H>@<fps>@<bitrate_bps>" string (X0.2)
  - exp  : absolute expiry (short TTL; the token is single-use in practice,
           consumed at the WHEP POST)
  - mediamtx permission claim so the JWT also satisfies mediamtx's
    action/path authorization for the read (WHEP) action.

The shared secret is injected via an environment variable (never a config
file, never in code). mediamtx holds the same secret (rendered into its
config); they must come from one source so signer and verifier never
diverge.
"""

from __future__ import annotations

import time

import jwt
from fessel_schemas import ModeTriplet, mode_to_canonical

# Key ID shared between the signed JWT header and the JWKS entry. mediamtx's
# JWKS validation matches the JWT header `kid` against a key in the JWKS, so
# both sides must carry the same kid.
WHEP_KID = "fessel-whep"


def mint_whep_token(
  *,
  secret: str,
  path: str,
  mode: ModeTriplet,
  ttl_seconds: int,
) -> str:
  """Sign an HS256 JWT authorising a WHEP read of `path` at `mode`.

  The signature is an HMAC over the canonical (path, mode, expiry) via the
  JWT claim set. mediamtx recomputes/verifies the HMAC against the same
  secret and checks exp; it reads `mode` from the WHEP query parameter and
  the request path from the URL.
  """
  now = int(time.time())
  claims = {
    "sub": path,
    "path": path,
    "mode": mode_to_canonical(mode),
    "iat": now,
    "exp": now + ttl_seconds,
    # mediamtx JWT authorization: grant the read (WHEP) action on this path.
    "mediamtx_permissions": [{"action": "read", "path": path}],
  }
  return jwt.encode(claims, secret, algorithm="HS256", headers={"kid": WHEP_KID})
