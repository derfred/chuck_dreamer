"""Operator identity from oauth2-proxy headers (B2.1).

The webui sits behind oauth2-proxy, which runs the GitHub OIDC flow and
forwards already-authenticated requests carrying identity headers
(X-Auth-Request-User / -Email). The backend does NOT run the OIDC code
flow; it trusts these headers when present and treats their absence as
unauthenticated.

Header names are config, not hardcoded:
  FESSEL_AUTH_USER_HEADER    default X-Auth-Request-User
  FESSEL_AUTH_EMAIL_HEADER   default X-Auth-Request-Email
  FESSEL_AUTH_GROUPS_HEADER  default X-Auth-Request-Groups

Trust model (single-listener variant, see webui/deploy/README.md): the
identity headers are set ONLY by oauth2-proxy. Endpoints that bypass the
proxy (`/jwks`) must reject any request that carries them, because a
direct in-cluster caller could otherwise forge an identity. The
`forbid_identity_headers` dependency enforces that.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from fastapi import Request


@dataclass(frozen=True)
class Identity:
  user: str
  email: str | None = None
  groups: tuple[str, ...] = ()


class AuthHeaders:
  """Configured identity-header names (read once at app construction)."""

  def __init__(self) -> None:
    self.user = os.environ.get("FESSEL_AUTH_USER_HEADER", "X-Auth-Request-User")
    self.email = os.environ.get("FESSEL_AUTH_EMAIL_HEADER", "X-Auth-Request-Email")
    self.groups = os.environ.get("FESSEL_AUTH_GROUPS_HEADER", "X-Auth-Request-Groups")

  def names(self) -> tuple[str, ...]:
    return (self.user, self.email, self.groups)

  def read(self, request: Request) -> Identity | None:
    """Return the forwarded operator identity, or None if unauthenticated."""
    user = request.headers.get(self.user)
    if not user:
      return None
    email = request.headers.get(self.email) or None
    raw_groups = request.headers.get(self.groups) or ""
    groups = tuple(g.strip() for g in raw_groups.split(",") if g.strip())
    return Identity(user=user, email=email, groups=groups)
