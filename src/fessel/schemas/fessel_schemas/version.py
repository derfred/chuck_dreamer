"""Runtime version resolution, shared by every Fessel process.

A release stamps the SAME version into both artifacts (webui image + Pi
dpkg) — see .github/workflows/fessel-release.yml. This module surfaces that
stamp at runtime so a deployed system can be checked for drift: the cluster
webui and the Pi supervisor should report the same string.

Resolution order (first hit wins):
  1. FESSEL_VERSION env var — the webui container sets this from the image
     build-arg; also lets any process override for tests/dev.
  2. /opt/fessel/VERSION — written by the dpkg postinst on the Pi.
  3. "unknown" — running from a source checkout with neither set.
"""

from __future__ import annotations

import os
from pathlib import Path

_VERSION_FILE = Path("/opt/fessel/VERSION")


def fessel_version() -> str:
  env = os.environ.get("FESSEL_VERSION")
  if env:
    return env.strip()
  try:
    text = _VERSION_FILE.read_text(encoding="utf-8").strip()
    if text:
      return text
  except OSError:
    pass
  return "unknown"
