"""Runtime version resolution, shared by every Fessel process.

The single source of truth is `fessel-schemas`'s version in its
pyproject.toml. A release bumps that one value; everything else derives
from it (see .github/workflows/fessel-release.yml and build-dpkg.sh), so
the webui image, the Pi dpkg, and the Python packages all carry one
version. This module surfaces it at runtime so a deployed system can be
checked for drift: the cluster webui and the Pi supervisor should report
the same string.

Resolution order (first hit wins):
  1. FESSEL_VERSION env var — explicit override for tests/dev, and what the
     webui container sets from its build-arg (belt-and-suspenders; the
     metadata lookup below would also work in the pip-installed image).
  2. installed `fessel-schemas` package metadata — the pyproject version,
     available wherever the package is pip-installed (the webui image, and
     any uv/venv checkout). This is the canonical source.
  3. /opt/fessel/VERSION — the Pi dpkg ships pure-Python files with NO pip
     install, so there's no package metadata there; build-dpkg.sh writes
     this file from the same pyproject version instead.
  4. "unknown" — none of the above (bare source checkout, package not
     installed, no file).
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_VERSION_FILE = Path("/opt/fessel/VERSION")


def fessel_version() -> str:
  env = os.environ.get("FESSEL_VERSION")
  if env:
    return env.strip()
  try:
    return version("fessel-schemas")
  except PackageNotFoundError:
    pass
  try:
    text = _VERSION_FILE.read_text(encoding="utf-8").strip()
    if text:
      return text
  except OSError:
    pass
  return "unknown"
