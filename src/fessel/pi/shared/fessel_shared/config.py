"""Minimal YAML config loading shared by the Pi processes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
  p = Path(path)
  if not p.exists():
    return {}
  with p.open() as f:
    return yaml.safe_load(f) or {}
