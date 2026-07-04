"""YAML config loading shared by the Pi processes.

Requirements §6 specifies a **single** YAML config file on the Pi. The three
processes (video, supervisor, uploader) each need a slice of it: the shared
sections (`mqtt`, `storage`) plus their own component subtree. Rather than
three files (which duplicated `mqtt` and `storage`), the layout is one file:

    mqtt:       {host, port}              # shared by all processes
    storage:    {ssd_path}                # shared by video, supervisor, uploader
    video:      {whip, camera, live, ring, recording}
    supervisor: {http, control}
    uploader:   {driver, ingest_url_base, upload}

`load_component_config(path, component)` returns the flat dict a process
already expects — the shared sections at top level, with the component's own
subtree merged in alongside — so process code keeps reading `config.get("mqtt")`,
`config.get("whip")`, `config.get("control")`, etc. unchanged. A component subtree
key wins over a shared key on collision (a component could override `storage`,
though none does).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Top-level sections shared across processes (not owned by one component).
SHARED_SECTIONS = ("mqtt", "storage")
# The component subtrees a single fessel.yaml carries.
COMPONENTS = ("video", "supervisor", "uploader")


def load_yaml_config(path: str | Path) -> dict[str, Any]:
  """Load a YAML file into a dict; empty/missing file -> {}."""
  p = Path(path)
  if not p.exists():
    return {}
  with p.open() as f:
    return yaml.safe_load(f) or {}


def load_component_config(path: str | Path, component: str) -> dict[str, Any]:
  """Load the single fessel.yaml and return `component`'s flattened view.

  The result is `{**shared_sections, **component_subtree}` — the shape each
  process's existing `config.get(...)` calls already expect. Tolerant of a
  legacy/flat file that has no component subtrees (returns shared + whatever
  top-level keys are present), so a hand-written flat config still loads.
  """
  raw = load_yaml_config(path)
  merged: dict[str, Any] = {}
  for section in SHARED_SECTIONS:
    if section in raw:
      merged[section] = raw[section]
  subtree = raw.get(component)
  if isinstance(subtree, dict):
    merged.update(subtree)
  else:
    # No `component:` subtree present. Treat any non-shared, non-component
    # top-level keys as this component's (supports a flat single-component file
    # in dev/tests, and is a no-op for a well-formed fessel.yaml).
    for key, value in raw.items():
      if key in SHARED_SECTIONS or key in COMPONENTS:
        continue
      merged[key] = value
  return merged
