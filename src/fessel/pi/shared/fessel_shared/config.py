"""YAML config loading shared by the Pi processes.

Requirements §6 specifies a **single** YAML config file on the Pi. The three
processes (video, supervisor, uploader) each need a slice of it: the shared
sections (`mqtt`, `storage`, `webui_base`) plus their own component subtree.
Rather than three files (which duplicated `mqtt` and `storage`), the layout
is one file:

    mqtt:       {host, port}              # shared by all processes
    storage:    {ssd_path}                # shared by video, supervisor, uploader
    webui_base: "http://..."              # shared by video, uploader (the
                                           # webui pod's tailnet endpoint)
    video:      {whip, camera, live, ring, recording}
    supervisor: {http, control}
    uploader:   {driver, ingest_url_base, upload}

`load_component_config(path, component)` returns the flat dict a process
already expects — the shared sections at top level, with the component's own
subtree merged in alongside — so process code keeps reading `config.get("mqtt")`,
`config.get("whip")`, `config.get("control")`, etc. unchanged. A component subtree
key wins over a shared key on collision (a component could override `storage`,
though none does).

`webui_base` is shared like `mqtt`/`storage` (not per-component config) since
video's WHIP/snapshot ingest and the uploader's recording ingest are all the
same webui-pod tailnet endpoint — one YAML key instead of three near-identical
ones. Each site derives its own value from `config["webui_base"]` in code
(video.whip.endpoint appends the fixed `/whip/ingest` path; video.snapshot and
uploader use it directly, each still able to override with their own explicit
`ingest_url_base` in their subtree — e.g. to disable snapshot push).

Any top-level scalar string key is ALSO available as a `${key}` placeholder in
string values anywhere else in the file (recursively through nested
dicts/lists) — useful for ad hoc shared fragments beyond the three built-in
shared sections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Top-level sections shared across processes (not owned by one component).
SHARED_SECTIONS = ("mqtt", "storage", "webui_base")
# The component subtrees a single fessel.yaml carries.
COMPONENTS = ("video", "supervisor", "uploader")


def _substitute(value: Any, variables: dict[str, str]) -> Any:
  if isinstance(value, str):
    for name, replacement in variables.items():
      value = value.replace(f"${{{name}}}", replacement)
    return value
  if isinstance(value, dict):
    return {k: _substitute(v, variables) for k, v in value.items()}
  if isinstance(value, list):
    return [_substitute(v, variables) for v in value]
  return value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
  """Load a YAML file into a dict; empty/missing file -> {}. Every top-level
  scalar string key is available as a `${key}` placeholder substituted into
  string values elsewhere in the file (component subtrees included)."""
  p = Path(path)
  if not p.exists():
    return {}
  with p.open() as f:
    raw = yaml.safe_load(f) or {}
  variables = {k: v for k, v in raw.items() if isinstance(v, str)}
  if variables:
    raw = _substitute(raw, variables)
  return raw


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
