"""Cross-cutting helpers shared across the codebase."""

from pathlib import Path

# Path to the SO-101 URDF used by the FK module
FK_MODEL_PATH = (
  Path(__file__).resolve().parents[3]
  / "assets" / "urdf" / "SO101" / "so101_new_calib.urdf"
)

# Pinned identities for external assets (SAM2 weights, …)
ASSET_MANIFEST_PATH = (
  Path(__file__).resolve().parents[3] / "assets" / "manifest.json"
)


def asset_manifest_entry(name: str) -> dict:
  """One external asset's pinned identity from ``assets/manifest.json``."""
  import json
  manifest = json.loads(ASSET_MANIFEST_PATH.read_text())
  entry    = manifest.get(name)
  if entry is None:
    raise KeyError(
      f"asset {name!r} not in {ASSET_MANIFEST_PATH}; "
      f"known: {sorted(k for k in manifest if not k.startswith('_'))}")
  return dict(entry)
