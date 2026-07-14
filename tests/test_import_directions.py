"""Import-direction lint (docs/trainer/package_structure.md).

Enforced rules:

  * ``chuck_dreamer.lerobot`` is imported ONLY by ``main.py`` (and tests) —
    no other ``chuck_dreamer`` package may depend on the import pipeline.
  * ``chuck_dreamer.perception`` is self-contained: it may not import
    ``lerobot``, ``store``, or any other feature package.
  * ``chuck_dreamer.common`` is the bottom layer: it may not import any
    feature package.

Scans source text for import statements rather than importing modules, so
optional heavy dependencies (torch, mujoco, lerobot) are never pulled in.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "chuck_dreamer"

# import chuck_dreamer.X / from chuck_dreamer.X import / from ..X import
_ABS = re.compile(r"^\s*(?:from|import)\s+chuck_dreamer\.(\w+)", re.MULTILINE)
_REL = re.compile(r"^\s*from\s+(\.+)(\w*)", re.MULTILINE)


def _imported_packages(path: Path) -> set[str]:
  """Top-level ``chuck_dreamer`` subpackages imported by one module."""
  text = path.read_text()
  pkgs = set(_ABS.findall(text))
  # Resolve relative imports against the module's location.
  parts = path.relative_to(SRC).parts
  for dots, name in _REL.findall(text):
    up = len(dots) - 1
    base = parts[: len(parts) - 1 - up]  # drop filename + `up` packages
    root = (base + (name,))[0] if (base or name) else None
    if root:
      pkgs.add(root)
  return pkgs


def _violations(package: str, forbidden: set[str]) -> list[str]:
  out = []
  for py in (SRC / package).rglob("*.py"):
    bad = _imported_packages(py) & forbidden
    if bad:
      out.append(f"{py.relative_to(SRC)} imports {sorted(bad)}")
  return out


def test_lerobot_is_imported_only_by_main():
  offenders = []
  for py in SRC.rglob("*.py"):
    if py.relative_to(SRC).parts[0] == "lerobot":
      continue
    if "lerobot" in _imported_packages(py):
      offenders.append(str(py.relative_to(SRC)))
  assert not offenders, (
    "only main.py may import chuck_dreamer.lerobot; violations: "
    f"{offenders}")


def test_perception_is_self_contained():
  assert not _violations("perception", {"lerobot", "store", "sim", "training",
                                        "runtime", "eval", "dreamer"})


def test_common_is_bottom_layer():
  assert not _violations("common", {"lerobot", "store", "perception", "sim",
                                    "training", "runtime", "eval", "dreamer"})


def test_store_imports_only_perception_and_common():
  assert not _violations("store", {"lerobot", "sim", "training", "runtime",
                                   "eval", "dreamer"})
