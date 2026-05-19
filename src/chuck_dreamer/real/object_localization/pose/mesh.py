"""Load and normalize the rhombicuboctahedron mesh."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


_UNIT_TO_MM = {
  "micron":     1e-3,
  "millimeter": 1.0,
  "millimeters": 1.0,
  "mm":         1.0,
  "centimeter": 10.0,
  "centimeters": 10.0,
  "cm":         10.0,
  "inch":       25.4,
  "inches":     25.4,
  "foot":       304.8,
  "feet":       304.8,
  "meter":      1000.0,
  "meters":     1000.0,
  "m":          1000.0,
}


@dataclass
class ObjectMesh:
  vertices_mm: np.ndarray   # (V, 3)
  faces:       np.ndarray   # (F, 3) triangulated
  edges:       np.ndarray   # (E, 2) unique edges
  face_groups: np.ndarray   # (F,) integer, faces of the same source polygon share a group


def _read_unit_from_3mf(path: Path) -> str:
  """Inspect the 3mf zip for the ``unit`` attribute on ``<model>``."""
  import xml.etree.ElementTree as ET
  import zipfile
  with zipfile.ZipFile(path) as z:
    for name in z.namelist():
      if name.endswith(".model") or name.endswith("3dmodel.model"):
        with z.open(name) as f:
          tree = ET.parse(f)
          root = tree.getroot()
          unit = root.attrib.get("unit") or root.attrib.get("{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}unit")
          if unit:
            return unit
  return "millimeter"


def _triangulate_with_groups(geom) -> tuple[np.ndarray, np.ndarray]:
  """Triangulate ``geom`` and return (faces, group_ids).

  ``geom`` is a trimesh.Trimesh. Group IDs identify faces from the same
  original polygon — useful for clustering tri faces back into the
  rhombicuboctahedron's 26 source faces.
  """
  faces = np.asarray(geom.faces, dtype=np.int64)
  # Trimesh stores all-triangle faces; without the original polygon info we
  # rebuild groups via coplanar adjacency.
  group = np.arange(len(faces), dtype=np.int64)
  try:
    facets = geom.facets   # list of arrays of face indices per coplanar group
    for k, f_idxs in enumerate(facets):
      for fi in f_idxs:
        group[fi] = k
    # Faces not in any facet (single triangle facets) get their own ids,
    # already initialized to their face index — relabel for compactness:
    unique: dict[int, int] = {}
    for i, g in enumerate(group):
      gi = int(g)
      if gi not in unique:
        unique[gi] = len(unique)
      group[i] = unique[gi]
  except Exception:
    pass
  return faces, group


def load_mesh(path: str | Path) -> ObjectMesh:
  """Load a ``.3mf`` mesh and return vertices in millimeters."""
  import trimesh

  p = Path(path)
  if not p.exists():
    raise FileNotFoundError(f"mesh not found: {p}")

  loaded = trimesh.load(str(p), file_type="3mf", force="mesh")
  if isinstance(loaded, trimesh.Scene):
    geom: Any = loaded.dump(concatenate=True)
  else:
    geom = loaded

  unit = _read_unit_from_3mf(p).lower()
  if unit not in _UNIT_TO_MM:
    logger.warning("unknown 3mf unit %r — assuming millimeter", unit)
    scale_mm = 1.0
  else:
    scale_mm = _UNIT_TO_MM[unit]

  vertices = np.asarray(geom.vertices, dtype=np.float64) * scale_mm
  faces, group = _triangulate_with_groups(geom)
  n_groups = int(group.max() + 1) if len(group) else 0
  if n_groups != 26:
    logger.warning("rhombicuboctahedron mesh has %d coplanar face groups "
                   "(expected 26) — may be retessellated", n_groups)

  edges = np.asarray(geom.edges_unique, dtype=np.int64)
  return ObjectMesh(vertices_mm=vertices, faces=faces, edges=edges, face_groups=group)
