"""Mesh loading for the pose estimator.

Supports ``.3mf`` (the supplied unit-stamped format) and ``.stl`` (the
converted asset). 3MF carries its unit in metadata (``unit`` attribute on
the root ``<model>``); we read it and convert vertices to millimeters.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_UNIT_TO_MM = {
  "micron":     1e-3,
  "millimeter": 1.0,
  "millimeters": 1.0,
  "mm":         1.0,
  "centimeter": 10.0,
  "centimeters": 10.0,
  "cm":         10.0,
  "meter":      1000.0,
  "meters":     1000.0,
  "m":          1000.0,
  "inch":       25.4,
  "inches":     25.4,
}


@dataclass
class Mesh:
  vertices_mm: np.ndarray   # (V, 3) float64, centered on bounding-box center, mm
  triangles:   np.ndarray   # (T, 3) int32, indices into vertices_mm
  face_normals: np.ndarray  # (T, 3) float64, unit normals

  def bbox_mm(self) -> tuple[np.ndarray, np.ndarray]:
    return self.vertices_mm.min(axis=0), self.vertices_mm.max(axis=0)


def load_mesh(path: Path) -> Mesh:
  """Load a mesh from ``.3mf`` (preferred) or ``.stl``. Returns mm-units."""
  path = Path(path)
  if path.suffix.lower() == ".3mf":
    return _load_3mf(path)
  if path.suffix.lower() == ".stl":
    return _load_stl(path)
  raise ValueError(f"Unsupported mesh format: {path.suffix} (expected .3mf or .stl)")


def _load_3mf(path: Path) -> Mesh:
  with zipfile.ZipFile(path) as z:
    with z.open("3D/3dmodel.model") as f:
      tree = ET.parse(f)
  root = tree.getroot()
  ns_match = re.match(r"\{.*?\}", root.tag)
  ns = ns_match.group(0) if ns_match else ""

  unit = (root.get("unit") or "millimeter").strip().lower()
  if unit not in _UNIT_TO_MM:
    raise ValueError(f"3MF unit {unit!r} not recognized; expected one of {sorted(_UNIT_TO_MM)}.")
  scale_mm = _UNIT_TO_MM[unit]

  obj = root.find(f"{ns}resources/{ns}object")
  if obj is None:
    raise ValueError(f"No <object> found in {path}")
  mesh = obj.find(f"{ns}mesh")
  if mesh is None:
    raise ValueError(f"No <mesh> in <object> in {path}")

  vlist = mesh.find(f"{ns}vertices")
  tlist = mesh.find(f"{ns}triangles")
  if vlist is None or tlist is None:
    raise ValueError(f"Malformed 3MF mesh in {path}")
  verts = np.array(
    [(float(v.get("x")), float(v.get("y")), float(v.get("z"))) for v in vlist],
    dtype=np.float64) * scale_mm
  tris = np.array(
    [(int(t.get("v1")), int(t.get("v2")), int(t.get("v3"))) for t in tlist],
    dtype=np.int32)
  return _finalize(verts, tris)


def _load_stl(path: Path) -> Mesh:
  try:
    import trimesh
  except Exception as e:
    raise RuntimeError(f"trimesh required to load STL meshes: {e}")
  tm = trimesh.load(str(path), force="mesh")
  if not hasattr(tm, "vertices") or not hasattr(tm, "faces"):
    raise ValueError(f"{path} loaded as something other than a triangle mesh.")
  # STL is unit-less; the converted .stl in this repo is in *meters* (per
  # scripts/convert_3mf_to_stl.py). Assume meters, convert to mm.
  verts = np.asarray(tm.vertices, dtype=np.float64) * 1000.0
  tris  = np.asarray(tm.faces, dtype=np.int32)
  return _finalize(verts, tris)


def _finalize(verts: np.ndarray, tris: np.ndarray) -> Mesh:
  center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
  verts = verts - center
  v0 = verts[tris[:, 0]]
  v1 = verts[tris[:, 1]]
  v2 = verts[tris[:, 2]]
  normals = np.cross(v1 - v0, v2 - v0)
  norms = np.linalg.norm(normals, axis=1, keepdims=True)
  norms = np.where(norms < 1e-9, 1.0, norms)
  normals = normals / norms
  return Mesh(vertices_mm=verts, triangles=tris, face_normals=normals)
