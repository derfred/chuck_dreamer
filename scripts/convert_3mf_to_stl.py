"""Convert a single-object 3MF file to a centred binary STL.

The 3MF format is just a zip containing an XML mesh definition. We parse the
vertex/triangle list directly, centre the mesh on its bounding-box origin,
convert millimetres to metres, and write a binary STL. This lets MuJoCo (which
has no native 3MF reader) consume the asset as a regular ``<mesh file="...">``.

Usage:
    python scripts/convert_3mf_to_stl.py <input.3mf> <output.stl>
"""

from __future__ import annotations

import re
import struct
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def convert(src: Path, dst: Path) -> None:
  with zipfile.ZipFile(src) as z:
    with z.open("3D/3dmodel.model") as f:
      tree = ET.parse(f)

  root = tree.getroot()
  ns_match = re.match(r"\{.*?\}", root.tag)
  ns = ns_match.group(0) if ns_match else ""

  obj = root.find(f"{ns}resources/{ns}object")
  if obj is None:
    raise ValueError(f"No <object> found in {src}")
  mesh = obj.find(f"{ns}mesh")
  if mesh is None:
    raise ValueError(f"No <mesh> in <object> in {src}")

  vertices = mesh.find(f"{ns}vertices")
  triangles = mesh.find(f"{ns}triangles")
  if vertices is None or triangles is None:
    raise ValueError(f"<mesh> in {src} is missing <vertices>/<triangles>")

  verts = [
      (float(v.get("x", 0.0)), float(v.get("y", 0.0)), float(v.get("z", 0.0)))
      for v in vertices
  ]
  tris = [
      (int(t.get("v1", 0)), int(t.get("v2", 0)), int(t.get("v3", 0)))
      for t in triangles
  ]

  # 3MF declares millimetres; the simulator works in metres.
  s = 1e-3
  verts_m = [(x * s, y * s, z * s) for x, y, z in verts]
  xs = [v[0] for v in verts_m]
  ys = [v[1] for v in verts_m]
  zs = [v[2] for v in verts_m]
  cx, cy, cz = (
      (min(xs) + max(xs)) / 2,
      (min(ys) + max(ys)) / 2,
      (min(zs) + max(zs)) / 2,
  )
  verts_c = [(x - cx, y - cy, z - cz) for x, y, z in verts_m]

  def normal(a, b, c):
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    L = (nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0
    return nx / L, ny / L, nz / L

  dst.parent.mkdir(parents=True, exist_ok=True)
  with open(dst, "wb") as f:
    f.write(b"\0" * 80)
    f.write(struct.pack("<I", len(tris)))
    for (i, j, k) in tris:
      a, b, c = verts_c[i], verts_c[j], verts_c[k]
      nx, ny, nz = normal(a, b, c)
      f.write(struct.pack("<3f", nx, ny, nz))
      for p in (a, b, c):
        f.write(struct.pack("<3f", *p))
      f.write(struct.pack("<H", 0))

  half = max(max(xs) - cx, max(ys) - cy, max(zs) - cz)
  print(f"wrote {dst} ({len(tris)} tris, half-extent {half:.4f} m)")


if __name__ == "__main__":
  if len(sys.argv) != 3:
    print(__doc__, file=sys.stderr)
    sys.exit(2)
  convert(Path(sys.argv[1]), Path(sys.argv[2]))
