#!/usr/bin/env python3
"""Print the FOV + per-view residuals implied by a cached intrinsics.json.

Usage:
  uv run python scripts/inspect_intrinsics.py [DATASET_ID ...]

Without args, prints every cached calibration under
``object_localization.cache_dir``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from chuck_dreamer.config import load_config
from chuck_dreamer.real.object_localization.runtime import init_from_config
from chuck_dreamer.real.object_localization.types import dataset_cache_dir


def report(path: Path) -> None:
  blob = json.loads(path.read_text())
  W, H = blob["image_size"]
  K = blob["K"]
  fx, fy = K[0][0], K[1][1]
  cx, cy = K[0][2], K[1][2]
  dist = blob["dist"]

  fov_x = 2 * math.degrees(math.atan(W / (2 * fx)))
  fov_y = 2 * math.degrees(math.atan(H / (2 * fy)))
  fov_d = 2 * math.degrees(math.atan(math.hypot(W, H) / (2 * (fx + fy) / 2)))

  print(f"\n=== {path} ===")
  print(f"image_size   : {W} x {H}")
  print(f"K            : fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
  print(f"dist (5)     : k1={dist[0]:+.4f}  k2={dist[1]:+.4f}  "
        f"p1={dist[2]:+.4f}  p2={dist[3]:+.4f}  k3={dist[4]:+.4f}")
  print(f"FOV          : horizontal={fov_x:.2f}°  vertical={fov_y:.2f}°  "
        f"diagonal={fov_d:.2f}°")
  print(f"Principal pt : offset from image center = "
        f"({cx - W/2:+.1f}, {cy - H/2:+.1f}) px")
  print(f"Aspect (fx/fy): {fx/fy:.4f}  (1.0 = square pixels)")
  print(f"global rms_px: {blob.get('rms_px', float('nan')):.3f}  "
        f"n_frames_used={blob.get('n_frames_used', '?')}")

  per_view = blob.get("per_view_rms_px") or []
  prov     = blob.get("view_provenance") or []
  if per_view and prov:
    print("\nper-view residuals (worst first):")
    rows = list(zip(per_view, prov))
    rows.sort(key=lambda x: x[0], reverse=True)
    for rms, p in rows:
      print(f"  rms={rms:6.3f}px  {p.get('dataset_id', '?'):42s} "
            f"f{p.get('frame_idx', -1):>5d}  bucket={tuple(p.get('bucket', []))}")


def main() -> int:
  cfg = load_config()
  ol_cfg = init_from_config(cfg)
  cache_root = Path(ol_cfg.cache_dir)

  ids = sys.argv[1:]
  if ids:
    paths = [dataset_cache_dir(cache_root, did) / "intrinsics.json" for did in ids]
  else:
    paths = sorted(cache_root.glob("*/intrinsics.json"))
  if not paths:
    print(f"no intrinsics.json found under {cache_root}.")
    return 1

  # All paths are the same blob under a pooled fit; show the first one.
  for p in paths[:1]:
    report(p)
  if len(paths) > 1:
    print(f"\n(skipped {len(paths) - 1} other intrinsics.json files; "
          f"they're copies of the pooled fit.)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
