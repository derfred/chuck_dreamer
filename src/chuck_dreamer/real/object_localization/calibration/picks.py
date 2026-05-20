"""On-disk picks: hand-selected frames + bboxes for pooled calibration.

Each dataset's picks live at ``<cache>/<slug>/picks.json`` with this
shape::

    {
      "dataset_id": "user/dataset",
      "picks": [
        {"frame_idx": 240, "bbox": [620, 180, 1290, 980]},
        ...
      ]
    }

The bbox is ``[x0, y0, x1, y1]`` in full-image pixel coords. It's
optional per pick — None / missing means "search the whole frame" at
calibration time. The picker UI writes this file; the calibrator reads
it via ``load_picks`` and feeds it through ``DatasetSpec``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..types import dataset_cache_dir


@dataclass
class Pick:
  frame_idx: int
  bbox: tuple[int, int, int, int] | None = None


def picks_path(cache_dir: Path | str, dataset_id: str) -> Path:
  return dataset_cache_dir(cache_dir, dataset_id) / "picks.json"


def load_picks(cache_dir: Path | str, dataset_id: str) -> list[Pick]:
  """Read picks.json. Returns ``[]`` if the file doesn't exist."""
  p = picks_path(cache_dir, dataset_id)
  if not p.exists():
    return []
  blob = json.loads(p.read_text())
  out: list[Pick] = []
  for entry in blob.get("picks", []):
    bbox = entry.get("bbox")
    out.append(Pick(
      frame_idx = int(entry["frame_idx"]),
      bbox      = (tuple(int(x) for x in bbox) if bbox else None),
    ))
  return out


def save_picks(cache_dir: Path | str, dataset_id: str, picks: list[Pick]) -> Path:
  """Write picks.json. Stable ordering by ``frame_idx``."""
  p = picks_path(cache_dir, dataset_id)
  p.parent.mkdir(parents=True, exist_ok=True)
  ordered = sorted(picks, key=lambda x: x.frame_idx)
  blob = {
    "dataset_id": dataset_id,
    "picks": [
      {"frame_idx": pk.frame_idx,
       "bbox":      (list(pk.bbox) if pk.bbox else None)}
      for pk in ordered
    ],
  }
  p.write_text(json.dumps(blob, indent=2))
  return p
