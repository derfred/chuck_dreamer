"""Single-file calibration bundle for the live runner.

``annotate-live`` writes a self-contained JSON pairing intrinsics,
extrinsics, and the mat detection used to solve them. The runner reads
this file via ``--calibration`` and reprojects the mat geometry over
its live feed without needing access to a dataset cache.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..types import Extrinsics, Intrinsics, MatDetection


@dataclass
class CalibrationBundle:
  intrinsics: Intrinsics
  extrinsics: Extrinsics
  detection:  MatDetection

  def to_json(self) -> dict[str, Any]:
    return {
      "intrinsics": self.intrinsics.to_json(),
      "extrinsics": self.extrinsics.to_json(),
      "detection":  self.detection.to_json(),
    }

  @classmethod
  def from_json(cls, blob: dict[str, Any]) -> "CalibrationBundle":
    return cls(
      intrinsics = Intrinsics.from_json(blob["intrinsics"]),
      extrinsics = Extrinsics.from_json(blob["extrinsics"]),
      detection  = MatDetection.from_json(blob["detection"]),
    )


def write_bundle(path: Path | str, bundle: CalibrationBundle,
                 extra: dict[str, Any] | None = None) -> Path:
  p = Path(path)
  p.parent.mkdir(parents=True, exist_ok=True)
  blob: dict[str, Any] = bundle.to_json()
  if extra:
    blob["meta"] = extra
  p.write_text(json.dumps(blob, indent=2))
  return p


def read_bundle(path: Path | str) -> CalibrationBundle:
  p = Path(path)
  if not p.exists():
    raise FileNotFoundError(f"calibration bundle not found: {p}")
  return CalibrationBundle.from_json(json.loads(p.read_text()))


def read_loose_intrinsics(path: Path | str) -> Intrinsics:
  """Load an ``intrinsics.json`` file, tolerating the shared-cache schema.

  Per-dataset ``intrinsics.json`` files use ``rms_px`` + ``n_frames_used``.
  The shared ``calibration_cache/intrinsics.json`` at the cache root uses
  ``intrinsic_rms_px`` + ``n_views`` and adds ``source_datasets``. We
  normalize both to the canonical ``Intrinsics`` dataclass.
  """
  p = Path(path)
  if not p.exists():
    raise FileNotFoundError(f"intrinsics file not found: {p}")
  blob = json.loads(p.read_text())
  if "rms_px" not in blob and "intrinsic_rms_px" in blob:
    blob["rms_px"] = blob["intrinsic_rms_px"]
  if "n_frames_used" not in blob and "n_views" in blob:
    blob["n_frames_used"] = blob["n_views"]
  return Intrinsics.from_json(blob)
