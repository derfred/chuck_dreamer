"""CameraCalibration dataclass + JSON load/save."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class CameraCalibration:
  dataset_id: str
  image_size: tuple[int, int]   # (W, H)
  K:    np.ndarray              # 3x3
  dist: np.ndarray              # (5,)  k1 k2 p1 p2 k3
  R:    np.ndarray              # 3x3, world -> camera
  t:    np.ndarray              # (3,)
  intrinsic_rms_px: float
  extrinsic_rms_px: float

  def to_json_dict(self) -> dict:
    d = asdict(self)
    d["image_size"] = list(self.image_size)
    for k in ("K", "dist", "R", "t"):
      d[k] = np.asarray(getattr(self, k)).tolist()
    return d

  @classmethod
  def from_json_dict(cls, d: dict) -> "CameraCalibration":
    return cls(
      dataset_id       = d["dataset_id"],
      image_size       = tuple(d["image_size"]),  # type: ignore[arg-type]
      K                = np.asarray(d["K"], dtype=np.float64),
      dist             = np.asarray(d["dist"], dtype=np.float64).reshape(-1),
      R                = np.asarray(d["R"], dtype=np.float64),
      t                = np.asarray(d["t"], dtype=np.float64).reshape(3),
      intrinsic_rms_px = float(d["intrinsic_rms_px"]),
      extrinsic_rms_px = float(d["extrinsic_rms_px"]),
    )


_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def slugify(dataset_id: str) -> str:
  return _SLUG_RE.sub("__", dataset_id.strip()).strip("_")


def calibration_dir(cache_dir: Path, dataset_id: str) -> Path:
  return Path(cache_dir) / slugify(dataset_id)


def save_calibration(cal: CameraCalibration, cache_dir: Path) -> Path:
  out_dir = calibration_dir(cache_dir, cal.dataset_id)
  out_dir.mkdir(parents=True, exist_ok=True)
  p = out_dir / "calibration.json"
  p.write_text(json.dumps(cal.to_json_dict(), indent=2))
  return p


def load_calibration(cache_dir: Path, dataset_id: str) -> CameraCalibration:
  p = calibration_dir(cache_dir, dataset_id) / "calibration.json"
  if not p.exists():
    raise FileNotFoundError(
      f"no calibration for {dataset_id!r} at {p}. Run `analyze-cameras` first.")
  return CameraCalibration.from_json_dict(json.loads(p.read_text()))
