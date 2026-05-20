"""Shared dataclasses and cache I/O for the object-localization pipeline.

The split between the three pipeline stages — per-camera intrinsics
(deliverable 1), per-recording extrinsics (deliverable 2), per-frame
poses (deliverable 3) — is reflected in three independent on-disk
artifacts. This module owns the dataclass <-> JSON conversion so the
CLIs and the runtime ``ObjectPoseEstimator`` all read and write the
same shape.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class CalibrationMissingError(FileNotFoundError):
  """Raised when a calibration deliverable is missing from the cache."""


@dataclass
class Intrinsics:
  image_size: tuple[int, int]   # (W, H)
  K: np.ndarray                 # (3, 3) float64
  dist: np.ndarray              # (5,) float64
  rms_px: float
  n_frames_used: int

  def to_json(self) -> dict[str, Any]:
    return {
      "image_size":    [int(self.image_size[0]), int(self.image_size[1])],
      "K":             self.K.astype(np.float64).tolist(),
      "dist":          self.dist.astype(np.float64).reshape(-1).tolist(),
      "rms_px":        float(self.rms_px),
      "n_frames_used": int(self.n_frames_used),
    }

  @classmethod
  def from_json(cls, blob: dict[str, Any]) -> "Intrinsics":
    return cls(
      image_size    = (int(blob["image_size"][0]), int(blob["image_size"][1])),
      K             = np.asarray(blob["K"], dtype=np.float64).reshape(3, 3),
      dist          = np.asarray(blob["dist"], dtype=np.float64).reshape(-1),
      rms_px        = float(blob["rms_px"]),
      n_frames_used = int(blob["n_frames_used"]),
    )


@dataclass
class Extrinsics:
  R: np.ndarray                 # (3, 3) float64, world -> camera
  t: np.ndarray                 # (3, 1) float64, mm
  rms_px: float

  def to_json(self) -> dict[str, Any]:
    return {
      "R":      self.R.astype(np.float64).tolist(),
      "t":      self.t.astype(np.float64).reshape(3).tolist(),
      "rms_px": float(self.rms_px),
    }

  @classmethod
  def from_json(cls, blob: dict[str, Any]) -> "Extrinsics":
    return cls(
      R      = np.asarray(blob["R"], dtype=np.float64).reshape(3, 3),
      t      = np.asarray(blob["t"], dtype=np.float64).reshape(3, 1),
      rms_px = float(blob["rms_px"]),
    )


@dataclass
class MatDetection:
  """Raw click + fitted-ellipse output from the mat annotation UI."""
  line1_near:        np.ndarray            # (2,) px — line on +Y, end near origin
  line1_far:         np.ndarray            # (2,) px — line on +Y, end at x=-L
  line2_near:        np.ndarray            # (2,) px — line on -Y, end near origin
  line2_far:         np.ndarray            # (2,) px — line on -Y, end at x=-L
  circle_center:     np.ndarray            # (2,) px — fitted ellipse center
  circle_axes:       tuple[float, float]   # full axis lengths (cv2 convention)
  circle_angle:      float                 # degrees, cv2 convention
  circle_samples_uv: np.ndarray            # (N, 2) px samples along the fitted ellipse
  circle_clicks:     np.ndarray = field(  # (>=5, 2) raw user clicks on the circle outline
    default_factory=lambda: np.zeros((0, 2), dtype=np.float64))

  def to_json(self) -> dict[str, Any]:
    return {
      "line1_near":        self.line1_near.astype(float).tolist(),
      "line1_far":         self.line1_far.astype(float).tolist(),
      "line2_near":        self.line2_near.astype(float).tolist(),
      "line2_far":         self.line2_far.astype(float).tolist(),
      "circle_center":     self.circle_center.astype(float).tolist(),
      "circle_axes":       [float(self.circle_axes[0]), float(self.circle_axes[1])],
      "circle_angle":      float(self.circle_angle),
      "circle_samples_uv": self.circle_samples_uv.astype(float).tolist(),
      "circle_clicks":     self.circle_clicks.astype(float).tolist(),
    }

  @classmethod
  def from_json(cls, blob: dict[str, Any]) -> "MatDetection":
    return cls(
      line1_near        = np.asarray(blob["line1_near"], dtype=np.float64),
      line1_far         = np.asarray(blob["line1_far"], dtype=np.float64),
      line2_near        = np.asarray(blob["line2_near"], dtype=np.float64),
      line2_far         = np.asarray(blob["line2_far"], dtype=np.float64),
      circle_center     = np.asarray(blob["circle_center"], dtype=np.float64),
      circle_axes       = (float(blob["circle_axes"][0]), float(blob["circle_axes"][1])),
      circle_angle      = float(blob["circle_angle"]),
      circle_samples_uv = np.asarray(blob["circle_samples_uv"], dtype=np.float64),
      circle_clicks     = np.asarray(blob.get("circle_clicks", []), dtype=np.float64).reshape(-1, 2),
    )


@dataclass
class CameraCalibration:
  dataset_id: str
  intrinsics: Intrinsics
  extrinsics: Extrinsics

  @classmethod
  def load(cls, cache_dir: Path, dataset_id: str) -> "CameraCalibration":
    """Strict loader. Missing artifacts raise CalibrationMissingError."""
    slug   = dataset_slug(dataset_id)
    root   = Path(cache_dir) / slug
    intr_p = root / "intrinsics.json"
    extr_p = root / "extrinsics.json"
    if not intr_p.exists():
      raise CalibrationMissingError(
        f"{intr_p} not found — run `calibrate-intrinsics {dataset_id}` first.")
    if not extr_p.exists():
      raise CalibrationMissingError(
        f"{extr_p} not found — run `annotate-mat {dataset_id}` first.")
    intrinsics = Intrinsics.from_json(json.loads(intr_p.read_text()))
    extrinsics = Extrinsics.from_json(json.loads(extr_p.read_text()))
    return cls(dataset_id=dataset_id, intrinsics=intrinsics, extrinsics=extrinsics)


def dataset_slug(dataset_id: str) -> str:
  """Filesystem-safe slug for a HF-style ``user/dataset`` id."""
  s = dataset_id.replace("/", "__")
  return re.sub(r"[^A-Za-z0-9_.\-]", "_", s)


def dataset_cache_dir(cache_dir: Path | str, dataset_id: str) -> Path:
  return Path(cache_dir) / dataset_slug(dataset_id)


def write_intrinsics(cache_dir: Path | str, dataset_id: str,
                     intrinsics: Intrinsics, extra: dict[str, Any] | None = None) -> Path:
  root = dataset_cache_dir(cache_dir, dataset_id)
  root.mkdir(parents=True, exist_ok=True)
  blob: dict[str, Any] = intrinsics.to_json()
  if extra:
    blob.update(extra)
  p = root / "intrinsics.json"
  p.write_text(json.dumps(blob, indent=2))
  return p


def read_intrinsics(cache_dir: Path | str, dataset_id: str) -> Intrinsics:
  p = dataset_cache_dir(cache_dir, dataset_id) / "intrinsics.json"
  if not p.exists():
    raise CalibrationMissingError(
      f"{p} not found — run `calibrate-intrinsics` for {dataset_id}.")
  return Intrinsics.from_json(json.loads(p.read_text()))


def write_extrinsics(cache_dir: Path | str, dataset_id: str,
                     extrinsics: Extrinsics, extra: dict[str, Any] | None = None) -> Path:
  root = dataset_cache_dir(cache_dir, dataset_id)
  root.mkdir(parents=True, exist_ok=True)
  blob: dict[str, Any] = extrinsics.to_json()
  if extra:
    blob.update(extra)
  p = root / "extrinsics.json"
  p.write_text(json.dumps(blob, indent=2))
  return p


def read_extrinsics(cache_dir: Path | str, dataset_id: str) -> Extrinsics:
  p = dataset_cache_dir(cache_dir, dataset_id) / "extrinsics.json"
  if not p.exists():
    raise CalibrationMissingError(
      f"{p} not found — run `annotate-mat` for {dataset_id}.")
  return Extrinsics.from_json(json.loads(p.read_text()))


def write_mat_annotation(cache_dir: Path | str, dataset_id: str,
                         detection: MatDetection, extra: dict[str, Any] | None = None) -> Path:
  root = dataset_cache_dir(cache_dir, dataset_id)
  root.mkdir(parents=True, exist_ok=True)
  blob: dict[str, Any] = detection.to_json()
  if extra:
    blob["meta"] = extra
  p = root / "mat_annotation.json"
  p.write_text(json.dumps(blob, indent=2))
  return p


def read_mat_annotation(cache_dir: Path | str, dataset_id: str) -> MatDetection:
  p = dataset_cache_dir(cache_dir, dataset_id) / "mat_annotation.json"
  if not p.exists():
    raise CalibrationMissingError(
      f"{p} not found — run `annotate-mat` (without --review) first.")
  return MatDetection.from_json(json.loads(p.read_text()))


def load_calibration(cache_dir: Path | str, dataset_id: str) -> CameraCalibration:
  """Public alias for ``CameraCalibration.load`` matching the historical API."""
  return CameraCalibration.load(Path(cache_dir), dataset_id)
