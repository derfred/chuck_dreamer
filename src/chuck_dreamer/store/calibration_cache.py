"""Legacy ``calibration_cache/`` layout I/O.

The provisional persistence layer for calibration artifacts: one directory
per dataset slug holding ``intrinsics.json`` / ``extrinsics.json`` /
``mat_annotation.json``. Both the import pipeline and the runtime read
calibration through this module; the annotation tools write through it.
It is the migration *source* for the artifact store proper
(``docs/trainer/artifact_store.md``) and dissolves into it once that lands.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from chuck_dreamer.perception.types import (
  CalibrationMissingError,
  CameraCalibration,
  Extrinsics,
  Intrinsics,
  MatDetection,
)


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
  """Strict loader for the full camera calibration. Missing artifacts raise
  :class:`CalibrationMissingError` naming the producing command."""
  root   = dataset_cache_dir(cache_dir, dataset_id)
  intr_p = root / "intrinsics.json"
  extr_p = root / "extrinsics.json"
  if not intr_p.exists():
    raise CalibrationMissingError(
      f"{intr_p} not found — run `calibrate-intrinsics {dataset_id}` first.")
  if not extr_p.exists():
    raise CalibrationMissingError(
      f"{extr_p} not found — run `annotate-mat {dataset_id}` first.")
  return CameraCalibration(
    dataset_id=dataset_id,
    intrinsics=Intrinsics.from_json(json.loads(intr_p.read_text())),
    extrinsics=Extrinsics.from_json(json.loads(extr_p.read_text())),
  )
