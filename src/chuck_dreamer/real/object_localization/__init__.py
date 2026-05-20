"""Monocular object-localization pipeline.

Three deliverables (see ``object_localization_spec.md``):

  1. ``calibrate-intrinsics`` CLI — per-camera intrinsics calibration.
  2. ``annotate-mat`` CLI — per-dataset extrinsics calibration.
  3. ``ObjectPoseEstimator`` — per-frame object pose estimation.
"""
from .estimator import ObjectPose, ObjectPoseEstimator
from .runtime import ObjectLocalizationConfig, active, init_from_config
from .types import (
  CalibrationMissingError,
  CameraCalibration,
  Extrinsics,
  Intrinsics,
  MatDetection,
  dataset_cache_dir,
  dataset_slug,
  load_calibration,
  read_extrinsics,
  read_intrinsics,
  read_mat_annotation,
  write_extrinsics,
  write_intrinsics,
  write_mat_annotation,
)

__all__ = [
  "ObjectPose",
  "ObjectPoseEstimator",
  "ObjectLocalizationConfig",
  "active",
  "init_from_config",
  "CalibrationMissingError",
  "CameraCalibration",
  "Extrinsics",
  "Intrinsics",
  "MatDetection",
  "dataset_cache_dir",
  "dataset_slug",
  "load_calibration",
  "read_extrinsics",
  "read_intrinsics",
  "read_mat_annotation",
  "write_extrinsics",
  "write_intrinsics",
  "write_mat_annotation",
]
