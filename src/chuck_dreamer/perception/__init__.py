"""Camera/object perception, shared by the import pipeline and the runtime.

Monocular object localization: SAM2 segmentation + analysis-by-synthesis
pose fitting against calibrated cameras. Self-contained — no imports from
``chuck_dreamer.lerobot`` (dataset access and annotation tooling live there).
"""
from .config import ObjectLocalizationConfig, active, init_from_config
from .estimator import ObjectPose, ObjectPoseEstimator
from .types import (
  CalibrationMissingError,
  CameraCalibration,
  Extrinsics,
  Intrinsics,
  MatDetection,
)

__all__ = [
  "ObjectLocalizationConfig",
  "active",
  "init_from_config",
  "ObjectPose",
  "ObjectPoseEstimator",
  "CalibrationMissingError",
  "CameraCalibration",
  "Extrinsics",
  "Intrinsics",
  "MatDetection",
]
