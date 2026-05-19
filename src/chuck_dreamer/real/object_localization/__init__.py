"""Object localization pipeline (chuck_dreamer.real.object_localization).

Two deliverables live here:

* :mod:`.calibration` — fit intrinsics + extrinsics per camera position
  from LeRobot episodes 0 (empty mat) and 1 (checkerboard).
* :mod:`.pose` — render-and-compare pose estimation for the
  rhombicuboctahedron, producing world-frame ``(x, y)`` with mm precision.

All runtime constants live under ``cfg.object_localization`` (see
``configs/default.yaml``). Call :func:`init_from_config` once before
using any other entry point — the chuck_dreamer ``main.py`` CLI does
this for you on every command.
"""

from .calibration import CameraCalibration, analyze_datasets, load_calibration
from .constants import ObjectLocalizationConfig, active, init_from_config
from .pose import ObjectPose, ObjectPoseEstimator, SegmentationPrompt

__all__ = [
  "CameraCalibration",
  "ObjectLocalizationConfig",
  "ObjectPose",
  "ObjectPoseEstimator",
  "SegmentationPrompt",
  "active",
  "analyze_datasets",
  "init_from_config",
  "load_calibration",
]
