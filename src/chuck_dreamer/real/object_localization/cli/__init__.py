"""Click entry points for the object-localization pipeline."""
from .annotate_mat import annotate_mat_cmd
from .augment_keyframes import augment_keyframes_cmd
from .calibrate_intrinsics import calibrate_intrinsics_cmd
from .prompt_episodes import prompt_episodes_cmd
from .test_calibration import test_calibration_cmd
from .test_object_pose import test_object_pose_cmd

__all__ = [
  "annotate_mat_cmd",
  "augment_keyframes_cmd",
  "calibrate_intrinsics_cmd",
  "prompt_episodes_cmd",
  "test_calibration_cmd",
  "test_object_pose_cmd",
]
