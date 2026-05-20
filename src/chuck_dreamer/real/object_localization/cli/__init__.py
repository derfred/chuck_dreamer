"""Click entry points for the object-localization pipeline."""
from .calibrate_intrinsics import calibrate_intrinsics_cmd
from .annotate_mat import annotate_mat_cmd

__all__ = ["calibrate_intrinsics_cmd", "annotate_mat_cmd"]
