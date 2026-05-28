"""Cross-cutting helpers shared across the codebase."""

from pathlib import Path

# Path to the SO-101 MuJoCo model used by the FK module. Shared with
# ``_build_arm_calibration`` / ``_doctor_import_lerobot`` in ``lerobot/cli.py``.
FK_MODEL_PATH = (
  Path(__file__).resolve().parents[3]
  / "assets" / "mujoco" / "so101_arm.xml"
)

