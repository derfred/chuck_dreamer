"""Smoke-test that the run module imports and the gripper rad/percent
conversion round-trips. No hardware required."""

import numpy as np

from chuck_dreamer.real.run.arm import (
    Arm,
    ArmConfig,
    LeaderArm,
    LeaderArmConfig,
    _bus_to_rad,
    _rad_to_bus,
)
from chuck_dreamer.real.run.policies import ManualPolicy, ManualPolicyConfig
from chuck_dreamer.real.run.runner import run  # noqa: F401


def main() -> None:
    for r in (-0.174, 0.0, 1.75):
        out = _rad_to_bus(
            np.array([0, 0, 0, 0, 0, r], dtype=np.float32),
            use_degrees=True,
        )
        back = _bus_to_rad(out, use_degrees=True)
        print(f"rad {r:+6.3f} -> pct {out[5]:6.2f} -> rad {back[5]:+6.3f}")
    print("imports ok")


if __name__ == "__main__":
    main()
