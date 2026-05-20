"""Direct follower probe: read current qpos, command shoulder_pan +0.2 rad,
ramp under max_relative_target, read back. No leader involved.

If the follower physically rotates ~11° at the base, the follower bus is
healthy and any "leader doesn't drive follower" complaint is in the
runner / mirror layer; if it doesn't move, the bug is on the follower
side (torque disabled, calibration, max_relative_target, etc.).
"""

import time

from chuck_dreamer.real.run.arm import Arm, ArmConfig


def main() -> None:
    a = Arm(ArmConfig(
        port="/dev/tty.usbmodem5B140318661",
        calibration_id="my_follower",
        max_relative_target=5.0,
    ))
    a.connect()
    try:
        q = a.read_joint_qpos()
        print("current:", q)
        q[0] += 0.2  # shoulder_pan +0.2 rad ≈ +11°
        print("command:", q)
        for _ in range(20):
            a.send_joint_qpos(q)
            time.sleep(0.1)
        print("final:  ", a.read_joint_qpos())
    finally:
        a.disconnect()


if __name__ == "__main__":
    main()
