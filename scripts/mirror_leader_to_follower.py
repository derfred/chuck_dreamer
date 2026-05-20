"""Direct leader→follower joint mirror — no cv2, no state machine.

Diagnostic for "the runner doesn't drive the follower": if THIS moves
the follower, the runner state machine is the problem; if it doesn't,
the bug is in Arm.send_joint_qpos / wiring / calibration.

Mirrors at 10 Hz for 30 s, then exits. Each step prints
``leader_qpos | sent_qpos`` in radians so you can see the command.
"""

import time

import numpy as np

from lerobot.utils.robot_utils import precise_sleep

from chuck_dreamer.real.run.arm import (
    Arm,
    ArmConfig,
    LeaderArm,
    LeaderArmConfig,
)


def main() -> None:
    leader = LeaderArm(LeaderArmConfig(
        port="/dev/tty.usbmodem5B141125391",
        calibration_id="my_leader",
    ))
    follower = Arm(ArmConfig(
        port="/dev/tty.usbmodem5B140318661",
        calibration_id="my_follower",
        # Match upstream lerobot-teleoperate: no software clamp on the
        # per-step delta. At 60 Hz the natural leader↔follower sample
        # delta is already tiny, so the clamp only serves to introduce
        # catch-up lag without adding safety.
        max_relative_target=None,
    ))

    leader.connect()
    follower.connect()
    fps = 60
    period = 1.0 / fps
    print(f"connected. Mirroring at {fps} Hz for 30 s. Move the leader.")
    print(f"{'leader (rad)':<48} | {'sent  (rad)':<48}")
    try:
        t0 = time.perf_counter()
        next_print = t0
        while time.perf_counter() - t0 < 30:
            tick = time.perf_counter()
            q = leader.read_joint_qpos()
            try:
                sent = follower.send_joint_qpos(q)
            except Exception as e:  # noqa: BLE001
                print(f"  send_joint_qpos FAILED: {e}")
                sent = np.full_like(q, np.nan)
            # Throttle printing to 5 Hz so the terminal isn't the
            # bottleneck — control loop itself stays at fps.
            if tick >= next_print:
                print(
                    "  ".join(f"{v:+6.2f}" for v in q)
                    + "   |  "
                    + "  ".join(f"{v:+6.2f}" for v in sent)
                )
                next_print = tick + 0.2
            precise_sleep(period - (time.perf_counter() - tick))
    finally:
        follower.disconnect()
        leader.disconnect()


if __name__ == "__main__":
    main()
