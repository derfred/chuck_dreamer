"""FK calibration: extract resting joint vectors from touch episodes.

Stage 1 of the FK-calibration pipeline. Reads a LeRobot dataset where
each episode is a single touch (gripper resting on one fiducial mat
point) and emits one ``joint_extraction.json`` per episode into the
dataset's ``calibration_cache/<slug>/`` directory. Stage 2 (forward
kinematics + Umeyama alignment) consumes those JSONs.
"""
