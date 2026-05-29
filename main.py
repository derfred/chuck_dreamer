#!/usr/bin/env python3
"""
Chuck Dreamer: A Robotics and Machine Learning Project

This is the main entry point for the project. It wires together the
per-module CLI commands (defined alongside the code they drive in
``chuck_dreamer/*/cli.py``) onto the root ``cli`` group.
"""

import os
import platform

if (
  platform.system() == "Linux"
  and "MUJOCO_GL" not in os.environ
  and not os.environ.get("DISPLAY")
):
  os.environ["MUJOCO_GL"] = "egl"

import logging

from chuck_dreamer.cli import cli
from chuck_dreamer.sim.cli import generate_scenes_cmd, show_scene_cmd
from chuck_dreamer.lerobot.cli import import_lerobot_cmd
from chuck_dreamer.training.cli import train_cmd
from chuck_dreamer.eval.cli import eval_cmd
from chuck_dreamer.lerobot.object_localization.cli import (
  annotate_mat_cmd,
  prompt_episodes_cmd,
)
from chuck_dreamer.real.fk_calibration.cli import (
  calibrate_arm_from_episodes_cmd,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for _cmd in (
  generate_scenes_cmd,
  show_scene_cmd,
  import_lerobot_cmd,
  train_cmd,
  eval_cmd,
  annotate_mat_cmd,
  prompt_episodes_cmd,
  calibrate_arm_from_episodes_cmd,
):
  cli.add_command(_cmd)


if __name__ == "__main__":
    cli()
