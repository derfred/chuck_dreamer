"""`calibrate-arm-from-episodes` CLI.

The episode-config → joint-extraction → T_world_arm flow depended on the
``extract_joints`` module, which has been removed. The command is kept
registered (so the CLI surface is stable) but now reports that the
feature is gone rather than failing with an opaque import error.
"""
from __future__ import annotations

import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command("calibrate-arm-from-episodes")
@click.argument("episode_config", type=click.Path(exists=False, dir_okay=False,
                                                  path_type=Path))
@click.pass_context
def calibrate_arm_from_episodes_cmd(ctx, episode_config: Path) -> None:
  """Removed: derived T_world_arm from LeRobot touch episodes.

  This depended on the ``extract_joints`` module, which no longer exists.
  """
  raise click.ClickException(
    "calibrate-arm-from-episodes has been removed (it relied on the "
    "deleted extract_joints module). Use the live-arm calibration flow "
    "instead.")
