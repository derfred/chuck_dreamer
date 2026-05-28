"""`train` CLI: train a world model with :class:`~chuck_dreamer.trainer.Trainer`."""
from __future__ import annotations

import logging

import click
from omegaconf import OmegaConf

from chuck_dreamer.config import load_config

from ..cli import override_option

logger = logging.getLogger(__name__)


@click.command("train")
@click.option("--name", "experiment_name", default=None, type=str,
              help="Experiment name (alias for -o logging.experiment_name=...). "
                   "Used as the checkpoint subdirectory ({save_dir}/{name}/) and as the logger run name.")
@click.option("--seed", default=None, type=int, help="Random seed (random if omitted)")
@click.option("--resume", "resume", default=None, is_flag=False, flag_value="__auto__",
              help="Resume from a checkpoint. Bare flag uses {save_dir}/{experiment}/latest.safetensors; "
                   "pass a path to load a specific file.")
@override_option
@click.pass_context
def train_cmd(ctx, experiment_name, seed, resume, overrides):
  """Train a model using the specified configuration."""
  from chuck_dreamer.trainer import Trainer

  cfg = load_config(
    ctx.obj["config_path"],
    overrides=overrides,
    aliases={
      "seed":                    seed,
      "logging.experiment_name": experiment_name,
    },
  )
  click.echo("Training with config:")
  click.echo(OmegaConf.to_yaml(cfg))
  trainer = Trainer(cfg)
  resume_arg: bool | str = True if resume == "__auto__" else (resume or False)
  trainer.train(resume=resume_arg)
