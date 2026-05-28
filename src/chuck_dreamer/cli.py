"""Root Click group and shared decorators for the chuck_dreamer CLI.

The ``cli`` group and ``override_option`` live here (rather than in
``main.py``) so that the per-module command files under
``chuck_dreamer/*/cli.py`` can import them without an import cycle:
``main.py`` imports the command modules, the command modules import
this, and this imports nothing from them.
"""
from __future__ import annotations

import logging

import click

logger = logging.getLogger(__name__)


override_option = click.option(
  "-o", "--override", "overrides", multiple=True, metavar="KEY=VALUE",
  help="Dotted-path config override, repeatable (e.g. -o training.optimizer.wm_lr=3e-4 "
       "-o model.rssm.hidden_size=256). Values are parsed as YAML scalars.",
)


@click.group()
@click.option(
  '--config', '-c', 'config_paths', type=str, multiple=True,
  help=(
    'Path to a YAML config. Repeatable: later files merge on top of '
    'earlier ones, all on top of configs/default.yaml. Use to factor '
    'out machine- or experiment-specific scale-out settings without '
    'editing default.yaml.'
  ),
)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config_paths, verbose):
  """Chuck Dreamer CLI - Robotics ML with MLX."""
  if verbose:
    logging.getLogger().setLevel(logging.DEBUG)

  ctx.ensure_object(dict)
  ctx.obj['config_path'] = list(config_paths)
