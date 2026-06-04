"""`run` CLI: launch the SO-101 pushing runtime (:class:`Runtime`)."""

from __future__ import annotations

import logging

import click
from omegaconf import OmegaConf

from chuck_dreamer.config import load_config

from ..cli import override_option

logger = logging.getLogger(__name__)


@click.command("run")
@click.option("--backend", default=None, type=str,
              help="Backend import path (alias for -o runtime.backend.target=...). "
                   "e.g. chuck_dreamer.runtime.mujoco_backend:MujocoBackend")
@click.option("--source", default=None, type=str,
              help="Scripted setpoint source kind: 'go_to_pose' | 'sine_sweep' "
                   "(alias for -o runtime.source.kind=...).")
@click.option("--viewer/--no-viewer", "viewer", default=None,
              help="Open the MuJoCo 3D viewer (MujocoBackend only). Off by default.")
@click.option("--duration", "duration_s", default=None, type=float,
              help="Run for this many seconds, then stop. Omit to run until Ctrl-C.")
@click.option("--log-path", "log_path", default=None, type=str,
              help="CSV telemetry output path (alias for -o runtime.logging.csv_path=...).")
@click.option("--seed", default=None, type=int, help="Random seed (random if omitted)")
@override_option
@click.pass_context
def run_cmd(ctx, backend, source, viewer, duration_s, log_path, seed, overrides):
  """Run the runtime: spin the control + policy loops against a backend."""
  from chuck_dreamer.runtime.harness import Runtime

  cfg = load_config(
    ctx.obj["config_path"],
    overrides=overrides,
    aliases={
      "seed":                    seed,
      "runtime.backend.target":  backend,
      "runtime.source.kind":     source,
      "runtime.viewer.enabled":  viewer,
      "runtime.duration_s":      duration_s,
      "runtime.logging.csv_path": log_path,
    },
  )
  click.echo("Runtime config:")
  click.echo(OmegaConf.to_yaml(cfg.runtime))
  Runtime(cfg).run()
