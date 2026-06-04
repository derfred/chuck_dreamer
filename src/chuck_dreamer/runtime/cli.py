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
              help="Setpoint source kind: 'go_to_pose' | 'sine_sweep' | 'manual' "
                   "(alias for -o runtime.source.kind=...).")
@click.option("--leader-port", "leader_port", default=None, type=str,
              help="Serial port for the real SO-101 leader arm. Enables the leader "
                   "sensor (the `leader_qpos` modality) and selects the real "
                   "LerobotLeaderReader on that port. The leader is decoupled from "
                   "the policy; pass --source manual (the default when --leader-port "
                   "is given) for pass-through teleop, or another source to merely "
                   "expose the modality.")
@click.option("--viewer/--no-viewer", "viewer", default=None,
              help="Open the MuJoCo 3D viewer (MujocoBackend only). Off by default.")
@click.option("--duration", "duration_s", default=None, type=float,
              help="Run for this many seconds, then stop. Omit to run until Ctrl-C.")
@click.option("--log-path", "log_path", default=None, type=str,
              help="CSV telemetry output path (alias for -o runtime.logging.csv_path=...).")
@click.option("--seed", default=None, type=int, help="Random seed (random if omitted)")
@override_option
@click.pass_context
def run_cmd(ctx, backend, source, leader_port, viewer, duration_s, log_path, seed, overrides):
  """Run the runtime: spin the control + policy loops against a backend."""
  from chuck_dreamer.runtime.harness import Runtime

  # --leader-port enables the real leader sensor on that port. The leader is
  # decoupled from the policy, but the common intent is pass-through teleop, so
  # default --source to manual when it isn't given (an explicit --source wins).
  leader_enabled = True if leader_port is not None else None
  leader_target = (
    "chuck_dreamer.runtime.teleop:LerobotLeaderReader" if leader_port is not None else None
  )
  if leader_port is not None and source is None:
    source = "manual"

  cfg = load_config(
    ctx.obj["config_path"],
    overrides=overrides,
    aliases={
      "seed":                       seed,
      "runtime.backend.target":     backend,
      "runtime.source.kind":        source,
      "runtime.leader.enabled":     leader_enabled,
      "runtime.leader.target":      leader_target,
      "runtime.leader.params.port": leader_port,
      "runtime.viewer.enabled":     viewer,
      "runtime.duration_s":         duration_s,
      "runtime.logging.csv_path":   log_path,
    },
  )
  click.echo("Runtime config:")
  click.echo(OmegaConf.to_yaml(cfg.runtime))
  Runtime(cfg).run()
