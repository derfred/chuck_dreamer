"""`run` CLI: launch the SO-101 pushing runtime (:class:`Runtime`)."""

from __future__ import annotations

import logging

import click
from omegaconf import OmegaConf

from chuck_dreamer.config import load_config

from ..cli import override_option

logger = logging.getLogger(__name__)

POLICY_SHORTHANDS = {
  "go_to_pose": "chuck_dreamer.runtime.sources:GoToPose",
  "sine_sweep": "chuck_dreamer.runtime.sources:SineSweep",
  "manual":     "chuck_dreamer.runtime.sources:ManualPolicy",
}


def _policy_target(policy: str | None) -> str | None:
  """Resolve a --policy shorthand to an import path; pass anything else through."""
  return None if policy is None else POLICY_SHORTHANDS.get(policy, policy)


@click.command("run")
@click.option("--backend", default=None, type=str,
              help="Backend import path (alias for -o runtime.backend.target=...). "
                   "e.g. chuck_dreamer.runtime.mujoco_backend:MujocoBackend")
@click.option("--policy", "policy", default=None, type=str,
              help="Policy to drive the arm: the shorthands 'go_to_pose' | "
                   "'sine_sweep' | 'manual', or any import path (alias for "
                   "-o runtime.policy.target=...). Params in config that the "
                   "chosen policy does not take are ignored with a warning; "
                   "tune it with -o runtime.policy.params.X=...")
@click.option("--leader-port", "leader_port", default=None, type=str, help="Serial port for the SO-101 leader arm")
@click.option("--viewer/--no-viewer", "viewer", default=None,
              help="Open the MuJoCo 3D viewer (MujocoBackend only). Off by default.")
@click.option("--duration", "duration_s", default=None, type=float,
              help="Run for this many seconds, then stop. Omit to run until Ctrl-C.")
@click.option("--rrd-dir", "rrd_dir", default=None, type=str,
              help="Rerun .rrd output directory "
                   "(alias for -o runtime.logging.rerun.rrd_dir=...).")
@click.option("--seed", default=None, type=int, help="Random seed (random if omitted)")
@override_option
@click.pass_context
def run_cmd(ctx, backend, policy, leader_port, viewer, duration_s, rrd_dir, seed, overrides):
  """Run the runtime: spin the control + policy loops against a backend."""
  from chuck_dreamer.runtime.harness import Runtime

  # --leader-port enables the real leader sensor on that port. The leader is
  # decoupled from the policy, but the common intent is pass-through teleop, so
  # default --policy to manual when it isn't given (an explicit --policy wins).
  leader_enabled = True if leader_port is not None else None
  leader_target = (
    "chuck_dreamer.runtime.teleop:LerobotLeaderReader" if leader_port is not None else None
  )
  if leader_port is not None and policy is None:
    policy = "manual"

  cfg = load_config(
    ctx.obj["config_path"],
    overrides=overrides,
    aliases={
      "seed":                       seed,
      "runtime.backend.target":     backend,
      "runtime.policy.target":      _policy_target(policy),
      "runtime.leader.enabled":     leader_enabled,
      "runtime.leader.target":      leader_target,
      "runtime.leader.params.port": leader_port,
      "runtime.viewer.enabled":     viewer,
      "runtime.duration_s":         duration_s,
      "runtime.logging.rerun.rrd_dir": rrd_dir,
    },
  )
  click.echo("Runtime config:")
  click.echo(OmegaConf.to_yaml(cfg.runtime))
  Runtime(cfg).run()
