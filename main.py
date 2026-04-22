#!/usr/bin/env python3
"""
Chuck Dreamer: A Robotics and Machine Learning Project

This is the main entry point for the project. It provides a CLI interface
for training models, processing data, and running experiments.
"""

import click
import logging
import numpy as np

from chuck_dreamer.config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', type=str, help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Chuck Dreamer CLI - Robotics ML with MLX."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)

def _resolve_sim_cfg(ctx, overrides: dict) -> dict:
  """Merge config file sim section with CLI overrides (None values are skipped)."""
  from omegaconf import OmegaConf
  base = OmegaConf.to_container(ctx.obj["config"].sim, resolve=True)
  merged = {k: v for k, v in overrides.items() if v is not None}
  base.update(merged)
  return base

def _resolve_seed(cfg: dict, cli_seed) -> int:
  """Return the seed from cfg if the user passed one via CLI, else a fresh random seed."""
  seed = cfg.get("seed") if cli_seed is not None else None
  if seed is None:
    seed = int(np.random.default_rng().integers(0, 2**31))
  return seed

def _parse_render_size(render_size: str) -> tuple[int, int]:
  """Parse a 'WxH' string and return (render_h, render_w) as expected by PushingEnv."""
  w_str, h_str = render_size.lower().split("x")
  return int(h_str), int(w_str)


@cli.command("generate-scenes")
@click.option("--episodes", default=10, type=int, help="Number of episodes to collect")
@click.option("--output", default=None, type=str, help="Output directory")
@click.option("--difficulty", default=None, type=str, help="Scene difficulty (easy/medium/hard)")
@click.option("--render-size", default=None, type=str, help="Render size WxH (e.g. 128x128)")
@click.option("--seed", default=None, type=int, help="Random seed")
@click.option("--max-steps", default=None, type=int, help="Per-episode step cap (overrides scene config)")
@click.option("--format", "fmt", default=None, type=click.Choice(["hdf5", "rerun"]),
              help="Episode output format (hdf5 or rerun)")
@click.pass_context
def generate_scenes(ctx, episodes, output, difficulty, render_size, seed, max_steps, fmt):
  """Generate pushing scenes using a random push policy."""
  from dataclasses import asdict
  from tqdm import tqdm

  from chuck_dreamer.sim import (
    EpisodeWriter,
    PushingEnv,
    RandomPushPolicy,
    SceneBuilder,
    SceneGenerator,
    ScenePlayer,
  )

  cfg = _resolve_sim_cfg(ctx, {
    "output_dir": output,
    "difficulty": difficulty,
    "render_size": render_size,
    "seed": seed,
    "format": fmt,
  })
  output             = cfg["output_dir"]
  difficulty         = cfg["difficulty"]
  resolved_seed      = _resolve_seed(cfg, seed)
  render_h, render_w = _parse_render_size(cfg["render_size"])
  fmt                = cfg.get("format", "hdf5")

  builder   = SceneBuilder()
  env       = PushingEnv(builder, render_size=(render_h, render_w))
  generator = SceneGenerator(table_size=cfg["table_size"], difficulty=difficulty)
  writer    = EpisodeWriter(output, format=fmt)
  rng       = np.random.default_rng(resolved_seed)

  click.echo(f"Collecting {episodes} episodes → {output}  (difficulty={difficulty}, format={fmt}, seed={resolved_seed})")
  outcome_counts = {"done": 0, "terminated": 0, "timeout": 0, "crashed": 0}

  for ep_idx in tqdm(range(episodes), desc="Collecting"):
    config = generator.sample(rng)
    policy = RandomPushPolicy(config, env.controller, rng)
    player = ScenePlayer(env, policy, config)

    episode_data, outcome = player.run_headless(max_steps=max_steps)
    outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

    if not episode_data:
      click.echo(f"[ep {ep_idx}] crashed on reset, skipping")
      continue

    writer.write_episode(
      episode_data,
      metadata={
        "config":  asdict(config),
        "seed":    resolved_seed + ep_idx,
        "source":  "sim",
        "outcome": outcome,
        "goal_xy": config.goal_pos,
      },
    )

  env.close()
  click.echo(f"Done. Outcomes: {outcome_counts}")

@cli.command("show-scene")
@click.option("--difficulty", default=None, type=str, help="Scene difficulty (easy/medium/hard)")
@click.option("--seed", default=None, type=int, help="Random seed (random if omitted)")
@click.option("--render-size", default=None, type=str, help="Render size WxH (e.g. 128x128)")
@click.option("--step-delay", default=0.05, type=float, help="Seconds to sleep between steps (default 0.05)")
@click.pass_context
def show_scene(ctx, difficulty, seed, render_size, step_delay):
  """Generate a scene and run it in the interactive MuJoCo viewer."""
  import mujoco
  import mujoco.viewer

  from chuck_dreamer.sim import (
    PushingEnv,
    RandomPushPolicy,
    SceneBuilder,
    SceneGenerator,
    ScenePlayer,
  )

  cfg = _resolve_sim_cfg(ctx, {"difficulty": difficulty, "seed": seed, "render_size": render_size})
  difficulty         = cfg["difficulty"]
  resolved_seed      = _resolve_seed(cfg, seed)
  render_h, render_w = _parse_render_size(cfg["render_size"])

  builder   = SceneBuilder()
  env       = PushingEnv(builder, render_size=(render_h, render_w))
  generator = SceneGenerator(table_size=cfg["table_size"], difficulty=difficulty)
  rng       = np.random.default_rng(resolved_seed)

  click.echo(f"difficulty={difficulty}  seed={resolved_seed}")
  config = generator.sample(rng)
  policy = RandomPushPolicy(config, env.controller, rng)
  player = ScenePlayer(env, policy, config)

  env.reset(config=config)

  def key_callback(keycode):
    if keycode == 32 and policy.state == "ready":  # Space bar to start the push
      policy.state = "approach"
      print("Policy state changed: ready → approach")
    return True

  click.echo("Launching MuJoCo viewer — close the window to exit.")
  with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as v:
    player.run_interactive(v, step_delay)

  env.close()


if __name__ == "__main__":
    cli()
