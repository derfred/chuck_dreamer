#!/usr/bin/env python3
"""
World Model Pusher: A Robotics and Machine Learning Project

This is the main entry point for the project. It provides a CLI interface
for training models, processing data, and running experiments.
"""

import click
import logging
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.world_model_pusher.config import load_config, save_config, get_default_config
from src.world_model_pusher.data.tfrecord_utils import TFRecordReader
from src.world_model_pusher.models.mlx_models import WorldModelEncoder
from src.world_model_pusher.training.trainer import Trainer, create_optimizer, create_data_loader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', type=str, help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """World Model Pusher CLI - Robotics ML with MLX."""
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


@cli.command("collect-data")
@click.option("--episodes", default=10, type=int, help="Number of episodes to collect")
@click.option("--output", default=None, type=str, help="Output directory")
@click.option("--difficulty", default=None, type=str, help="Scene difficulty (easy/medium/hard)")
@click.option("--render-size", default=None, type=str, help="Render size WxH (e.g. 128x128)")
@click.option("--seed", default=None, type=int, help="Random seed")
@click.pass_context
def collect_data(ctx, episodes, output, difficulty, render_size, seed):
  """Collect pushing episodes using a random push policy."""
  from dataclasses import asdict
  from tqdm import tqdm

  from src.world_model_pusher.sim import (
    EpisodeWriter,
    PushingEnv,
    SceneBuilder,
    SceneGenerator,
    random_push_policy,
  )

  cfg = _resolve_sim_cfg(ctx, {
    "output_dir": output,
    "difficulty": difficulty,
    "render_size": render_size,
    "seed": seed,
  })
  output             = cfg["output_dir"]
  difficulty         = cfg["difficulty"]
  resolved_seed      = _resolve_seed(cfg, seed)
  render_h, render_w = _parse_render_size(cfg["render_size"])

  builder   = SceneBuilder()
  env       = PushingEnv(builder, render_size=(render_h, render_w))
  generator = SceneGenerator(difficulty=difficulty)
  writer    = EpisodeWriter(output, format="hdf5")
  rng       = np.random.default_rng(resolved_seed)

  click.echo(f"Collecting {episodes} episodes → {output}  (difficulty={difficulty}, seed={resolved_seed})")
  total_reward = 0.0

  for ep_idx in tqdm(range(episodes), desc="Collecting"):
    config       = generator.sample(rng)
    obs, _       = env.reset(config=config)
    episode_data = []
    done         = False

    while not done:
      action = random_push_policy(obs, config, rng)
      next_obs, reward, terminated, truncated, info = env.step(action)
      episode_data.append({
        "pre_image": obs["image"],
        "post_image": next_obs["image"],
        "action": action,
        "reward": reward,
      })
      total_reward += reward
      done = terminated or truncated
      obs = next_obs

    writer.write_episode(
      episode_data,
      metadata={"config": asdict(config), "seed": resolved_seed + ep_idx, "source": "sim"},
    )

  env.close()
  click.echo(f"Done. Avg reward per step: {total_reward / max(1, episodes):.4f}")


@cli.command("rl-loop")
@click.option("--episodes", default=None, type=int, help="Number of episodes")
@click.option("--difficulty", default=None, type=str, help="Scene difficulty (easy/medium/hard)")
@click.option("--seed", default=None, type=int, help="Random seed")
@click.option("--render-size", default=None, type=str, help="Render size WxH (e.g. 128x128)")
@click.pass_context
def rl_loop(ctx, episodes, difficulty, seed, render_size):
  """Run online RL loop with a random policy, logging episode rewards."""
  from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
  )

  from src.world_model_pusher.sim import (
    PushingEnv,
    SceneBuilder,
    SceneGenerator,
    random_push_policy,
  )

  cfg = _resolve_sim_cfg(ctx, {
    "rl_episodes": episodes,
    "rl_difficulty": difficulty,
    "seed": seed,
    "render_size": render_size,
  })
  episodes           = cfg["rl_episodes"]
  difficulty         = cfg["rl_difficulty"]
  resolved_seed      = _resolve_seed(cfg, seed)
  render_h, render_w = _parse_render_size(cfg["render_size"])

  builder   = SceneBuilder()
  env       = PushingEnv(builder, render_size=(render_h, render_w))
  generator = SceneGenerator(difficulty=difficulty)
  rng       = np.random.default_rng(resolved_seed)

  click.echo(f"Starting RL loop: {episodes} episodes, difficulty={difficulty}, seed={resolved_seed}")

  with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TextColumn("ep_reward={task.fields[ep_reward]:.3f} avg={task.fields[avg_reward]:.3f}"),
  ) as progress:
    task = progress.add_task(
      "RL loop", total=episodes, ep_reward=0.0, avg_reward=0.0
    )
    total_reward = 0.0

    for ep_idx in range(episodes):
      config = generator.sample(rng)
      obs, _ = env.reset(config=config)
      done = False
      ep_reward = 0.0

      while not done:
        action = random_push_policy(obs, config, rng)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward
        done = terminated or truncated

      total_reward += ep_reward
      avg_reward = total_reward / (ep_idx + 1)
      progress.update(task, advance=1, ep_reward=ep_reward, avg_reward=avg_reward)

  env.close()
  click.echo(f"Finished. Average episode reward: {total_reward / episodes:.4f}")

@cli.command("show-scene")
@click.option("--difficulty", default=None, type=str, help="Scene difficulty (easy/medium/hard)")
@click.option("--seed", default=None, type=int, help="Random seed (random if omitted)")
@click.option("--render-size", default=None, type=str, help="Render size WxH (e.g. 128x128)")
@click.option("--step-delay", default=0.5, type=float, help="Seconds to sleep between steps (default 0.05)")
@click.pass_context
def show_scene(ctx, difficulty, seed, render_size, step_delay):
  """Generate a scene and run it in the interactive MuJoCo viewer."""
  import time
  import mujoco
  import mujoco.viewer

  from src.world_model_pusher.sim import PushingEnv, SceneBuilder, SceneGenerator, RandomPushPolicy

  cfg = _resolve_sim_cfg(ctx, {"difficulty": difficulty, "seed": seed, "render_size": render_size})
  difficulty         = cfg["difficulty"]
  resolved_seed      = _resolve_seed(cfg, seed)
  render_h, render_w = _parse_render_size(cfg["render_size"])

  builder   = SceneBuilder()
  env       = PushingEnv(builder, render_size=(render_h, render_w))
  generator = SceneGenerator(difficulty=difficulty)
  rng       = np.random.default_rng(resolved_seed)

  click.echo(f"difficulty={difficulty}  seed={resolved_seed}")
  config = generator.sample(rng)
  policy = RandomPushPolicy(config, env.controller, rng)
  obs, _ = env.reset(config=config)

  def key_callback(keycode):
    print(f"Key pressed: {keycode}")
    # if keycode == mujoco.mjtKey.mjKEY_ESCAPE:
    #   return False  # signal to close viewer
    return True

  click.echo("Launching MuJoCo viewer — close the window to exit.")
  with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as v:
    while v.is_running():
      action = policy.act(obs)
      obs, _, terminated, truncated, _ = env.step(action)
      v.sync()
      time.sleep(step_delay)

  env.close()


if __name__ == "__main__":
    cli()
