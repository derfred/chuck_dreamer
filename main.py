#!/usr/bin/env python3
"""
Chuck Dreamer: A Robotics and Machine Learning Project

This is the main entry point for the project. It provides a CLI interface
for training models, processing data, and running experiments.
"""

import click
import logging
from pathlib import Path

from omegaconf import OmegaConf

from chuck_dreamer.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


override_option = click.option(
  "-o", "--override", "overrides", multiple=True, metavar="KEY=VALUE",
  help="Dotted-path config override, repeatable (e.g. -o training.optimizer.wm_lr=3e-4 "
       "-o model.rssm.hidden_size=256). Values are parsed as YAML scalars.",
)


@click.group()
@click.option('--config', '-c', type=str, help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
  """Chuck Dreamer CLI - Robotics ML with MLX."""
  if verbose:
    logging.getLogger().setLevel(logging.DEBUG)

  ctx.ensure_object(dict)
  ctx.obj['config_path'] = config


def _collect_one_episode(cfg, output, fmt, ep_idx, episode_seed):
  """Worker entry point: build env + writer, roll one episode, write it.

  Returns ``(ep_idx, outcome, crashed_on_reset)``. Runs in a child process,
  so it imports the heavy deps lazily and owns its own MuJoCo handles.
  """
  from dataclasses import asdict

  from chuck_dreamer.sim import (
    EpisodeCollector,
    EpisodeWriter,
    PushingEnv,
    ScriptedPolicy,
  )

  worker_cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"seed={episode_seed}"]))

  env    = PushingEnv(worker_cfg)
  policy = ScriptedPolicy(auto_advance_from_ready=True)
  collector = EpisodeCollector(env, policy)
  writer = EpisodeWriter(output, format=fmt)
  try:
    scene = collector.reset()
    if worker_cfg.env.max_steps is not None:
      scene.max_steps = int(worker_cfg.env.max_steps)
    episode_data, outcome = collector.run()

    if episode_data is None:
      return ep_idx, outcome, True

    writer.write_episode(
      episode_data,
      metadata={
        "config":  asdict(scene),
        "seed":    episode_seed,
        "source":  "sim",
        "outcome": outcome,
        "goal_xy": scene.goal_pos,
      },
      name_suffix=f"{ep_idx:05d}",
    )
    return ep_idx, outcome, False
  finally:
    env.close()


@cli.command("generate-scenes")
@click.option("--episodes", default=10, type=int, help="Number of episodes to collect")
@click.option("--output", required=True, type=str, help="Output directory")
@click.option("--seed", default=None, type=int, help="Random seed")
@click.option("--format", "fmt", default="rerun", type=click.Choice(["hdf5", "rerun"]),
              help="Episode output format (hdf5 or rerun)")
@click.option("--workers", default=None, type=int,
              help="Number of parallel worker processes. Default: one per CPU. "
                   "Pass 1 to force the original single-process path; >1 fans episodes "
                   "out over a ProcessPoolExecutor, each worker building its own "
                   "PushingEnv. Each episode's seed is cfg.seed + ep_idx, so output is "
                   "deterministic across worker counts.")
@override_option
@click.pass_context
def generate_scenes(ctx, episodes, output, seed, fmt, workers, overrides):
  """Generate pushing scenes using the scripted heuristic policy."""
  import os

  cfg = load_config(ctx.obj["config_path"], overrides=overrides, aliases={"seed": seed})

  if workers is None:
    workers = os.cpu_count() or 1

  click.echo(f"Collecting {episodes} episodes → {output}  (difficulty={cfg.env.difficulty}, "
             f"format={fmt}, seed={cfg.seed}, workers={workers})")
  outcome_counts: dict[str, int] = {}

  from tqdm import tqdm

  if workers <= 1:
    for ep_idx in tqdm(range(episodes), desc="Collecting"):
      _, outcome, crashed_on_reset = _collect_one_episode(
        cfg, output, fmt, ep_idx, cfg.seed + ep_idx,
      )
      outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
      if crashed_on_reset:
        click.echo(f"[ep {ep_idx}] crashed on reset, skipping")
  else:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=workers) as pool:
      futures = [
        pool.submit(_collect_one_episode, cfg, output, fmt, ep_idx, cfg.seed + ep_idx)
        for ep_idx in range(episodes)
      ]
      for fut in tqdm(as_completed(futures), total=episodes, desc="Collecting"):
        ep_idx, outcome, crashed_on_reset = fut.result()
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        if crashed_on_reset:
          click.echo(f"[ep {ep_idx}] crashed on reset, skipping")

  click.echo(f"Done. Outcomes: {outcome_counts}")


@cli.command("import-lerobot")
@click.argument("repo_id", type=str)
@click.option("--output", required=True, type=str, help="Output directory")
@click.option("--format", "fmt", default="rerun", type=click.Choice(["hdf5", "rerun"]),
              help="Episode output format (hdf5 or rerun)")
@click.option("--video-key", default=None, type=str,
              help="Video feature key (e.g. observation.images.wrist). Defaults to the first one.")
@click.option("--max-episodes", default=None, type=int,
              help="Cap on number of episodes to convert (default: all)")
@click.option("--derive-target-mask/--no-derive-target-mask", default=True,
              help="Derive a segmentation_target mask per frame via white-blob detection "
                   "+ Kalman smoothing (default: on). Provides the focus-loss path with "
                   "a target mask for datasets that ship without segmentation.")
@click.option("--tag", "tags", multiple=True, metavar="TAG",
              help="Tag to stamp onto each written episode's metadata. Repeatable. "
                   "The replay buffer reads these for protected_tags (no-evict) and "
                   "tag_weights (sample upweighting). Typical use: --tag real for "
                   "physical-robot recordings.")
@click.option("--process", "processor_paths", multiple=True,
              type=click.Path(exists=True, dir_okay=False),
              metavar="PATH",
              help="Path to a Python file defining `process(episode, metadata) -> episode`. "
                   "Repeatable; processors run in order as the final step before "
                   "the episode is written. Use for rescaling joint values, computing "
                   "ee_pos via forward kinematics, image cropping, etc.")
@click.pass_context
def import_lerobot(ctx, repo_id, output, fmt, video_key, max_episodes,
                   derive_target_mask, tags, processor_paths):
  """Convert a LeRobot v3 HF dataset (REPO_ID) into the repo's episode files.

  Pulls the parquet + MP4 from Hugging Face, slices per episode using
  ``meta/episodes/*.parquet``, decodes the matching video range, and
  writes one episode file per LeRobot episode under --output.

  Fields the repo's format needs but LeRobot teleop data lacks (reward,
  ee_pos, ee_quat, object_xy) are zero-filled — fine for image-mode
  training, not for state-mode.
  """
  from tqdm import tqdm

  from chuck_dreamer.sim.lerobot_import import import_dataset, load_processor

  processors = tuple(load_processor(p) for p in processor_paths)

  click.echo(
    f"Importing {repo_id} → {output}  (format={fmt}, "
    f"derive_target_mask={derive_target_mask}, tags={list(tags)}, "
    f"processors={list(processor_paths)})"
  )
  count = 0
  for ep_idx, out_path in tqdm(
    import_dataset(
      repo_id, output,
      format=fmt, video_key=video_key, max_episodes=max_episodes,
      derive_target_mask=derive_target_mask, tags=tuple(tags),
      processors=processors,
    ),
    desc="Episodes",
    total=max_episodes,
  ):
    count += 1
    logger.info("episode %d → %s", ep_idx, out_path)
  click.echo(f"Done. Wrote {count} episodes.")


@cli.command("show-scene")
@click.option("--seed", default=None, type=int, help="Random seed (random if omitted)")
@click.option("--step-delay", default=0.05, type=float, help="Seconds to sleep between steps (default 0.05)")
@override_option
@click.pass_context
def show_scene(ctx, seed, step_delay, overrides):
  """Generate a scene and run it in the interactive MuJoCo viewer.

  Drives the scripted policy directly so the user can press Space to
  advance from ``ready`` → ``approach`` and see the heuristic push play
  out, with hint geoms drawn while the policy is in ``ready``.
  """
  import time
  import mujoco
  import mujoco.viewer

  from chuck_dreamer.sim import PushingEnv, ScriptedPolicy

  cfg = load_config(ctx.obj["config_path"], overrides=overrides, aliases={"seed": seed})

  click.echo(f"difficulty={cfg.env.difficulty}  seed={cfg.seed}")
  env    = PushingEnv(cfg)
  policy = ScriptedPolicy()
  scene  = env.generate_scene()
  obs, _ = env.reset(scene=scene)
  policy.reset(scene)

  def key_callback(keycode):
    if keycode == 32 and policy.state == "ready":  # Space bar
      policy.advance_from_ready()
      print("Policy state changed: ready → approach")
    return True

  click.echo("Launching MuJoCo viewer — close the window to exit.")
  with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as v:
    while v.is_running():
      prev_state = policy.state
      action = policy.act(obs)
      obs, _, terminated, truncated, _ = env.step(action)
      if policy.state != prev_state:
        print(f"Policy state changed: {prev_state} → {policy.state}")

      policy.insert_hints(v)
      v.sync()
      if step_delay > 0:
        time.sleep(step_delay)

      if terminated or truncated or policy.state == "done":
        break

  env.close()


@cli.command("train")
@click.option("--name", "experiment_name", default=None, type=str,
              help="Experiment name (alias for -o logging.experiment_name=...). "
                   "Used as the checkpoint subdirectory ({save_dir}/{name}/) and as the logger run name.")
@click.option("--seed", default=None, type=int, help="Random seed (random if omitted)")
@click.option("--resume", "resume", default=None, is_flag=False, flag_value="__auto__",
              help="Resume from a checkpoint. Bare flag uses {save_dir}/{experiment}/latest.safetensors; "
                   "pass a path to load a specific file.")
@override_option
@click.pass_context
def train(ctx, experiment_name, seed, resume, overrides):
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


@cli.command("record")
@click.option("--output", required=True, type=str, help="Session output directory")
@click.option("--storage", type=click.Choice(["lerobot", "episodes"]), default="lerobot",
              help="How phase B's data is preserved. 'lerobot' keeps the LeRobotDataset "
                   "under {output}/dataset/. 'episodes' re-imports it into "
                   "{output}/episodes/ via the project's EpisodeWriter.")
@click.option("--format", "fmt", default="rerun", type=click.Choice(["hdf5", "rerun"]),
              help="Episode output format when --storage=episodes (ignored otherwise).")
@click.option("--skip-calibration", is_flag=True,
              help="Skip phase A.1 (camera intrinsics).")
@click.option("--skip-markers", is_flag=True,
              help="Skip phase A.2 (marker clip capture).")
@click.option("--seed", default=None, type=int, help="Random seed (random if omitted)")
@override_option
@click.pass_context
def record_cmd(ctx, output, storage, fmt, skip_calibration, skip_markers, seed, overrides):
  """Record a real-hardware session: scene registration + lerobot teleop episodes."""
  from chuck_dreamer.real import run as run_session

  cfg = load_config(ctx.obj["config_path"], overrides=overrides, aliases={"seed": seed})
  click.echo(f"Recording session → {output}  (storage={storage}, format={fmt})")
  out = run_session(
    cfg, Path(output),
    storage=storage,
    episodes_format=fmt,
    skip_calibration=skip_calibration,
    skip_markers=skip_markers,
  )
  click.echo(f"Done. Session written to {out}")


def _list_evals() -> dict[str, Path]:
  """Return a mapping of eval-name → notebook path under chuck_dreamer/evals/."""
  evals_dir = Path(__file__).parent / "src" / "chuck_dreamer" / "evals"
  return {p.stem: p for p in sorted(evals_dir.glob("*.ipynb"))}


@cli.command("eval")
@click.argument("name", required=False)
@click.option("--checkpoint", "checkpoint_path", default=None, type=str,
              help="Path to a trained checkpoint .safetensors file. "
                   "Defaults to {save_dir}/{experiment}/latest.safetensors.")
@click.option("--num-episodes", default=20, type=int, help="Number of episodes to evaluate on.")
@click.option("--burn-in", default=5, type=int, help="Closed-loop burn-in steps before open-loop rollout.")
@click.option("--horizon", default=15, type=int, help="Open-loop horizon length.")
@click.option("--seed", default=None, type=int, help="Random seed (random if omitted).")
@click.option("--output", "output_path", default=None, type=str,
              help="Where to write the executed notebook (default: ./<name>_<ckpt-stem>.ipynb in cwd).")
@click.option("-p", "--param", "extra_params", multiple=True, metavar="KEY=VALUE",
              help="Additional papermill parameter override (repeatable).")
@override_option
@click.pass_context
def eval_cmd(ctx, name, checkpoint_path, num_episodes,
             burn_in, horizon, seed, output_path, extra_params, overrides):
  """Run an evaluation notebook on a trained checkpoint via papermill.

  NAME selects which notebook under ``src/chuck_dreamer/evals/`` to execute
  (e.g. ``open_loop_rollout``). Run without NAME to list available evals.
  """
  import os

  evals = _list_evals()
  if not name:
    click.echo("Available evals:")
    for n, p in evals.items():
      click.echo(f"  {n:<30s} {p}")
    return
  if name not in evals:
    raise click.BadParameter(f"unknown eval {name!r}. Available: {sorted(evals)}")
  nb_in = evals[name]

  cfg = load_config(ctx.obj["config_path"], overrides=overrides, aliases={"seed": seed})

  if checkpoint_path is None:
    experiment = cfg.logging.experiment_name or "default"
    checkpoint_path = os.path.join(cfg.logging.save_dir, experiment, "latest.safetensors")
  if not Path(checkpoint_path).exists():
    raise click.ClickException(f"checkpoint not found: {checkpoint_path}")

  # The eval notebook expects one data path. Pull from the first warmup source.
  first_source = cfg.training.data.warmup_source[0]
  data_path    = first_source["path"]
  data_format  = first_source["format"]

  if output_path is None:
    output_path = f"{name}_{Path(checkpoint_path).stem}.ipynb"
  else:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

  parameters: dict = {
    "checkpoint_path": str(checkpoint_path),
    "data_path":       str(data_path),
    "data_format":     str(data_format),
    "num_episodes":    int(num_episodes),
    "burn_in":         int(burn_in),
    "horizon":         int(horizon),
    "seed":            int(cfg.seed),
  }
  for kv in extra_params:
    if "=" not in kv:
      raise click.BadParameter(f"--param expects KEY=VALUE, got {kv!r}")
    k, v = kv.split("=", 1)
    parameters[k.strip()] = v

  click.echo(f"Executing {nb_in} → {output_path}")
  click.echo(f"Parameters: {parameters}")

  import papermill as pm
  pm.execute_notebook(
    input_path=str(nb_in),
    output_path=str(output_path),
    parameters=parameters,
    cwd=str(Path(__file__).parent),
  )
  click.echo(f"Done. Executed notebook: {output_path}")


if __name__ == "__main__":
    cli()
