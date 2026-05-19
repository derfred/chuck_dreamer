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


@cli.command("analyze-cameras")
@click.option("--dataset", "dataset_ids", multiple=True, required=True, metavar="REPO_ID",
              help="LeRobot dataset id (HuggingFace repo) for one camera position. "
                   "Pass --dataset repeatedly to calibrate several cameras in one run.")
@click.option("--cache-dir", default=None, type=click.Path(),
              help="Directory under which calibrations + verify overlays are stored. "
                   "Defaults to cfg.object_localization.calibration_cache.")
@click.option("--doctor-interactive/--doctor-no-interactive",
              "doctor_interactive", default=True,
              help="Open matplotlib review windows for every detection step "
                   "(per-frame checkerboard accept/reject/click-fallback, "
                   "mat detection auto-then-edit, final verify-overlay sign-off). "
                   "Default: on. Pass --doctor-no-interactive for headless / batch runs.")
@click.option("--force", is_flag=True,
              help="Recompute calibrations even if a cached entry exists.")
@click.option("--intrinsic-samples", default=60, type=int,
              help="How many frames to sample from the checkerboard episode for intrinsics.")
@click.option("--doctor", is_flag=True,
              help="Diagnostic mode: inspect the datasets without fitting anything. "
                   "Reports image dims, which expected episodes are present, "
                   "whether checkerboards are detectable, and whether the mat "
                   "auto-detector finds the fiducials. Diagnostic PNGs are "
                   "written under <cache-dir>/doctor/<dataset>/ unless "
                   "--doctor-no-images is set.")
@click.option("--doctor-dir", default=None, type=click.Path(),
              help="Where to write doctor diagnostic PNGs. Defaults to "
                   "<cache-dir>/doctor/. Only used when --doctor is set.")
@click.option("--doctor-no-images", is_flag=True,
              help="Skip writing diagnostic PNGs in --doctor mode.")
@override_option
@click.pass_context
def analyze_cameras(ctx, dataset_ids, cache_dir, doctor_interactive, force,
                    intrinsic_samples, doctor, doctor_dir, doctor_no_images,
                    overrides):
  """Calibrate cameras from LeRobot calibration datasets (one per camera position).

  Each dataset must follow the layout described in cfg.object_localization.episodes:
    episode 0 — empty mat (extrinsics fit)
    episode 1 — checkerboard (intrinsics fit)
    episodes 2-4 — object placements (used for verify overlays)

  Writes ``calibration.json`` per dataset under ``--cache-dir`` along with
  verify overlays under ``<cache-dir>/<dataset>/verify/*.png``.

  Use ``--doctor`` to dry-run the inspection. Diagnostic PNGs land under
  ``<cache-dir>/doctor/<dataset>/`` so a user can eyeball the inputs.
  """
  from pathlib import Path as _Path

  from chuck_dreamer.real.object_localization import init_from_config
  from chuck_dreamer.real.object_localization.calibration import (
    analyze_datasets, diagnose_datasets,
  )

  cfg = load_config(ctx.obj["config_path"], overrides=overrides)
  ol_cfg = init_from_config(cfg)
  resolved_cache = _Path(cache_dir) if cache_dir else _Path(ol_cfg.calibration_cache)

  if doctor:
    resolved_image_dir: _Path | None
    if doctor_no_images:
      resolved_image_dir = None
    elif doctor_dir:
      resolved_image_dir = _Path(doctor_dir)
    else:
      resolved_image_dir = resolved_cache / "doctor"
    where = "no PNGs (--doctor-no-images)" if resolved_image_dir is None else str(resolved_image_dir)
    click.echo(f"Doctor: inspecting {len(dataset_ids)} dataset(s) — diagnostic images: {where}")
    reports = diagnose_datasets(list(dataset_ids), image_dir=resolved_image_dir)
    for r in reports:
      click.echo(r.format())
    failed = [r for r in reports if not r.loaded]
    if failed:
      raise click.ClickException(
        f"{len(failed)} dataset(s) failed to load — see report above")
    return

  click.echo(f"Calibrating {len(dataset_ids)} dataset(s) → {resolved_cache} "
             f"(doctor_interactive={doctor_interactive}, force={force})")
  results = analyze_datasets(
    list(dataset_ids),
    cache_dir=resolved_cache,
    interactive=doctor_interactive,
    force=force,
  )
  for did, cal in results.items():
    click.echo(f"  {did}: intrinsic RMS={cal.intrinsic_rms_px:.3f}px  "
               f"extrinsic RMS={cal.extrinsic_rms_px:.2f}px  "
               f"image={cal.image_size}")
  click.echo("Done.")


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


@cli.command("eval")
@click.argument("name", required=False)
@click.option("--all", "run_all", is_flag=True,
              help="Run every notebook listed under cfg.eval.deep.notebooks. "
                   "Mutually exclusive with NAME.")
@click.option("--recordings/--no-recordings", default=None,
              help="Also dump per-episode rerun recordings of the rollout into "
                   "<output-dir>/recordings/. Default: follow cfg.eval.deep.recordings. "
                   "Can be used WITHOUT a notebook to only dump recordings.")
@click.option("--checkpoint", "checkpoint_path", default=None, type=str,
              help="Path to a trained checkpoint .safetensors file. "
                   "Defaults to {save_dir}/{experiment}/latest.safetensors.")
@click.option("--cleanup/--no-cleanup", default=None,
              help="Whether to clean up intermediate files after evaluation. Defaults to cfg.eval.deep.cleanup.")
@click.option("--num-episodes", default=None, type=int,
              help="Episodes to evaluate on. Defaults to cfg.eval.deep.num_episodes.")
@click.option("--burn-in", default=None, type=int,
              help="Closed-loop burn-in steps. Defaults to cfg.eval.burn_in.")
@click.option("--horizon", default=None, type=int,
              help="Open-loop horizon length. Defaults to cfg.eval.deep.horizon.")
@click.option("--seed", default=None, type=int, help="Random seed (random if omitted).")
@click.option("--output-dir", "output_dir", default=None, type=str,
              help="Directory to write executed notebooks + recordings. "
                   "Default: ./eval_runs/<ckpt-stem>/.")
@click.option("-p", "--param", "extra_params", multiple=True, metavar="KEY=VALUE",
              help="Additional papermill parameter override (repeatable).")
@override_option
@click.pass_context
def eval_cmd(ctx, name, run_all, recordings, checkpoint_path, num_episodes,
             burn_in, horizon, seed, output_dir, extra_params, cleanup, overrides):
  """Run deep eval on a trained checkpoint.

  Three modes:

    * ``main.py eval``                          — list available notebooks.
    * ``main.py eval <notebook>``               — execute one notebook.
    * ``main.py eval --all``                    — execute every notebook
                                                  in ``cfg.eval.deep.notebooks``.
    * Add ``--recordings`` to also dump per-episode rerun rollout files.
      ``--recordings`` works without a notebook too — useful for just
      generating viewer-ready recordings against a checkpoint.
  """
  import os

  from chuck_dreamer.eval import DeepEvalRunner, available_notebooks

  notebooks = available_notebooks()
  if not name and not run_all and recordings is None:
    click.echo("Available notebooks:")
    for n, p in notebooks.items():
      click.echo(f"  {n:<30s} {p}")
    click.echo("\nUse `eval <name>` to run one, `eval --all` to run every configured "
               "notebook, or `eval --recordings` to dump recordings without notebooks.")
    return
  if name and run_all:
    raise click.BadParameter("--all is mutually exclusive with a notebook NAME argument.")
  if name and name not in notebooks:
    raise click.BadParameter(f"unknown notebook {name!r}. Available: {sorted(notebooks)}")

  cfg = load_config(ctx.obj["config_path"], overrides=overrides, aliases={"seed": seed})

  if checkpoint_path is None:
    experiment = cfg.logging.experiment_name or "default"
    checkpoint_path = os.path.join(cfg.logging.save_dir, experiment, "latest.safetensors")
  if not Path(checkpoint_path).exists():
    raise click.ClickException(f"checkpoint not found: {checkpoint_path}")

  if name:
    chosen_notebooks: tuple[str, ...] = (name,)
  elif run_all:
    chosen_notebooks = tuple(cfg.eval.deep.notebooks or ())
    if not chosen_notebooks:
      click.echo("cfg.eval.deep.notebooks is empty — only recordings will be dumped.")
  else:
    chosen_notebooks = ()

  effective_recordings = (
    bool(cfg.eval.deep.recordings) if recordings is None else bool(recordings)
  )
  if not chosen_notebooks and not effective_recordings:
    raise click.BadParameter(
      "nothing to do: pass a notebook NAME, --all, or --recordings."
    )

  if output_dir is None:
    output_dir = os.path.join("eval_runs", Path(checkpoint_path).stem)
  Path(output_dir).mkdir(parents=True, exist_ok=True)

  extras: dict = {}
  for kv in extra_params:
    if "=" not in kv:
      raise click.BadParameter(f"--param expects KEY=VALUE, got {kv!r}")
    k, v = kv.split("=", 1)
    extras[k.strip()] = v

  click.echo(f"Deep eval on checkpoint: {checkpoint_path}")
  click.echo(f"Output dir:              {output_dir}")
  click.echo(f"Notebooks:               {list(chosen_notebooks) or '—'}")
  click.echo(f"Recordings:              {'on' if effective_recordings else 'off'}")

  runner = DeepEvalRunner(cfg)
  summary = runner.run_sync(
    checkpoint_path=str(checkpoint_path),
    output_dir=output_dir,
    notebooks=chosen_notebooks,
    recordings=effective_recordings,
    num_episodes=num_episodes,
    burn_in=burn_in,
    horizon=horizon,
    extra_params=extras,
  )
  click.echo("Done.")
  for nb in summary["executed"]:
    click.echo(f"  notebook → {nb}")
  if summary["recordings_path"]:
    click.echo(f"  recordings → {summary['recordings_path']}")

  if cleanup:
    click.echo("Cleaning up checkpoint...")
    os.unlink(checkpoint_path)
    click.echo("Cleanup done.")

if __name__ == "__main__":
    cli()
