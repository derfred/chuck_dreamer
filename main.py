#!/usr/bin/env python3
"""
Chuck Dreamer: A Robotics and Machine Learning Project

This is the main entry point for the project. It provides a CLI interface
for training models, processing data, and running experiments.
"""

import os
import platform

if (
  platform.system() == "Linux"
  and "MUJOCO_GL" not in os.environ
  and not os.environ.get("DISPLAY")
):
  os.environ["MUJOCO_GL"] = "egl"

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


def _collect_one_episode(cfg, output, fmt, ep_idx, episode_seed, min_steps):
  """Worker entry point: build env + writer, roll one episode, write it.

  Returns ``(ep_idx, outcome, crashed_on_reset, skipped_short)``. Runs in a
  child process, so it imports the heavy deps lazily and owns its own
  MuJoCo handles.
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
      return ep_idx, outcome, True, False

    if len(episode_data["reward"]) < min_steps:
      return ep_idx, outcome, False, True

    writer.write_episode(
      episode_data,
      metadata={
        "config":  asdict(scene),
        "seed":    episode_seed,
        "source":  "sim",
        "outcome": outcome,
        "goal_xy": scene.goal_xy,
        "goal_side": scene.goal_side,
      },
      name_suffix=f"{ep_idx:05d}",
    )
    return ep_idx, outcome, False, False
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
@click.option("--min-steps", default=20, type=int,
              help="Drop episodes shorter than this many steps instead of writing them.")
@override_option
@click.pass_context
def generate_scenes(ctx, episodes, output, seed, fmt, workers, min_steps, overrides):
  """Generate pushing scenes using the scripted heuristic policy."""
  import os

  cfg = load_config(ctx.obj["config_path"], overrides=overrides, aliases={"seed": seed})

  if workers is None:
    workers = os.cpu_count() or 1

  click.echo(f"Collecting {episodes} episodes → {output}  (difficulty={cfg.env.difficulty}, "
             f"format={fmt}, seed={cfg.seed}, workers={workers})")
  outcome_counts: dict[str, int] = {}

  from tqdm import tqdm

  short_skipped = 0
  if workers <= 1:
    for ep_idx in tqdm(range(episodes), desc="Collecting"):
      _, outcome, crashed_on_reset, skipped_short = _collect_one_episode(
        cfg, output, fmt, ep_idx, cfg.seed + ep_idx, min_steps,
      )
      outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
      if crashed_on_reset:
        click.echo(f"[ep {ep_idx}] crashed on reset, skipping")
      if skipped_short:
        short_skipped += 1
        click.echo(f"[ep {ep_idx}] shorter than {min_steps} steps, skipping write")
  else:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=workers) as pool:
      futures = [
        pool.submit(_collect_one_episode, cfg, output, fmt, ep_idx, cfg.seed + ep_idx, min_steps)
        for ep_idx in range(episodes)
      ]
      for fut in tqdm(as_completed(futures), total=episodes, desc="Collecting"):
        ep_idx, outcome, crashed_on_reset, skipped_short = fut.result()
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        if crashed_on_reset:
          click.echo(f"[ep {ep_idx}] crashed on reset, skipping")
        if skipped_short:
          short_skipped += 1
          click.echo(f"[ep {ep_idx}] shorter than {min_steps} steps, skipping write")

  click.echo(f"Done. Outcomes: {outcome_counts}  (skipped {short_skipped} short episodes)")


@cli.command("import-lerobot")
@click.argument("repo_id", type=str, metavar="REPO_ID[#EPISODES]")
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
@click.option("--name-prefix", default=None, type=str,
              help="String prepended to each written episode's filename. "
                   "Use to import multiple datasets into the same --output "
                   "directory without filename collisions (default "
                   "names are episode-00000, etc. — identical across "
                   "datasets). Typical: --name-prefix task_2_1805_2")
@click.pass_context
def import_lerobot(ctx, repo_id, output, fmt, video_key, max_episodes,
                   derive_target_mask, tags, processor_paths, name_prefix):
  """Convert a LeRobot v3 HF dataset (REPO_ID) into the repo's episode files.

  Pulls the parquet + MP4 from Hugging Face, slices per episode using
  ``meta/episodes/*.parquet``, decodes the matching video range, and
  writes one episode file per LeRobot episode under --output.

  ``REPO_ID`` accepts the same ``DATASET[#EPISODES]`` syntax used by
  ``calibrate-intrinsics`` and ``prompt-episodes`` — e.g.
  ``user/dataset#8-99`` to import only episodes 8 onward. ``:FRAMES``
  syntax is rejected here (frame-range filtering doesn't make sense
  for the importer).

  Fields the repo's format needs but LeRobot teleop data lacks (reward,
  ee_pos, ee_quat, object_xy) are zero-filled — fine for image-mode
  training, not for state-mode.
  """
  from tqdm import tqdm

  from chuck_dreamer.real.object_localization.calibration.spec_parser import (
    parse_calibration_source,
  )
  from chuck_dreamer.sim.lerobot_import import import_dataset, load_processor

  try:
    spec = parse_calibration_source(repo_id)
  except ValueError as e:
    raise click.ClickException(str(e))
  if spec.frames is not None:
    raise click.ClickException(
      "import-lerobot doesn't accept the :FRAMES portion of the spec "
      "(only #EPISODES is meaningful). Got: "
      f"{repo_id!r}")
  parsed_repo_id = spec.dataset_id
  episode_filter: set[int] | None = (
    set(spec.episodes) if spec.episodes is not None else None
  )

  processors = tuple(load_processor(p) for p in processor_paths)

  click.echo(
    f"Importing {parsed_repo_id} → {output}  (format={fmt}, "
    f"derive_target_mask={derive_target_mask}, tags={list(tags)}, "
    f"processors={list(processor_paths)}, name_prefix={name_prefix!r}, "
    f"episodes={'all' if episode_filter is None else sorted(episode_filter)})"
  )
  count = 0
  for ep_idx, out_path in tqdm(
    import_dataset(
      parsed_repo_id, output,
      format=fmt, video_key=video_key, max_episodes=max_episodes,
      derive_target_mask=derive_target_mask, tags=tuple(tags),
      name_prefix=name_prefix,
      processors=processors,
      episode_filter=episode_filter,
    ),
    desc="Episodes",
    total=(len(episode_filter) if episode_filter is not None else max_episodes),
  ):
    count += 1
    logger.info("episode %d → %s", ep_idx, out_path)
  click.echo(f"Done. Wrote {count} episodes.")


@cli.command("show-scene")
@click.option("--seed", default=None, type=int, help="Random seed (random if omitted)")
@click.option("--step-delay", default=0.05, type=float, help="Seconds to sleep between steps (default 0.05)")
@click.option("--policy", "policy_type", default="scripted",
              type=click.Choice(["scripted", "cem", "dreamer", "cem_probe"]),
              help="Which policy to drive the scene with. cem/dreamer/cem_probe "
                   "load a trained world model from --checkpoint and are "
                   "wrapped in a GatedPolicy that holds the arm still until "
                   "you press Space. ``cem_probe`` replaces the WM's "
                   "reward_head with a small probe MLP + Python reward — see "
                   "scripts/train_probe.py.")
@click.option("--checkpoint", "checkpoint_path", default=None, type=str,
              help="Path to a trained .safetensors checkpoint. Required for "
                   "--policy cem, --policy dreamer, --policy cem_probe.")
@click.option("--probe", "probe_path", default=None, type=str,
              help="Path to a probe .pt file from scripts/train_probe.py. "
                   "Required for --policy cem_probe.")
@click.option("--reward-kind", "reward_kind", default="push_shaped", type=str,
              help="Reward function name passed through build_reward_fn for "
                   "--policy cem_probe. Default: push_shaped.")
@click.option("--goal-xy", "goal_xy_str", default="0.15,0.18", type=str,
              help="Fixed goal_xy for --policy cem_probe, as 'x,y' (metres).")
@click.option("--close", "close_on_done", is_flag=True, default=False,
              help="Close the viewer as soon as the episode terminates. "
                   "Default: leave the window open so you can inspect the final state.")
@override_option
@click.pass_context
def show_scene(ctx, seed, step_delay, policy_type, checkpoint_path,
               probe_path, reward_kind, goal_xy_str,
               close_on_done, overrides):
  """Generate a scene and run it in the interactive MuJoCo viewer.

  With ``--policy scripted`` (default) the user presses Space to advance
  the heuristic policy from ``ready`` → ``approach``. With ``--policy
  cem`` or ``--policy dreamer`` the policy is wrapped in
  :class:`~chuck_dreamer.policy.GatedPolicy`, which holds the arm in its
  initial pose until Space is pressed, then hands off to the underlying
  planner / actor.

  After the episode terminates the window stays open by default so the
  final state can be inspected; pass ``--close`` to exit immediately.
  """
  import time
  import mujoco
  import mujoco.viewer
  import numpy as np

  from chuck_dreamer.policy import GatedPolicy
  from chuck_dreamer.sim import PushingEnv, ScriptedPolicy

  cfg = load_config(ctx.obj["config_path"], overrides=overrides, aliases={"seed": seed})

  click.echo(f"difficulty={cfg.env.difficulty}  seed={cfg.seed}  policy={policy_type}")
  env    = PushingEnv(cfg)
  scene  = env.generate_scene()
  obs, _ = env.reset(scene=scene)

  if policy_type == "scripted":
    policy = ScriptedPolicy()
    policy.reset(scene)
  else:
    if not checkpoint_path:
      raise click.BadParameter(f"--checkpoint is required for --policy {policy_type}")
    from chuck_dreamer.eval.checkpoint import load_checkpoint

    loaded = load_checkpoint(checkpoint_path)
    if policy_type == "cem":
      from chuck_dreamer.dreamer import CEMPolicy
      c = cfg.real.policy.cem
      inner = CEMPolicy(
        loaded.model,
        horizon=int(c.horizon),
        num_samples=int(c.num_samples),
        num_elites=int(c.num_elites),
        num_iterations=int(c.num_iterations),
        init_std=float(c.get("init_std", 0.02)),
        min_std=float(c.get("min_std", 0.002)),
        discount=float(c.get("discount", 1.0)),
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        ee_max_delta=float(c.get("ee_max_delta", c.get("max_delta", 0.005))),
        ee_freeze_quat=bool(c.get("ee_freeze_quat", True)),
      )
    elif policy_type == "cem_probe":
      if not probe_path:
        raise click.BadParameter("--probe is required for --policy cem_probe")
      from chuck_dreamer.dreamer.cem_probe_policy import CEMProbePolicy, ProbeBundle
      from chuck_dreamer.reward import build_reward_fn
      c = cfg.real.policy.cem
      try:
        gx, gy = (float(v) for v in goal_xy_str.split(","))
      except Exception as exc:
        raise click.BadParameter(
          f"--goal-xy must be 'x,y' (metres); got {goal_xy_str!r}: {exc}",
        )
      inner = CEMProbePolicy(
        loaded.model,
        probe=ProbeBundle.from_file(probe_path),
        reward_fn=build_reward_fn(reward_kind),
        goal_xy=(gx, gy),
        horizon=int(c.horizon),
        num_samples=int(c.num_samples),
        num_elites=int(c.num_elites),
        num_iterations=int(c.num_iterations),
        discount=float(c.get("discount", 1.0)),
        action_low=env.action_space.low,
        action_high=env.action_space.high,
      )
    else:
      from chuck_dreamer.dreamer import DreamerPolicy
      inner = DreamerPolicy(loaded.model, act_mode=env.act_mode)
    def _hold_action(o):
      if env.act_mode == "joint":
        return np.asarray(o["arm_qpos"], dtype=np.float32)
      return np.concatenate([
        np.asarray(o["ee_pos"],  dtype=np.float32),
        np.asarray(o["ee_quat"], dtype=np.float32),
      ])

    policy = GatedPolicy(inner, hold_action=_hold_action, project=env.policy_obs)
    policy.reset(scene)

    if policy_type == "cem":
      inner.seed_action(_hold_action(obs))

  def key_callback(keycode):
    if keycode != 32:  # Space bar
      return True
    if isinstance(policy, ScriptedPolicy) and policy.state == "ready":
      policy.advance_from_ready()
      print("Policy state changed: ready → approach")
    elif isinstance(policy, GatedPolicy) and not policy.released:
      policy.release()
      print(f"Released gated policy → {policy_type} now in control")
    return True

  click.echo("Launching MuJoCo viewer — close the window to exit.")
  with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as v:
    # Switch the viewer from its free orbit camera to the scene's main_camera
    # so randomised positions (e.g. the "real" preset's 110° front arc) are
    # actually what the user sees.
    main_cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "main_camera")
    if main_cam_id >= 0:
      v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
      v.cam.fixedcamid = main_cam_id

    done = False
    while v.is_running():
      if not done:
        prev_state = policy.state if isinstance(policy, ScriptedPolicy) else None

        if policy_type == "cem" and env.act_mode == "ee":
          inner.seed_action(_hold_action(obs))

        action = policy.act(obs)
        obs, _, terminated, truncated, _ = env.step(action)

        if isinstance(policy, ScriptedPolicy):
          if policy.state != prev_state:
            print(f"Policy state changed: {prev_state} → {policy.state}")
          policy.insert_hints(v)

        if terminated or truncated or (
          isinstance(policy, ScriptedPolicy) and policy.state == "done"
        ):
          done = True
          if close_on_done:
            break
          click.echo("Episode finished — window stays open; close it (or pass --close) to exit.")

      v.sync()
      if step_delay > 0:
        time.sleep(step_delay)

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


@cli.command("run")
@click.option("--policy", "policy_type", default=None, type=str,
              help="Policy implementation: manual | dreamer | cem. "
                   "Overrides cfg.real.policy.type.")
@click.option("--checkpoint", "checkpoint_path", default=None, type=str,
              help="Checkpoint path for dreamer / cem policies. "
                   "Overrides cfg.real.policy.<type>.checkpoint.")
@click.option("--camera", "camera_source", default=None, type=str,
              help="Camera index or device path. Overrides cfg.real.camera.source.")
@click.option("--port", "arm_port", default=None, type=str,
              help="USB port for the SO-101 controller. Overrides cfg.real.arm.port.")
@override_option
@click.pass_context
def run_cmd(ctx, policy_type, checkpoint_path, camera_source, arm_port, overrides):
  """Run a policy on the real SO-101 arm.

  Opens the configured camera, instantiates the policy, and feeds the
  live observation stream to it. The arm only follows the policy after
  the operator hits the start key (default ``s``).
  """
  from chuck_dreamer.real.run import run as run_loop

  aliases: dict = {
    "real.policy.type":     policy_type,
    "real.camera.source":   camera_source,
    "real.arm.port":        arm_port,
  }
  if checkpoint_path is not None:
    ptype = policy_type or "dreamer"
    if ptype not in ("dreamer", "cem"):
      raise click.BadParameter(
        f"--checkpoint only applies to dreamer/cem policies; got {ptype!r}"
      )
    aliases[f"real.policy.{ptype}.checkpoint"] = checkpoint_path

  cfg = load_config(ctx.obj["config_path"], overrides=overrides, aliases=aliases)

  click.echo("Running real arm with config:")
  click.echo(OmegaConf.to_yaml(cfg.real))
  click.echo("Press 's' to start the policy, 'q' or ESC to quit.")
  run_loop(cfg)


from chuck_dreamer.real.object_localization.cli import (
  annotate_mat_cmd,
  calibrate_intrinsics_cmd,
  prompt_episodes_cmd,
  test_calibration_cmd,
  test_object_pose_cmd,
)

cli.add_command(calibrate_intrinsics_cmd)
cli.add_command(annotate_mat_cmd)
cli.add_command(prompt_episodes_cmd)
cli.add_command(test_calibration_cmd)
cli.add_command(test_object_pose_cmd)


if __name__ == "__main__":
    cli()
