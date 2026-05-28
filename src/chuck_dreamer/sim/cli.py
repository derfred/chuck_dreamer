"""CLI entry points for the simulation pipeline: ``generate-scenes`` and
``show-scene``.

Both commands drive the MuJoCo :class:`~chuck_dreamer.sim.PushingEnv`:
``generate-scenes`` rolls episodes with the scripted policy (optionally
fanned across worker processes) and writes them to disk; ``show-scene``
opens the interactive viewer with a scripted or trained policy.
"""
from __future__ import annotations

import logging

import click
from omegaconf import OmegaConf

from chuck_dreamer.config import load_config

from ..cli import override_option

logger = logging.getLogger(__name__)


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


@click.command("generate-scenes")
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
def generate_scenes_cmd(ctx, episodes, output, seed, fmt, workers, min_steps, overrides):
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


@click.command("show-scene")
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
def show_scene_cmd(ctx, seed, step_delay, policy_type, checkpoint_path,
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
