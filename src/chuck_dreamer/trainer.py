import logging
import os
from collections import defaultdict
import numpy as np
import tqdm

from .reward import build_reward_fn
from .sim.pushing_env import PushingEnv
from .sim.episode_collector import EpisodeCollector
from .sim.collector_backend import CollectResult, build_collector_backend

from .eval import OnlineEvaluator
from .training.episode_processor import processor_for
from .training.replay_buffer import ReplayBuffer
from .training.tracker import Tracker

from .sim.scripted_policy import ScriptedPolicy
from .dreamer import build_model, DreamerPolicy

logger = logging.getLogger(__name__)


class Trainer:
  def __init__(self, config):
    self.config = config
    self.env    = PushingEnv(config)

    self._replay_buffer = ReplayBuffer.from_config(
      config,
      processor=processor_for(config),
      reward_fn=build_reward_fn(config.env.reward),
    )

    obs_shape      = self.env.model_obs_shape
    action_dim     = int(self.env.action_space.shape[0])
    self.model     = build_model(config, obs_shape=obs_shape, action_dim=action_dim)
    self.policy    = self._build_policy(config, obs_shape, action_dim)
    self.collector = EpisodeCollector(self.env, self.policy)
    self.evaluator = OnlineEvaluator(config, self.model)
    self.tracker   = Tracker(config)
    self.tracker.init()

    self._backend = build_collector_backend(config, self.collector)

    # Replay-ratio bookkeeping. Both axes useful: cumulative is the
    # long-run rate that matters for stability; per-iter is the
    # instantaneous balance you tune against after async lands.
    self._cumulative_env_steps:  int = 0
    self._cumulative_grad_steps: int = 0

  def _warmup(self):
    for src in self.config.training.data.warmup_source:
      if not os.path.exists(src["path"]):
        logger.warning(f"Warmup path {src['path']} does not exist. Skipping.")
        continue
      logger.info(
        f"Loading warmup episodes from {src['path']} "
        f"(format={src['format']}, num_episodes={src['num_episodes']})"
      )
      inserted = self._replay_buffer.load_sim_episodes(
        src["path"],
        src["format"],
        progress=True,
        num_episodes=src["num_episodes"],
      )
      logger.info(
        f"  inserted {inserted} episodes from {src['path']} "
        f"→ buffer now {self._replay_buffer.num_episodes} eps / "
        f"{len(self._replay_buffer)} steps"
      )

    by_tag = self._replay_buffer.size_by_tag()
    if by_tag:
      logger.info("Warmup buffer by tag (episodes counted under each of their tags):")
      for tag in sorted(by_tag):
        n_eps, n_steps = by_tag[tag]
        logger.info(f"  {tag}: {n_eps} eps / {n_steps} steps")

    # Eval episodes load is independent of the buffer warmup — preload
    # them here so the first eval phase doesn't pay the cost (and
    # subsequent phases reuse the cached set).
    self.evaluator.warmup()

  def _build_policy(self, config, obs_shape, action_dim):
    if config.training.policy == "dreamer":
      return DreamerPolicy(self.model, act_mode=self.env.act_mode)
    elif config.training.policy == "scripted":
      return ScriptedPolicy(auto_advance_from_ready=True)
    else:
      raise ValueError(f"Unknown policy type {config.training.policy!r}")

  def _absorb(self, results: list[CollectResult], iteration: int) -> int:
    """Insert collected episodes into the replay buffer and log outcomes.

    Returns the number of env-steps actually added (sum of action
    lengths across non-None episodes). Used by ``train`` to update the
    replay-ratio counters.
    """
    collect_data: defaultdict[str, int] = defaultdict(int)
    env_steps_added = 0
    with self.tracker.scope(
      {"phase": "collect", "iteration": iteration}, step=self.tracker.current_step,
    ) as tracker:
      for r in results:
        collect_data[r.outcome] += 1
        if r.episode is not None:
          self._replay_buffer.add_sim_episode(r.episode)
          # Raw episode actions are keyed by their mode (``joint_action``
          # / ``ee_action``); reward is always present and shares the
          # action time axis, so it's the safe length-of-episode source.
          env_steps_added += int(r.episode["reward"].shape[0])
          if r.scene is not None:
            tracker.maybe_log_collect_episode(
              r.episode, r.scene, r.outcome, {"iteration": iteration},
            )

      tracker.log({
        "num_episodes":   sum(collect_data.values()),
        "env_steps_iter": env_steps_added,
        **{f"outcome/{k}": v for k, v in collect_data.items()},
      })
    return env_steps_added

  def _train_phase(self, iteration: int) -> int:
    """Run one train phase and return the number of gradient steps taken.

    Returns 0 when the buffer was too small to sample (phase skipped).
    """
    with self.tracker.scope(
      {"phase": "train", "iteration": iteration}, step=self.tracker.current_step,
    ) as tracker:
      tracker.log_replay_buffer_size(self._replay_buffer)
      if not self._replay_buffer.can_sample(self.config.training.batch_size, self.config.training.seq_len):
        logger.warning("Buffer too small to sample (have %d steps); skipping train phase.", len(self._replay_buffer))
        return 0

      num_steps = self.config.training.num_gradient_steps
      phase_losses: list[float] = []
      for _ in tqdm.tqdm(range(num_steps), desc=f"Training - iteration {iteration}", total=num_steps):
        # tick() advances the global step counter and returns a child
        # tracker bound to the new step, so batch_tags + wm_update logs
        # land on the same row.
        step_tracker = tracker.tick()
        batch = self._replay_buffer.sample(self.config.training.batch_size, self.config.training.seq_len)
        step_tracker.log_batch_tags(batch.pop("tags", []))
        _, loss_val = self.model.wm_update(batch, tracker=step_tracker)
        if loss_val is not None:
          phase_losses.append(loss_val)

      # Plateau diagnostic: normalized slope of wm_loss across this phase.
      # |slope_normalized| ~ 0 means flat (over-training); strongly
      # negative means still descending (under-training). Log on the last
      # grad step of the phase so it shares an axis with wm/loss.
      if len(phase_losses) >= 2:
        losses = np.asarray(phase_losses, dtype=np.float64)
        x      = np.arange(len(losses), dtype=np.float64)
        slope  = float(np.polyfit(x, losses, 1)[0])
        denom  = max(abs(float(losses[0])), 1e-8)
        tracker.at_step(tracker.current_step).log({
          "wm/loss_slope":            slope,
          "wm/loss_slope_normalized": slope / denom,
          "wm/loss_phase_first":      float(losses[0]),
          "wm/loss_phase_last":       float(losses[-1]),
        })

      return num_steps

  def _eval_phase(self, iteration: int, checkpoint_path: str | None = None, cleanup: bool = False):
    self.evaluator.evaluate(
      iteration, self.tracker.at_step(self.tracker.current_step),
      checkpoint_path=checkpoint_path,
      cleanup=cleanup,
    )

  def _checkpoint_dir(self) -> str:
    name = self.config.logging.experiment_name or "default"
    return os.path.join(self.config.logging.save_dir, name)

  def _resume(self, resume: bool | str) -> int:
    """Load weights before training and return the iteration to start from.

    ``resume`` may be:
      - False/None: no-op, start from 0
      - True:       load ``{checkpoint_dir}/latest.safetensors`` if present
      - str path:   load that exact file

    The starting iteration is read from the safetensors metadata block
    (``"iteration"`` key) written by :meth:`_checkpoint`. Missing key →
    start from 0.
    """
    if not resume:
      return 0
    if resume is True:
      path = os.path.join(self._checkpoint_dir(), "latest.safetensors")
    else:
      path = resume
    if not os.path.exists(path):
      logger.warning(f"Resume requested but {path} does not exist; starting from scratch.")
      return 0
    logger.info(f"Resuming from {path}")
    extra = self.model.load(path)
    iteration = extra.get("iteration")
    if iteration is None:
      return 0
    start = int(iteration) + 1
    # Approximate the grad-step counter from iteration count so post-resume
    # logs land beyond the previous run's step axis. Off-by-a-few when
    # earlier train phases were skipped (buffer too small), but monotonic.
    self.tracker.set_step(start * int(self.config.training.num_gradient_steps))
    logger.info(f"Resuming at iteration {start}")
    return start

  def _checkpoint_paths(self, step: int, temporary: bool) -> list[str]:
    if temporary:
      temp_dir = os.path.join(self._checkpoint_dir(), "temp")
      os.makedirs(temp_dir, exist_ok=True)
      return [os.path.join(temp_dir, f"step_{step:06d}.safetensors")]
    else:
      ckpt_dir = self._checkpoint_dir()
      os.makedirs(ckpt_dir, exist_ok=True)
      step_path   = os.path.join(ckpt_dir, f"step_{step:06d}.safetensors")
      latest_path = os.path.join(ckpt_dir, "latest.safetensors")
      return [latest_path, step_path]

  def _checkpoint(self, step: int, temporary: bool = False) -> str:
    extra = {"iteration": str(int(step))}
    path  = ""
    for path in self._checkpoint_paths(step, temporary):
      self.model.save(path, extra_metadata=extra)
    if not temporary:
      logger.info(f"Saved checkpoint to {path}")
    return path

  def train(self, resume: bool | str = False):
    start = self._resume(resume)
    self._warmup()

    broadcast_every = int(self.config.training.collector.weight_broadcast_every_n_iters)
    is_dreamer = self.config.training.policy == "dreamer"
    target = int(self.config.training.num_collect_episodes)

    try:
      for i in range(start, self.config.training.num_iterations):
        # ---- collect (sync inline, or drain async pool) ----
        with tqdm.tqdm(
          total=target, desc=f"Collecting episodes - iteration {i}",
        ) as bar:
          results = self._backend.drain(target=target)
          bar.update(len(results))
        env_steps = self._absorb(results, i)
        self._cumulative_env_steps += env_steps

        # ---- train ----
        grad_steps = self._train_phase(i)
        self._cumulative_grad_steps += grad_steps

        # ---- replay-ratio logging (per-iter + cumulative) ----
        if env_steps > 0 or grad_steps > 0:
          self.tracker.at_step(self.tracker.current_step).log({
            "replay_ratio/iter":            grad_steps / max(env_steps, 1),
            "replay_ratio/cumulative":      self._cumulative_grad_steps / max(self._cumulative_env_steps, 1),
            "replay_ratio/env_steps_total": self._cumulative_env_steps,
            "replay_ratio/grad_steps_total": self._cumulative_grad_steps,
          })

        # ---- broadcast updated weights to async workers ----
        if (
          is_dreamer
          and broadcast_every > 0
          and (i + 1) % broadcast_every == 0
        ):
          self._backend.broadcast_weights(self.model.state_bytes())

        # ---- checkpoint / eval ----
        checkpoint_path = None
        if i % self.config.training.save_every == 0:
          checkpoint_path = self._checkpoint(i)
        if i % self.config.training.eval_every == 0:
          present_checkpoint = checkpoint_path is not None
          if not present_checkpoint:
            checkpoint_path = self._checkpoint(i, temporary=True)
          self._eval_phase(i, checkpoint_path=checkpoint_path, cleanup=not present_checkpoint)
    finally:
      self._backend.shutdown()

    final_path = os.path.join(self._checkpoint_dir(), "final.safetensors")
    os.makedirs(self._checkpoint_dir(), exist_ok=True)
    self.model.save(final_path)
    logger.info(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
  from chuck_dreamer.config import load_config
  config = load_config()

  trainer = Trainer(config)
  trainer.train()
