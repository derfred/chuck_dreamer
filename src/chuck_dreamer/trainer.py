import logging
import os
from collections import defaultdict
from dataclasses import asdict
import tqdm

from .reward import build_reward_fn
from .sim.pushing_env import PushingEnv
from .sim.episode_collector import EpisodeCollector
from .sim.episode_writer import EpisodeWriter

from .training.episode_processor import processor_for
from .training.evaluator import Evaluator
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
    self.evaluator = Evaluator(config, self.model)
    self.tracker   = Tracker(config)
    self.tracker.init()

  def _warmup(self):
    data_cfg = self.config.training.data
    if os.path.exists(data_cfg.warmup_path):
      logger.info(
        f"Loading warmup episodes from {data_cfg.warmup_path} "
        f"(format={data_cfg.warmup_format}, num_episodes={data_cfg.warmup_num_episodes})"
      )
      self._replay_buffer.load_sim_episodes(
        data_cfg.warmup_path,
        data_cfg.warmup_format,
        progress=True,
        num_episodes=data_cfg.warmup_num_episodes,
      )
    else:
      logger.warning(f"Warmup path {data_cfg.warmup_path} does not exist. Skipping buffer warmup.")

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

  def _collect_phase(self, iteration: int):
    collect_data = defaultdict(int)
    for _ in tqdm.tqdm(range(self.config.training.num_collect_episodes), desc=f"Collecting episodes - iteration {iteration}", total=self.config.training.num_collect_episodes):
      scene = self.collector.reset()
      if self.config.env.max_steps is not None:
        scene.max_steps = int(self.config.env.max_steps)

      episode_data, outcome = self.collector.run()
      collect_data[outcome] += 1

      if episode_data is not None:
        self._replay_buffer.add_sim_episode(episode_data)
        self.tracker.maybe_log_collect_episode(episode_data, scene, outcome, { "iteration": iteration })

    self.tracker.log({
      "phase": "collect",
      "iteration": iteration,
      "num_episodes": sum(collect_data.values()),
      **{f"outcome/{k}": v for k, v in collect_data.items()},
    })

  def _train_phase(self, iteration: int):
    with self.tracker.scope({"phase": "train", "iteration": iteration}) as tracker:
      tracker.log({"replay_buffer_size": len(self._replay_buffer)})
      if not self._replay_buffer.can_sample(self.config.training.batch_size, self.config.training.seq_len):
        logger.warning("Buffer too small to sample (have %d steps); skipping train phase.", len(self._replay_buffer))
        return

      for _ in tqdm.tqdm(range(self.config.training.num_gradient_steps), desc=f"Training - iteration {iteration}", total=self.config.training.num_gradient_steps):
        batch = self._replay_buffer.sample(self.config.training.batch_size, self.config.training.seq_len)
        self.model.wm_update(batch, tracker=tracker)

  def _eval_phase(self, iteration: int):
    self.evaluator.evaluate(iteration, self.tracker)

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
    logger.info(f"Resuming at iteration {start}")
    return start

  def _checkpoint(self, step: int):
    ckpt_dir = self._checkpoint_dir()
    os.makedirs(ckpt_dir, exist_ok=True)
    step_path   = os.path.join(ckpt_dir, f"step_{step:06d}.safetensors")
    latest_path = os.path.join(ckpt_dir, "latest.safetensors")
    extra = {"iteration": str(int(step))}
    self.model.save(step_path,   extra_metadata=extra)
    self.model.save(latest_path, extra_metadata=extra)
    logger.info(f"Saved checkpoint to {step_path}")

  def train(self, resume: bool | str = False):
    start = self._resume(resume)
    self._warmup()
    for i in range(start, self.config.training.num_iterations):
      self._collect_phase(i)
      self._train_phase(i)
      if i % self.config.training.eval_every == 0:
        self._eval_phase(i)
      if i % self.config.training.save_every == 0:
        self._checkpoint(i)

    final_path = os.path.join(self._checkpoint_dir(), "final.safetensors")
    os.makedirs(self._checkpoint_dir(), exist_ok=True)
    self.model.save(final_path)
    logger.info(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
  from chuck_dreamer.config import load_config
  config = load_config()

  trainer = Trainer(config)
  trainer.train()
