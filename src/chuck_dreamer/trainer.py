import logging
import os
from collections import defaultdict

from .reward import build_reward_fn
from .sim.pushing_env import PushingEnv
from .sim.episode_collector import EpisodeCollector

from .training.episode_processor import processor_for
from .training.replay_buffer import ReplayBuffer
from .training.tracker import Tracker

from .dreamer import build_model, DreamerPolicy

logger = logging.getLogger(__name__)


class Trainer:
  def __init__(self, config):
    self.config = config
    self.env    = PushingEnv(config)

    self._replay_buffer = ReplayBuffer(
      capacity_steps=config.training.data.buffer_size,
      min_episode_len=config.training.min_episode_len,
      processor=processor_for(config),
      reward_fn=build_reward_fn(config.env.reward),
      seed=config.seed,
    )

    obs_shape  = self.env.model_obs_shape
    action_dim = int(self.env.action_space.shape[0])

    self.model  = build_model(config, obs_shape=obs_shape, action_dim=action_dim)
    self.policy = DreamerPolicy(self.model, act_mode=self.env.act_mode)
    self.collector = EpisodeCollector(self.env, self.policy)
    self.tracker = Tracker(config)
    self.tracker.init()

  def _warmup(self):
    data_cfg = self.config.training.data
    if not os.path.exists(data_cfg.warmup_path):
      logger.warning(f"Warmup path {data_cfg.warmup_path} does not exist. Skipping warmup.")
      return
    logger.info(f"Loading warmup episodes from {data_cfg.warmup_path} (format={data_cfg.warmup_format})")
    self._replay_buffer.load_sim_episodes(data_cfg.warmup_path, data_cfg.warmup_format, progress=True)

  def _collect_phase(self):
    collect_data = defaultdict(int)
    for _ in range(self.config.training.num_collect_episodes):
      scene = self.collector.reset()
      if self.config.env.max_steps is not None:
        scene.max_steps = int(self.config.env.max_steps)
      episode_data, outcome = self.collector.run()
      collect_data[outcome] += 1
      if episode_data is not None:
        self._replay_buffer.add_sim_episode(episode_data)
    self.tracker.derive({"phase": "collect"}).log({
      "num_episodes": self.config.training.num_collect_episodes,
      **{f"outcome/{k}": v for k, v in collect_data.items()},
    })

  def _train_phase(self):
    with self.tracker.scope({"phase": "train"}) as tracker:
      tracker.log({"replay_buffer_size": len(self._replay_buffer)})
      if not self._replay_buffer.can_sample(self.config.training.batch_size, self.config.training.seq_len):
        logger.warning("Buffer too small to sample (have %d steps); skipping train phase.", len(self._replay_buffer))
        return

      for epoch in range(self.config.training.num_gradient_steps):
        batch = self._replay_buffer.sample(self.config.training.batch_size, self.config.training.seq_len)

        # get the world model predictions for this batch
        self.model.wm_update(batch, tracker=tracker.derive({"epoch": epoch}))

  def _eval_phase(self):
    pass

  def _checkpoint_dir(self) -> str:
    name = self.config.logging.experiment_name or "default"
    return os.path.join(self.config.logging.save_dir, name)

  def _resume(self, resume: bool | str):
    """Load weights before training. ``resume`` may be:
      - False/None: no-op
      - True:       load ``{checkpoint_dir}/latest.safetensors`` if present
      - str path:   load that exact file
    """
    if not resume:
      return
    if resume is True:
      path = os.path.join(self._checkpoint_dir(), "latest.safetensors")
    else:
      path = resume
    if not os.path.exists(path):
      logger.warning(f"Resume requested but {path} does not exist; starting from scratch.")
      return
    logger.info(f"Resuming from {path}")
    self.model.load(path)

  def _checkpoint(self, step: int):
    ckpt_dir = self._checkpoint_dir()
    os.makedirs(ckpt_dir, exist_ok=True)
    step_path   = os.path.join(ckpt_dir, f"step_{step:06d}.safetensors")
    latest_path = os.path.join(ckpt_dir, "latest.safetensors")
    self.model.save(step_path)
    self.model.save(latest_path)
    logger.info(f"Saved checkpoint to {step_path}")

  def train(self, resume: bool | str = False):
    self._resume(resume)
    self._warmup()
    for i in range(self.config.training.num_iterations):
      self._collect_phase()
      self._train_phase()
      if i % self.config.training.eval_every == 0:
        self._eval_phase()
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
