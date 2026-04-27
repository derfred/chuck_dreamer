import logging
import os

from chuck_dreamer.sim.data_collection import RandomPushPolicy
from chuck_dreamer.sim.pushing_env import PushingEnv
from chuck_dreamer.sim.scene_player import ScenePlayer

from .dreamer.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class Trainer:
  def __init__(self, config):
    self.config         = config
    self._replay_buffer = ReplayBuffer(
      capacity_steps=config.data.buffer_size,
      min_episode_len=config.training.min_episode_len,
      seed=config.seed,
    )
    self.env    = PushingEnv(config)
    self.policy = RandomPushPolicy()
    self.player = ScenePlayer(config, self.env, self.policy)

  def _warmup(self):
    if not os.path.exists(self.config.data.warmup_path):
      logger.warning(f"Warmup path {self.config.data.warmup_path} does not exist. Skipping warmup.")
      return
    logger.info(f"Loading warmup episodes from {self.config.data.warmup_path} (format={self.config.data.warmup_format})")
    self._replay_buffer.load_sim_episodes(self.config.data.warmup_path, self.config.data.warmup_format, progress=True)

  def _collect_phase(self):
    for _ in range(self.config.training.num_collect_episodes):
      self.player.reset()
      episode_data, outcome = self.player.run_headless(max_steps=self.config.sim.max_steps)
      logger.info(f"Collected episode with outcome: {outcome}")
      if episode_data is None:
        logger.warning("Collected episode is empty, skipping.")
        continue
      self._replay_buffer.add_sim_episode(episode_data)

  def _train_phase(self):
    pass

  def _eval_phase(self):
    pass

  def _checkpoint(self):
    pass

  def train(self):
    self._warmup()
    for i in range(self.config.training.num_iterations):
      self._collect_phase()
      self._train_phase()
      if i % self.config.training.eval_every == 0:
        self._eval_phase()
      if i % self.config.training.save_every == 0:
        self._checkpoint()


if __name__ == "__main__":
  from chuck_dreamer.config import load_config
  config = load_config()

  trainer = Trainer(config)
  trainer.train()
