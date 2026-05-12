from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime

from ..sim.episode_writer import EpisodeWriter


class Tracker:
  def __init__(self, config, data={}, parent=None):
    self.config  = config
    self.data    = data
    self._parent = parent

    self._collect_writer = None
    self._collect_episode_count = 0
    if parent is None:
      ep_cfg = config.logging.episodes
      if ep_cfg.every_n_collects and ep_cfg.dump_path:
        self._collect_writer = EpisodeWriter(ep_cfg.dump_path, format=ep_cfg.dump_format)

  def _collect_root(self):
    node = self
    while node._parent is not None:
      node = node._parent
    return node

  def init(self, data={}):
    self.data = data

    run_name = getattr(self.config.logging, "experiment_name", None)
    run_name = f"{run_name}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    if self.config.logging.logger == "wandb":
      import wandb
      wandb.init(project=self.config.logging.project_name, name=run_name, config=self.config)
    elif self.config.logging.logger == "trackio":
      import trackio
      trackio.init(project=self.config.logging.project_name, name=run_name, config=self.config)
    else:
      self._tracker = None

  def maybe_log_collect_episode(self, episode_data, scene, outcome, data={}):
    root = self._collect_root()
    if root._collect_writer is None:
      return

    every = self.config.logging.episodes.every_n_collects
    root._collect_episode_count += 1
    if root._collect_episode_count % every != 0:
      return

    experiment_name = self.config.logging.experiment_name
    name   = "" if experiment_name is None else f"{experiment_name}-"
    number = root._collect_episode_count // every

    root._collect_writer.write_episode(
          episode_data,
          metadata={
            "config":  asdict(scene),
            "seed":    self.config.seed,
            "source":  "collect",
            "outcome": outcome,
            "goal_xy": scene.goal_pos,
            **data
          },
          name_suffix=f"{name}{number:03d}",
        )

  def log(self, data: dict, **kwargs):
    if self._parent:
      self._parent.log({**self.data, **data}, **kwargs)
    elif self.config.logging.logger == "wandb":
      import wandb
      wandb.log({**data, **kwargs})
    elif self.config.logging.logger == "trackio":
      import trackio
      trackio.log({**data, **kwargs})

  def derive(self, data: dict):
    return Tracker(self.config, data=data, parent=self)

  @contextmanager
  def scope(self, data: dict):
    yield self.derive(data)
