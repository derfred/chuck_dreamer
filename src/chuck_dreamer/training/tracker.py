from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime

from ..sim.episode_writer import EpisodeWriter


class Tracker:
  def __init__(self, config, data={}, parent=None):
    self.config  = config
    self.data    = data
    self._parent = parent

    # One writer instance backs both collect and eval dumps — the file
    # naming differs but the format/output dir are shared.
    self._episode_writer        = None
    self._collect_episode_count = 0
    self._eval_episode_count    = 0
    if parent is None:
      ep_cfg = config.logging.episodes
      collect_on = bool(ep_cfg.every_n_collects)
      eval_on    = ep_cfg.every_n_evals is not None
      if (collect_on or eval_on) and ep_cfg.dump_path:
        self._episode_writer = EpisodeWriter(ep_cfg.dump_path, format=ep_cfg.dump_format)

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

  def _maybe_log_episode(
    self,
    *,
    write_method: str,
    counter_attr: str,
    every: int | None,
    metadata: dict,
    episode_data,
    suffix_width: int,
  ):
    """Shared logic for periodic episode dumps.

    Increments ``counter_attr`` on the root tracker and, every ``every``
    calls, asks the root's episode writer to persist ``episode_data``
    via ``write_method`` (e.g. ``"write_episode"`` or
    ``"write_eval_episode"``). No-op when the writer is absent or
    ``every`` is falsy.
    """
    root = self._collect_root()
    if root._episode_writer is None or not every:
      return

    count = getattr(root, counter_attr) + 1
    setattr(root, counter_attr, count)
    if count % every != 0:
      return

    experiment_name = self.config.logging.experiment_name
    name   = "" if experiment_name is None else f"{experiment_name}-"
    number = count // every
    getattr(root._episode_writer, write_method)(
      episode_data,
      metadata=metadata,
      name_suffix=f"{name}{number:0{suffix_width}d}",
    )

  def maybe_log_collect_episode(self, episode_data, scene, outcome, data={}):
    self._maybe_log_episode(
      write_method = "write_episode",
      counter_attr = "_collect_episode_count",
      every        = self.config.logging.episodes.every_n_collects,
      suffix_width = 3,
      episode_data = episode_data,
      metadata     = {
        "config":  asdict(scene),
        "seed":    self.config.seed,
        "source":  "collect",
        "outcome": outcome,
        "goal_xy": scene.goal_pos,
        **data,
      },
    )

  def maybe_log_eval_episode(self, episode_data, data={}):
    """Persist a per-step record of one eval rollout (raw obs, processed
    obs, posterior/prior reconstructions, latent ``h`` / ``s``).

    Mirrors :meth:`maybe_log_collect_episode`: writes one file every
    ``logging.episodes.every_n_evals`` eval episodes seen on the root
    tracker.
    """
    self._maybe_log_episode(
      write_method = "write_eval_episode",
      counter_attr = "_eval_episode_count",
      every        = self.config.logging.episodes.every_n_evals,
      suffix_width = 4,
      episode_data = episode_data,
      metadata     = {
        "seed":   self.config.seed,
        "source": "eval",
        **data,
      },
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
