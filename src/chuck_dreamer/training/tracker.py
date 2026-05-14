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
    self._episode_writer         = None
    self._collect_episode_count  = 0
    # Eval logging is gated per *phase*: every Nth eval phase dumps every
    # episode it sees. We track phases by stamping the iteration of the
    # most recent eval phase that opened, and a running phase counter.
    self._eval_phase_count        = 0
    self._eval_phase_iteration    = None
    self._eval_phase_logging      = False
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

  def maybe_log_collect_episode(self, episode_data, scene, outcome, data={}):
    """Persist one collect-phase episode every Nth call, where N is
    ``logging.episodes.every_n_collects``. No-op when the writer is
    absent or ``every`` is falsy.
    """
    root = self._collect_root()
    if root._episode_writer is None:
      return
    every = self.config.logging.episodes.every_n_collects
    if not every:
      return

    root._collect_episode_count += 1
    if root._collect_episode_count % every != 0:
      return

    experiment_name = self.config.logging.experiment_name
    name   = "" if experiment_name is None else f"{experiment_name}-"
    number = root._collect_episode_count // every
    root._episode_writer.write_episode(
      episode_data,
      metadata = {
        "config":  asdict(scene),
        "seed":    self.config.seed,
        "source":  "collect",
        "outcome": outcome,
        "goal_xy": scene.goal_pos,
        **data,
      },
      name_suffix = f"{name}{number:03d}",
    )

  def maybe_log_eval_episode(self, episode_data, *, iteration, source_filename, data={}):
    """Persist one eval rollout (raw obs, processed obs, posterior/prior
    reconstructions, latent ``h`` / ``s``).

    Gating is per *phase*, not per episode: when ``every_n_evals`` is N,
    every Nth eval phase dumps every episode it sees. Phases are
    distinguished by ``iteration`` — the first call seen for a new
    iteration opens a phase and decides whether it logs.

    ``source_filename`` is the eval episode's source stem (e.g.
    ``"episode-027"``); it shows up verbatim in the output filename as
    ``eval-iteration{iteration}-{source_filename}.{ext}``.
    """
    root = self._collect_root()
    if root._episode_writer is None:
      return
    every = self.config.logging.episodes.every_n_evals
    if not every:
      return

    # First call for a new iteration → open a new phase. Decide once
    # whether this phase logs and stick to that for the rest of the
    # episodes seen at this iteration.
    if root._eval_phase_iteration != iteration:
      root._eval_phase_count    += 1
      root._eval_phase_iteration = iteration
      root._eval_phase_logging   = (root._eval_phase_count % every == 0)

    if not root._eval_phase_logging:
      return

    root._episode_writer.write_eval_episode(
      episode_data,
      metadata = {
        "seed":      self.config.seed,
        "source":    "eval",
        "iteration": iteration,
        **data,
      },
      name_suffix = f"iteration{iteration:04d}-{source_filename}",
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
