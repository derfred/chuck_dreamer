from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime

from ..sim.episode_writer import EpisodeWriter


_UNSET: object = object()


class Tracker:
  def __init__(self, config, data={}, parent=None, *, step=_UNSET):
    self.config  = config
    self.data    = data
    self._parent = parent
    # When set on a derived tracker, all log() calls through this node
    # (and its children that don't override) use this step value. Walks
    # up the parent chain in _resolve_step. Sentinel distinguishes
    # "explicitly set to None" from "not set on this node".
    self._step   = step

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

      # Owning the gradient-step counter here keeps the trainer ignorant
      # of the wandb/trackio step axis: it just calls `tick()` to
      # advance and bind the step on a derived tracker. Lives on the
      # root only; children read/write via :meth:`_root`.
      self._step_counter = 0

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
      metadata={
        "config":     asdict(scene),
        "seed":       self.config.seed,
        "source":     "collect",
        "outcome":    outcome,
        "goal_xy":    scene.goal_pos,
        "mat_centre": scene.mat_centre,
        **data,
      },
      name_suffix=f"{name}{number:03d}",
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
      metadata={
        "seed":      self.config.seed,
        "source":    "eval",
        "iteration": iteration,
        **data,
      },
      name_suffix=f"iteration{iteration:04d}-{source_filename}",
    )

  def _resolve_step(self, explicit) -> int | None:
    # Explicit kwarg wins. Otherwise walk up the chain for the nearest
    # node with a bound step. Falls back to None (wandb auto-increment).
    if explicit is not _UNSET:
      return explicit
    node = self
    while node is not None:
      if node._step is not _UNSET:
        return node._step
      node = node._parent
    return None

  def log(self, data: dict, *, step=_UNSET):
    # `step` keeps wandb/trackio rows on a single monotonic axis owned
    # by the trainer (gradient-step counter). Without it, every log()
    # auto-increments wandb's internal step and rows without a
    # `wm_loss` key produce visible gaps in the wm_loss series. Inner
    # code usually omits `step=` and inherits from a step-bound parent
    # tracker created via :meth:`at_step`.
    resolved = self._resolve_step(step)
    if self._parent:
      self._parent.log({**self.data, **data}, step=resolved)
    elif self.config.logging.logger == "wandb":
      import wandb
      wandb.log(data, step=resolved)
    elif self.config.logging.logger == "trackio":
      import trackio
      trackio.log(data, step=resolved)

  def log_replay_buffer_size(self, replay_buffer):
    """Log total replay-buffer size plus a per-tag breakdown of episodes/steps."""
    data = {"replay_buffer_size": len(replay_buffer)}
    for tag, (n_eps, n_steps) in replay_buffer.size_by_tag().items():
      data[f"replay_buffer_size/{tag}/episodes"] = n_eps
      data[f"replay_buffer_size/{tag}/steps"] = n_steps
    self.log(data)

  def log_batch_tags(self, batch_tags):
    """Log per-batch episode counts grouped by tag.

    ``batch_tags`` is the list of tag tuples returned in
    :meth:`ReplayBuffer.sample`'s ``"tags"`` entry — one tuple per
    sampled sequence. Untagged sequences land in the ``"untagged"``
    bucket. Multi-tag sequences increment every matching bucket, so the
    sum across buckets can exceed batch_size; that's deliberate so each
    bucket answers "how many sequences carried this tag?" independently.
    """
    counts: dict[str, int] = {}
    for tags in batch_tags:
      keys = tags if tags else ("",)
      for t in keys:
        counts[t] = counts.get(t, 0) + 1
    if counts:
      self.log({f"batch_tag/{t or 'untagged'}": n for t, n in counts.items()})

  def derive(self, data: dict, *, step=_UNSET):
    return Tracker(self.config, data=data, parent=self, step=step)

  def at_step(self, step: int | None) -> "Tracker":
    """Return a child tracker that binds ``step`` for all log calls.

    Use at the boundary of a logical "tick" (e.g. one gradient step or
    one eval phase). Inner code that takes the returned tracker as a
    parameter no longer needs to pass ``step=`` explicitly — the
    derived tracker carries it via the parent chain. Combine with
    :meth:`scope` to attach both step and phase data.
    """
    return self.derive({}, step=step)

  @property
  def current_step(self) -> int:
    """Current value of the root tracker's gradient-step counter."""
    return self._collect_root()._step_counter

  def set_step(self, step: int) -> None:
    """Reset the gradient-step counter (e.g. on resume).

    Subsequent :meth:`tick` calls advance from this value.
    """
    self._collect_root()._step_counter = int(step)

  def tick(self) -> "Tracker":
    """Advance the gradient-step counter and return a step-bound tracker.

    Combined op — folds the common ``counter += 1; tracker.at_step(counter)``
    pattern at the top of a gradient-step loop into one call. The returned
    tracker is a child of ``self`` so any scope data (``phase``,
    ``iteration``) is inherited; only the bound step changes.
    """
    root = self._collect_root()
    root._step_counter += 1
    return self.at_step(root._step_counter)

  @contextmanager
  def scope(self, data: dict, *, step=_UNSET):
    yield self.derive(data, step=step)
