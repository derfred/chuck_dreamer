"""Online (in-training) evaluator — fast scalar metrics per eval phase.

Replaces the previous ``training.evaluator.Evaluator``. Same role —
runs every ``training.eval_every`` iterations on the main thread, logs
to the :class:`~chuck_dreamer.training.tracker.Tracker`, optionally
dumps per-episode artifacts — but the rollout + metric code is shared
with the deep evaluator via :mod:`chuck_dreamer.eval.core` and
:mod:`chuck_dreamer.eval.metrics`.

The evaluator also drives deep-eval scheduling: when
``cfg.eval.deep.every_n_evals`` is set and this phase falls on it,
:meth:`evaluate` fires :class:`DeepEvalRunner` against the latest
checkpoint. By default the runner spawns a detached subprocess so deep
eval never blocks training.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tqdm

from ..training.episode_dataset import EpisodeDataset
from ..training.episode_processor import processor_for
from .core import EvalEpisode, RolloutOutputs, run_split_rollout
from .deep import DeepEvalRunner
from .metrics import recon_l2, reward_abs_error, stack_curves, zero_dynamics_prediction

logger = logging.getLogger(__name__)


@dataclass
class _EvalSource:
  """One ``eval.source`` entry paired with its lazy :class:`EpisodeDataset`."""

  path:         str
  num_episodes: int | None
  dataset:      EpisodeDataset
  label:        str


class OnlineEvaluator:
  """Runs burn-in / open-loop rollouts on every eval phase.

  Sources are read from ``config.eval.source``. The dataset for each
  source is cached on first :meth:`warmup` and reused across phases.
  """

  def __init__(self, config, model):
    self.config     = config
    self.model      = model
    self._obs_mode  = str(config.env.obs_mode)
    self._processor = processor_for(config)
    self._sources: list[_EvalSource] = []
    for src in config.eval.source:
      path = src["path"]
      self._sources.append(_EvalSource(
        path=path,
        num_episodes=src["num_episodes"],
        dataset=EpisodeDataset(path, format=src["format"]),
        label=(Path(path).name or path),
      ))
    self._deep = DeepEvalRunner(config)

  # ---------- public API ----------

  def warmup(self) -> None:
    """Eagerly load every eval source into memory. Idempotent."""
    for src in self._sources:
      if src.dataset.loaded:
        continue
      if not src.path or not os.path.exists(src.path):
        logger.warning(f"Eval source {src.path!r} does not exist. Skipping.")
        continue
      logger.info(
        f"Loading eval episodes from {src.path!r} "
        f"(format={src.dataset.format}, num_episodes={src.num_episodes})"
      )
      src.dataset.load(num_episodes=src.num_episodes)
      if len(src.dataset) == 0:
        logger.warning(f"Eval source {src.path!r} contains no episodes.")

  def evaluate(self, iteration: int, tracker, *, checkpoint_path: str | None = None, cleanup: bool = False) -> None:
    """Run one eval pass across all sources and log to ``tracker``.

    No-op if every eval source ended up empty. When the deep-eval
    cadence fires (``cfg.eval.deep.every_n_evals``) the deep runner is
    invoked at the end with ``latest_checkpoint`` — pass the path to the
    most recent ``step_*.safetensors`` so the subprocess loads a stable
    snapshot independent of the live training weights.
    """
    eval_cfg = self.config.eval
    active = [s for s in self._sources if s.dataset.loaded and len(s.dataset) > 0]
    if not active:
      return

    burn_in = int(eval_cfg.burn_in)
    if burn_in < 0:
      raise ValueError(f"eval.burn_in must be >= 0; got {burn_in}")

    with tracker.scope({"phase": "eval", "iteration": iteration}) as eval_tracker:
      self._run_phase(active, iteration, eval_tracker, burn_in)

    self._deep.maybe_fire(iteration=iteration, checkpoint_path=checkpoint_path, cleanup=cleanup)

  # ---------- internals ----------

  def _run_phase(
    self,
    active:       list[_EvalSource],
    iteration:    int,
    eval_tracker, # parent-scoped tracker
    burn_in:      int,
  ) -> None:
    per_episode_posterior: list[np.ndarray] = []
    per_episode_prior:     list[np.ndarray] = []
    reward_prior_curves:   list[np.ndarray] = []
    zero_dyn_curves:       list[np.ndarray] = []

    ep_count   = 0
    global_idx = 0

    for src in active:
      src_posterior: list[np.ndarray] = []
      src_prior:     list[np.ndarray] = []

      for ep_idx, raw_episode in enumerate(tqdm.tqdm(
        src.dataset,
        desc=f"Evaluating {src.label} - iteration {iteration}",
      )):
        try:
          processed = self._processor(raw_episode.data)
        except ValueError as exc:
          logger.warning(f"Skipping eval episode {src.label}/{raw_episode.stem!r}: {exc}")
          continue
        episode = EvalEpisode(
          raw=raw_episode.data,
          processed=processed,
          stem=raw_episode.stem,
          source_dir=src.label,
        )

        result = run_split_rollout(
          self.model, episode,
          burn_in=burn_in,
          obs_mode=self._obs_mode,
          decode=True,
          predict_reward=True,
        )
        if result is None:
          logger.debug(
            f"Eval episode {src.label}/{episode.stem!r} too short for burn_in={burn_in}; skipping"
          )
          continue

        posterior_err = recon_l2(result.recon_posterior, result.target_obs)
        prior_err     = recon_l2(result.recon_prior,     result.target_obs)
        per_episode_posterior.append(posterior_err)
        per_episode_prior.append(prior_err)
        src_posterior.append(posterior_err)
        src_prior.append(prior_err)

        log_dict: dict[str, Any] = {
          "episode_index":         global_idx,
          "source_dir":            src.label,
          "source":                episode.stem,
          "recon/posterior_mean":  float(posterior_err.mean()),
          "recon/prior_mean":      float(prior_err.mean()),
          "recon/posterior_final": float(posterior_err[-1]),
          "recon/prior_final":     float(prior_err[-1]),
          "horizon":               int(prior_err.shape[0]),
        }

        if result.reward_prior is not None:
          rew_err_prior = reward_abs_error(result.reward_prior, result.reward)
          reward_prior_curves.append(rew_err_prior)
          log_dict["reward/prior_mae_mean"]  = float(rew_err_prior.mean())
          log_dict["reward/prior_mae_final"] = float(rew_err_prior[-1])

        if burn_in > 0:
          zero_pred = zero_dynamics_prediction(result.full_target_obs, burn_in - 1)
          full_err  = recon_l2(zero_pred, result.full_target_obs)
          zero_open = full_err[burn_in:]
          if zero_open.size > 0:
            zero_dyn_curves.append(zero_open)
            log_dict["recon/zero_dyn_mean"] = float(zero_open.mean())

        eval_tracker.log(log_dict)

        eval_tracker.maybe_log_eval_episode(
          self._episode_artifacts(result),
          iteration=iteration,
          source_filename=(
            f"{src.label}__{episode.stem}" if len(active) > 1 else episode.stem
          ),
          data={
            "episode_index": ep_idx,
            "source_dir":    src.label,
            "burn_in":       burn_in,
          },
        )

        ep_count   += 1
        global_idx += 1

      if src_posterior and len(active) > 1:
        src_post_curve  = stack_curves(src_posterior)
        src_prior_curve = stack_curves(src_prior)
        eval_tracker.log({
          "source_dir":            src.label,
          "num_episodes":          len(src_posterior),
          "recon/posterior_mean":  float(np.nanmean(src_post_curve)),
          "recon/prior_mean":      float(np.nanmean(src_prior_curve)),
          "recon/posterior_final": float(np.nanmean(src_post_curve[:, -1])),
          "recon/prior_final":     float(np.nanmean(src_prior_curve[:, -1])),
        })

    if ep_count == 0:
      logger.warning(f"Eval phase produced no usable episodes (burn_in={burn_in}).")
      return

    posterior_curve = stack_curves(per_episode_posterior)
    prior_curve     = stack_curves(per_episode_prior)
    summary: dict[str, Any] = {
      "num_episodes":          ep_count,
      "num_sources":           len(active),
      "recon/posterior_mean":  float(np.nanmean(posterior_curve)),
      "recon/prior_mean":      float(np.nanmean(prior_curve)),
      "recon/posterior_final": float(np.nanmean(posterior_curve[:, -1])),
      "recon/prior_final":     float(np.nanmean(prior_curve[:, -1])),
    }
    if reward_prior_curves:
      rew_curve = stack_curves(reward_prior_curves)
      summary["reward/prior_mae_mean"]  = float(np.nanmean(rew_curve))
      summary["reward/prior_mae_final"] = float(np.nanmean(rew_curve[:, -1]))
    if zero_dyn_curves:
      zd_curve = stack_curves(zero_dyn_curves)
      summary["recon/zero_dyn_mean"] = float(np.nanmean(zd_curve))
      # Ratio "prior beats zero-dyn baseline" — < 1 means the model helps.
      prior_mean = summary["recon/prior_mean"]
      summary["recon/prior_over_zero_dyn"] = (
        prior_mean / max(float(np.nanmean(zd_curve)), 1e-9)
      )
    eval_tracker.log(summary)

  def _episode_artifacts(self, result: RolloutOutputs) -> dict[str, Any]:
    """Pack the per-step record consumed by the rerun episode writer."""
    return {
      "obs":             result.target_obs,
      "recon_posterior": result.recon_posterior,
      "recon_prior":     result.recon_prior,
      "h_posterior":     result.h_posterior,
      "s_posterior":     result.s_posterior,
      "h_prior":         result.h_prior,
      "s_prior":         result.s_prior,
    }
