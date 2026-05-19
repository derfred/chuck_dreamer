"""Evaluator — burn-in posterior + open-loop prior comparison on held-out episodes.

The :class:`Evaluator` reads episodes from one or more directories listed
under ``eval.source`` (mirrors ``training.data.warmup_source``), runs each
through the world model under two regimes that share a burn-in:

  * **Posterior tail** — the latent is refined with the encoder embedding at
    every step (teacher-forced reconstruction).
  * **Prior tail** — after the burn-in, the model rolls forward purely with
    its own dynamics (open-loop reconstruction).

It logs scalar reconstruction errors (per-step and aggregate) to the
:class:`Tracker`, and dumps per-episode artifacts via
``Tracker.maybe_log_eval_episode`` when ``logging.episodes.dump_path`` is
set — observation input, posterior/prior reconstructions, and the latent
``h`` / ``s`` trajectories.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tqdm

import numpy as np

from ..dreamer.world_model import EvalRollout, eval_split_rollout
from .episode_dataset import EpisodeDataset
from .episode_processor import processor_for
from .observation import Observation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reconstruction error helpers
# ---------------------------------------------------------------------------


def _recon_error(recon: Any, target: Any) -> np.ndarray:
  """Per-step squared error summed over the obs feature axes.

  Both ``recon`` (decoder output) and ``target`` (processor output) are in
  the same float32 ``[-0.5, 0.5]`` range for image obs, so no
  normalization step is needed here. Both have a leading time axis.
  """
  if isinstance(recon, dict):
    img_err = _recon_error(recon["image"],   target["image"])
    pro_err = _recon_error(recon["proprio"], target["proprio"])
    return img_err + pro_err

  recon = np.asarray(recon, dtype=np.float32)
  tgt   = np.asarray(target, dtype=np.float32)
  diff  = recon - tgt
  axes  = tuple(range(1, diff.ndim))
  return np.asarray((diff ** 2).sum(axis=axes), dtype=np.float32)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


@dataclass
class _EvalSource:
  """One ``eval.source`` entry paired with its lazy ``EpisodeDataset``."""
  path:         str
  num_episodes: int | None
  dataset:      EpisodeDataset


class Evaluator:
  """Runs burn-in / open-loop eval rollouts on one or more held-out sets.

  Reads its sources from ``eval.source`` (string, dict, or list of
  either — see ``configs/default.yaml``). One evaluator instance is
  reusable across iterations.
  """

  def __init__(self, config, model):
    self.config     = config
    self.model      = model
    self._obs_mode  = str(config.env.obs_mode)
    self._processor = processor_for(config)
    self._sources: list[_EvalSource] = [
      _EvalSource(
        path=src["path"],
        num_episodes=src["num_episodes"],
        dataset=EpisodeDataset(src["path"], format=src["format"]),
      )
      for src in config.eval.source
    ]

  # ---------- public API ----------

  def warmup(self):
    """Eagerly load every eval source into memory.

    Idempotent. Sources whose directory is missing or empty are logged
    and skipped; :meth:`evaluate` is a silent no-op only if *every*
    source ends up empty.
    """
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

  def evaluate(self, iteration: int, tracker) -> None:
    """Run one eval pass across all sources and log to ``tracker``.

    No-op if every eval source is missing or empty.
    """
    eval_cfg = self.config.eval
    active = [s for s in self._sources if s.dataset.loaded and len(s.dataset) > 0]
    if not active:
      return

    burn_in = int(eval_cfg.burn_in)
    if burn_in < 0:
      raise ValueError(f"eval.burn_in must be >= 0; got {burn_in}")

    with tracker.scope({"phase": "eval", "iteration": iteration}) as eval_tracker:
      per_episode_posterior: list[np.ndarray] = []
      per_episode_prior:     list[np.ndarray] = []
      per_source_counts: dict[str, int] = {}
      ep_count   = 0
      global_idx = 0

      for src in active:
        src_label = Path(src.path).name or src.path
        src_posterior: list[np.ndarray] = []
        src_prior:     list[np.ndarray] = []

        for ep_idx, raw_episode in enumerate(tqdm.tqdm(
          src.dataset,
          desc=f"Evaluating {src_label} - iteration {iteration}",
        )):
          stem = raw_episode.stem
          try:
            episode = self._processor(raw_episode.data)
          except ValueError as exc:
            logger.warning(f"Skipping eval episode {src_label}/{stem!r}: {exc}")
            continue

          result = self._run_episode(episode, burn_in)
          if result is None:
            logger.debug(
              f"Eval episode {src_label}/{stem!r} too short for burn_in={burn_in}; skipping"
            )
            continue

          posterior_err = result["error_posterior"]
          prior_err     = result["error_prior"]
          per_episode_posterior.append(posterior_err)
          per_episode_prior.append(prior_err)
          src_posterior.append(posterior_err)
          src_prior.append(prior_err)

          eval_tracker.log({
            "episode_index":         global_idx,
            "source_dir":            src_label,
            "source":                stem,
            "recon/posterior_mean":  float(posterior_err.mean()),
            "recon/prior_mean":      float(prior_err.mean()),
            "recon/posterior_final": float(posterior_err[-1]),
            "recon/prior_final":     float(prior_err[-1]),
            "horizon":               int(prior_err.shape[0]),
          })

          eval_tracker.maybe_log_eval_episode(
            self._episode_artifacts(episode, result, burn_in),
            iteration=iteration,
            source_filename=f"{src_label}__{stem}" if len(active) > 1 else stem,
            data={
              "episode_index": ep_idx,
              "source_dir":    src_label,
              "burn_in":       burn_in,
            },
          )

          ep_count   += 1
          global_idx += 1

        per_source_counts[src_label] = len(src_posterior)
        if src_posterior and len(active) > 1:
          src_post_curve  = self._stack_curve(src_posterior)
          src_prior_curve = self._stack_curve(src_prior)
          eval_tracker.log({
            "source_dir":            src_label,
            "num_episodes":          len(src_posterior),
            "recon/posterior_mean":  float(np.nanmean(src_post_curve)),
            "recon/prior_mean":      float(np.nanmean(src_prior_curve)),
            "recon/posterior_final": float(np.nanmean(src_post_curve[:, -1])),
            "recon/prior_final":     float(np.nanmean(src_prior_curve[:, -1])),
          })

      if ep_count == 0:
        logger.warning(
          f"Eval phase produced no usable episodes (burn_in={burn_in})."
        )
        return

      posterior_curve = self._stack_curve(per_episode_posterior)
      prior_curve     = self._stack_curve(per_episode_prior)
      eval_tracker.log({
        "num_episodes":          ep_count,
        "num_sources":           len(active),
        "recon/posterior_mean":  float(np.nanmean(posterior_curve)),
        "recon/prior_mean":      float(np.nanmean(prior_curve)),
        "recon/posterior_final": float(np.nanmean(posterior_curve[:, -1])),
        "recon/prior_final":     float(np.nanmean(prior_curve[:, -1])),
      })

  # ---------- internals ----------

  def _run_episode(self, episode: dict[str, Any], burn_in: int) -> dict[str, Any] | None:
    """Run one episode through ``eval_split_rollout``.

    Returns ``None`` if the episode is shorter than ``burn_in + 1`` steps.
    """
    obs    = Observation.from_buffer_value(episode["obs"], self._obs_mode)  # type: ignore[arg-type]
    action = np.asarray(episode["action"], dtype=np.float32)
    T      = action.shape[0]
    horizon = T - burn_in
    if horizon < 1:
      return None

    # Align embeddings with actions: T entries of both. The window we
    # actually evaluate over is [0, T) — burn_in steps then horizon prior
    # steps.
    obs_batched    = obs[:T].add_batch_axis()
    action_batched = action[None]  # (1, T, action_dim)

    embeds = self.model.encode(obs_batched.to_buffer_value())

    rollout: EvalRollout = eval_split_rollout(
      self.model,
      embeds=embeds,
      actions=action_batched,
      burn_in=burn_in,
      horizon=horizon,
      sample=False,
      decode=True,
      predict_reward=False,
    )

    recon_posterior = self._unbatch(rollout.posterior.obs_pred_numpy())
    recon_prior     = self._unbatch(rollout.prior.obs_pred_numpy())

    target          = obs[burn_in:T].to_buffer_value()
    error_posterior = _recon_error(recon_posterior, target)
    error_prior     = _recon_error(recon_prior,     target)

    h_posterior = self._unbatch(rollout.posterior.h_numpy())
    s_posterior = self._unbatch(rollout.posterior.s_numpy())
    h_prior     = self._unbatch(rollout.prior.h_numpy())
    s_prior     = self._unbatch(rollout.prior.s_numpy())

    return {
      "recon_posterior": recon_posterior,
      "recon_prior":     recon_prior,
      "target_window":   target,
      "error_posterior": error_posterior,
      "error_prior":     error_prior,
      "h_posterior":     h_posterior,
      "s_posterior":     s_posterior,
      "h_prior":         h_prior,
      "s_prior":         s_prior,
    }

  def _episode_artifacts(
    self, episode: dict[str, Any], result: dict[str, Any], burn_in: int,
  ) -> dict[str, Any]:
    """Pack the per-step record consumed by :class:`EvalEpisodeWriter`.

    We log only the post-burn-in window so the writer's arrays all line
    up on the same time axis — that's the segment where posterior vs
    prior actually diverge.
    """
    obs = Observation.from_buffer_value(episode["obs"], self._obs_mode)  # type: ignore[arg-type]
    hi  = burn_in + result["error_prior"].shape[0]
    return {
      "obs":             obs[burn_in:hi].to_buffer_value(),
      "recon_posterior": result["recon_posterior"],
      "recon_prior":     result["recon_prior"],
      "h_posterior":     result["h_posterior"],
      "s_posterior":     result["s_posterior"],
      "h_prior":         result["h_prior"],
      "s_prior":         result["s_prior"],
    }

  @staticmethod
  def _unbatch(x: Any) -> Any:
    """Strip the leading batch axis from a backend tensor or dict thereof."""
    if isinstance(x, dict):
      return {k: Evaluator._unbatch(v) for k, v in x.items()}
    return x[0]

  @staticmethod
  def _stack_curve(curves: list[np.ndarray]) -> np.ndarray:
    """Stack per-episode error curves into ``(num_episodes, T_max)`` with NaN padding."""
    if not curves:
      return np.empty((0, 0), dtype=np.float32)
    T_max = max(c.shape[0] for c in curves)
    out = np.full((len(curves), T_max), np.nan, dtype=np.float32)
    for i, c in enumerate(curves):
      out[i, : c.shape[0]] = c
    return out
