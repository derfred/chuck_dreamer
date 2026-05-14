"""Evaluator — burn-in posterior + open-loop prior comparison on held-out episodes.

The :class:`Evaluator` reads episodes from ``eval.data_path``, runs each
through the world model under two regimes that share a burn-in:

  * **Posterior tail** — the latent is refined with the encoder embedding at
    every step (teacher-forced reconstruction).
  * **Prior tail** — after the burn-in, the model rolls forward purely with
    its own dynamics (open-loop reconstruction).

It logs scalar reconstruction errors (per-step and aggregate) to the
:class:`Tracker`, and dumps per-episode artifacts via
``Tracker.maybe_log_eval_episode`` when ``logging.episodes.dump_path`` is
set — observation input, the processed image fed to the encoder,
posterior/prior reconstructions, and the latent ``h`` / ``s`` trajectories.
"""

from __future__ import annotations

import logging
import os
from typing import Any
import tqdm

import numpy as np

from ..dreamer.world_model import EvalRollout, eval_split_rollout
from .episode_dataset import EpisodeDataset
from .episode_processor import processor_for

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image preprocessing — match the encoder's normalization
# ---------------------------------------------------------------------------


def _normalize_image(img: np.ndarray) -> np.ndarray:
  """Match :class:`CNNEncoder`'s input normalization (uint8 → [-0.5, 0.5])."""
  arr = np.asarray(img)
  if arr.dtype == np.uint8:
    return arr.astype(np.float32) / 255.0 - 0.5
  return arr.astype(np.float32, copy=False)


def _processed_view(obs: Any) -> Any:
  """The tensor as the encoder actually consumes it (post-normalize).

  For dict obs, recurse leaf-by-leaf and only normalize the image leaf.
  """
  if isinstance(obs, dict):
    out: dict[str, np.ndarray] = {}
    for k, v in obs.items():
      out[k] = _normalize_image(v) if k == "image" else np.asarray(v, dtype=np.float32)
    return out
  return _normalize_image(obs)


# ---------------------------------------------------------------------------
# Reconstruction error helpers
# ---------------------------------------------------------------------------


def _recon_error(recon: Any, target: Any) -> np.ndarray:
  """Per-step squared error summed over the obs feature axes.

  ``recon`` is the decoder output (already in normalized space for image
  obs); ``target`` is the raw episode obs that needs the same
  normalization applied before comparison. Both have a leading time axis.
  """
  if isinstance(recon, dict):
    img_err = _recon_error(recon["image"],   target["image"])
    pro_err = _recon_error(recon["proprio"], target["proprio"])
    return img_err + pro_err

  recon = np.asarray(recon, dtype=np.float32)
  tgt   = _normalize_image(target) if recon.ndim >= 3 else np.asarray(target, dtype=np.float32)
  diff  = recon - tgt
  axes  = tuple(range(1, diff.ndim))
  return np.asarray((diff ** 2).sum(axis=axes), dtype=np.float32)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
  """Runs burn-in / open-loop eval rollouts on a held-out episode set.

  One evaluator instance is reusable across iterations: the episodes are
  re-loaded on every :meth:`evaluate` call so freshly-recorded eval data
  is picked up automatically when the path changes contents.
  """

  def __init__(self, config, model):
    self.config     = config
    self.model      = model
    self._processor = processor_for(config)
    eval_cfg        = config.eval
    self._dataset   = EpisodeDataset(eval_cfg.data_path, format=eval_cfg.data_format)

  # ---------- public API ----------

  def warmup(self):
    """Eagerly load the eval episode set into memory.

    Idempotent. On a missing or empty eval directory the dataset stays
    empty and :meth:`evaluate` is a silent no-op.
    """
    if self._dataset.loaded:
      return
    eval_cfg = self.config.eval
    if not eval_cfg.data_path or not os.path.exists(eval_cfg.data_path):
      logger.warning(f"Eval data path {eval_cfg.data_path!r} does not exist. Eval phase will be skipped.")
      return
    logger.info(
      f"Evaluator will read eval episodes from {eval_cfg.data_path!r} "
      f"with format {eval_cfg.data_format!r}"
    )
    self._dataset.load()
    if len(self._dataset) == 0:
      logger.warning(
        f"Eval data path {eval_cfg.data_path!r} contains no episodes. Eval phase will be skipped."
      )

  def evaluate(self, iteration: int, tracker) -> None:
    """Run one eval pass and log to ``tracker``.

    No-op if the eval directory is missing or no episode is long enough
    for one burn-in + one prediction step.
    """
    eval_cfg = self.config.eval
    if not self._dataset.loaded or len(self._dataset) == 0:
      return

    burn_in = int(eval_cfg.burn_in)
    if burn_in < 0:
      raise ValueError(f"eval.burn_in must be >= 0; got {burn_in}")

    with tracker.scope({"phase": "eval", "iteration": iteration}) as eval_tracker:
      per_episode_post: list[np.ndarray] = []
      per_episode_prior: list[np.ndarray] = []
      ep_count = 0

      for ep_idx, raw_episode in enumerate(tqdm.tqdm(
        self._dataset, desc=f"Evaluating episodes - iteration {iteration}",
      )):
        stem = raw_episode.stem
        try:
          episode = self._processor(raw_episode.data)
        except ValueError as exc:
          logger.warning(f"Skipping eval episode {stem!r}: {exc}")
          continue

        result = self._run_episode(episode, burn_in)
        if result is None:
          logger.debug(f"Eval episode {stem!r} too short for burn_in={burn_in}; skipping")
          continue

        # Log per-episode aggregates and persist artifacts.
        post_err  = result["error_posterior"]
        prior_err = result["error_prior"]
        per_episode_post.append(post_err)
        per_episode_prior.append(prior_err)

        eval_tracker.log({
          "episode_index":         ep_idx,
          "source":                stem,
          "recon/posterior_mean":  float(post_err.mean()),
          "recon/prior_mean":      float(prior_err.mean()),
          "recon/posterior_final": float(post_err[-1]),
          "recon/prior_final":     float(prior_err[-1]),
          "horizon":               int(prior_err.shape[0]),
        })

        eval_tracker.maybe_log_eval_episode(
          self._episode_artifacts(episode, result, burn_in),
          iteration       = iteration,
          source_filename = stem,
          data            = {
            "episode_index": ep_idx,
            "burn_in":       burn_in,
          },
        )

        ep_count += 1

      if ep_count == 0:
        logger.warning(
          f"Eval phase produced no usable episodes (burn_in={burn_in})."
        )
        return

      post_curve  = self._stack_curve(per_episode_post)
      prior_curve = self._stack_curve(per_episode_prior)
      eval_tracker.log({
        "num_episodes":          ep_count,
        "recon/posterior_mean":  float(np.nanmean(post_curve)),
        "recon/prior_mean":      float(np.nanmean(prior_curve)),
        "recon/posterior_final": float(np.nanmean(post_curve[:, -1])),
        "recon/prior_final":     float(np.nanmean(prior_curve[:, -1])),
      })

  # ---------- internals ----------

  def _run_episode(self, episode: dict[str, Any], burn_in: int) -> dict[str, Any] | None:
    """Run one episode through ``eval_split_rollout``.

    Returns ``None`` if the episode is shorter than ``burn_in + 1`` steps.
    """
    obs    = episode["obs"]
    action = np.asarray(episode["action"], dtype=np.float32)
    T      = action.shape[0]
    horizon = T - burn_in
    if horizon < 1:
      return None

    # Align embeddings with actions: T entries of both. The window we
    # actually evaluate over is [0, T) — burn_in steps then horizon prior
    # steps.
    obs_batched    = self._add_batch(obs, T)
    action_batched = action[None]  # (1, T, action_dim)

    embeds = self.model.encode(obs_batched)

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

    recon_post  = self._unbatch(rollout.posterior.obs_pred_numpy())
    recon_prior = self._unbatch(rollout.prior.obs_pred_numpy())

    target      = self._slice_obs(obs, burn_in, T)
    error_post  = _recon_error(recon_post,  target)
    error_prior = _recon_error(recon_prior, target)

    h_post  = self._unbatch(rollout.posterior.h_numpy())
    s_post  = self._unbatch(rollout.posterior.s_numpy())
    h_prior = self._unbatch(rollout.prior.h_numpy())
    s_prior = self._unbatch(rollout.prior.s_numpy())

    return {
      "recon_posterior": recon_post,
      "recon_prior":     recon_prior,
      "target_window":   target,
      "error_posterior": error_post,
      "error_prior":     error_prior,
      "h_post":          h_post,
      "s_post":          s_post,
      "h_prior":         h_prior,
      "s_prior":         s_prior,
    }

  def _episode_artifacts(
    self, episode: dict[str, Any], result: dict[str, Any], burn_in: int,
  ) -> dict[str, Any]:
    """Pack the per-step record consumed by :class:`EvalEpisodeWriter`."""
    # We log only the post-burn-in window so the writer's arrays all line
    # up on the same time axis — that's the segment where posterior vs
    # prior actually diverge.
    obs        = self._slice_obs(episode["obs"], burn_in, burn_in + result["error_prior"].shape[0])
    processed  = _processed_view(obs)
    return {
      "raw_obs":         obs,
      "processed_obs":   processed,
      "recon_posterior": result["recon_posterior"],
      "recon_prior":     result["recon_prior"],
      "h_post":          result["h_post"],
      "s_post":          result["s_post"],
      "h_prior":         result["h_prior"],
      "s_prior":         result["s_prior"],
    }

  @staticmethod
  def _add_batch(obs: Any, T: int) -> Any:
    """Add a leading batch axis and slice to the first ``T`` steps.

    ``obs`` from the processor has ``T+1`` entries; we drop the last to
    align with the buffer's training-time convention (obs[t] paired with
    action[t]).
    """
    if isinstance(obs, dict):
      return {k: np.asarray(v[:T])[None] for k, v in obs.items()}
    return np.asarray(obs[:T])[None]

  @staticmethod
  def _slice_obs(obs: Any, lo: int, hi: int) -> Any:
    if isinstance(obs, dict):
      return {k: np.asarray(v[lo:hi]) for k, v in obs.items()}
    return np.asarray(obs[lo:hi])

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
