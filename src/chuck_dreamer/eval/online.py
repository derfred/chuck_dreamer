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
    from ..training.observation import normalize_obs_mode
    self._obs_mode  = normalize_obs_mode(config.env.obs_mode)
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

  def evaluate(
    self,
    iteration: int,
    tracker,
    *,
    checkpoint_path: str | None = None,
    cleanup: bool = False,
  ) -> None:
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
    # Linear-probe inputs: per-episode (latent, target) pairs, pooled at
    # phase end into a single ridge fit. Episode-level train/test split
    # prevents the probe from cheating by interpolating consecutive
    # frames within an episode. See _fit_linear_probe below.
    probe_records: list[dict[str, np.ndarray]] = []

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

        # Per-episode rows live under a separate ``eval/per_episode/*``
        # namespace so the headline ``recon/*`` series stays one point
        # per eval phase. Without this, each phase wrote ~30 closely
        # spaced rows followed by a long gap, which makes the dashboard
        # smoother go haywire and the trend illegible.
        log_dict: dict[str, Any] = {
          "eval/per_episode/episode_index":         global_idx,
          "eval/per_episode/source_dir":            src.label,
          "eval/per_episode/source":                episode.stem,
          "eval/per_episode/recon/posterior_mean":  float(posterior_err.mean()),
          "eval/per_episode/recon/prior_mean":      float(prior_err.mean()),
          "eval/per_episode/recon/posterior_final": float(posterior_err[-1]),
          "eval/per_episode/recon/prior_final":     float(prior_err[-1]),
          "eval/per_episode/horizon":               int(prior_err.shape[0]),
        }

        if result.reward_prior is not None:
          rew_err_prior = reward_abs_error(result.reward_prior, result.reward)
          reward_prior_curves.append(rew_err_prior)
          log_dict["eval/per_episode/reward/prior_mae_mean"]  = float(rew_err_prior.mean())
          log_dict["eval/per_episode/reward/prior_mae_final"] = float(rew_err_prior[-1])

        if burn_in > 0:
          zero_pred = zero_dynamics_prediction(result.full_target_obs, burn_in - 1)
          full_err  = recon_l2(zero_pred, result.full_target_obs)
          zero_open = full_err[burn_in:]
          if zero_open.size > 0:
            zero_dyn_curves.append(zero_open)
            log_dict["eval/per_episode/recon/zero_dyn_mean"] = float(zero_open.mean())

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

        # Stash latents + privileged targets for the phase-level linear
        # probes. Window = post-action open-loop, matching the recon
        # target slice (raw[t] for t in burn_in+1 .. T_window+1).
        probe_rec = self._extract_probe_record(episode.raw, result, burn_in)
        if probe_rec is not None:
          probe_records.append(probe_rec)

        ep_count   += 1
        global_idx += 1

      if src_posterior and len(active) > 1:
        # Per-source aggregates ride under a source-specific prefix so
        # multiple sources don't overwrite each other (or the headline
        # phase summary) at the same step. Keys look like e.g.
        # ``eval/by_source/eval_real/recon/posterior_mean``.
        src_post_curve  = stack_curves(src_posterior)
        src_prior_curve = stack_curves(src_prior)
        prefix = f"eval/by_source/{src.label}"
        eval_tracker.log({
          f"{prefix}/num_episodes":          len(src_posterior),
          f"{prefix}/recon/posterior_mean":  float(np.nanmean(src_post_curve)),
          f"{prefix}/recon/prior_mean":      float(np.nanmean(src_prior_curve)),
          f"{prefix}/recon/posterior_final": float(np.nanmean(src_post_curve[:, -1])),
          f"{prefix}/recon/prior_final":     float(np.nanmean(src_prior_curve[:, -1])),
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

    # Linear "probe-lite": three closed-form ridge fits over pooled
    # (latent, privileged-target) pairs. R² > 0 means the probe beats
    # predicting the mean. The post→object_xy probe says "does the
    # encoder extract ball position from the image"; the prior probe
    # says "does the dynamics module preserve it across the open-loop
    # rollout". ee_xy_prior is the control: action-conditional, should
    # be ~1.0 in a healthy model.
    summary.update(self._compute_probe_metrics(probe_records))

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

  # ---------- linear probe ----------

  def _extract_probe_record(
    self,
    raw:     dict[str, Any],
    result:  RolloutOutputs,
    burn_in: int,
  ) -> dict[str, np.ndarray] | None:
    """Build the (latent, target) arrays for the linear probe.

    Both latent and target are sliced to the open-loop window
    ``[burn_in+1 : T_window+1]`` so they align with the rollout outputs
    that already use that same window. Returns ``None`` if the episode
    is missing a privileged target key (the probe is sim-only — real
    data without object_xy / ee_pos simply doesn't contribute).
    """
    if "object_xy" not in raw or "ee_pos" not in raw:
      return None
    horizon = int(result.horizon)
    T_window = burn_in + horizon
    object_xy = np.asarray(raw["object_xy"], dtype=np.float32)[burn_in + 1 : T_window + 1]
    ee_xy     = np.asarray(raw["ee_pos"],    dtype=np.float32)[burn_in + 1 : T_window + 1, :2]
    if object_xy.shape[0] != horizon or ee_xy.shape[0] != horizon:
      # Privileged target arrays shorter than the rollout — episode
      # truncated mid-window. Skip rather than misalign.
      return None
    return {
      "post":      np.concatenate([result.h_posterior, result.s_posterior], axis=-1),
      "prior":     np.concatenate([result.h_prior,     result.s_prior],     axis=-1),
      "object_xy": object_xy,
      "ee_xy":     ee_xy,
    }

  def _compute_probe_metrics(
    self,
    records: list[dict[str, np.ndarray]],
  ) -> dict[str, float]:
    """Episode-level train/test split → three closed-form ridge probes."""
    if len(records) < 2:
      # Need at least one episode each side of the split.
      return {}
    rng = np.random.default_rng(0)  # deterministic so the same eval set
    # produces the same split — apples-to-apples across phases.
    n = len(records)
    perm = rng.permutation(n)
    n_test = max(1, n // 4)
    test_idx = set(perm[:n_test].tolist())
    train_idx = [i for i in range(n) if i not in test_idx]
    test_idx_list = sorted(test_idx)

    def _pool(idx_list: list[int], latent_key: str, target_key: str):
      X = np.concatenate([records[i][latent_key] for i in idx_list], axis=0)
      Y = np.concatenate([records[i][target_key] for i in idx_list], axis=0)
      return X.astype(np.float32), Y.astype(np.float32)

    out: dict[str, float] = {}
    for latent_key, target_key, name in [
      ("post",  "object_xy", "eval/probe/object_xy_post_linear_r2"),
      ("prior", "object_xy", "eval/probe/object_xy_prior_linear_r2"),
      ("prior", "ee_xy",     "eval/probe/ee_xy_prior_linear_r2"),
    ]:
      Xtr, Ytr = _pool(train_idx,      latent_key, target_key)
      Xte, Yte = _pool(test_idx_list,  latent_key, target_key)
      out[name] = _fit_linear_probe_r2(Xtr, Ytr, Xte, Yte)
    return out


def _fit_linear_probe_r2(
  Xtr: np.ndarray,
  Ytr: np.ndarray,
  Xte: np.ndarray,
  Yte: np.ndarray,
  alpha: float = 1e-3,
) -> float:
  """Closed-form ridge in standardized space; return test R².

  ``alpha`` is small — for diagnostic probes we want to measure how
  much information is *present*, not how much survives heavy
  regularization. Bias is unpenalized (last column of the design
  matrix has its ridge term zeroed).
  """
  xm = Xtr.mean(0, keepdims=True)
  xs = Xtr.std(0,  keepdims=True) + 1e-6
  ym = Ytr.mean(0, keepdims=True)
  ys = Ytr.std(0,  keepdims=True) + 1e-6
  Xtr_n = (Xtr - xm) / xs
  Ytr_n = (Ytr - ym) / ys
  Xb = np.concatenate([Xtr_n, np.ones((Xtr_n.shape[0], 1), dtype=Xtr_n.dtype)], axis=1)
  d  = Xb.shape[1]
  A  = Xb.T @ Xb + alpha * np.eye(d, dtype=Xb.dtype)
  A[-1, -1] -= alpha  # unpenalized bias
  W  = np.linalg.solve(A, Xb.T @ Ytr_n)

  Xte_n = (Xte - xm) / xs
  Xb_te = np.concatenate([Xte_n, np.ones((Xte_n.shape[0], 1), dtype=Xte_n.dtype)], axis=1)
  pred  = (Xb_te @ W) * ys + ym

  ss_res = float(((Yte - pred) ** 2).sum())
  ss_tot = float(((Yte - Yte.mean(0, keepdims=True)) ** 2).sum())
  return 1.0 - ss_res / max(ss_tot, 1e-12)
