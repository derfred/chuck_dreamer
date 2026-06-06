"""Shared eval primitives: episode loading + one-shot split rollout.

The online evaluator, the deep evaluator, and the notebooks all call
through here. Two pieces:

* :func:`load_eval_episodes` — iterate over ``cfg.eval.source`` (or one
  explicit override path), apply the obs-mode processor, drop episodes
  shorter than ``min_len``, and yield :class:`EvalEpisode` records. Each
  record carries the raw arrays alongside the processed dict so callers
  that need ground-truth columns the modal processor drops (object_xy,
  ee_pos, …) can still get them.
* :func:`run_split_rollout` — single call that wraps obs, encodes the
  embedding stream, calls :func:`eval_split_rollout`, and returns a
  numpy-only :class:`RolloutOutputs` so post-processing on CPU doesn't
  pay for mlx/torch round-trips per metric.

Timing convention: the RSSM step ``t`` consumes the embedding of the
observation *after* ``action_t``. The processor returns ``T+1`` obs and
``T`` actions, so we feed ``encode(obs)[1:T+1]`` as the embed stream.
This matches the notebook convention; the previous Evaluator used
``obs[:T]`` and was off by one — :func:`run_split_rollout` is the
canonical reference for new callers.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..dreamer.world_model import EvalRollout, as_numpy, eval_split_rollout
from ..training.episode_dataset import EpisodeDataset
from ..training.episode_processor import processor_for
from ..training.observation import Observation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvalEpisode:
  """One eval episode in both raw and processed form.

  ``raw`` is the on-disk arrays dict (object_xy, ee_pos, joint_qpos, …)
  before the modal processor; ``processed`` is the buffer-shaped dict
  the encoder consumes (``obs``, ``action``, ``reward`` keys, with obs
  resized + normalized to match the trained model). ``source_dir`` is
  the directory the episode came from, for multi-source eval logging.
  """

  # `raw` may be a plain arrays dict or an Episode Mapping container (from the
  # dataset loader's `.data`); read-only here, so Mapping accepts both.
  raw:        Mapping[str, Any]
  processed:  dict[str, Any]
  stem:       str
  source_dir: str

  @property
  def num_actions(self) -> int:
    return int(np.asarray(self.processed["action"]).shape[0])


def _iter_source(
  path: str,
  fmt: str,
  num_episodes: int | None,
  processor,
  min_len: int,
  *,
  rng: np.random.Generator | None = None,
) -> Iterator[EvalEpisode]:
  """Yield :class:`EvalEpisode` records from one directory."""
  if not path or not os.path.exists(path):
    logger.warning(f"Eval source {path!r} does not exist. Skipping.")
    return
  source_label = Path(path).name or path
  dataset = EpisodeDataset(path, format=fmt)
  for episode in dataset.stream(num_episodes=num_episodes, rng=rng):
    try:
      processed = processor(episode.data)
    except ValueError as exc:
      logger.warning(f"Skipping eval episode {source_label}/{episode.stem!r}: {exc}")
      continue
    n = int(np.asarray(processed["action"]).shape[0])
    if n < min_len:
      logger.debug(
        f"Skipping eval episode {source_label}/{episode.stem!r}: "
        f"{n} actions < min_len {min_len}"
      )
      continue
    yield EvalEpisode(
      raw=episode.data,
      processed=processed,
      stem=episode.stem,
      source_dir=source_label,
    )


def load_eval_episodes(
  config,
  *,
  sources:      list[dict] | None = None,
  min_len:      int = 0,
  max_episodes: int | None = None,
  rng:          np.random.Generator | None = None,
) -> Iterator[EvalEpisode]:
  """Yield processed eval episodes from one or more directories.

  ``sources`` defaults to ``config.eval.source`` (the normalized
  ``list[dict]`` form with ``path`` / ``format`` / ``num_episodes`` keys).
  Pass an explicit list to point at one specific directory — useful when
  invoking from the notebook with the data_path papermill parameter.

  ``min_len`` is in action units (``T``, not ``T+1``). Episodes shorter
  than ``min_len`` are skipped with a debug log.

  ``max_episodes`` caps the total yield across all sources combined; the
  default ``None`` is "no cap on top of each source's own ``num_episodes``".
  """
  processor = processor_for(config)
  src_list = sources if sources is not None else list(config.eval.source)
  yielded = 0
  for src in src_list:
    for episode in _iter_source(
      src["path"], src["format"], src.get("num_episodes"),
      processor, min_len, rng=rng,
    ):
      yield episode
      yielded += 1
      if max_episodes is not None and yielded >= max_episodes:
        return


# ---------------------------------------------------------------------------
# Split rollout
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RolloutOutputs:
  """Per-episode rollout result, fully numpy-side.

  All arrays are unbatched (no leading batch axis) and aligned along the
  post-burn-in window: time index ``k`` here corresponds to global step
  ``burn_in + k`` in the episode. ``target_obs`` is the ground-truth
  processed obs sliced to that window for direct subtraction in metrics.

  ``recon_*`` and ``reward_*`` may be ``None`` if decoding / reward
  prediction was disabled. ``h_*`` / ``s_*`` always populate.
  """

  burn_in:           int
  horizon:           int
  target_obs:        Any                       # (horizon, ...) or dict
  full_target_obs:   Any                       # (burn_in + horizon, ...) for baselines
  reward:            np.ndarray                # (horizon,)
  recon_posterior:   Any | None
  recon_prior:       Any | None
  reward_posterior:  np.ndarray | None
  reward_prior:      np.ndarray | None
  h_posterior:       np.ndarray                # (horizon, deter_dim)
  s_posterior:       np.ndarray                # (horizon, stoch_dim)
  h_prior:           np.ndarray
  s_prior:           np.ndarray


def _unbatch(x: Any) -> Any:
  if isinstance(x, dict):
    return {k: _unbatch(v) for k, v in x.items()}
  return np.asarray(x)[0]


def _slice_obs_window(obs_value: Any, lo: int, hi: int) -> Any:
  if isinstance(obs_value, dict):
    return {k: v[lo:hi] for k, v in obs_value.items()}
  return obs_value[lo:hi]


def run_split_rollout(
  model,
  episode:        EvalEpisode,
  *,
  burn_in:        int,
  horizon:        int | None = None,
  obs_mode:       Any,
  decode:         bool = True,
  predict_reward: bool = True,
) -> RolloutOutputs | None:
  """Run one episode through :func:`eval_split_rollout`.

  Returns ``None`` if the episode is shorter than ``burn_in + 1`` action
  steps (no horizon to roll). When ``horizon`` is ``None`` it defaults to
  "as many steps as the episode allows after burn-in".

  Image obs are normalized to ``[-0.5, 0.5]`` via the
  :class:`Observation` wrapper before encoding, so the returned
  ``target_obs`` matches the float space the decoder produces.
  """
  action = np.asarray(episode.processed["action"], dtype=np.float32)
  T_act  = action.shape[0]
  if horizon is None:
    horizon = T_act - burn_in
  if horizon < 1:
    return None
  T_window = burn_in + horizon
  if T_window > T_act:
    return None

  obs = Observation.from_buffer_value(episode.processed["obs"], obs_mode)  # type: ignore[arg-type]
  obs = obs.normalize_image()

  # encode(obs) → (T_act + 1, ...). RSSM step t consumes embed of obs
  # AFTER action_t, so we feed embeds[1 : T_window + 1].
  obs_batched = obs.add_batch_axis()
  embeds_full = model.encode(obs_batched.to_buffer_value())
  embeds      = embeds_full[:, 1:T_window + 1]

  action_batched = action[:T_window][None]

  rollout: EvalRollout = eval_split_rollout(
    model,
    embeds=embeds,
    actions=action_batched,
    burn_in=burn_in,
    horizon=horizon,
    sample=False,
    decode=decode,
    predict_reward=predict_reward,
  )

  recon_posterior = _unbatch(rollout.posterior.obs_pred_numpy()) if decode else None
  recon_prior     = _unbatch(rollout.prior.obs_pred_numpy())     if decode else None

  reward_posterior: np.ndarray | None = None
  reward_prior:     np.ndarray | None = None
  if predict_reward and rollout.posterior.rewards is not None:
    reward_posterior = _unbatch(as_numpy(rollout.posterior.rewards))
  if predict_reward and rollout.prior.rewards is not None:
    reward_prior = _unbatch(as_numpy(rollout.prior.rewards))

  # Targets: post-action obs are obs[1:], so the prior-tail target is
  # obs[burn_in + 1 : T_window + 1]. The "full" target carries the
  # burn-in window too, for the zero-dynamics baseline.
  target_obs = _slice_obs_window(
    obs.to_buffer_value(), burn_in + 1, T_window + 1,
  )
  full_target_obs = _slice_obs_window(
    obs.to_buffer_value(), 1, T_window + 1,
  )

  reward_arr = np.asarray(episode.processed["reward"], dtype=np.float32).reshape(-1)
  reward = reward_arr[burn_in:T_window]

  return RolloutOutputs(
    burn_in=burn_in,
    horizon=horizon,
    target_obs=target_obs,
    full_target_obs=full_target_obs,
    reward=reward,
    recon_posterior=recon_posterior,
    recon_prior=recon_prior,
    reward_posterior=reward_posterior,
    reward_prior=reward_prior,
    h_posterior=_unbatch(rollout.posterior.h_numpy()),
    s_posterior=_unbatch(rollout.posterior.s_numpy()),
    h_prior=_unbatch(rollout.prior.h_numpy()),
    s_prior=_unbatch(rollout.prior.s_numpy()),
  )
