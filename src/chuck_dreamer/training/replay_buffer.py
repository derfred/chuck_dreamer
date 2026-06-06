"""Episodic replay buffer for Dreamer v1.

See `replay_buffer.md` in the project root for the full design doc.
"""

from __future__ import annotations

import os
import pickle
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..reward import RewardFn
from ..sim.step_info import StepInfo
from .episode_dataset import EpisodeDataset, Progress
from .episode_processor import EpisodeProcessor
from .observation import Observation

Episode = dict[str, Any]


def _coerce_to_observation(raw_obs: Any, focus_mask: Any = None) -> Observation:
  """Wrap a buffer-side obs value (dict or Observation) into an Observation.

  Buffer episodes always carry obs as a component dict keyed by
  component name. The mode is implied by the dict's keys, so this
  function doesn't need a separate ``obs_mode`` argument.

  ``focus_mask`` (per-step ``(T+1, H, W)`` float32) rides alongside
  observations that include an image component when supplied; passing
  one without an image present raises.
  """
  if isinstance(raw_obs, Observation):
    if focus_mask is not None and raw_obs.focus_mask is None:
      return Observation.from_components(raw_obs.components, focus_mask=focus_mask)
    return raw_obs
  if not isinstance(raw_obs, dict):
    raise ValueError(
      f"buffer obs must be a dict keyed by component name; got {type(raw_obs).__name__}"
    )
  return Observation.from_components(raw_obs, focus_mask=focus_mask)


class ReplayBuffer:
  """List-of-episodes replay buffer with step-count capacity.

  Variable-length episodes are stored as dicts of numpy arrays. Sampling
  yields fixed-length sequences of shape ``(B, T, ...)`` that are
  guaranteed to come from a single episode (no ``done`` crossings).
  """

  def __init__(
    self,
    capacity_steps: int,
    min_episode_len: int,
    processor: EpisodeProcessor | None = None,
    reward_fn: RewardFn | None = None,
    seed: int | None = None,
    default_num_workers: int = 0,
    protected_tags: frozenset[str] | set[str] | list[str] | tuple[str, ...] = (),
    tag_weights: dict[str, float] | None = None,
  ) -> None:
    """Construct an episodic replay buffer.

    ``reward_fn`` recomputes reward from stored ``step_info`` at sample
    time — swap it to change the training signal without re-collecting.
    When None, sample() returns the reward recorded with the episode.
    Recomputation additionally requires ``goal_xy``; episodes missing
    that fall back to stored reward.

    ``default_num_workers`` is the default pool size used by
    :meth:`load_sim_episodes` when its ``num_workers`` arg is omitted.
    ``from_config`` resolves this from ``training.data.warmup_num_workers``.

    ``protected_tags`` — episodes whose tags intersect this set are
    never evicted. ``tag_weights`` — per-tag sample-weight multiplier;
    an episode's effective weight is ``max(tag_weights[t] for t in tags)``
    (default 1.0 when no tags or no matches). Tags themselves come from
    the episode files via :class:`EpisodeProcessor`.
    """
    if min_episode_len < 1:
      raise ValueError("min_episode_len must be >= 1")
    self.capacity_steps  = capacity_steps
    self.min_episode_len = min_episode_len
    self.reward_fn       = reward_fn

    self._episodes: deque[Episode] = deque()
    self._total_steps: int         = 0
    self._rng                      = np.random.default_rng(seed)

    self._sim_processor       = processor if processor is not None else EpisodeProcessor(("joints",))
    self._default_num_workers = default_num_workers
    self._protected_tags      = frozenset(protected_tags)
    self._tag_weights: dict[str, float] = dict(tag_weights or {})

  @classmethod
  def from_config(
    cls,
    config,
    *,
    processor: EpisodeProcessor | None = None,
    reward_fn: RewardFn | None = None,
  ) -> "ReplayBuffer":
    """Build a ReplayBuffer from a project config.

    Reads ``training.data.buffer_size``, ``training.min_episode_len``,
    ``seed``, and ``training.data.warmup_num_workers``. A ``null``
    ``warmup_num_workers`` resolves to ``max(1, os.cpu_count() // 2)``;
    ``0`` keeps loading sequential.

    ``processor`` and ``reward_fn`` stay as explicit args because they
    aren't expressible in the config schema.
    """
    data_cfg = config.training.data
    cfg_workers = data_cfg.get("warmup_num_workers")
    if cfg_workers is None:
      num_workers = os.cpu_count() or 1
    else:
      num_workers = int(cfg_workers)
    protected_tags = tuple(str(t) for t in (data_cfg.get("protected_tags") or ()))
    tag_weights = {
      str(k): float(v) for k, v in (data_cfg.get("tag_weights") or {}).items()
    }
    return cls(
      capacity_steps=data_cfg.buffer_size,
      min_episode_len=config.training.min_episode_len,
      processor=processor,
      reward_fn=reward_fn,
      seed=config.seed,
      default_num_workers=num_workers,
      protected_tags=protected_tags,
      tag_weights=tag_weights,
    )

  # ---------------------------------------------------------------------
  # Write side
  # ---------------------------------------------------------------------

  def add_episode(self, episode: Episode) -> None:
    """Insert a complete pre-recorded episode."""
    validated = self._validate_episode(episode)
    T = validated["action"].shape[0]
    if T < self.min_episode_len:
      return
    self._append_episode(validated)

  def _validate_episode(self, episode: Episode) -> Episode:
    for key in ("obs", "action", "reward", "done", "step_info"):
      if key not in episode:
        raise ValueError(f"episode missing required key: {key!r}")
    si = episode["step_info"]
    for col in ("object_xy", "ee_pos"):
      if col not in si:
        raise ValueError(f"step_info missing required column: {col!r}")

    raw_obs    = episode["obs"]
    focus_mask = episode.get("focus_mask")
    action     = np.asarray(episode["action"], dtype=np.float32)
    reward     = np.asarray(episode["reward"], dtype=np.float32)
    done       = np.asarray(episode["done"], dtype=bool)
    T          = action.shape[0]

    obs = _coerce_to_observation(raw_obs, focus_mask=focus_mask)
    leading = obs.leading_shape
    if leading[0] != T + 1:
      raise ValueError(f"obs must have T+1 entries; got {leading[0]}, action={T}")
    if obs.focus_mask is not None and obs.focus_mask.shape[0] != T + 1:
      raise ValueError(
        f"focus_mask must have T+1 entries; got {obs.focus_mask.shape[0]}, action={T}"
      )

    if reward.shape != (T,):
      raise ValueError(f"reward must have shape ({T},); got {reward.shape}")
    if done.shape != (T,):
      raise ValueError(f"done must have shape ({T},); got {done.shape}")

    out: Episode = {
      "obs": obs, "action": action, "reward": reward, "done": done,
      "step_info": episode["step_info"],
    }
    if "goal_xy" in episode:
      out["goal_xy"] = episode["goal_xy"]
    if "mat_centre" in episode:
      out["mat_centre"] = episode["mat_centre"]
    # Tags carry through unchanged; missing → empty tuple so all callers
    # can treat ep["tags"] as iterable without a None check.
    out["tags"] = tuple(str(t) for t in episode.get("tags", ()))
    return out

  def _append_episode(self, episode: Episode) -> None:
    self._episodes.append(episode)
    self._total_steps += episode["action"].shape[0]
    self._evict()

  def _evict(self) -> None:
    # Walk left-to-right and drop the oldest unprotected episode each
    # pass. Keep at least one episode even if oversized. The for/else
    # bails out once every remaining episode is protected — without it
    # we'd spin forever when protected tags pin >capacity_steps in place.
    while self._total_steps > self.capacity_steps and len(self._episodes) > 1:
      for i, ep in enumerate(self._episodes):
        if not self._is_protected(ep):
          del self._episodes[i]
          self._total_steps -= ep["action"].shape[0]
          break
      else:
        break

  def _is_protected(self, ep: Episode) -> bool:
    if not self._protected_tags:
      return False
    tags = ep.get("tags") or ()
    return bool(self._protected_tags.intersection(tags))

  def _tag_weight(self, ep: Episode) -> float:
    if not self._tag_weights:
      return 1.0
    tags = ep.get("tags") or ()
    matched = [self._tag_weights[t] for t in tags if t in self._tag_weights]
    return max(matched) if matched else 1.0

  # ---------------------------------------------------------------------
  # Read side
  # ---------------------------------------------------------------------

  def __len__(self) -> int:
    return self._total_steps

  @property
  def num_episodes(self) -> int:
    return len(self._episodes)

  def size_by_tag(self) -> dict[str, tuple[int, int]]:
    """Return ``{tag: (num_episodes, num_steps)}`` with double-counting.

    Episodes with multiple tags contribute to each of their tags;
    untagged episodes land in ``"untagged"``. Mirrors the convention in
    :meth:`Tracker.log_batch_tags` so warmup summaries and per-batch
    tag counts use the same bucket names.
    """
    counts: dict[str, tuple[int, int]] = {}
    for ep in self._episodes:
      steps = int(ep["action"].shape[0])
      tags  = ep.get("tags") or ()
      keys  = tags if tags else ("untagged",)
      for t in keys:
        n_eps, n_steps = counts.get(t, (0, 0))
        counts[t]      = (n_eps + 1, n_steps + steps)
    return counts

  def can_sample(self, batch_size: int, seq_len: int) -> bool:
    del batch_size  # any eligible episode is enough; weighted sampling with replacement
    for ep in self._episodes:
      if ep["action"].shape[0] >= seq_len:
        return True
    return False

  def sample(self, batch_size: int, seq_len: int) -> dict[str, Any]:
    """Sample a batch of ``(B, T, ...)`` sequences from single episodes.

    The returned batch contains ``aux_target`` of shape ``(B, T, 5)`` —
    object_xy ‖ ee_pos, the privileged target for the model's aux head.
    Step-info columns align with the action axis (``object_xy[t]`` is
    the state AFTER ``action_t``).
    """
    if batch_size < 1 or seq_len < 1:
      raise ValueError("batch_size and seq_len must be >= 1")

    eligible: list[tuple[Episode, int]] = []
    weights: list[float] = []
    for ep in self._episodes:
      T = ep["action"].shape[0]
      valid_starts = T - seq_len + 1
      if valid_starts > 0:
        eligible.append((ep, valid_starts))
        # Per-tag multiplier on top of the valid-starts weight, so tagged
        # episodes are oversampled relative to their length. Untagged
        # episodes get the default 1.0 (i.e. plain length-proportional).
        weights.append(valid_starts * self._tag_weight(ep))

    if not eligible:
      raise RuntimeError(
        f"no episode long enough for seq_len={seq_len}; "
        f"have {len(self._episodes)} episodes"
      )

    weights_arr = np.asarray(weights, dtype=np.float64)
    probs       = weights_arr / weights_arr.sum()
    ep_indices  = self._rng.choice(len(eligible), size=batch_size, p=probs)

    obs_slices: list[Observation]     = []
    action_batch                      = []
    reward_batch                      = []
    done_batch                        = []
    aux_batch: list[np.ndarray]       = []
    tags_batch: list[tuple[str, ...]] = []

    for idx in ep_indices:
      ep, valid_starts = eligible[idx]
      start            = int(self._rng.integers(0, valid_starts))
      end              = start + seq_len
      obs_slices.append(ep["obs"][start:end])
      action_batch.append(ep["action"][start:end])
      reward_batch.append(self._reward_slice(ep, start, end))
      done_batch.append(ep["done"][start:end])
      si = ep["step_info"]
      aux_batch.append(np.concatenate(
        [si["object_xy"][start:end], si["ee_pos"][start:end]], axis=-1,
      ))
      tags_batch.append(tuple(ep.get("tags") or ()))

    # The buffer's wire format keeps obs as a plain ndarray or dict so
    # backends lift them via their ``coerce`` method (see WorldModel.encode
    # and rollout()'s coerce hook). Internally we stack via Observation so
    # the dict-vs-array branching lives in one place. ``focus_mask`` rides
    # in a sibling key — present only when the stored episodes have it.
    obs_batched = Observation.stack(obs_slices, axis=0)

    batch: dict[str, Any] = {
      "obs": obs_batched.to_buffer_value(),
      "action": np.stack(action_batch, axis=0),
      "reward": np.stack(reward_batch, axis=0),
      "done": np.stack(done_batch, axis=0),
      "aux_target": np.stack(aux_batch, axis=0).astype(np.float32),
      "tags": tags_batch,
    }
    if obs_batched.focus_mask is not None:
      batch["focus_mask"] = obs_batched.focus_mask
    return batch

  def _reward_slice(self, ep: Episode, start: int, end: int) -> np.ndarray:
    """Return ``reward[start:end]`` recomputed via ``self.reward_fn`` if possible.

    Falls back to the stored reward when ``reward_fn`` is None or the
    episode lacks ``goal_xy`` (which is episode-scalar metadata and may
    not always be set).
    """
    if self.reward_fn is None or "goal_xy" not in ep:
      return np.asarray(ep["reward"][start:end], dtype=np.float32)

    si      = ep["step_info"]
    goal_xy = np.asarray(ep["goal_xy"], dtype=np.float32)
    out     = np.empty((end - start,), dtype=np.float32)
    for i, t in enumerate(range(start, end)):
      info = StepInfo(
        object_xy=si["object_xy"][t],
        ee_pos=si["ee_pos"][t],
        ee_quat=si["ee_quat"][t],
        goal_xy=goal_xy,
        step=int(si["step"][t]) if "step" in si else t,
        time=float(si["time"][t]) if "time" in si else 0.0,
      )
      out[i] = self.reward_fn(info)
    return out

  # ---------------------------------------------------------------------
  # Persistence
  # ---------------------------------------------------------------------

  def save(self, path: str | Path) -> None:
    with open(path, "wb") as f:
      pickle.dump(list(self._episodes), f)

  def load(self, path: str | Path) -> None:
    with open(path, "rb") as f:
      episodes = pickle.load(f)
    for ep in episodes:
      self.add_episode(ep)

  # ---------------------------------------------------------------------
  # Load sim-collected episodes
  # ---------------------------------------------------------------------

  def add_sim_episode(self, raw: Mapping[str, Any]) -> None:
    # raw may be a plain dict (live sim collection) or an Episode Mapping
    # container (replayed recording); the processor reads it without mutating.
    episode = self._sim_processor(raw)
    self.add_episode(episode)

  def load_sim_episodes(
    self,
    directory: str | Path,
    format: str = "hdf5",
    progress: "Progress" = False,
    num_episodes: int | None = None,
  ) -> int:
    """Ingest episodes from a sim ``EpisodeWriter`` output directory.

    ``processor`` converts each raw sim episode into the buffer's
    ``(obs, action, reward, done)`` schema. Defaults to
    ``EpisodeProcessor("joints")`` (low-dim joint obs). Returns the number
    of episodes successfully inserted (short episodes that fail the
    ``min_episode_len`` check are skipped, like ``add_episode``).

    ``progress`` is forwarded to :meth:`EpisodeDataset.stream` — pass
    ``True`` for a tqdm/print progress bar or a ``(i, total, path)``
    callable for custom reporting.

    ``num_episodes`` optionally limits ingestion to a random subset of
    that many episode files (drawn without replacement using the
    buffer's RNG, so it's deterministic given ``seed``).

    Parallel-load pool size is set on the buffer at construction time
    via ``default_num_workers`` (or :meth:`from_config`).
    """
    dataset = EpisodeDataset(directory, format=format)
    count   = 0
    for _path, processed in dataset.stream_processed(
      self._sim_processor,
      progress=progress,
      num_episodes=num_episodes,
      rng=self._rng,
      num_workers=self._default_num_workers,
    ):
      before = self.num_episodes
      self.add_episode(processed)
      if self.num_episodes > before:
        count += 1
    return count
