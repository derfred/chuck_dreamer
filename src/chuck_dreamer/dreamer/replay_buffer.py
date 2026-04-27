"""Episodic replay buffer for Dreamer v1.

See `replay_buffer.md` in the project root for the full design doc.
"""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .episode_loader import StateVectorProcessor, EpisodeProcessor, Progress, iter_episodes

Episode = dict[str, np.ndarray]


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
    seed: int | None = None,
  ) -> None:
    if min_episode_len < 1:
      raise ValueError("min_episode_len must be >= 1")
    self.capacity_steps  = capacity_steps
    self.min_episode_len = min_episode_len

    self._episodes: deque[Episode] = deque()
    self._total_steps: int = 0
    self._rng = np.random.default_rng(seed)

    self._current: dict[str, list[Any]] | None = None
    self._sim_processor = processor if processor is not None else StateVectorProcessor()

  # ---------------------------------------------------------------------
  # Write side — online collection
  # ---------------------------------------------------------------------

  def start_episode(self, initial_obs: np.ndarray) -> None:
    """Begin a new episode with its first observation ``o_0``."""
    obs = np.asarray(initial_obs, dtype=np.float32)
    self._current = {
      "obs": [obs],
      "action": [],
      "reward": [],
      "done": [],
    }

  def add(
    self,
    action: np.ndarray,
    next_obs: np.ndarray,
    reward: float,
    done: bool,
  ) -> None:
    """Append ``(a_t, o_{t+1}, r_t, done_t)`` to the in-progress episode.

    Finalizes the episode when ``done`` is True.
    """
    if self._current is None:
      raise RuntimeError("add() called before start_episode()")

    self._current["action"].append(np.asarray(action, dtype=np.float32))
    self._current["obs"].append(np.asarray(next_obs, dtype=np.float32))
    self._current["reward"].append(float(reward))
    self._current["done"].append(bool(done))

    if done:
      self._finalize_current()

  def _finalize_current(self) -> None:
    assert self._current is not None
    T = len(self._current["action"])
    if T < self.min_episode_len:
      self._current = None
      return

    episode: Episode = {
      "obs": np.stack(self._current["obs"], axis=0).astype(np.float32),
      "action": np.stack(self._current["action"], axis=0).astype(np.float32),
      "reward": np.asarray(self._current["reward"], dtype=np.float32),
      "done": np.asarray(self._current["done"], dtype=bool),
    }
    self._current = None
    self._append_episode(episode)

  # ---------------------------------------------------------------------
  # Write side — offline seeding
  # ---------------------------------------------------------------------

  def add_episode(self, episode: Episode) -> None:
    """Insert a complete pre-recorded episode."""
    validated = self._validate_episode(episode)
    T = validated["action"].shape[0]
    if T < self.min_episode_len:
      return
    self._append_episode(validated)

  def _validate_episode(self, episode: Episode) -> Episode:
    for key in ("obs", "action", "reward", "done"):
      if key not in episode:
        raise ValueError(f"episode missing required key: {key!r}")

    obs    = np.asarray(episode["obs"], dtype=np.float32)
    action = np.asarray(episode["action"], dtype=np.float32)
    reward = np.asarray(episode["reward"], dtype=np.float32)
    done   = np.asarray(episode["done"], dtype=bool)

    T = action.shape[0]
    if obs.shape[0] != T + 1:
      raise ValueError(
        f"obs must have T+1 entries; got obs={obs.shape[0]}, action={T}"
      )
    if reward.shape != (T,):
      raise ValueError(f"reward must have shape ({T},); got {reward.shape}")
    if done.shape != (T,):
      raise ValueError(f"done must have shape ({T},); got {done.shape}")

    return {"obs": obs, "action": action, "reward": reward, "done": done}

  def _append_episode(self, episode: Episode) -> None:
    self._episodes.append(episode)
    self._total_steps += episode["action"].shape[0]
    self._evict()

  def _evict(self) -> None:
    # Keep at least one episode even if oversized.
    while self._total_steps > self.capacity_steps and len(self._episodes) > 1:
      dropped = self._episodes.popleft()
      self._total_steps -= dropped["action"].shape[0]

  # ---------------------------------------------------------------------
  # Read side
  # ---------------------------------------------------------------------

  def __len__(self) -> int:
    return self._total_steps

  @property
  def num_episodes(self) -> int:
    return len(self._episodes)

  def can_sample(self, batch_size: int, seq_len: int) -> bool:
    del batch_size  # any eligible episode is enough; weighted sampling with replacement
    for ep in self._episodes:
      if ep["action"].shape[0] >= seq_len:
        return True
    return False

  def sample(self, batch_size: int, seq_len: int) -> dict[str, mx.array]:
    """Sample a batch of ``(B, T, ...)`` sequences from single episodes."""
    if batch_size < 1 or seq_len < 1:
      raise ValueError("batch_size and seq_len must be >= 1")

    eligible: list[tuple[Episode, int]] = []
    weights: list[int] = []
    for ep in self._episodes:
      T = ep["action"].shape[0]
      valid_starts = T - seq_len + 1
      if valid_starts > 0:
        eligible.append((ep, valid_starts))
        weights.append(valid_starts)

    if not eligible:
      raise RuntimeError(
        f"no episode long enough for seq_len={seq_len}; "
        f"have {len(self._episodes)} episodes"
      )

    weights_arr = np.asarray(weights, dtype=np.float64)
    probs = weights_arr / weights_arr.sum()
    ep_indices = self._rng.choice(len(eligible), size=batch_size, p=probs)

    obs_batch = []
    action_batch = []
    reward_batch = []
    done_batch = []

    for idx in ep_indices:
      ep, valid_starts = eligible[idx]
      start = int(self._rng.integers(0, valid_starts))
      end = start + seq_len
      obs_batch.append(ep["obs"][start:end])
      action_batch.append(ep["action"][start:end])
      reward_batch.append(ep["reward"][start:end])
      done_batch.append(ep["done"][start:end])

    return {
      "obs": mx.array(np.stack(obs_batch, axis=0)),
      "action": mx.array(np.stack(action_batch, axis=0)),
      "reward": mx.array(np.stack(reward_batch, axis=0)),
      "done": mx.array(np.stack(done_batch, axis=0)),
    }

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

  def add_sim_episode(self, raw: dict[str, Any]) -> None:
    episode = self._sim_processor(raw)
    self.add_episode(episode)

  def load_sim_episodes(
    self,
    directory: str | Path,
    format: str = "hdf5",
    progress: "Progress" = False,
  ) -> int:
    """Ingest episodes from a sim ``EpisodeWriter`` output directory.

    ``processor`` converts each raw sim episode into the buffer's
    ``(obs, action, reward, done)`` schema. Defaults to
    ``StateVectorProcessor`` (low-dim state obs). Returns the number
    of episodes successfully inserted (short episodes that fail the
    ``min_episode_len`` check are skipped, like ``add_episode``).

    ``progress`` is forwarded to :func:`iter_episodes` — pass ``True``
    for a tqdm/print progress bar or a ``(i, total, path)`` callable
    for custom reporting.
    """

    count = 0
    for raw in iter_episodes(directory, format=format, progress=progress):
      before = self.num_episodes
      self.add_sim_episode(raw)
      if self.num_episodes > before:
        count += 1
    return count
