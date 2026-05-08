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

from ..reward import RewardFn
from ..sim.step_info import StepInfo
from .episode_loader import Progress, iter_episodes
from .episode_processor import EpisodeProcessor, StateVectorProcessor

Episode = dict[str, Any]


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
  ) -> None:
    """Construct an episodic replay buffer.

    ``reward_fn`` recomputes reward from stored ``step_info`` at sample
    time — swap it to change the training signal without re-collecting.
    When None, sample() returns the reward recorded with the episode.
    Recomputation additionally requires ``goal_xy``; episodes missing
    that fall back to stored reward.
    """
    if min_episode_len < 1:
      raise ValueError("min_episode_len must be >= 1")
    self.capacity_steps  = capacity_steps
    self.min_episode_len = min_episode_len
    self.reward_fn       = reward_fn

    self._episodes: deque[Episode] = deque()
    self._total_steps: int = 0
    self._rng = np.random.default_rng(seed)

    self._sim_processor = processor if processor is not None else StateVectorProcessor()

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

    raw_obs = episode["obs"]
    action  = np.asarray(episode["action"], dtype=np.float32)
    reward  = np.asarray(episode["reward"], dtype=np.float32)
    done    = np.asarray(episode["done"], dtype=bool)

    T = action.shape[0]
    obs: dict[str, np.ndarray] | np.ndarray
    if isinstance(raw_obs, dict):
      # Composite obs (e.g. image_proprio): each leaf is (T+1, ...). Image
      # leaves stay uint8; everything else is cast to float32.
      obs = {}
      for k, v in raw_obs.items():
        arr = np.asarray(v)
        if arr.dtype != np.uint8:
          arr = arr.astype(np.float32, copy=False)
        if arr.shape[0] != T + 1:
          raise ValueError(
            f"obs[{k!r}] must have T+1 entries; got {arr.shape[0]}, action={T}"
          )
        obs[k] = arr
    else:
      arr = np.asarray(raw_obs)
      if arr.dtype != np.uint8:
        arr = arr.astype(np.float32, copy=False)
      if arr.shape[0] != T + 1:
        raise ValueError(
          f"obs must have T+1 entries; got obs={arr.shape[0]}, action={T}"
        )
      obs = arr

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
    return out

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

    is_dict_obs = isinstance(eligible[0][0]["obs"], dict)
    obs_batch: list = [] if not is_dict_obs else None  # type: ignore[assignment]
    obs_batch_dict: dict[str, list] | None = (
      {k: [] for k in eligible[0][0]["obs"].keys()} if is_dict_obs else None
    )
    action_batch = []
    reward_batch = []
    done_batch = []
    aux_batch: list[np.ndarray] = []

    for idx in ep_indices:
      ep, valid_starts = eligible[idx]
      start = int(self._rng.integers(0, valid_starts))
      end = start + seq_len
      if obs_batch_dict is not None:
        for k, lst in obs_batch_dict.items():
          lst.append(ep["obs"][k][start:end])
      else:
        obs_batch.append(ep["obs"][start:end])  # type: ignore[union-attr]
      action_batch.append(ep["action"][start:end])
      reward_batch.append(self._reward_slice(ep, start, end))
      done_batch.append(ep["done"][start:end])
      si = ep["step_info"]
      aux_batch.append(np.concatenate(
        [si["object_xy"][start:end], si["ee_pos"][start:end]], axis=-1,
      ))

    obs_out: dict | mx.array
    if obs_batch_dict is not None:
      obs_out = {k: mx.array(np.stack(v, axis=0)) for k, v in obs_batch_dict.items()}
    else:
      obs_out = mx.array(np.stack(obs_batch, axis=0))

    return {
      "obs": obs_out,
      "action": mx.array(np.stack(action_batch, axis=0)),
      "reward": mx.array(np.stack(reward_batch, axis=0)),
      "done": mx.array(np.stack(done_batch, axis=0)),
      "aux_target": mx.array(np.stack(aux_batch, axis=0).astype(np.float32)),
    }

  def _reward_slice(self, ep: Episode, start: int, end: int) -> np.ndarray:
    """Return ``reward[start:end]`` recomputed via ``self.reward_fn`` if possible.

    Falls back to the stored reward when ``reward_fn`` is None or the
    episode lacks ``goal_xy`` (which is episode-scalar metadata and may
    not always be set).
    """
    if self.reward_fn is None or "goal_xy" not in ep:
      return np.asarray(ep["reward"][start:end], dtype=np.float32)

    si = ep["step_info"]
    goal_xy = np.asarray(ep["goal_xy"], dtype=np.float32)
    out = np.empty((end - start,), dtype=np.float32)
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
