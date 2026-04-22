"""Tests for the Dreamer replay buffer."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from src.world_model_pusher.dreamer import ReplayBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(
  episode_id: int,
  length: int,
  obs_shape: tuple[int, ...] = (4,),
  action_dim: int = 2,
) -> dict[str, np.ndarray]:
  """Build an episode where obs[t, 0] = episode_id * 1000 + t.

  This encoding lets tests verify that sampled sequences stay inside a
  single episode (thousands digit constant, remainder contiguous).
  """
  T = length
  obs = np.zeros((T + 1, *obs_shape), dtype=np.float32)
  for t in range(T + 1):
    obs[t, 0] = episode_id * 1000 + t
  action = np.full((T, action_dim), float(episode_id), dtype=np.float32)
  reward = np.arange(T, dtype=np.float32)
  done = np.zeros((T,), dtype=bool)
  done[-1] = True
  return {"obs": obs, "action": action, "reward": reward, "done": done}


# ---------------------------------------------------------------------------
# Identity / no-cross-episode-boundary
# ---------------------------------------------------------------------------


def test_sampled_sequences_stay_in_single_episode():
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=10, seed=0)
  for ep_id in range(5):
    buf.add_episode(_make_episode(ep_id, length=30))

  batch = buf.sample(batch_size=16, seq_len=10)
  obs = np.asarray(batch["obs"])  # (B, T, obs_dim)

  for b in range(obs.shape[0]):
    seq = obs[b, :, 0]
    ep_ids = np.floor(seq / 1000).astype(int)
    assert np.all(ep_ids == ep_ids[0]), f"cross-episode: {seq}"
    within = seq - ep_ids * 1000
    diffs = np.diff(within)
    assert np.all(diffs == 1), f"non-contiguous within episode: {seq}"


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------


def test_sample_shapes_and_dtypes():
  obs_shape = (7,)
  action_dim = 3
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=50, seed=1)
  for ep_id in range(4):
    buf.add_episode(
      _make_episode(ep_id, length=100, obs_shape=obs_shape, action_dim=action_dim)
    )

  B, T = 50, 50
  batch = buf.sample(batch_size=B, seq_len=T)
  assert batch["obs"].shape == (B, T, *obs_shape)
  assert batch["action"].shape == (B, T, action_dim)
  assert batch["reward"].shape == (B, T)
  assert batch["done"].shape == (B, T)

  assert batch["obs"].dtype == mx.float32
  assert batch["action"].dtype == mx.float32
  assert batch["reward"].dtype == mx.float32
  assert batch["done"].dtype == mx.bool_


# ---------------------------------------------------------------------------
# Capacity / eviction
# ---------------------------------------------------------------------------


def test_capacity_triggers_episode_eviction():
  # Capacity 1000 steps, episodes of 100 → at most 10 stored.
  buf = ReplayBuffer(capacity_steps=1_000, min_episode_len=10, seed=2)
  for ep_id in range(25):
    buf.add_episode(_make_episode(ep_id, length=100))

  assert len(buf) <= 1_000
  assert buf.num_episodes <= 10

  # Oldest episodes should be gone: no sample should draw from episode 0..5.
  batch = buf.sample(batch_size=64, seq_len=10)
  obs = np.asarray(batch["obs"])
  min_ep_id = int(np.floor(obs[:, 0, 0].min() / 1000))
  assert min_ep_id >= 15


def test_oversize_single_episode_is_retained():
  # A single episode larger than capacity must still be kept.
  buf = ReplayBuffer(capacity_steps=100, min_episode_len=10, seed=0)
  buf.add_episode(_make_episode(0, length=500))
  assert buf.num_episodes == 1
  assert len(buf) == 500


# ---------------------------------------------------------------------------
# Online collection path
# ---------------------------------------------------------------------------


def test_online_collection_round_trip():
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=5, seed=0)

  obs = np.zeros((3,), dtype=np.float32)
  buf.start_episode(obs)
  for t in range(20):
    action = np.ones((2,), dtype=np.float32) * t
    next_obs = np.full((3,), t + 1, dtype=np.float32)
    done = t == 19
    buf.add(action, next_obs, reward=float(t), done=done)

  assert buf.num_episodes == 1
  assert len(buf) == 20
  batch = buf.sample(batch_size=4, seq_len=5)
  assert batch["obs"].shape == (4, 5, 3)


def test_short_episode_dropped_on_finalize():
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=10, seed=0)
  buf.start_episode(np.zeros((3,), dtype=np.float32))
  for t in range(5):
    buf.add(
      np.zeros((2,), dtype=np.float32),
      np.zeros((3,), dtype=np.float32),
      reward=0.0,
      done=(t == 4),
    )
  assert buf.num_episodes == 0
  assert len(buf) == 0


def test_add_without_start_raises():
  buf = ReplayBuffer()
  with pytest.raises(RuntimeError):
    buf.add(
      np.zeros((2,), dtype=np.float32),
      np.zeros((3,), dtype=np.float32),
      reward=0.0,
      done=False,
    )


# ---------------------------------------------------------------------------
# can_sample / gating
# ---------------------------------------------------------------------------


def test_can_sample_gating():
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=10, seed=0)
  assert not buf.can_sample(batch_size=1, seq_len=10)

  buf.add_episode(_make_episode(0, length=15))
  assert buf.can_sample(batch_size=1, seq_len=10)
  assert buf.can_sample(batch_size=1, seq_len=15)
  assert not buf.can_sample(batch_size=1, seq_len=16)


def test_sample_raises_when_no_eligible_episode():
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=5, seed=0)
  buf.add_episode(_make_episode(0, length=8))
  with pytest.raises(RuntimeError):
    buf.sample(batch_size=4, seq_len=50)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_with_seed():
  def build(seed):
    buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=10, seed=seed)
    for ep_id in range(5):
      buf.add_episode(_make_episode(ep_id, length=30))
    return buf

  buf_a = build(42)
  buf_b = build(42)
  ba = buf_a.sample(batch_size=8, seq_len=10)
  bb = buf_b.sample(batch_size=8, seq_len=10)
  np.testing.assert_array_equal(np.asarray(ba["obs"]), np.asarray(bb["obs"]))
  np.testing.assert_array_equal(np.asarray(ba["action"]), np.asarray(bb["action"]))


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=10, seed=0)
  for ep_id in range(3):
    buf.add_episode(_make_episode(ep_id, length=25))
  path = tmp_path / "buffer.pkl"
  buf.save(path)

  restored = ReplayBuffer(capacity_steps=10_000, min_episode_len=10, seed=0)
  restored.load(path)

  assert restored.num_episodes == buf.num_episodes
  assert len(restored) == len(buf)

  # Every episode is byte-identical.
  for orig, new in zip(buf._episodes, restored._episodes):
    for key in ("obs", "action", "reward", "done"):
      np.testing.assert_array_equal(orig[key], new[key])


# ---------------------------------------------------------------------------
# Validation of offline-seeded episodes
# ---------------------------------------------------------------------------


def test_add_episode_validates_shapes():
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=1, seed=0)

  bad = {
    "obs": np.zeros((10, 4), dtype=np.float32),  # should be T+1 = 11
    "action": np.zeros((10, 2), dtype=np.float32),
    "reward": np.zeros((10,), dtype=np.float32),
    "done": np.zeros((10,), dtype=bool),
  }
  with pytest.raises(ValueError):
    buf.add_episode(bad)

  missing_key = {
    "obs": np.zeros((11, 4), dtype=np.float32),
    "action": np.zeros((10, 2), dtype=np.float32),
    "reward": np.zeros((10,), dtype=np.float32),
  }
  with pytest.raises(ValueError):
    buf.add_episode(missing_key)


# ---------------------------------------------------------------------------
# Length-proportional sampling
# ---------------------------------------------------------------------------


def test_episode_sampling_is_length_proportional():
  # One episode of length 100, one of length 1000. With seq_len=10 the
  # long episode offers ~10x more valid start indices and should be
  # picked ~10x as often.
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=10, seed=123)
  buf.add_episode(_make_episode(0, length=100))
  buf.add_episode(_make_episode(1, length=1000))

  batch = buf.sample(batch_size=2000, seq_len=10)
  obs = np.asarray(batch["obs"])
  ep_ids = np.floor(obs[:, 0, 0] / 1000).astype(int)
  long_count = int((ep_ids == 1).sum())
  short_count = int((ep_ids == 0).sum())

  # Expected ratio: (1000-9) / (100-9) ≈ 10.89. Allow generous tolerance.
  ratio = long_count / max(short_count, 1)
  assert 7.0 < ratio < 15.0, f"ratio={ratio} long={long_count} short={short_count}"
