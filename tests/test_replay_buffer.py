"""Tests for the Dreamer replay buffer."""

from __future__ import annotations

import numpy as np
import pytest

from chuck_dreamer.training.replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step_info(episode_id: int, length: int) -> dict[str, np.ndarray]:
  """Step-info columns aligned with the action axis.

  ``object_xy[t, 0]`` encodes (episode_id, t) and ``ee_pos[t, 0]`` mirrors
  it with flipped sign — gives the aux-target tests a deterministic way
  to verify alignment and component layout.
  """
  object_xy = np.zeros((length, 2), dtype=np.float32)
  ee_pos    = np.zeros((length, 3), dtype=np.float32)
  for t in range(length):
    object_xy[t, 0] = episode_id * 1000 + t
    ee_pos[t, 0]    = -(episode_id * 1000 + t)
  return {"object_xy": object_xy, "ee_pos": ee_pos}


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
  return {
    "obs": obs, "action": action, "reward": reward, "done": done,
    "step_info": _make_step_info(episode_id, T),
  }


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

  assert batch["obs"].dtype == np.float32
  assert batch["action"].dtype == np.float32
  assert batch["reward"].dtype == np.float32
  assert batch["done"].dtype == np.bool_


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
# Min-length filter
# ---------------------------------------------------------------------------


def test_short_episode_dropped_by_add_episode():
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=10, seed=0)
  buf.add_episode(_make_episode(0, length=5))
  assert buf.num_episodes == 0
  assert len(buf) == 0


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
    np.testing.assert_array_equal(orig["obs"].state, new["obs"].state)
    for key in ("action", "reward", "done"):
      np.testing.assert_array_equal(orig[key], new[key])
    for col in ("object_xy", "ee_pos"):
      np.testing.assert_array_equal(orig["step_info"][col], new["step_info"][col])


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
    "step_info": _make_step_info(0, 10),
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


def test_add_episode_requires_step_info():
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=1, seed=0)
  ep = _make_episode(0, length=10)
  del ep["step_info"]
  with pytest.raises(ValueError, match="step_info"):
    buf.add_episode(ep)


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


# ---------------------------------------------------------------------------
# Dict (image_proprio) obs support
# ---------------------------------------------------------------------------


def _make_image_proprio_episode(
  episode_id: int,
  length: int,
  image_size: int = 8,
  proprio_dim: int = 13,
) -> dict:
  """Episode whose obs is a dict {'image': uint8, 'proprio': float32}.

  The image is filled with a constant per timestep so we can later check
  that sampled sequences stay contiguous within an episode.
  """
  T = length
  # Image fill values stay in [0, 255] (uint8), so use modulo. The sequence
  # contiguity check uses the proprio channel below; the image is only here
  # to exercise the dict-obs round-trip.
  images = np.stack([
    np.full((image_size, image_size, 3), (episode_id * 100 + t) % 256, dtype=np.uint8)
    for t in range(T + 1)
  ])
  proprio = np.zeros((T + 1, proprio_dim), dtype=np.float32)
  for t in range(T + 1):
    proprio[t, 0] = episode_id * 1000 + t
  action = np.full((T, 2), float(episode_id), dtype=np.float32)
  reward = np.arange(T, dtype=np.float32)
  done = np.zeros((T,), dtype=bool)
  done[-1] = True
  return {
    "obs": {"image": images, "proprio": proprio},
    "action": action,
    "reward": reward,
    "done": done,
    "step_info": _make_step_info(episode_id, T),
  }


def test_dict_obs_round_trips_through_buffer():
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=5, seed=0)
  buf.add_episode(_make_image_proprio_episode(episode_id=0, length=20))

  ep = buf._episodes[0]
  obs = ep["obs"]
  assert obs.mode == "image_proprio"
  assert obs.image.dtype == np.uint8
  assert obs.image.shape == (21, 8, 8, 3)
  assert obs.proprio.shape == (21, 13)


def test_dict_obs_sample_yields_dict_batch_with_correct_shapes():
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=5, seed=0)
  for ep_id in range(3):
    buf.add_episode(_make_image_proprio_episode(ep_id, length=30))

  batch = buf.sample(batch_size=4, seq_len=10)
  assert isinstance(batch["obs"], dict)
  img = np.asarray(batch["obs"]["image"])
  pro = np.asarray(batch["obs"]["proprio"])
  assert img.shape == (4, 10, 8, 8, 3)
  assert pro.shape == (4, 10, 13)
  # Image dtype survives the round trip.
  assert img.dtype == np.uint8


def test_dict_obs_sampled_sequences_stay_within_episode():
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=10, seed=0)
  for ep_id in range(5):
    buf.add_episode(_make_image_proprio_episode(ep_id, length=30))

  batch = buf.sample(batch_size=16, seq_len=10)
  pro = np.asarray(batch["obs"]["proprio"])  # (B, T, P)
  for b in range(pro.shape[0]):
    seq = pro[b, :, 0]
    ep_ids = np.floor(seq / 1000).astype(int)
    assert np.all(ep_ids == ep_ids[0]), f"cross-episode: {seq}"
    within = seq - ep_ids * 1000
    diffs = np.diff(within)
    assert np.all(diffs == 1), f"non-contiguous within episode: {seq}"


def test_focus_mask_round_trips_through_buffer():
  # Episode carries a (T+1, H, W) focus mask; verify it survives validation,
  # storage, slicing, and stacking into a (B, T, H, W) batch field.
  T, H = 20, 8
  ep = _make_image_proprio_episode(episode_id=0, length=T, image_size=H)
  fm = np.zeros((T + 1, H, H), dtype=np.float32)
  fm[:, :2, :2] = 1.0  # corner patch lit at every step
  ep["focus_mask"] = fm

  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=5, seed=0)
  buf.add_episode(ep)

  stored = buf._episodes[0]
  assert stored["obs"].focus_mask.shape == (T + 1, H, H)

  batch = buf.sample(batch_size=4, seq_len=10)
  assert "focus_mask" in batch
  assert batch["focus_mask"].shape == (4, 10, H, H)
  # Patch survives slicing/stacking.
  assert np.allclose(batch["focus_mask"][:, :, :2, :2], 1.0)


def test_focus_mask_absent_when_episode_lacks_it():
  # When no episode carries focus_mask, sample() should not include the key.
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=5, seed=0)
  for ep_id in range(3):
    buf.add_episode(_make_episode(ep_id, length=30))
  batch = buf.sample(batch_size=4, seq_len=10)
  assert "focus_mask" not in batch


def test_focus_mask_length_mismatch_rejected():
  T = 10
  ep = _make_image_proprio_episode(episode_id=0, length=T)
  ep["focus_mask"] = np.zeros((T, 8, 8), dtype=np.float32)  # WRONG: T not T+1
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=5, seed=0)
  with pytest.raises(ValueError):
    buf.add_episode(ep)


def test_aux_target_shape():
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=10, seed=0)
  for ep_id in range(3):
    buf.add_episode(_make_episode(ep_id, length=30))

  batch = buf.sample(batch_size=8, seq_len=10)
  aux = np.asarray(batch["aux_target"])
  assert aux.shape == (8, 10, 5)  # 2 (object_xy) + 3 (ee_pos)


def test_aux_target_aligns_with_action_axis():
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=10, seed=0)
  for ep_id in range(3):
    buf.add_episode(_make_episode(ep_id, length=30))

  batch = buf.sample(batch_size=4, seq_len=10)
  aux = np.asarray(batch["aux_target"])  # (B, T, 5)

  # Mirrors the cross-episode contiguity test on `obs`: within each row,
  # object_xy[:,0] increments by 1 per timestep, and ee_pos[:,0] is its
  # negation (the test fixture encodes them that way in _make_step_info).
  for b in range(aux.shape[0]):
    obj0 = aux[b, :, 0]
    ee0  = aux[b, :, 2]  # first entry of ee_pos, after the 2-D object_xy
    np.testing.assert_array_equal(np.diff(obj0), np.ones(aux.shape[1] - 1))
    np.testing.assert_array_equal(ee0, -obj0)


def test_dict_obs_rejects_mismatched_leaf_lengths():
  T = 10
  ep = {
    "obs": {
      "image":   np.zeros((T + 1, 4, 4, 3), dtype=np.uint8),
      "proprio": np.zeros((T,    13),        dtype=np.float32),  # WRONG: T not T+1
    },
    "action": np.zeros((T, 2), dtype=np.float32),
    "reward": np.zeros((T,),   dtype=np.float32),
    "done":   np.array([False] * (T - 1) + [True]),
    "step_info": _make_step_info(0, T),
  }
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=5, seed=0)
  with pytest.raises(ValueError):
    buf.add_episode(ep)
