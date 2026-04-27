"""Tests for loading sim-writer episodes into the replay buffer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from chuck_dreamer.dreamer import (  # noqa: E402
  ImageProcessor,
  ReplayBuffer,
  StateVectorProcessor,
  iter_episodes,
  load_hdf5_episode,
  load_rerun_episode,
)
from chuck_dreamer.dreamer.episode_loader import _drop_last_and_pack  # noqa: E402
from chuck_dreamer.sim.data_collection import EpisodeWriter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_episode(T: int, img_hw: tuple[int, int] = (16, 16)) -> dict:
  """An episode in the dict-of-stacked-arrays shape the sim writers consume."""
  H, W = img_hw
  ts = np.arange(T)
  return {
    "image":      np.stack([np.full((H, W, 3), t, dtype=np.uint8) for t in ts]),
    "action":     np.stack([np.array([0.1 * t, 0.2 * t, 0.3 * t], dtype=np.float32) for t in ts]),
    "reward":     ts.astype(np.float32),
    "timestamp":  (ts * 0.05).astype(np.float32),
    "joint_qpos": np.stack([np.full((6,), 0.1 * t, dtype=np.float32) for t in ts]),
    "ee_pos":     np.stack([np.array([0.1, 0.2, 0.3], dtype=np.float32) * t for t in ts]),
    "ee_quat":    np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 1)),
    "object_xy":  np.tile(np.array([0.5, 0.5], dtype=np.float32), (T, 1)),
  }


def _write_sim_episode(dir_path: Path, fmt: str, T: int = 20) -> None:
  writer = EpisodeWriter(str(dir_path), format=fmt)
  writer.write_episode(
    _make_raw_episode(T),
    metadata={"seed": 7, "source": "test", "outcome": "done", "goal_xy": [0.1, 0.2]},
  )


# ---------------------------------------------------------------------------
# Drop-last / shape invariants — tested without touching the writers
# ---------------------------------------------------------------------------


def test_drop_last_and_pack_enforces_buffer_invariants():
  N = 10
  obs = np.arange(N * 4, dtype=np.float32).reshape(N, 4)
  raw = {
    "action": np.arange(N * 2, dtype=np.float32).reshape(N, 2),
    "reward": np.arange(N, dtype=np.float32),
  }
  ep = _drop_last_and_pack(obs, raw)

  assert ep["obs"].shape == (N, 4)
  assert ep["action"].shape == (N - 1, 2)
  assert ep["reward"].shape == (N - 1,)
  assert ep["done"].shape == (N - 1,)
  assert ep["done"][-1]
  assert not ep["done"][:-1].any()


def test_drop_last_rejects_single_step_episode():
  with pytest.raises(ValueError):
    _drop_last_and_pack(
      np.zeros((1, 3), dtype=np.float32),
      {"action": np.zeros((1, 2), dtype=np.float32), "reward": np.zeros((1,))},
    )


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------


def test_state_vector_processor_concatenates_state_fields():
  N = 5
  raw = {
    "image": np.zeros((N, 4, 4, 3), dtype=np.uint8),
    "action": np.zeros((N, 3), dtype=np.float32),
    "reward": np.zeros((N,), dtype=np.float32),
    "timestamp": np.zeros((N,), dtype=np.float32),
    "joint_qpos": np.ones((N, 6), dtype=np.float32),
    "ee_pos": np.full((N, 3), 2.0, dtype=np.float32),
    "ee_quat": np.full((N, 4), 3.0, dtype=np.float32),
    "object_xy": np.full((N, 2), 4.0, dtype=np.float32),
  }
  ep = StateVectorProcessor()(raw)

  # 3 + 4 + 2 + 6 = 15.
  assert ep["obs"].shape == (N, 15)
  # Order is ee_pos | ee_quat | object_xy | joint_qpos.
  np.testing.assert_array_equal(ep["obs"][0, 0:3], [2.0, 2.0, 2.0])
  np.testing.assert_array_equal(ep["obs"][0, 3:7], [3.0, 3.0, 3.0, 3.0])
  np.testing.assert_array_equal(ep["obs"][0, 7:9], [4.0, 4.0])
  np.testing.assert_array_equal(ep["obs"][0, 9:15], [1.0] * 6)


def test_image_processor_returns_image_as_obs():
  N = 5
  raw = {
    "image": np.stack([np.full((8, 8, 3), t, dtype=np.uint8) for t in range(N)]),
    "action": np.zeros((N, 3), dtype=np.float32),
    "reward": np.zeros((N,), dtype=np.float32),
    "timestamp": np.zeros((N,), dtype=np.float32),
    "joint_qpos": np.zeros((N, 6), dtype=np.float32),
    "ee_pos": np.zeros((N, 3), dtype=np.float32),
    "ee_quat": np.zeros((N, 4), dtype=np.float32),
    "object_xy": np.zeros((N, 2), dtype=np.float32),
  }
  ep = ImageProcessor()(raw)
  assert ep["obs"].shape == (N, 8, 8, 3)
  assert ep["obs"].dtype == np.float32
  assert float(ep["obs"][3].mean()) == 3.0


# ---------------------------------------------------------------------------
# HDF5 round-trip
# ---------------------------------------------------------------------------


def test_load_hdf5_episode_round_trip(tmp_path):
  _write_sim_episode(tmp_path, "hdf5", T=20)
  raw = load_hdf5_episode(tmp_path / "episode_00000.hdf5")

  assert raw["image"].shape == (20, 16, 16, 3)
  assert raw["action"].shape == (20, 3)
  assert raw["reward"].shape == (20,)
  # Step t's joint_qpos was 0.1 * t (broadcast over 6 joints).
  np.testing.assert_allclose(raw["joint_qpos"][7], [0.7] * 6, rtol=0, atol=1e-6)


def test_replay_buffer_loads_hdf5_directory(tmp_path):
  _write_sim_episode(tmp_path, "hdf5", T=20)
  _write_sim_episode(tmp_path, "hdf5", T=20)

  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=5, seed=0)
  n = buf.load_sim_episodes(tmp_path, format="hdf5")

  assert n == 2
  assert buf.num_episodes == 2
  # 20 raw steps → 20 obs + 19 actions per episode.
  assert len(buf) == 2 * 19
  ep = buf._episodes[0]
  assert ep["obs"].shape == (20, 15)
  assert ep["action"].shape == (20 - 1, 3)
  assert bool(ep["done"][-1])

  batch = buf.sample(batch_size=4, seq_len=10)
  assert batch["obs"].shape == (4, 10, 15)


# ---------------------------------------------------------------------------
# Rerun round-trip
# ---------------------------------------------------------------------------


def test_load_rerun_episode_round_trip(tmp_path):
  _write_sim_episode(tmp_path, "rerun", T=20)
  raw = load_rerun_episode(tmp_path / "episode_00000.rrd")

  assert raw["image"].shape == (20, 16, 16, 3)
  assert raw["action"].shape == (20, 3)
  assert raw["reward"].shape == (20,)
  # Chunks arrive unordered; the loader sorts by step, so values should
  # still follow the generator pattern joint_qpos[t] = 0.1 * t.
  np.testing.assert_allclose(raw["joint_qpos"][7], [0.7] * 6, rtol=0, atol=1e-5)
  assert float(raw["image"][3].mean()) == 3.0


def test_replay_buffer_loads_rerun_directory(tmp_path):
  _write_sim_episode(tmp_path, "rerun", T=20)
  _write_sim_episode(tmp_path, "rerun", T=20)

  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=5, processor=ImageProcessor(), seed=0)
  n = buf.load_sim_episodes(tmp_path, format="rerun")

  assert n == 2
  ep = buf._episodes[0]
  assert ep["obs"].shape == (20, 16, 16, 3)
  assert ep["action"].shape == (19, 3)
  # Image[t] is filled with constant t — a good check that step ordering
  # survives the rerun chunk round-trip.
  for t in range(20):
    assert float(ep["obs"][t].mean()) == float(t)


# ---------------------------------------------------------------------------
# Directory iteration / error paths
# ---------------------------------------------------------------------------


def test_iter_episodes_rejects_unknown_format(tmp_path):
  with pytest.raises(ValueError):
    list(iter_episodes(tmp_path, format="parquet"))


def test_iter_episodes_yields_nothing_for_empty_dir(tmp_path):
  assert list(iter_episodes(tmp_path, format="hdf5")) == []
  assert list(iter_episodes(tmp_path, format="rerun")) == []


def test_load_sim_episodes_skips_too_short_episodes(tmp_path):
  # T=3 raw steps → 2 actions after drop-last. min_episode_len=5 filters it.
  writer = EpisodeWriter(str(tmp_path), format="hdf5")
  writer.write_episode(
    _make_raw_episode(3),
    metadata={"seed": 0, "source": "test"},
  )
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=5, seed=0)
  inserted = buf.load_sim_episodes(tmp_path, format="hdf5")
  assert inserted == 0
  assert buf.num_episodes == 0


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


def test_iter_episodes_progress_callback_sees_every_file(tmp_path):
  for _ in range(3):
    _write_sim_episode(tmp_path, "hdf5", T=10)

  calls: list[tuple[int, int, str]] = []

  def cb(i, total, path):
    calls.append((i, total, path.name))

  list(iter_episodes(tmp_path, format="hdf5", progress=cb))

  assert [i for i, _, _ in calls] == [1, 2, 3]
  assert all(total == 3 for _, total, _ in calls)
  assert [name for _, _, name in calls] == [
    "episode_00000.hdf5",
    "episode_00001.hdf5",
    "episode_00002.hdf5",
  ]


def test_iter_episodes_progress_callback_is_not_called_for_empty_dir(tmp_path):
  calls: list[tuple[int, int, Path]] = []
  list(iter_episodes(tmp_path, format="hdf5", progress=lambda *a: calls.append(a)))
  assert calls == []


def test_load_sim_episodes_forwards_progress(tmp_path):
  for _ in range(2):
    _write_sim_episode(tmp_path, "hdf5", T=10)

  seen: list[int] = []
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=5, seed=0)
  buf.load_sim_episodes(
    tmp_path,
    format="hdf5",
    progress=lambda i, total, path: seen.append(i),
  )
  assert seen == [1, 2]


def test_iter_episodes_progress_true_does_not_crash(tmp_path, capsys):
  # progress=True should work whether or not tqdm is installed: we only
  # assert that no exception escapes and something gets reported.
  _write_sim_episode(tmp_path, "hdf5", T=10)
  eps = list(iter_episodes(tmp_path, format="hdf5", progress=True))
  assert len(eps) == 1
