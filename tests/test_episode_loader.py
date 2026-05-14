"""Tests for loading sim-writer episodes into the replay buffer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chuck_dreamer.config import (
  CNN_BOTTLENECK_HW,
  derive_image_size,
  load_config,
)
from chuck_dreamer.training.episode_loader import (
  iter_episodes,
  load_hdf5_episode,
  load_rerun_episode,
)
from chuck_dreamer.training.episode_processor import (
  ImageProcessor,
  ImageProprioProcessor,
  StateVectorProcessor,
  _drop_last_and_pack,
  processor_for,
)
from chuck_dreamer.training.replay_buffer import ReplayBuffer
from chuck_dreamer.sim.episode_writer import EpisodeWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_episode(T: int, img_hw: tuple[int, int] = (16, 16)) -> dict:
  """An episode in the dict-of-stacked-arrays shape the sim writers consume."""
  H, W = img_hw
  ts = np.arange(T)
  return {
    "image":        np.stack([np.full((H, W, 3), t, dtype=np.uint8) for t in ts]),
    "joint_action": np.stack([np.array([0.1 * t, 0.2 * t, 0.3 * t], dtype=np.float32) for t in ts]),
    "reward":       ts.astype(np.float32),
    "timestamp":    (ts * 0.05).astype(np.float32),
    "joint_qpos":   np.stack([np.full((6,), 0.1 * t, dtype=np.float32) for t in ts]),
    "ee_pos":       np.stack([np.array([0.1, 0.2, 0.3], dtype=np.float32) * t for t in ts]),
    "ee_quat":      np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 1)),
    "object_xy":    np.tile(np.array([0.5, 0.5], dtype=np.float32), (T, 1)),
  }


def _write_sim_episode(dir_path: Path, fmt: str, T: int = 20, idx: int = 0) -> None:
  writer = EpisodeWriter(str(dir_path), format=fmt)
  writer.write_episode(
    _make_raw_episode(T),
    metadata={"seed": 7, "source": "test", "outcome": "done", "goal_xy": [0.1, 0.2]},
    name_suffix=f"{idx:05d}",
  )


# ---------------------------------------------------------------------------
# Drop-last / shape invariants — tested without touching the writers
# ---------------------------------------------------------------------------


def _step_info_columns(N: int) -> dict:
  return {
    "object_xy": np.zeros((N, 2), dtype=np.float32),
    "ee_pos":    np.zeros((N, 3), dtype=np.float32),
    "ee_quat":   np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (N, 1)),
  }


def test_drop_last_and_pack_enforces_buffer_invariants():
  N = 10
  obs = np.arange(N * 4, dtype=np.float32).reshape(N, 4)
  raw = {
    "joint_action": np.arange(N * 2, dtype=np.float32).reshape(N, 2),
    "reward": np.arange(N, dtype=np.float32),
    **_step_info_columns(N),
  }
  ep = _drop_last_and_pack(obs, raw)

  assert ep["obs"].shape == (N, 4)
  assert ep["action"].shape == (N - 1, 2)
  assert ep["reward"].shape == (N - 1,)
  assert ep["done"].shape == (N - 1,)
  assert ep["done"][-1]
  assert not ep["done"][:-1].any()
  # step_info columns align with reward (T = N-1): step_info[t] is the
  # post-action info matching reward[t].
  assert ep["step_info"]["object_xy"].shape == (N - 1, 2)


def test_drop_last_rejects_single_step_episode():
  with pytest.raises(ValueError):
    _drop_last_and_pack(
      np.zeros((1, 3), dtype=np.float32),
      {"joint_action": np.zeros((1, 2), dtype=np.float32),
       "reward": np.zeros((1,)),
       **_step_info_columns(1)},
    )


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------


def test_state_vector_processor_concatenates_state_fields():
  N = 5
  raw = {
    "image": np.zeros((N, 4, 4, 3), dtype=np.uint8),
    "joint_action": np.zeros((N, 3), dtype=np.float32),
    "reward": np.zeros((N,), dtype=np.float32),
    "timestamp": np.zeros((N,), dtype=np.float32),
    "joint_qpos": np.ones((N, 6), dtype=np.float32),
    "ee_pos": np.full((N, 3), 2.0, dtype=np.float32),
    "ee_quat": np.full((N, 4), 3.0, dtype=np.float32),
    "object_xy": np.full((N, 2), 4.0, dtype=np.float32),
  }
  ep = StateVectorProcessor()(raw)

  assert ep["obs"].shape == (N, 15)
  np.testing.assert_array_equal(ep["obs"][0, 0:3], [2.0, 2.0, 2.0])
  np.testing.assert_array_equal(ep["obs"][0, 3:7], [3.0, 3.0, 3.0, 3.0])
  np.testing.assert_array_equal(ep["obs"][0, 7:9], [4.0, 4.0])
  np.testing.assert_array_equal(ep["obs"][0, 9:15], [1.0] * 6)


def test_image_processor_returns_image_as_uint8():
  N = 5
  raw = {
    "image": np.stack([np.full((8, 8, 3), t, dtype=np.uint8) for t in range(N)]),
    "joint_action": np.zeros((N, 3), dtype=np.float32),
    "reward": np.zeros((N,), dtype=np.float32),
    "timestamp": np.zeros((N,), dtype=np.float32),
    "joint_qpos": np.zeros((N, 6), dtype=np.float32),
    "ee_pos": np.zeros((N, 3), dtype=np.float32),
    "ee_quat": np.zeros((N, 4), dtype=np.float32),
    "object_xy": np.zeros((N, 2), dtype=np.float32),
  }
  ep = ImageProcessor(image_size=8)(raw)
  assert ep["obs"].shape == (N, 8, 8, 3)
  assert ep["obs"].dtype == np.uint8
  assert int(ep["obs"][3].mean()) == 3


# ---------------------------------------------------------------------------
# HDF5 round-trip
# ---------------------------------------------------------------------------


def test_load_hdf5_episode_round_trip(tmp_path):
  _write_sim_episode(tmp_path, "hdf5", T=20)
  raw = load_hdf5_episode(tmp_path / "episode-00000.hdf5")

  assert raw["image"].shape == (20, 16, 16, 3)
  assert raw["joint_action"].shape == (20, 3)
  assert raw["reward"].shape == (20,)
  assert raw["act_mode"] == "joint"
  np.testing.assert_allclose(raw["joint_qpos"][7], [0.7] * 6, rtol=0, atol=1e-6)


def test_replay_buffer_loads_hdf5_directory(tmp_path):
  _write_sim_episode(tmp_path, "hdf5", T=20, idx=0)
  _write_sim_episode(tmp_path, "hdf5", T=20, idx=1)

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
  raw = load_rerun_episode(tmp_path / "episode-00000.rrd")

  assert raw["image"].shape == (20, 16, 16, 3)
  assert raw["joint_action"].shape == (20, 3)
  assert raw["reward"].shape == (20,)
  np.testing.assert_allclose(raw["joint_qpos"][7], [0.7] * 6, rtol=0, atol=1e-5)
  assert float(raw["image"][3].mean()) == 3.0


def test_replay_buffer_loads_rerun_directory(tmp_path):
  _write_sim_episode(tmp_path, "rerun", T=20, idx=0)
  _write_sim_episode(tmp_path, "rerun", T=20, idx=1)

  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=5, processor=ImageProcessor(image_size=16), seed=0)
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
    name_suffix="00000",
  )
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=5, seed=0)
  inserted = buf.load_sim_episodes(tmp_path, format="hdf5")
  assert inserted == 0
  assert buf.num_episodes == 0


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


def test_iter_episodes_progress_callback_sees_every_file(tmp_path):
  for idx in range(3):
    _write_sim_episode(tmp_path, "hdf5", T=10, idx=idx)

  calls: list[tuple[int, int, str]] = []

  def cb(i, total, path):
    calls.append((i, total, path.name))

  list(iter_episodes(tmp_path, format="hdf5", progress=cb))

  assert [i for i, _, _ in calls] == [1, 2, 3]
  assert all(total == 3 for _, total, _ in calls)
  assert [name for _, _, name in calls] == [
    "episode-00000.hdf5",
    "episode-00001.hdf5",
    "episode-00002.hdf5",
  ]


def test_iter_episodes_progress_callback_is_not_called_for_empty_dir(tmp_path):
  calls: list[tuple[int, int, Path]] = []
  list(iter_episodes(tmp_path, format="hdf5", progress=lambda *a: calls.append(a)))
  assert calls == []


def test_load_sim_episodes_forwards_progress(tmp_path):
  for idx in range(2):
    _write_sim_episode(tmp_path, "hdf5", T=10, idx=idx)

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


def test_iter_episodes_parallel_preserves_order_and_progress(tmp_path):
  # Parallel loading must yield in submission order and call the
  # progress callback once per file with monotonically increasing i.
  for idx in range(5):
    _write_sim_episode(tmp_path, "hdf5", T=10, idx=idx)

  calls: list[tuple[int, int, str]] = []
  eps = list(iter_episodes(
    tmp_path, format="hdf5", num_workers=3,
    progress=lambda i, total, path: calls.append((i, total, path.name)),
  ))

  assert len(eps) == 5
  assert [i for i, _, _ in calls] == [1, 2, 3, 4, 5]
  assert all(total == 5 for _, total, _ in calls)
  assert [name for _, _, name in calls] == [
    f"episode-{i:05d}.hdf5" for i in range(5)
  ]


# ---------------------------------------------------------------------------
# derive_image_size
# ---------------------------------------------------------------------------


def test_derive_image_size_default_4_strides_yields_64():
  # Default config: 4 stride-2 layers => 4 * 16 = 64.
  assert derive_image_size((2, 2, 2, 2)) == CNN_BOTTLENECK_HW * 16


def test_derive_image_size_scales_with_layer_count():
  # Adding a stride-2 layer doubles the input resolution.
  assert derive_image_size((2,)) == CNN_BOTTLENECK_HW * 2
  assert derive_image_size((2, 2)) == CNN_BOTTLENECK_HW * 4
  assert derive_image_size((2, 2, 2)) == CNN_BOTTLENECK_HW * 8


def test_derive_image_size_handles_mixed_strides():
  # Non-2 strides multiply the same way.
  assert derive_image_size((2, 3)) == CNN_BOTTLENECK_HW * 6
  assert derive_image_size((4, 2, 2)) == CNN_BOTTLENECK_HW * 16


def test_derive_image_size_empty_strides():
  # No strided layers => image_size collapses to the bottleneck.
  assert derive_image_size(()) == CNN_BOTTLENECK_HW


# ---------------------------------------------------------------------------
# ImageProcessor / ImageProprioProcessor: resize + obs shape
# ---------------------------------------------------------------------------


def _state_only_raw(N: int, img_hw: tuple[int, int]) -> dict:
  H, W = img_hw
  return {
    "image": np.stack([np.full((H, W, 3), t, dtype=np.uint8) for t in range(N)]),
    "joint_action": np.zeros((N, 3), dtype=np.float32),
    "reward": np.zeros((N,), dtype=np.float32),
    "timestamp": np.zeros((N,), dtype=np.float32),
    "joint_qpos": np.full((N, 6), 0.5, dtype=np.float32),
    "ee_pos": np.full((N, 3), 0.25, dtype=np.float32),
    "ee_quat": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (N, 1)),
    "object_xy": np.zeros((N, 2), dtype=np.float32),
  }


def test_image_processor_resizes_to_target_size():
  raw = _state_only_raw(N=4, img_hw=(32, 32))
  ep = ImageProcessor(image_size=8)(raw)
  assert ep["obs"].shape == (4, 8, 8, 3)
  assert ep["obs"].dtype == np.uint8


def test_image_processor_passthrough_when_already_target_size():
  raw = _state_only_raw(N=3, img_hw=(8, 8))
  ep = ImageProcessor(image_size=8)(raw)
  assert ep["obs"].shape == (3, 8, 8, 3)
  # When sizes match the resize is a no-op.
  for t in range(3):
    assert int(ep["obs"][t].mean()) == t


def test_image_proprio_processor_returns_dict_obs():
  raw = _state_only_raw(N=5, img_hw=(16, 16))
  ep = ImageProprioProcessor(image_size=8)(raw)

  assert isinstance(ep["obs"], dict)
  assert set(ep["obs"].keys()) == {"image", "proprio"}
  assert ep["obs"]["image"].shape == (5, 8, 8, 3)
  assert ep["obs"]["image"].dtype == np.uint8
  # proprio = ee_pos (3) + ee_quat (4) + joint_qpos (6) = 13.
  assert ep["obs"]["proprio"].shape == (5, 13)
  assert ep["obs"]["proprio"].dtype == np.float32


def test_image_proprio_processor_excludes_object_xy_from_proprio():
  raw = _state_only_raw(N=3, img_hw=(8, 8))
  raw["object_xy"] = np.full((3, 2), 999.0, dtype=np.float32)
  ep = ImageProprioProcessor(image_size=8)(raw)
  # proprio is body-internal, so object_xy must not bleed in.
  assert not np.any(np.isclose(ep["obs"]["proprio"], 999.0))


# ---------------------------------------------------------------------------
# processor_for: dispatch on config.env.obs_mode
# ---------------------------------------------------------------------------


def test_processor_for_state_mode_returns_state_processor():
  cfg = load_config()
  cfg.env.obs_mode = "state"
  p = processor_for(cfg)
  assert isinstance(p, StateVectorProcessor)


def test_processor_for_image_mode_uses_derived_size():
  cfg = load_config()
  cfg.env.obs_mode = "image"
  cfg.model.encoder.cnn_strides = [2, 2, 2]   # 3 layers -> image_size = 32
  p = processor_for(cfg)
  assert isinstance(p, ImageProcessor)
  assert p.image_size == derive_image_size((2, 2, 2))


def test_processor_for_image_proprio_uses_derived_size():
  cfg = load_config()
  cfg.env.obs_mode = "image_proprio"
  cfg.model.encoder.cnn_strides = [2, 2]      # 2 layers -> image_size = 16
  p = processor_for(cfg)
  assert isinstance(p, ImageProprioProcessor)
  assert p.image_size == derive_image_size((2, 2))


def test_processor_for_rejects_unknown_obs_mode():
  cfg = load_config()
  cfg.env.obs_mode = "depth"
  with pytest.raises(ValueError):
    processor_for(cfg)
