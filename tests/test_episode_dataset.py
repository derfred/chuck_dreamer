"""Tests for loading sim-writer episodes via the EpisodeDataset abstraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chuck_dreamer.config import (
  CNN_BOTTLENECK_HW,
  derive_image_size,
  load_config,
)
from chuck_dreamer.training.episode_dataset import Episode, EpisodeDataset
from chuck_dreamer.training.episode_processor import (
  EpisodeProcessor,
  _pack,
  processor_for,
)
from chuck_dreamer.training.observation import Observation
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


def test_pack_enforces_buffer_invariants():
  N = 10
  obs = Observation.from_components(
    {"joints": np.arange(N * 4, dtype=np.float32).reshape(N, 4)},
  )
  raw = {
    "joint_action": np.arange(N * 2, dtype=np.float32).reshape(N, 2),
    "reward": np.arange(N, dtype=np.float32),
    **_step_info_columns(N),
  }
  ep = _pack(obs, raw, "joint")

  assert ep["obs"]["joints"].shape == (N, 4)
  assert ep["action"].shape == (N - 1, 2)
  assert ep["reward"].shape == (N - 1,)
  assert ep["done"].shape == (N - 1,)
  assert ep["done"][-1]
  assert not ep["done"][:-1].any()
  # step_info columns align with reward (T = N-1): step_info[t] is the
  # post-action info matching reward[t].
  assert ep["step_info"]["object_xy"].shape == (N - 1, 2)


def test_pack_rejects_single_step_episode():
  with pytest.raises(ValueError):
    _pack(
      Observation.from_components({"joints": np.zeros((1, 3), dtype=np.float32)}),
      {"joint_action": np.zeros((1, 2), dtype=np.float32),
       "reward": np.zeros((1,)),
       **_step_info_columns(1)},
      "joint",
    )


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------


def test_joints_processor_returns_joint_qpos():
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
  ep = EpisodeProcessor(("joints",), act_mode="joint")(raw)

  assert ep["obs"]["joints"].shape == (N, 6)
  np.testing.assert_array_equal(ep["obs"]["joints"][0], [1.0] * 6)


def test_object_xy_processor_returns_object_xy():
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
  ep = EpisodeProcessor(("object_xy",), act_mode="joint")(raw)

  assert ep["obs"]["object_xy"].shape == (N, 2)
  np.testing.assert_array_equal(ep["obs"]["object_xy"][0], [4.0, 4.0])


def test_object_uv_processor_returns_object_uv():
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
    "object_uv": np.full((N, 2), 50.0, dtype=np.float32),
  }
  ep = EpisodeProcessor(("object_uv",), act_mode="joint")(raw)

  assert ep["obs"]["object_uv"].shape == (N, 2)
  np.testing.assert_array_equal(ep["obs"]["object_uv"][0], [50.0, 50.0])


def test_image_processor_returns_normalized_float32():
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
  ep = EpisodeProcessor(("image",), image_size=8, act_mode="joint")(raw)
  assert ep["obs"]["image"].shape == (N, 8, 8, 3)
  assert ep["obs"]["image"].dtype == np.float32
  # Each timestep's uint8 fill ``t`` maps to ``t/255 - 0.5``.
  for t in range(N):
    np.testing.assert_allclose(ep["obs"]["image"][t].mean(), t / 255.0 - 0.5, atol=1e-6)


def test_image_processor_attaches_focus_mask_when_sources_present():
  N, H, W = 4, 8, 8
  raw = {
    "image":                 np.zeros((N, H, W, 3), dtype=np.uint8),
    "segmentation_target":   np.zeros((N, H, W),    dtype=bool),
    "segmentation_arm":      np.zeros((N, H, W),    dtype=bool),
    "joint_action": np.zeros((N, 3), dtype=np.float32),
    "reward":       np.zeros((N,),   dtype=np.float32),
    "timestamp":    np.zeros((N,),   dtype=np.float32),
    "joint_qpos":   np.zeros((N, 6), dtype=np.float32),
    "ee_pos":       np.zeros((N, 3), dtype=np.float32),
    "ee_quat":      np.zeros((N, 4), dtype=np.float32),
    "object_xy":    np.zeros((N, 2), dtype=np.float32),
  }
  # target lights up a (2,2) corner at each step; arm lights up a different corner.
  raw["segmentation_target"][:, :2, :2] = True
  raw["segmentation_arm"][:,    -2:, -2:] = True

  proc = EpisodeProcessor(
    "image", image_size=H,
    focus_mask_sources=("segmentation_target", "segmentation_arm"),
    act_mode="joint",
  )
  ep = proc(raw)
  assert "focus_mask" in ep
  fm = ep["focus_mask"]
  assert fm.shape == (N, H, W)
  assert fm.dtype == np.float32
  # Union of the two masks ⇒ both corners on.
  assert np.allclose(fm[:, :2, :2], 1.0)
  assert np.allclose(fm[:, -2:, -2:], 1.0)
  # Off-mask regions stay 0.
  assert np.allclose(fm[:, 3:5, 3:5], 0.0)


def test_image_processor_omits_focus_mask_when_no_sources_match():
  # Sources named but no segmentation_* key present → focus_mask absent.
  N = 3
  raw = {
    "image": np.zeros((N, 4, 4, 3), dtype=np.uint8),
    "joint_action": np.zeros((N, 2), dtype=np.float32),
    "reward":       np.zeros((N,),   dtype=np.float32),
    "timestamp":    np.zeros((N,),   dtype=np.float32),
    "joint_qpos":   np.zeros((N, 6), dtype=np.float32),
    "ee_pos":       np.zeros((N, 3), dtype=np.float32),
    "ee_quat":      np.zeros((N, 4), dtype=np.float32),
    "object_xy":    np.zeros((N, 2), dtype=np.float32),
  }
  ep = EpisodeProcessor("image", image_size=4, focus_mask_sources=("segmentation_target",), act_mode="joint")(raw)
  assert "focus_mask" not in ep


def test_focus_mask_resizes_alongside_image():
  # Mask of shape (N, 32, 32) should be downsampled to (N, 8, 8) with the image.
  N = 2
  raw_H = 32
  raw = {
    "image":               np.zeros((N, raw_H, raw_H, 3), dtype=np.uint8),
    "segmentation_target": np.zeros((N, raw_H, raw_H),    dtype=bool),
    "joint_action": np.zeros((N, 2), dtype=np.float32),
    "reward":       np.zeros((N,),   dtype=np.float32),
    "timestamp":    np.zeros((N,),   dtype=np.float32),
    "joint_qpos":   np.zeros((N, 6), dtype=np.float32),
    "ee_pos":       np.zeros((N, 3), dtype=np.float32),
    "ee_quat":      np.zeros((N, 4), dtype=np.float32),
    "object_xy":    np.zeros((N, 2), dtype=np.float32),
  }
  # 16x16 patch in the top-left ⇒ should remain on after a 4x downsample.
  raw["segmentation_target"][:, :16, :16] = True

  ep = EpisodeProcessor(
    "image", image_size=8, focus_mask_sources=("segmentation_target",),
    act_mode="joint",
  )(raw)
  fm = ep["focus_mask"]
  assert fm.shape == (N, 8, 8)
  # Nearest-neighbor downsample keeps 0/1 — no in-between values from blur.
  assert set(np.unique(fm).tolist()).issubset({0.0, 1.0})
  # The half that was masked is still masked.
  assert np.allclose(fm[:, :4, :4], 1.0)
  assert np.allclose(fm[:, 4:, 4:], 0.0)


# ---------------------------------------------------------------------------
# Episode.from_file — HDF5 / Rerun round-trip
# ---------------------------------------------------------------------------


def test_episode_from_file_hdf5_round_trip(tmp_path):
  _write_sim_episode(tmp_path, "hdf5", T=20)
  ep = Episode.from_file(tmp_path / "episode-00000.hdf5")

  assert ep.format == "hdf5"
  assert ep.stem == "episode-00000"
  assert ep.num_steps == 20
  assert ep.data["image"].shape == (20, 16, 16, 3)
  assert ep.data["joint_action"].shape == (20, 3)
  assert ep.data["reward"].shape == (20,)
  np.testing.assert_allclose(ep.data["joint_qpos"][7], [0.7] * 6, rtol=0, atol=1e-6)


def test_episode_from_file_rerun_round_trip(tmp_path):
  _write_sim_episode(tmp_path, "rerun", T=20)
  ep = Episode.from_file(tmp_path / "episode-00000.rrd")

  assert ep.format == "rerun"
  assert ep.num_steps == 20
  assert ep.data["image"].shape == (20, 16, 16, 3)
  assert ep.data["joint_action"].shape == (20, 3)
  assert ep.data["reward"].shape == (20,)
  np.testing.assert_allclose(ep.data["joint_qpos"][7], [0.7] * 6, rtol=0, atol=1e-5)
  assert float(ep.data["image"][3].mean()) == 3.0


def test_episode_from_file_infers_format_from_suffix(tmp_path):
  _write_sim_episode(tmp_path, "hdf5", T=10)
  ep = Episode.from_file(tmp_path / "episode-00000.hdf5")
  assert ep.format == "hdf5"


def test_episode_from_file_explicit_format_overrides_suffix(tmp_path):
  _write_sim_episode(tmp_path, "hdf5", T=10)
  # Explicit format wins even if it disagrees with the suffix — the
  # extension check is just a sniff for convenience.
  ep = Episode.from_file(tmp_path / "episode-00000.hdf5", format="hdf5")
  assert ep.format == "hdf5"


def test_episode_from_file_rejects_unknown_suffix(tmp_path):
  bogus = tmp_path / "episode-00000.parquet"
  bogus.write_text("")
  with pytest.raises(ValueError):
    Episode.from_file(bogus)


# ---------------------------------------------------------------------------
# Replay-buffer ingestion — HDF5
# ---------------------------------------------------------------------------


def test_replay_buffer_loads_hdf5_directory(tmp_path):
  _write_sim_episode(tmp_path, "hdf5", T=20, idx=0)
  _write_sim_episode(tmp_path, "hdf5", T=20, idx=1)

  buf = ReplayBuffer(
    capacity_steps=10_000, min_episode_len=5, seed=0,
    processor=EpisodeProcessor(("joints",), act_mode="joint"),
  )
  n = buf.load_sim_episodes(tmp_path, format="hdf5")

  assert n == 2
  assert buf.num_episodes == 2
  # 20 raw steps → 20 obs + 19 actions per episode.
  assert len(buf) == 2 * 19
  ep = buf._episodes[0]
  # joints = joint_qpos(6) under the new vocabulary.
  assert ep["obs"].components["joints"].shape == (20, 6)
  assert ep["action"].shape == (20 - 1, 3)
  assert bool(ep["done"][-1])

  batch = buf.sample(batch_size=4, seq_len=10)
  assert batch["obs"]["joints"].shape == (4, 10, 6)


# ---------------------------------------------------------------------------
# Replay-buffer ingestion — Rerun
# ---------------------------------------------------------------------------


def test_replay_buffer_loads_rerun_directory(tmp_path):
  _write_sim_episode(tmp_path, "rerun", T=20, idx=0)
  _write_sim_episode(tmp_path, "rerun", T=20, idx=1)

  buf = ReplayBuffer(
    capacity_steps=10_000, min_episode_len=5,
    processor=EpisodeProcessor(("image",), image_size=16, act_mode="joint"),
    seed=0,
  )
  n = buf.load_sim_episodes(tmp_path, format="rerun")

  assert n == 2
  ep = buf._episodes[0]
  img = ep["obs"].components["image"]
  assert img.shape == (20, 16, 16, 3)
  assert ep["action"].shape == (19, 3)
  # Image[t] is filled with constant ``t`` uint8 → ``t/255 - 0.5`` after
  # the processor normalizes. The ordering check survives normalization
  # since the mapping is monotonic.
  for t in range(20):
    np.testing.assert_allclose(float(img[t].mean()), t / 255.0 - 0.5, atol=1e-6)


# ---------------------------------------------------------------------------
# EpisodeDataset — discovery, load, stream, error paths
# ---------------------------------------------------------------------------


def test_dataset_rejects_unknown_format(tmp_path):
  with pytest.raises(ValueError):
    EpisodeDataset(tmp_path, format="parquet")


def test_dataset_empty_dir_has_zero_length(tmp_path):
  for fmt in ("hdf5", "rerun"):
    ds = EpisodeDataset(tmp_path, format=fmt)
    assert len(ds) == 0
    assert ds.paths == []
    assert not ds.loaded
    # load() on an empty dir is a no-crash no-op.
    ds.load()
    assert len(ds) == 0
    assert ds.loaded


def test_dataset_missing_dir_has_zero_length(tmp_path):
  ds = EpisodeDataset(tmp_path / "does-not-exist", format="hdf5")
  assert len(ds) == 0
  assert list(ds.stream()) == []


def test_dataset_discovery_does_not_load(tmp_path):
  for idx in range(3):
    _write_sim_episode(tmp_path, "hdf5", T=10, idx=idx)
  ds = EpisodeDataset(tmp_path, format="hdf5")
  assert len(ds) == 3
  assert not ds.loaded
  assert [p.name for p in ds.paths] == [f"episode-{i:05d}.hdf5" for i in range(3)]


def test_dataset_load_caches_after_first_call(tmp_path):
  _write_sim_episode(tmp_path, "hdf5", T=10)
  ds = EpisodeDataset(tmp_path, format="hdf5")
  ds.load()
  loaded_once = list(ds)
  ds.load()  # idempotent: must not reload
  loaded_twice = list(ds)
  assert loaded_once[0] is loaded_twice[0]


def test_dataset_getitem_by_index_and_stem(tmp_path):
  for idx in range(3):
    _write_sim_episode(tmp_path, "hdf5", T=10, idx=idx)
  ds = EpisodeDataset(tmp_path, format="hdf5").load()

  assert ds[0].stem == "episode-00000"
  assert ds["episode-00002"].stem == "episode-00002"
  with pytest.raises(KeyError):
    ds["episode-99999"]


def test_dataset_iter_requires_load(tmp_path):
  _write_sim_episode(tmp_path, "hdf5", T=10)
  ds = EpisodeDataset(tmp_path, format="hdf5")
  with pytest.raises(RuntimeError):
    iter(ds)
  with pytest.raises(RuntimeError):
    ds[0]


def test_dataset_stream_does_not_populate_loaded(tmp_path):
  _write_sim_episode(tmp_path, "hdf5", T=10)
  ds = EpisodeDataset(tmp_path, format="hdf5")
  eps = list(ds.stream())
  assert len(eps) == 1
  assert not ds.loaded   # streaming is non-retentive


def test_dataset_load_skips_too_short_episodes_via_buffer(tmp_path):
  # T=3 raw steps → 2 actions after drop-last. min_episode_len=5 filters
  # it out at the buffer side; the dataset still yields the episode.
  writer = EpisodeWriter(str(tmp_path), format="hdf5")
  writer.write_episode(
    _make_raw_episode(3),
    metadata={"seed": 0, "source": "test"},
    name_suffix="00000",
  )
  buf = ReplayBuffer(
    capacity_steps=10_000, min_episode_len=5, seed=0,
    processor=EpisodeProcessor("joints", act_mode="joint"),
  )
  inserted = buf.load_sim_episodes(tmp_path, format="hdf5")
  assert inserted == 0
  assert buf.num_episodes == 0


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


def test_dataset_stream_progress_callback_sees_every_file(tmp_path):
  for idx in range(3):
    _write_sim_episode(tmp_path, "hdf5", T=10, idx=idx)

  calls: list[tuple[int, int, str]] = []

  def cb(i, total, path):
    calls.append((i, total, path.name))

  list(EpisodeDataset(tmp_path, format="hdf5").stream(progress=cb))

  assert [i for i, _, _ in calls] == [1, 2, 3]
  assert all(total == 3 for _, total, _ in calls)
  assert [name for _, _, name in calls] == [
    "episode-00000.hdf5",
    "episode-00001.hdf5",
    "episode-00002.hdf5",
  ]


def test_dataset_stream_progress_callback_is_not_called_for_empty_dir(tmp_path):
  calls: list[tuple[int, int, Path]] = []
  list(EpisodeDataset(tmp_path, format="hdf5").stream(progress=lambda *a: calls.append(a)))
  assert calls == []


def test_load_sim_episodes_forwards_progress(tmp_path):
  for idx in range(2):
    _write_sim_episode(tmp_path, "hdf5", T=10, idx=idx)

  seen: list[int] = []
  buf = ReplayBuffer(
    capacity_steps=10_000, min_episode_len=5, seed=0,
    processor=EpisodeProcessor("joints", act_mode="joint"),
  )
  buf.load_sim_episodes(
    tmp_path,
    format="hdf5",
    progress=lambda i, total, path: seen.append(i),
  )
  assert seen == [1, 2]


def test_dataset_stream_progress_true_does_not_crash(tmp_path, capsys):
  # progress=True should work whether or not tqdm is installed: we only
  # assert that no exception escapes and something gets reported.
  _write_sim_episode(tmp_path, "hdf5", T=10)
  eps = list(EpisodeDataset(tmp_path, format="hdf5").stream(progress=True))
  assert len(eps) == 1


def test_dataset_stream_parallel_preserves_order_and_progress(tmp_path):
  # Parallel loading must yield in submission order and call the
  # progress callback once per file with monotonically increasing i.
  for idx in range(5):
    _write_sim_episode(tmp_path, "hdf5", T=10, idx=idx)

  calls: list[tuple[int, int, str]] = []
  eps = list(EpisodeDataset(tmp_path, format="hdf5").stream(
    num_workers=3,
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
  ep = EpisodeProcessor(("image",), image_size=8, act_mode="joint")(raw)
  assert ep["obs"]["image"].shape == (4, 8, 8, 3)
  assert ep["obs"]["image"].dtype == np.float32


def test_image_processor_passthrough_when_already_target_size():
  raw = _state_only_raw(N=3, img_hw=(8, 8))
  ep = EpisodeProcessor(("image",), image_size=8, act_mode="joint")(raw)
  assert ep["obs"]["image"].shape == (3, 8, 8, 3)
  # When sizes match the resize is a no-op; values are then normalized.
  for t in range(3):
    np.testing.assert_allclose(ep["obs"]["image"][t].mean(), t / 255.0 - 0.5, atol=1e-6)


def test_image_proprio_processor_returns_dict_obs():
  raw = _state_only_raw(N=5, img_hw=(16, 16))
  ep = EpisodeProcessor(("image", "proprio"), image_size=8, act_mode="joint")(raw)

  assert isinstance(ep["obs"], dict)
  assert set(ep["obs"].keys()) == {"image", "proprio"}
  assert ep["obs"]["image"].shape == (5, 8, 8, 3)
  assert ep["obs"]["image"].dtype == np.float32
  # proprio = joint_qpos (6) under the new vocabulary; ee fields live
  # in a separate ``ee`` component.
  assert ep["obs"]["proprio"].shape == (5, 6)
  assert ep["obs"]["proprio"].dtype == np.float32


def test_image_proprio_processor_excludes_object_xy_from_proprio():
  raw = _state_only_raw(N=3, img_hw=(8, 8))
  raw["object_xy"] = np.full((3, 2), 999.0, dtype=np.float32)
  ep = EpisodeProcessor(("image", "proprio"), image_size=8, act_mode="joint")(raw)
  # proprio is body-internal, so object_xy must not bleed in.
  assert not np.any(np.isclose(ep["obs"]["proprio"], 999.0))


def test_ee_processor_concatenates_ee_pos_and_quat():
  raw = _state_only_raw(N=4, img_hw=(8, 8))
  ep = EpisodeProcessor(("ee",), act_mode="joint")(raw)
  assert ep["obs"]["ee"].shape == (4, 7)
  np.testing.assert_allclose(ep["obs"]["ee"][:, :3], 0.25)
  np.testing.assert_allclose(ep["obs"]["ee"][:, 3], 1.0)


# ---------------------------------------------------------------------------
# processor_for: dispatch on config.env.obs_mode
# ---------------------------------------------------------------------------


def test_processor_for_joints_mode_returns_joints_processor():
  cfg = load_config()
  cfg.env.obs_mode = ["joints"]
  p = processor_for(cfg)
  assert isinstance(p, EpisodeProcessor)
  assert p.obs_mode == ("joints",)
  assert p.image_size is None


def test_processor_for_image_mode_uses_derived_size():
  cfg = load_config()
  cfg.env.obs_mode = ["image"]
  cfg.model.encoder.cnn_strides = [2, 2, 2]   # 3 layers -> image_size = 32
  p = processor_for(cfg)
  assert isinstance(p, EpisodeProcessor)
  assert p.obs_mode == ("image",)
  assert p.image_size == derive_image_size((2, 2, 2))


def test_processor_for_image_proprio_uses_derived_size():
  cfg = load_config()
  cfg.env.obs_mode = ["image", "proprio"]
  cfg.model.encoder.cnn_strides = [2, 2]      # 2 layers -> image_size = 16
  p = processor_for(cfg)
  assert isinstance(p, EpisodeProcessor)
  assert p.obs_mode == ("image", "proprio")
  assert p.image_size == derive_image_size((2, 2))


def test_processor_for_rejects_unknown_obs_mode():
  cfg = load_config()
  cfg.env.obs_mode = ["depth"]
  with pytest.raises(ValueError):
    processor_for(cfg)

# ---------------------------------------------------------------------------
# Tags — round-trip through writer → episode_dataset → processor → buffer
# ---------------------------------------------------------------------------


def _write_tagged_episode(dir_path: Path, fmt: str, tags: tuple, *, T: int = 20, idx: int = 0) -> None:
  writer = EpisodeWriter(str(dir_path), format=fmt)
  writer.write_episode(
    _make_raw_episode(T),
    metadata={"seed": 7, "source": "test", "outcome": "done",
              "goal_xy": [0.1, 0.2], "tags": tags},
    name_suffix=f"{idx:05d}",
  )


@pytest.mark.parametrize("fmt,suffix", [("hdf5", ".hdf5"), ("rerun", ".rrd")])
def test_episode_tags_round_trip_through_metadata(tmp_path, fmt, suffix):
  _write_tagged_episode(tmp_path, fmt, tags=("real", "demo"))
  ep = Episode.from_file(tmp_path / f"episode-00000{suffix}")
  assert ep.metadata.get("tags") == ("real", "demo")
  # Tags are also surfaced on raw data for the processor to lift.
  assert ep.data.get("tags") == ("real", "demo")


@pytest.mark.parametrize("fmt,suffix", [("hdf5", ".hdf5"), ("rerun", ".rrd")])
def test_episode_without_tags_has_no_tags_key(tmp_path, fmt, suffix):
  # Default _write_sim_episode does not stamp tags.
  _write_sim_episode(tmp_path, fmt, T=10)
  ep = Episode.from_file(tmp_path / f"episode-00000{suffix}")
  assert "tags" not in ep.metadata
  assert "tags" not in ep.data


@pytest.mark.parametrize("fmt", ["hdf5", "rerun"])
def test_processor_propagates_tags_to_buffer_episode(tmp_path, fmt):
  _write_tagged_episode(tmp_path, fmt, tags=("real",), T=12)
  processor = EpisodeProcessor("image", image_size=16, act_mode="joint")
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=5, processor=processor, seed=0)
  buf.load_sim_episodes(tmp_path, format=fmt)
  assert buf.num_episodes == 1
  assert buf._episodes[0]["tags"] == ("real",)


@pytest.mark.parametrize("fmt", ["hdf5", "rerun"])
def test_untagged_episodes_get_empty_tag_tuple_in_buffer(tmp_path, fmt):
  _write_sim_episode(tmp_path, fmt, T=12)
  processor = EpisodeProcessor("image", image_size=16, act_mode="joint")
  buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=5, processor=processor, seed=0)
  buf.load_sim_episodes(tmp_path, format=fmt)
  assert buf._episodes[0]["tags"] == ()


# ---------------------------------------------------------------------------
# Action streams — both can be written, processor picks based on act_mode
# ---------------------------------------------------------------------------


def _make_dual_action_raw(T: int) -> dict:
  """Raw episode carrying BOTH joint_action and ee_action — the real-recording shape."""
  raw = _make_raw_episode(T)
  raw["ee_action"] = np.stack([
    np.array([1.0 + 0.01 * t, 2.0 + 0.01 * t, 3.0 + 0.01 * t,
              1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for t in range(T)
  ])
  return raw


@pytest.mark.parametrize("fmt,suffix", [("hdf5", ".hdf5"), ("rerun", ".rrd")])
def test_writer_round_trips_both_action_streams(tmp_path, fmt, suffix):
  writer = EpisodeWriter(str(tmp_path), format=fmt)
  writer.write_episode(_make_dual_action_raw(T=8), metadata={}, name_suffix="00000")
  ep = Episode.from_file(tmp_path / f"episode-00000{suffix}")
  assert ep.data["joint_action"].shape == (8, 3)
  assert ep.data["ee_action"].shape    == (8, 7)


@pytest.mark.parametrize("fmt,suffix", [("hdf5", ".hdf5"), ("rerun", ".rrd")])
def test_processor_picks_joint_action_when_act_mode_joint(tmp_path, fmt, suffix):
  writer = EpisodeWriter(str(tmp_path), format=fmt)
  writer.write_episode(_make_dual_action_raw(T=8), metadata={}, name_suffix="00000")
  raw = Episode.from_file(tmp_path / f"episode-00000{suffix}").data

  ep = EpisodeProcessor("joints", act_mode="joint")(raw)
  # joint_action[t] = [0.1t, 0.2t, 0.3t] from _make_raw_episode.
  assert ep["action"].shape == (7, 3)
  np.testing.assert_allclose(ep["action"][0], [0.0, 0.0, 0.0], atol=1e-6)
  np.testing.assert_allclose(ep["action"][1], [0.1, 0.2, 0.3], atol=1e-6)


@pytest.mark.parametrize("fmt,suffix", [("hdf5", ".hdf5"), ("rerun", ".rrd")])
def test_processor_picks_ee_action_when_act_mode_ee(tmp_path, fmt, suffix):
  writer = EpisodeWriter(str(tmp_path), format=fmt)
  writer.write_episode(_make_dual_action_raw(T=8), metadata={}, name_suffix="00000")
  raw = Episode.from_file(tmp_path / f"episode-00000{suffix}").data

  ep = EpisodeProcessor("joints", act_mode="ee")(raw)
  # ee_action[t] starts with [1.0+0.01t, 2.0+0.01t, 3.0+0.01t, 1, 0, 0, 0].
  assert ep["action"].shape == (7, 7)
  np.testing.assert_allclose(ep["action"][0][:3], [1.0, 2.0, 3.0], atol=1e-5)


def test_processor_raises_when_requested_action_stream_missing():
  # Episode carries joint_action only; asking for ee_action must hard-error.
  raw = _make_raw_episode(T=5)
  with pytest.raises(KeyError, match="ee_action"):
    EpisodeProcessor("joints", act_mode="ee")(raw)


def test_processor_for_reads_act_mode_from_config():
  cfg = load_config()
  cfg.env.act_mode = "joint"
  proc = processor_for(cfg)
  assert proc.act_mode == "joint"

  cfg.env.act_mode = "ee"
  proc = processor_for(cfg)
  assert proc.act_mode == "ee"


def test_act_mode_metadata_is_no_longer_written(tmp_path):
  # Sim writer should not stamp act_mode anywhere; consumers pick based on
  # training config.
  writer = EpisodeWriter(str(tmp_path), format="hdf5")
  writer.write_episode(_make_raw_episode(T=4), metadata={}, name_suffix="00000")
  ep = Episode.from_file(tmp_path / "episode-00000.hdf5")
  assert "act_mode" not in ep.metadata
  assert "act_mode" not in ep.data


def test_writer_rejects_episode_with_no_action_stream(tmp_path):
  writer = EpisodeWriter(str(tmp_path), format="hdf5")
  raw = _make_raw_episode(T=4)
  del raw["joint_action"]  # leave neither joint_action nor ee_action
  with pytest.raises(KeyError, match="action"):
    writer.write_episode(raw, metadata={}, name_suffix="00000")
