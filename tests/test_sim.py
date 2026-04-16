"""Tests for the MuJoCo pushing simulation."""

from __future__ import annotations

import json
import os
import tempfile

import h5py
import mujoco
import numpy as np
import pytest

from src.world_model_pusher.sim import (
  CameraConfig,
  EpisodeWriter,
  LightingConfig,
  ObjectConfig,
  PushingEnv,
  SceneBuilder,
  SceneConfig,
  SceneGenerator,
  create_default_base_mjcf,
  random_push_policy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_config() -> SceneConfig:
  """Return a minimal valid SceneConfig."""
  return SceneConfig(
    table_size=[0.30, 0.25, 0.02],
    table_friction=0.5,
    table_color=[0.6, 0.5, 0.4, 1.0],
    robot_type="stick",
    robot_base_pos=[-0.30, 0.0, 0.04],
    robot_initial_qpos=None,
    target=ObjectConfig(
      shape="box",
      size=[0.03, 0.03, 0.03],
      mass=0.1,
      friction=0.5,
      pos=[0.05, 0.0],
      orientation=0.0,
      color=[1.0, 0.0, 0.0, 1.0],
    ),
    goal_pos=[0.10, 0.05],
    goal_tolerance=0.04,
    obstacles=[],
    clutter=[],
    camera=CameraConfig(pos=[0.0, -0.40, 0.55], look_at=[0.0, 0.0, 0.04], fov=60.0),
    lighting=LightingConfig(direction=[0.0, -0.5, -1.0], intensity=0.8, ambient=0.3),
    max_steps=50,
    control_dt=0.1,
  )


@pytest.fixture
def base_mjcf_path(tmp_path):
  path = str(tmp_path / "base.xml")
  create_default_base_mjcf(path)
  return path


@pytest.fixture
def builder(base_mjcf_path):
  return SceneBuilder(base_mjcf_path)


@pytest.fixture
def env(builder):
  e = PushingEnv(builder, render_size=(64, 64))
  yield e
  e.close()


# ---------------------------------------------------------------------------
# SceneConfig dataclass tests
# ---------------------------------------------------------------------------

class TestSceneConfig:
  def test_construction(self):
    cfg = _make_simple_config()
    assert cfg.table_size == [0.30, 0.25, 0.02]
    assert cfg.target.shape == "box"
    assert cfg.target.mass == 0.1
    assert cfg.goal_pos == [0.10, 0.05]
    assert cfg.max_steps == 50
    assert cfg.obstacles == []

  def test_object_config_fields(self):
    obj = ObjectConfig(
      shape="cylinder",
      size=[0.03, 0.05],
      mass=0.15,
      friction=0.6,
      pos=[0.1, 0.0],
      orientation=1.0,
      color=[0.0, 1.0, 0.0, 1.0],
    )
    assert obj.shape == "cylinder"
    assert obj.mesh_path is None

  def test_camera_config(self):
    cam = CameraConfig(pos=[0.0, -0.4, 0.5], look_at=[0.0, 0.0, 0.04], fov=60.0)
    assert cam.fov == 60.0

  def test_lighting_config(self):
    light = LightingConfig(direction=[0.0, -0.5, -1.0], intensity=0.8, ambient=0.3)
    assert 0.0 <= light.intensity <= 1.0


# ---------------------------------------------------------------------------
# SceneGenerator tests
# ---------------------------------------------------------------------------

class TestSceneGenerator:
  def test_sample_easy(self):
    gen = SceneGenerator(difficulty="easy")
    rng = np.random.default_rng(42)
    cfg = gen.sample(rng)
    assert isinstance(cfg, SceneConfig)
    assert cfg.target.shape in ["box", "cylinder"]
    assert len(cfg.obstacles) == 0

  def test_sample_medium(self):
    gen = SceneGenerator(difficulty="medium")
    rng = np.random.default_rng(42)
    cfg = gen.sample(rng)
    assert isinstance(cfg, SceneConfig)
    assert cfg.target.shape in ["box", "cylinder", "capsule"]

  def test_sample_hard(self):
    gen = SceneGenerator(difficulty="hard")
    rng = np.random.default_rng(42)
    cfg = gen.sample(rng)
    assert isinstance(cfg, SceneConfig)

  def test_invalid_difficulty(self):
    with pytest.raises(ValueError):
      SceneGenerator(difficulty="extreme")

  def test_validity_checks_reject_bad_target(self):
    gen = SceneGenerator(difficulty="easy")
    rng = np.random.default_rng(42)
    cfg = gen.sample(rng)
    # Place target out of reach (too far from robot base)
    cfg.target.pos = [0.50, 0.50]
    assert not gen._check_reachability(cfg)

  def test_validity_goal_on_table(self):
    gen = SceneGenerator(difficulty="easy")
    rng = np.random.default_rng(42)
    cfg = gen.sample(rng)
    cfg.goal_pos = [5.0, 5.0]  # way off table
    assert not gen._check_goal_on_table(cfg)

  def test_multiple_samples_are_valid(self):
    gen = SceneGenerator(difficulty="easy")
    rng = np.random.default_rng(0)
    for _ in range(10):
      cfg = gen.sample(rng)
      assert gen._is_valid(cfg), "Generated config should be valid"


# ---------------------------------------------------------------------------
# SceneBuilder tests
# ---------------------------------------------------------------------------

class TestSceneBuilder:
  def test_build_returns_mjmodel(self, builder):
    cfg = _make_simple_config()
    model = builder.build(cfg)
    assert isinstance(model, mujoco.MjModel)

  def test_target_object_in_model(self, builder):
    cfg = _make_simple_config()
    model = builder.build(cfg)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    assert body_id >= 0

  def test_obstacles_in_model(self, builder):
    cfg = _make_simple_config()
    cfg.obstacles = [
      ObjectConfig(
        shape="box",
        size=[0.03, 0.03, 0.03],
        mass=0.2,
        friction=0.5,
        pos=[0.08, 0.08],
        orientation=0.0,
        color=[0.0, 0.0, 1.0, 1.0],
      )
    ]
    model = builder.build(cfg)
    obs_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_0")
    assert obs_id >= 0

  def test_create_default_base_mjcf(self, tmp_path):
    path = str(tmp_path / "test_base.xml")
    create_default_base_mjcf(path)
    assert os.path.exists(path)
    with open(path) as f:
      content = f.read()
    assert "mocap_target" in content
    assert "ee_link" in content
    assert "weld" in content


# ---------------------------------------------------------------------------
# PushingEnv tests
# ---------------------------------------------------------------------------

class TestPushingEnv:
  def test_reset_returns_correct_shapes(self, env):
    cfg = _make_simple_config()
    obs, info = env.reset(config=cfg)
    assert obs["image"].shape == (64, 64, 3)
    assert obs["image"].dtype == np.uint8
    assert obs["ee_pos"].shape == (3,)
    assert obs["object_pos"].shape == (2,)

  def test_step_returns_correct_shapes(self, env):
    cfg = _make_simple_config()
    env.reset(config=cfg)
    action = np.array([0.01, 0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs["image"].shape == (64, 64, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "object_pos" in info
    assert "ee_pos" in info

  def test_action_clipping(self, env):
    cfg = _make_simple_config()
    env.reset(config=cfg)
    # Oversized action should be clipped without error
    action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs["image"].shape == (64, 64, 3)

  def test_episode_terminates_on_timeout(self, env):
    cfg = _make_simple_config()
    cfg.max_steps = 3
    env.reset(config=cfg)
    action = np.zeros(3, dtype=np.float32)
    done = False
    for _ in range(5):
      _, _, term, _, _ = env.step(action)
      if term:
        done = True
        break
    assert done

  def test_reset_requires_config(self, env):
    with pytest.raises((ValueError, TypeError)):
      env.reset()

  def test_observation_and_action_spaces(self, env):
    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.action_space.shape == (3,)


# ---------------------------------------------------------------------------
# EpisodeWriter tests
# ---------------------------------------------------------------------------

class TestEpisodeWriter:
  def _make_fake_episode(self, T: int = 5, H: int = 64, W: int = 64):
    rng = np.random.default_rng(42)
    return [
      {
        "pre_image": rng.integers(0, 256, (H, W, 3), dtype=np.uint8),
        "post_image": rng.integers(0, 256, (H, W, 3), dtype=np.uint8),
        "action": rng.uniform(-0.02, 0.02, (3,)).astype(np.float32),
        "reward": float(rng.uniform(-1.0, 0.0)),
      }
      for _ in range(T)
    ]

  def test_write_and_read_back(self, tmp_path):
    writer = EpisodeWriter(str(tmp_path), format="hdf5")
    episode = self._make_fake_episode(T=5, H=64, W=64)
    cfg = _make_simple_config()
    path = writer.write_episode(episode, metadata={"config": cfg, "seed": 42, "source": "sim"})

    assert path.exists()
    with h5py.File(path, "r") as f:
      assert f["pre_images"].shape == (5, 64, 64, 3)
      assert f["pre_images"].dtype == np.uint8
      assert f["post_images"].shape == (5, 64, 64, 3)
      assert f["actions"].shape == (5, 3)
      assert f["rewards"].shape == (5,)
      assert "metadata" in f
      assert "config" in f["metadata"]
      assert "seed" in f["metadata"]
      assert "source" in f["metadata"]
      config_str = str(f["metadata/config"][()])
      seed_val = int(f["metadata/seed"][()])
      source_val = str(f["metadata/source"][()])
    assert seed_val == 42
    assert "sim" in source_val

  def test_config_is_valid_json(self, tmp_path):
    writer = EpisodeWriter(str(tmp_path))
    episode = self._make_fake_episode(T=3)
    cfg = _make_simple_config()
    path = writer.write_episode(episode, metadata={"config": cfg, "seed": 0, "source": "sim"})
    with h5py.File(path, "r") as f:
      raw = f["metadata/config"][()]
      if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
      parsed = json.loads(raw)
    assert "target" in parsed

  def test_episode_counter_increments(self, tmp_path):
    writer = EpisodeWriter(str(tmp_path))
    ep = self._make_fake_episode(T=2)
    p1 = writer.write_episode(ep)
    p2 = writer.write_episode(ep)
    assert p1 != p2
    assert p1.name == "episode_00000.hdf5"
    assert p2.name == "episode_00001.hdf5"

  def test_unsupported_format_raises(self, tmp_path):
    with pytest.raises(ValueError):
      EpisodeWriter(str(tmp_path), format="csv")
