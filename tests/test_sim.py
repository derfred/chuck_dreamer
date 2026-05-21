"""Tests for the MuJoCo pushing simulation."""

from __future__ import annotations

import json

import h5py
import mujoco
import numpy as np
import pytest
from omegaconf import OmegaConf

from chuck_dreamer.sim import (
    CameraConfig,
    EpisodeCollector,
    EpisodeWriter,
    LightingConfig,
    ObjectConfig,
    PushingEnv,
    ScriptedPolicy,
    SceneBuilder,
    SceneConfig,
    SceneGenerator,
)
from chuck_dreamer.sim.step_info import StepInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_TABLE_SIZE = [0.60, 0.5, 0.02]


def _make_simple_config(robot_type: str = "so100") -> SceneConfig:
  """Return a minimal valid SceneConfig. Uses so100 arm by default since
  PushingEnv/Controller require an arm with named actuators and ee_site."""
  return SceneConfig(
      table_size=[0.30, 0.25, 0.02],
      table_friction=0.5,
      table_color=[0.6, 0.5, 0.4, 1.0],
      robot_type=robot_type,
      robot_base_pos=[-0.30, 0.0, 0.04],
      robot_initial_qpos=None,
      target=ObjectConfig(
          shape="box",
          size=[0.03, 0.03, 0.03],
          mass=0.1,
          friction=0.5,
          pos=[0.05, 0.0, 0.07],
          orientation=0.0,
          color=[1.0, 0.0, 0.0, 1.0],
      ),
      goal_xy=[0.10, 0.05],
      goal_tolerance=0.04,
      obstacles=[],
      clutter=[],
      camera=CameraConfig(pos=[0.0, -0.40, 0.55], look_at=[0.0, 0.0, 0.04], fov=60.0),
      lighting=LightingConfig(
          direction=[0.0, -0.5, -1.0], intensity=0.8, ambient=0.3),
      max_steps=50,
      control_dt=0.1,
  )


def _make_env_cfg(
    difficulty: str = "easy",
    render_size: str = "64x64",
    seed: int = 42,
    obs_mode: str = "joints",
    act_mode: str = "joint",
):
  """Build a DictConfig that PushingEnv / SceneGenerator accept."""
  return OmegaConf.create({
      "seed": seed,
      "env": {
          "difficulty":  difficulty,
          "render_size": render_size,
          "table_size":  list(_DEFAULT_TABLE_SIZE),
          "max_steps":   50,
          "obs_mode":    obs_mode,
          "act_mode":    act_mode,
      },
  })


@pytest.fixture
def builder():
  return SceneBuilder()


@pytest.fixture
def env():
  e = PushingEnv(_make_env_cfg())
  yield e
  e.close()


@pytest.fixture
def env_ee():
  e = PushingEnv(_make_env_cfg(act_mode="ee"))
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
    assert cfg.goal_xy == [0.10, 0.05]
    assert cfg.max_steps == 50
    assert cfg.obstacles == []

  def test_object_config_fields(self):
    obj = ObjectConfig(
        shape="cylinder",
        size=[0.03, 0.05],
        mass=0.15,
        friction=0.6,
        pos=[0.1, 0.0, 0.09],
        orientation=1.0,
        color=[0.0, 1.0, 0.0, 1.0],
    )
    assert obj.shape == "cylinder"
    assert obj.mesh_path is None

  def test_camera_config(self):
    cam = CameraConfig(pos=[0.0, -0.4, 0.5],
                       look_at=[0.0, 0.0, 0.04], fov=60.0)
    assert cam.fov == 60.0

  def test_lighting_config(self):
    light = LightingConfig(
        direction=[
            0.0, -0.5, -1.0], intensity=0.8, ambient=0.3)
    assert 0.0 <= light.intensity <= 1.0


# ---------------------------------------------------------------------------
# SceneGenerator tests
# ---------------------------------------------------------------------------

class TestSceneGenerator:
  def test_sample_easy(self):
    gen = SceneGenerator(_make_env_cfg(difficulty="easy"))
    cfg = gen.sample()
    assert isinstance(cfg, SceneConfig)
    assert cfg.target.shape in ["box", "cylinder"]
    assert len(cfg.obstacles) == 0

  def test_sample_medium(self):
    gen = SceneGenerator(_make_env_cfg(difficulty="medium"))
    cfg = gen.sample()
    assert isinstance(cfg, SceneConfig)
    assert cfg.target.shape in ["box", "cylinder", "capsule", "mesh"]

  def test_sample_hard(self):
    gen = SceneGenerator(_make_env_cfg(difficulty="hard"))
    cfg = gen.sample()
    assert isinstance(cfg, SceneConfig)

  def test_invalid_difficulty(self):
    with pytest.raises(ValueError):
      SceneGenerator(_make_env_cfg(difficulty="extreme"))

  def test_validity_checks_reject_bad_target(self):
    gen = SceneGenerator(_make_env_cfg(difficulty="easy"))
    cfg = gen.sample()
    cfg.target.pos = [0.50, 0.50, cfg.target.pos[2]]
    assert not gen._check_reachability(cfg)

  def test_validity_goal_on_table(self):
    gen = SceneGenerator(_make_env_cfg(difficulty="easy"))
    cfg = gen.sample()
    cfg.goal_xy = [5.0, 5.0]
    assert not gen._check_goal_on_table(cfg)

  def test_multiple_samples_are_valid(self):
    gen = SceneGenerator(_make_env_cfg(difficulty="easy", seed=0))
    for _ in range(10):
      cfg = gen.sample()
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
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    assert body_id >= 0

  def test_obstacles_in_model(self, builder):
    cfg = _make_simple_config()
    cfg.obstacles = [
        ObjectConfig(
            shape="box",
            size=[0.03, 0.03, 0.03],
            mass=0.2,
            friction=0.5,
            pos=[0.08, 0.08, 0.07],
            orientation=0.0,
            color=[0.0, 0.0, 1.0, 1.0],
        )
    ]
    model = builder.build(cfg)
    obs_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_0")
    assert obs_id >= 0

# ---------------------------------------------------------------------------
# PushingEnv tests — joint action mode (default for these fixtures)
# ---------------------------------------------------------------------------


def _zero_joint_action(n_joints: int) -> np.ndarray:
  return np.zeros(n_joints, dtype=np.float32)


def _ee_pose_action(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
  return np.concatenate([
    np.asarray(pos, dtype=np.float32),
    np.asarray(quat, dtype=np.float32),
  ])


class TestPushingEnv:
  def test_reset_returns_correct_shapes(self, env):
    cfg = _make_simple_config()
    obs, info = env.reset(scene=cfg)
    assert obs["image"].image.shape == (64, 64, 3)
    assert obs["image"].image.dtype == np.uint8
    assert obs["image"].target_mask.shape == (64, 64)
    assert obs["image"].target_mask.dtype == bool
    assert obs["image"].goal_mask.shape == (64, 64)
    assert obs["ee_pos"].shape == (3,)
    assert obs["object_xy"].shape == (2,)

  def test_step_returns_correct_shapes(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    action = _zero_joint_action(len(cfg.joint_names))
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs["image"].image.shape == (64, 64, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    step_info = info["step_info"]
    assert step_info.object_xy.shape == (2,)
    assert step_info.ee_pos.shape == (3,)

  def test_action_clipping(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    # Oversized joint action should be clipped without error.
    action = np.full(len(cfg.joint_names), 100.0, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs["image"].image.shape == (64, 64, 3)

  def test_episode_truncates_on_timeout(self, env):
    cfg = _make_simple_config()
    cfg.max_steps = 3
    env.reset(scene=cfg)
    action = _zero_joint_action(len(cfg.joint_names))
    done = False
    for _ in range(5):
      _, _, term, trunc, _ = env.step(action)
      if term or trunc:
        done = True
        break
    assert done

  def test_reset_requires_config(self, env):
    with pytest.raises((AssertionError, ValueError, TypeError)):
      env.reset()

  def test_observation_and_action_spaces_joint(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.action_space.shape == (len(cfg.joint_names),)

  def test_action_space_ee_mode(self, env_ee):
    cfg = _make_simple_config()
    env_ee.reset(scene=cfg)
    assert env_ee.action_space.shape == (7,)

  def test_observation_space_joints_mode(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    # joints = arm_qpos(n_joints) under the new vocabulary.
    space = env.observation_space
    assert set(space.spaces.keys()) == {"joints"}
    assert space["joints"].shape == (len(cfg.joint_names),)

  def test_render_returns_image(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    img = env.render()
    assert img is not None
    assert img.image.shape == (64, 64, 3)
    assert img.image.dtype == np.uint8
    assert img.target_mask.shape == (64, 64)

  def test_render_before_reset_returns_none(self, env):
    assert env.render() is None

  def test_close_is_idempotent(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    env.close()
    env.close()

  def test_reset_twice_releases_previous_renderer(self, env):
    """Second reset() must close the previous renderer."""
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    first_renderer = env.renderer
    env.reset(scene=cfg)
    assert env.renderer is not None
    assert env.renderer is not first_renderer

  def test_step_info_contents(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    action = _zero_joint_action(len(cfg.joint_names))
    _, _, _, _, info = env.step(action)
    si = info["step_info"]
    assert si.object_xy.shape == (2,)
    assert si.ee_pos.shape == (3,)
    assert si.ee_quat.shape == (4,)
    assert si.goal_xy.shape == (2,)
    assert si.step == 1

  def test_step_increments_step_count(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    action = _zero_joint_action(len(cfg.joint_names))
    for expected in (1, 2, 3):
      _, _, _, _, info = env.step(action)
      assert info["step_info"].step == expected

  def test_reset_clears_step_count(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    action = _zero_joint_action(len(cfg.joint_names))
    env.step(action)
    env.step(action)
    env.reset(scene=cfg)
    _, _, _, _, info = env.step(action)
    assert info["step_info"].step == 1

  def test_reward_is_negative_distance_to_goal(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    action = _zero_joint_action(len(cfg.joint_names))
    _, reward, _, _, info = env.step(action)
    si = info["step_info"]
    expected = -float(np.linalg.norm(si.object_xy - si.goal_xy))
    assert reward == pytest.approx(expected, abs=1e-5)

  def test_goal_reached_terminates(self, env):
    cfg = _make_simple_config()
    cfg.goal_xy = [cfg.target.pos[0], cfg.target.pos[1]]
    cfg.goal_tolerance = 0.5
    env.reset(scene=cfg)
    action = _zero_joint_action(len(cfg.joint_names))
    _, _, terminated, _, _ = env.step(action)
    assert terminated

  def test_step_without_reset_asserts(self, env):
    with pytest.raises(AssertionError):
      env.step(_zero_joint_action(6))

  def test_observation_keys(self, env):
    cfg = _make_simple_config()
    obs, _ = env.reset(scene=cfg)
    expected = {"image", "ee_pos", "ee_quat", "arm_qpos", "object_xy", "goal_xy"}
    assert expected.issubset(obs.keys())
    # step/time live in StepInfo, not obs.
    assert "qpos" not in obs
    assert "step" not in obs
    assert "time" not in obs

  def test_initial_qpos_is_applied(self, env):
    cfg = _make_simple_config()
    cfg.robot_initial_qpos = [0.1, -0.2, 0.3, -0.1, 0.2, -0.3]
    obs, _ = env.reset(scene=cfg)
    assert obs["arm_qpos"] == pytest.approx(cfg.robot_initial_qpos, abs=1e-3)


class TestController:
  """Tests for PushingEnv.Controller — IK with the new ik_for_pose API."""

  def test_ik_position_only_at_current_returns_current_qpos(self, env):
    """Position-only IK at the current EE position returns ≈ the current qpos."""
    cfg = _make_simple_config()
    obs, _ = env.reset(scene=cfg)
    current_ee = obs["ee_pos"].astype(np.float64)
    q = env.controller.ik_for_pose(env.data, current_ee, target_quat=None)
    assert np.all(np.isfinite(q))
    assert q.shape == obs["arm_qpos"].shape
    np.testing.assert_allclose(q, obs["arm_qpos"], atol=1e-3)

  def test_ik_position_only_moves_toward_target(self, env):
    """Position IK for a nearby reachable target should bring the EE closer."""
    cfg = _make_simple_config()
    obs, _ = env.reset(scene=cfg)
    current_ee = obs["ee_pos"].astype(np.float64)
    target = current_ee + np.array([0.02, 0.0, 0.0])
    q_new = env.controller.ik_for_pose(env.data, target, target_quat=None)

    env.controller.reset_initial_qpos(env.data, q_new)
    new_ee = env.controller.get_ee_pos(env.data).astype(np.float64)
    assert np.linalg.norm(new_ee - target) < np.linalg.norm(current_ee - target)

  def test_ik_pose_with_current_quat_converges(self, env):
    """6-DOF IK at the current pose (pos + quat) converges to current qpos."""
    cfg = _make_simple_config()
    obs, _ = env.reset(scene=cfg)
    current_ee   = obs["ee_pos"].astype(np.float64)
    current_quat = obs["ee_quat"].astype(np.float64)
    q = env.controller.ik_for_pose(env.data, current_ee, target_quat=current_quat)
    assert np.all(np.isfinite(q))
    np.testing.assert_allclose(q, obs["arm_qpos"], atol=1e-3)

  def test_ik_raises_when_unreachable(self, env):
    """A wildly-unreachable target should raise RuntimeError."""
    cfg = _make_simple_config()
    obs, _ = env.reset(scene=cfg)
    unreachable = np.array([100.0, 100.0, 100.0])
    with pytest.raises(RuntimeError):
      env.controller.ik_for_pose(env.data, unreachable, target_quat=None)

  def test_ee_step_runs_through_ik(self, env_ee):
    """End-to-end: an EE-mode step uses IK and lands close to the target."""
    cfg = _make_simple_config()
    obs, _ = env_ee.reset(scene=cfg)
    target_pos = obs["ee_pos"] + np.array([0.01, 0.0, 0.0], dtype=np.float32)
    action = _ee_pose_action(target_pos, obs["ee_quat"])
    next_obs, _, _, _, _ = env_ee.step(action)
    # The actuator may not fully track the IK solution in one step; we
    # only require the EE moved toward the target.
    assert np.linalg.norm(next_obs["ee_pos"] - target_pos) <= np.linalg.norm(obs["ee_pos"] - target_pos)


class _FakeViewerWithGeoms:
  """Minimal viewer stand-in for insert_hints."""

  def __init__(self, maxgeom: int):
    self.user_scn = type("UserScn", (), {})()
    self.user_scn.ngeom = 0
    self.user_scn.maxgeom = maxgeom
    self.user_scn.geoms = [mujoco.MjvGeom() for _ in range(maxgeom)]


class TestPolicyInsertHints:
  """insert_hints lives on the policy, not the env."""

  def _make_policy(self):
    cfg = _make_simple_config()
    cfg.target.pos = [0.05, 0.00, 0.03]
    cfg.goal_xy = [0.20, 0.00]
    p = ScriptedPolicy()
    p.reset(cfg)
    p.state = "ready"
    return p

  def test_insert_hints_adds_two_geoms(self):
    policy = self._make_policy()
    viewer = _FakeViewerWithGeoms(maxgeom=10)
    policy.insert_hints(viewer)
    assert viewer.user_scn.ngeom == 2

  def test_insert_hints_noop_when_geom_buffer_full(self):
    policy = self._make_policy()
    viewer = _FakeViewerWithGeoms(maxgeom=2)
    viewer.user_scn.ngeom = 2
    policy.insert_hints(viewer)
    assert viewer.user_scn.ngeom == 2

  def test_insert_hints_noop_when_not_ready(self):
    policy = self._make_policy()
    policy.state = "approach"
    viewer = _FakeViewerWithGeoms(maxgeom=10)
    policy.insert_hints(viewer)
    assert viewer.user_scn.ngeom == 0


# ---------------------------------------------------------------------------
# EpisodeWriter tests
# ---------------------------------------------------------------------------

class TestEpisodeWriter:
  def _make_fake_episode(
      self,
      T: int = 5,
      H: int = 64,
      W: int = 64,
      n_joints: int = 6,
      action_kind: str = "joint_action",
  ):
    rng = np.random.default_rng(42)
    if action_kind == "joint_action":
      action = rng.uniform(-0.02, 0.02, (T, n_joints)).astype(np.float32)
    else:
      action = rng.uniform(-0.5, 0.5, (T, 7)).astype(np.float32)
    return {
        "image":      rng.integers(0, 256, (T, H, W, 3), dtype=np.uint8),
        action_kind:  action,
        "reward":     rng.uniform(-1.0, 0.0, (T,)).astype(np.float32),
        "timestamp":  (np.arange(T) * 0.1).astype(np.float32),
        "joint_qpos": rng.uniform(-1.0, 1.0, (T, n_joints)).astype(np.float32),
        "ee_pos":     rng.uniform(-0.5, 0.5, (T, 3)).astype(np.float32),
        "ee_quat":    np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 1)),
        "object_xy":  rng.uniform(-0.3, 0.3, (T, 2)).astype(np.float32),
    }

  def test_write_and_read_back_joint_mode(self, tmp_path):
    writer = EpisodeWriter(str(tmp_path), format="hdf5")
    episode = self._make_fake_episode(T=5, H=64, W=64, n_joints=6, action_kind="joint_action")
    cfg = _make_simple_config()
    path = writer.write_episode(
        episode,
        metadata={
            "config":  cfg,
            "seed":    42,
            "source":  "sim",
            "outcome": "done",
            "goal_xy": cfg.goal_xy,
        },
        name_suffix="00000")

    assert path.exists()
    with h5py.File(path, "r") as f:
      assert f["images"].shape         == (5, 64, 64, 3)
      assert f["images"].dtype         == np.uint8
      assert f["joint_action"].shape   == (5, 6)
      assert "ee_action" not in f
      assert f["rewards"].shape        == (5,)
      assert f["timestamps"].shape     == (5,)
      assert f["joint_qpos"].shape     == (5, 6)
      assert f["ee_pos"].shape         == (5, 3)
      assert f["ee_quat"].shape        == (5, 4)
      assert f["object_xy"].shape      == (5, 2)
      meta = f["metadata"]
      assert "act_mode" not in meta
      assert int(meta["seed"][()])     == 42
      assert "sim" in str(meta["source"][()])
      assert "done" in str(meta["outcome"][()])
      assert meta["goal_xy"].shape == (2,)

  def test_write_and_read_back_ee_mode(self, tmp_path):
    writer = EpisodeWriter(str(tmp_path), format="hdf5")
    episode = self._make_fake_episode(T=5, action_kind="ee_action")
    path = writer.write_episode(episode, metadata={"seed": 0, "source": "sim"}, name_suffix="00000")
    with h5py.File(path, "r") as f:
      assert f["ee_action"].shape == (5, 7)
      assert "joint_action" not in f
      assert "act_mode" not in f["metadata"]

  def test_config_is_valid_json(self, tmp_path):
    writer = EpisodeWriter(str(tmp_path))
    episode = self._make_fake_episode(T=3)
    cfg = _make_simple_config()
    path = writer.write_episode(
        episode,
        metadata={
            "config": cfg,
            "seed": 0,
            "source": "sim"},
        name_suffix="00000")
    with h5py.File(path, "r") as f:
      raw = f["metadata/config"][()]
      if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
      parsed = json.loads(raw)
    assert "target" in parsed

  def test_caller_supplied_suffix_drives_filename(self, tmp_path):
    writer = EpisodeWriter(str(tmp_path))
    ep = self._make_fake_episode(T=2)
    p1 = writer.write_episode(ep, name_suffix="00000")
    p2 = writer.write_episode(ep, name_suffix="step012-003")
    assert p1.name == "episode-00000.hdf5"
    assert p2.name == "episode-step012-003.hdf5"

  def test_unsupported_format_raises(self, tmp_path):
    with pytest.raises(ValueError):
      EpisodeWriter(str(tmp_path), format="csv")

  def test_rerun_writes_rrd_file(self, tmp_path):
    pytest.importorskip("rerun")
    writer = EpisodeWriter(str(tmp_path), format="rerun")
    episode = self._make_fake_episode(T=3, H=32, W=32, n_joints=4)
    cfg = _make_simple_config()
    path = writer.write_episode(
        episode,
        metadata={
            "config":  cfg,
            "seed":    7,
            "source":  "sim",
            "outcome": "done",
            "goal_xy": cfg.goal_xy,
        },
        name_suffix="00000")
    assert path.exists()
    assert path.suffix == ".rrd"
    assert path.stat().st_size > 0

  def test_rerun_caller_supplied_suffix_drives_filename(self, tmp_path):
    pytest.importorskip("rerun")
    writer = EpisodeWriter(str(tmp_path), format="rerun")
    ep = self._make_fake_episode(T=2)
    p1 = writer.write_episode(ep, name_suffix="00000")
    p2 = writer.write_episode(ep, name_suffix="step012-003")
    assert p1.name == "episode-00000.rrd"
    assert p2.name == "episode-step012-003.rrd"


# ---------------------------------------------------------------------------
# ScriptedPolicy tests
# ---------------------------------------------------------------------------

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _policy_obs(ee_pos, qpos=None, n_joints: int = 6, ee_quat: np.ndarray = _IDENTITY_QUAT):
  qpos_arr = (np.zeros(n_joints, dtype=np.float32)
              if qpos is None else np.asarray(qpos, dtype=np.float32))
  return {
      "ee_pos":   np.asarray(ee_pos, dtype=np.float32),
      "ee_quat":  np.asarray(ee_quat, dtype=np.float32),
      "arm_qpos": qpos_arr,
  }


def _fresh_policy():
  cfg = _make_simple_config()
  cfg.target.pos = [0.05, 0.00, 0.03]
  cfg.goal_xy = [0.20, 0.00]
  p = ScriptedPolicy()
  p.reset(cfg)
  return p


class TestScriptedPolicy:
  """Exercises the state-machine logic. Policy outputs (7,) EE poses."""

  def test_initial_state(self):
    p = _fresh_policy()
    assert p.state == "initial"

  def test_approach_xy_offsets_behind_object(self):
    p = _fresh_policy()
    assert p.scene is not None
    obj = np.array(p.scene.target.pos[:2])
    assert p.approach_xy[0] < obj[0]
    d = float(np.linalg.norm(p.approach_xy - obj))
    assert d == pytest.approx(ScriptedPolicy._STANDOFF, rel=1e-3)

  def test_approach_xyz_uses_push_z(self):
    p = _fresh_policy()
    assert p.approach_xyz[2] == pytest.approx(ScriptedPolicy._PUSH_Z)

  def test_goal_xyz_uses_push_z(self):
    p = _fresh_policy()
    assert p.goal_xyz[2] == pytest.approx(ScriptedPolicy._PUSH_Z)

  def test_transition_initial_to_ready(self):
    p = _fresh_policy()
    obs = _policy_obs(np.append(p.ready_xy, ScriptedPolicy._PUSH_Z))
    p.act(obs)
    assert p.state == "ready"

  def test_ready_is_sticky_without_external_signal(self):
    """ready → approach is a manual transition; policy itself does not self-advance."""
    p = _fresh_policy()
    p.state = "ready"
    obs = _policy_obs(np.append(p.ready_xy, ScriptedPolicy._PUSH_Z))
    p.act(obs)
    assert p.state == "ready"

  def test_transition_approach_to_push(self):
    p = _fresh_policy()
    p.state = "approach"
    obs = _policy_obs(p.approach_xyz)
    p.act(obs)
    assert p.state == "push"

  def test_transition_push_to_done(self):
    p = _fresh_policy()
    p.state = "push"
    obs = _policy_obs(p.goal_xyz)
    p.act(obs)
    assert p.state == "done"

  def test_done_is_terminal(self):
    p = _fresh_policy()
    p.state = "done"
    obs = _policy_obs(p.goal_xyz)
    p.act(obs)
    assert p.state == "done"

  def test_act_returns_pose_action(self):
    p = _fresh_policy()
    obs = _policy_obs(np.array([0.0, 0.0, 0.1]))
    action = p.act(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == (7,)
    # Quat half should match the captured hold_quat (set on first act).
    np.testing.assert_array_equal(action[3:], _IDENTITY_QUAT)


# ---------------------------------------------------------------------------
# EpisodeCollector tests (fake env + fake policy — no MuJoCo)
# ---------------------------------------------------------------------------


def _fake_obs(step: int, n_joints: int = 6):
  from chuck_dreamer.sim.observation_image import ObservationImage
  H = W = 4
  image_obs = ObservationImage(
    image=np.full((H, W, 3), step % 256, dtype=np.uint8),
    target_mask=np.zeros((H, W), dtype=bool),
    goal_mask=np.zeros((H, W), dtype=bool),
    clutter_masks=[],
    background_mask=np.ones((H, W), dtype=bool),
  )
  return {
      "image":    image_obs,
      "ee_pos":   np.array([0.0, 0.0, 0.1], dtype=np.float32),
      "ee_quat":  np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
      "arm_qpos": np.zeros(n_joints, dtype=np.float32),
      "object_xy": np.array([0.05, 0.0], dtype=np.float32),
      "goal_xy":  np.array([0.2, 0.0], dtype=np.float32),
  }


def _fake_step_info(step: int):
  return StepInfo(
    object_xy=np.array([0.05, 0.0], dtype=np.float32),
    ee_pos=np.array([0.0, 0.0, 0.1], dtype=np.float32),
    ee_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    goal_xy=np.array([0.2, 0.0], dtype=np.float32),
    step=step,
    time=float(step) * 0.1,
  )


class _FakeEnv:
  """Minimal PushingEnv stand-in. The act_mode controls action_kind in the recorded episode."""

  def __init__(
      self,
      terminate_at: int | None = None,
      truncate_at: int | None = None,
      crash_at: int | None = None,
      max_steps: int = 20,
      n_joints: int = 6,
      act_mode: str = "ee",
  ):
    self.terminate_at = terminate_at
    self.truncate_at  = truncate_at
    self.crash_at     = crash_at
    self.max_steps    = max_steps
    self.n_joints     = n_joints
    self.act_mode     = act_mode
    self._step        = 0

  def generate_scene(self):
    cfg = _make_simple_config()
    cfg.max_steps = self.max_steps
    return cfg

  def policy_obs(self, full_obs):
    return full_obs  # tests don't care about projection

  def reset(self, *, scene=None, seed=None):
    self._step = 0
    return _fake_obs(0, self.n_joints), {}

  def step(self, action):
    self._step += 1
    if self.crash_at is not None and self._step >= self.crash_at:
      raise RuntimeError("simulated IK failure")
    terminated = self.terminate_at is not None and self._step >= self.terminate_at
    truncated  = self.truncate_at  is not None and self._step >= self.truncate_at
    info = {"step_info": _fake_step_info(self._step)}
    return _fake_obs(self._step, self.n_joints), -0.1, terminated, truncated, info


class _FakePolicy:
  """Stub policy: returns zeros of the right shape and never reads obs."""

  def __init__(self, action_dim: int = 7):
    self.action_dim = action_dim
    self.reset_calls = 0
    self.act_calls = 0

  def reset(self, scene):
    self.reset_calls += 1

  def act(self, obs):
    self.act_calls += 1
    return np.zeros(self.action_dim, dtype=np.float32)


def _make_collector(env, policy):
  collector = EpisodeCollector(env, policy)
  collector.reset()
  return collector


class TestEpisodeCollector:
  def test_outcome_done_when_env_terminates(self):
    env    = _FakeEnv(terminate_at=3, max_steps=20, act_mode="ee")
    policy = _FakePolicy(action_dim=7)
    collector = _make_collector(env, policy)

    episode, outcome = collector.run()
    assert outcome == "done"
    assert episode is not None
    assert episode["ee_action"].shape == (3, 7)

  def test_outcome_timeout_when_env_truncates(self):
    env    = _FakeEnv(truncate_at=2, max_steps=20, act_mode="ee")
    policy = _FakePolicy(action_dim=7)
    collector = _make_collector(env, policy)

    episode, outcome = collector.run()
    assert outcome == "timeout"
    assert episode is not None
    assert episode["ee_action"].shape[0] == 2

  def test_outcome_timeout_when_max_steps_exhausted(self):
    env    = _FakeEnv(max_steps=4, act_mode="ee")
    policy = _FakePolicy(action_dim=7)
    collector = _make_collector(env, policy)

    episode, outcome = collector.run()
    assert outcome == "timeout"
    assert episode is not None
    assert episode["ee_action"].shape[0] == 4

  def test_outcome_crashed_returns_partial_data(self):
    env    = _FakeEnv(crash_at=3, max_steps=20, act_mode="ee")
    policy = _FakePolicy(action_dim=7)
    collector = _make_collector(env, policy)

    episode, outcome = collector.run()
    assert outcome == "crashed"
    # 2 successful steps before the crash on step 3.
    assert episode is not None
    assert episode["ee_action"].shape[0] == 2

  def test_records_joint_action_when_act_mode_joint(self):
    env    = _FakeEnv(terminate_at=2, max_steps=20, act_mode="joint", n_joints=6)
    policy = _FakePolicy(action_dim=6)
    collector = _make_collector(env, policy)

    episode, outcome = collector.run()
    assert outcome == "done"
    assert episode is not None
    assert "joint_action" in episode
    assert "ee_action" not in episode
    assert episode["joint_action"].shape == (2, 6)

  def test_episode_has_all_required_fields(self):
    env    = _FakeEnv(terminate_at=3, max_steps=20, act_mode="ee")
    policy = _FakePolicy(action_dim=7)
    collector = _make_collector(env, policy)

    episode, outcome = collector.run()
    assert episode is not None
    expected_keys = {
        "image", "ee_action", "reward",
        "timestamp", "joint_qpos", "ee_pos", "ee_quat", "object_xy",
        "step_info",
    }
    assert expected_keys.issubset(episode.keys())
    si = episode["step_info"]
    T = episode["ee_action"].shape[0]
    assert si["object_xy"].shape == (T, 2)
    assert si["ee_pos"].shape == (T, 3)


# ---------------------------------------------------------------------------
# Target-mask centroid → object_uv
# ---------------------------------------------------------------------------


class TestObjectUVCentroid:
  def test_centroid_of_single_pixel_recovers_its_uv(self):
    from chuck_dreamer.sim.episode_collector import mask_centroids_uv
    masks = np.zeros((1, 8, 8), dtype=bool)
    masks[0, 5, 3] = True  # row 5, col 3 → u=3, v=5
    uv = mask_centroids_uv(masks)
    assert uv.shape == (1, 2)
    np.testing.assert_allclose(uv[0], [3.0, 5.0])

  def test_centroid_of_square_block_is_geometric_centre(self):
    from chuck_dreamer.sim.episode_collector import mask_centroids_uv
    masks = np.zeros((1, 10, 10), dtype=bool)
    masks[0, 2:6, 4:8] = True  # rows 2..5, cols 4..7 → mean v=3.5, u=5.5
    uv = mask_centroids_uv(masks)
    np.testing.assert_allclose(uv[0], [5.5, 3.5])

  def test_centroid_is_nan_when_mask_is_empty(self):
    from chuck_dreamer.sim.episode_collector import mask_centroids_uv
    masks = np.zeros((3, 4, 4), dtype=bool)
    masks[1, 1, 2] = True  # only the middle step has a target
    uv = mask_centroids_uv(masks)
    assert np.isnan(uv[0]).all()
    np.testing.assert_allclose(uv[1], [2.0, 1.0])
    assert np.isnan(uv[2]).all()

  def test_collector_writes_object_uv_alongside_object_xy(self):
    # Use a fake env whose target mask is non-empty so the centroid is finite.
    class _MaskedFakeEnv(_FakeEnv):
      def _obs_with_mask(self, step: int):
        obs = _fake_obs(step, self.n_joints)
        H, W = obs["image"].image.shape[:2]
        tm = np.zeros((H, W), dtype=bool)
        tm[H // 2, W // 2] = True  # centre pixel
        # Rebuild ObservationImage with a non-empty target mask.
        from chuck_dreamer.sim.observation_image import ObservationImage
        obs["image"] = ObservationImage(
          image=obs["image"].image,
          target_mask=tm,
          goal_mask=obs["image"].goal_mask,
          clutter_masks=obs["image"].clutter_masks,
          arm_mask=obs["image"].arm_mask,
          background_mask=obs["image"].background_mask,
        )
        return obs

      def reset(self, *, scene=None, seed=None):
        self._step = 0
        return self._obs_with_mask(0), {}

      def step(self, action):
        self._step += 1
        info = {"step_info": _fake_step_info(self._step)}
        terminated = self.terminate_at is not None and self._step >= self.terminate_at
        return self._obs_with_mask(self._step), -0.1, terminated, False, info

    env    = _MaskedFakeEnv(terminate_at=3, max_steps=20, act_mode="ee")
    policy = _FakePolicy(action_dim=7)
    collector = _make_collector(env, policy)

    episode, _outcome = collector.run()
    assert episode is not None
    assert "object_uv" in episode
    T = episode["ee_action"].shape[0]
    assert episode["object_uv"].shape == (T, 2)
    # Centre pixel of a 4x4 image is (u=2, v=2).
    np.testing.assert_allclose(episode["object_uv"], 2.0)

  def test_hdf5_round_trips_object_uv(self, tmp_path):
    from chuck_dreamer.training.episode_dataset import Episode

    writer = EpisodeWriter(str(tmp_path), format="hdf5")
    T = 4
    rng = np.random.default_rng(0)
    episode = {
      "image":        rng.integers(0, 256, (T, 8, 8, 3), dtype=np.uint8),
      "joint_action": rng.uniform(-1, 1, (T, 6)).astype(np.float32),
      "reward":       np.zeros((T,), dtype=np.float32),
      "timestamp":    np.arange(T, dtype=np.float32) * 0.1,
      "joint_qpos":   np.zeros((T, 6), dtype=np.float32),
      "ee_pos":       np.zeros((T, 3), dtype=np.float32),
      "ee_quat":      np.tile([1.0, 0.0, 0.0, 0.0], (T, 1)).astype(np.float32),
      "object_xy":    np.zeros((T, 2), dtype=np.float32),
      "object_uv":    np.arange(T * 2, dtype=np.float32).reshape(T, 2),
    }
    path = writer.write_episode(episode, metadata={"seed": 0}, name_suffix="00000")
    raw = Episode.from_file(path).data
    assert "object_uv" in raw
    np.testing.assert_allclose(raw["object_uv"], episode["object_uv"])

  def test_hdf5_round_trip_without_object_uv_still_works(self, tmp_path):
    # Legacy episodes (no object_uv) must still load.
    from chuck_dreamer.training.episode_dataset import Episode

    writer = EpisodeWriter(str(tmp_path), format="hdf5")
    T = 3
    episode = {
      "image":        np.zeros((T, 4, 4, 3), dtype=np.uint8),
      "joint_action": np.zeros((T, 6), dtype=np.float32),
      "reward":       np.zeros((T,), dtype=np.float32),
      "timestamp":    np.zeros((T,), dtype=np.float32),
      "joint_qpos":   np.zeros((T, 6), dtype=np.float32),
      "ee_pos":       np.zeros((T, 3), dtype=np.float32),
      "ee_quat":      np.tile([1.0, 0.0, 0.0, 0.0], (T, 1)).astype(np.float32),
      "object_xy":    np.zeros((T, 2), dtype=np.float32),
    }
    path = writer.write_episode(episode, metadata={"seed": 0}, name_suffix="00000")
    raw = Episode.from_file(path).data
    assert "object_uv" not in raw

  def test_rerun_round_trips_object_uv(self, tmp_path):
    pytest.importorskip("rerun")
    from chuck_dreamer.training.episode_dataset import Episode

    writer = EpisodeWriter(str(tmp_path), format="rerun")
    T = 4
    rng = np.random.default_rng(1)
    episode = {
      "image":        rng.integers(0, 256, (T, 8, 8, 3), dtype=np.uint8),
      "joint_action": rng.uniform(-1, 1, (T, 6)).astype(np.float32),
      "reward":       np.zeros((T,), dtype=np.float32),
      "timestamp":    np.arange(T, dtype=np.float32) * 0.1,
      "joint_qpos":   np.zeros((T, 6), dtype=np.float32),
      "ee_pos":       np.zeros((T, 3), dtype=np.float32),
      "ee_quat":      np.tile([1.0, 0.0, 0.0, 0.0], (T, 1)).astype(np.float32),
      "object_xy":    np.zeros((T, 2), dtype=np.float32),
      "object_uv":    np.arange(T * 2, dtype=np.float32).reshape(T, 2),
    }
    path = writer.write_episode(episode, metadata={"seed": 1}, name_suffix="00000")
    raw = Episode.from_file(path).data
    assert "object_uv" in raw
    np.testing.assert_allclose(raw["object_uv"], episode["object_uv"])
