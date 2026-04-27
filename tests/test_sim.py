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
    EpisodeWriter,
    LightingConfig,
    ObjectConfig,
    PushingEnv,
    RandomPushPolicy,
    SceneBuilder,
    SceneConfig,
    SceneGenerator,
    ScenePlayer,
)
from chuck_dreamer.policy import Action


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
      goal_pos=[0.10, 0.05],
      goal_tolerance=0.04,
      obstacles=[],
      clutter=[],
      camera=CameraConfig(pos=[0.0, -0.40, 0.55], look_at=[0.0, 0.0, 0.04], fov=60.0),
      lighting=LightingConfig(
          direction=[0.0, -0.5, -1.0], intensity=0.8, ambient=0.3),
      max_steps=50,
      control_dt=0.1,
  )


def _make_env_cfg(difficulty: str = "easy", render_size: str = "64x64", seed: int = 42):
  """Build a DictConfig that PushingEnv / SceneGenerator accept."""
  return OmegaConf.create({
      "seed": seed,
      "sim": {
          "difficulty": difficulty,
          "render_size": render_size,
          "table_size": list(_DEFAULT_TABLE_SIZE),
          "max_steps": 50,
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
    assert cfg.target.shape in ["box", "cylinder", "capsule"]

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
    # Place target out of reach (too far from robot base)
    cfg.target.pos = [0.50, 0.50, cfg.target.pos[2]]
    assert not gen._check_reachability(cfg)

  def test_validity_goal_on_table(self):
    gen = SceneGenerator(_make_env_cfg(difficulty="easy"))
    cfg = gen.sample()
    cfg.goal_pos = [5.0, 5.0]  # way off table
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

  def test_simple_arm_xml_exists(self):
    from chuck_dreamer.sim.scene_builder import _SIMPLE_ARM_XML
    assert _SIMPLE_ARM_XML.exists()
    content = _SIMPLE_ARM_XML.read_text()
    assert "ee_frame" in content
    assert "joint1" in content


# ---------------------------------------------------------------------------
# PushingEnv tests
# ---------------------------------------------------------------------------

def _zero_action(n_joints: int) -> Action:
  return Action.from_qpos(np.zeros(n_joints, dtype=np.float32))


class TestPushingEnv:
  def test_reset_returns_correct_shapes(self, env):
    cfg = _make_simple_config()
    obs, info = env.reset(scene=cfg)
    assert obs["image"].shape == (64, 64, 3)
    assert obs["image"].dtype == np.uint8
    assert obs["ee_pos"].shape == (3,)
    assert obs["object_xy"].shape == (2,)

  def test_step_returns_correct_shapes(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    action = _zero_action(len(cfg.joint_names))
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs["image"].shape == (64, 64, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "object_xy" in info
    assert "ee_pos" in info

  def test_action_clipping(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    # Oversized action in joint-space should be clipped without error
    action = Action.from_qpos(np.full(len(cfg.joint_names), 100.0, dtype=np.float32))
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs["image"].shape == (64, 64, 3)

  def test_episode_terminates_on_timeout(self, env):
    cfg = _make_simple_config()
    cfg.max_steps = 3
    env.reset(scene=cfg)
    action = _zero_action(len(cfg.joint_names))
    done = False
    for _ in range(5):
      _, _, term, _, _ = env.step(action)
      if term:
        done = True
        break
    assert done

  def test_reset_requires_config(self, env):
    # reset() without scene hits an assert — any of these is fine.
    with pytest.raises((AssertionError, ValueError, TypeError)):
      env.reset()

  def test_observation_and_action_spaces(self, env):
    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.action_space.shape == (3,)

  def test_render_returns_image(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    img = env.render()
    assert img is not None
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8

  def test_render_before_reset_returns_none(self, env):
    # Renderer is only constructed in reset(); before that, render() no-ops.
    assert env.render() is None

  def test_close_is_idempotent(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    env.close()
    # Second close must not raise even though renderer is already gone.
    env.close()

  def test_reset_twice_releases_previous_renderer(self, env):
    """Second reset() must close the previous renderer."""
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    first_renderer = env.renderer
    env.reset(scene=cfg)
    # A new renderer is created on the second reset.
    assert env.renderer is not None
    assert env.renderer is not first_renderer

  def test_step_info_contents(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    action = _zero_action(len(cfg.joint_names))
    _, _, _, _, info = env.step(action)
    assert info["object_xy"].shape == (2,)
    assert info["ee_pos"].shape == (3,)
    assert info["goal_xy"].shape == (2,)
    assert info["step"] == 1

  def test_step_increments_step_count(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    action = _zero_action(len(cfg.joint_names))
    for expected in (1, 2, 3):
      _, _, _, _, info = env.step(action)
      assert info["step"] == expected

  def test_reset_clears_step_count(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    action = _zero_action(len(cfg.joint_names))
    env.step(action)
    env.step(action)
    env.reset(scene=cfg)
    _, _, _, _, info = env.step(action)
    assert info["step"] == 1

  def test_reward_is_negative_distance_to_goal(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    action = _zero_action(len(cfg.joint_names))
    _, reward, _, _, info = env.step(action)
    # Reward = -||object_xy - goal_xy||.
    expected = -float(np.linalg.norm(info["object_xy"] - info["goal_xy"]))
    assert reward == pytest.approx(expected, abs=1e-5)

  def test_goal_reached_terminates(self, env):
    cfg = _make_simple_config()
    # Put the goal right on top of the target so the env is "at goal" immediately.
    cfg.goal_pos = [cfg.target.pos[0], cfg.target.pos[1]]
    cfg.goal_tolerance = 0.5
    env.reset(scene=cfg)
    action = _zero_action(len(cfg.joint_names))
    _, _, terminated, _, _ = env.step(action)
    assert terminated

  def test_step_without_reset_asserts(self, env):
    with pytest.raises(AssertionError):
      env.step(_zero_action(6))

  def test_observation_keys(self, env):
    cfg = _make_simple_config()
    obs, _ = env.reset(scene=cfg)
    expected = {"image", "ee_pos", "ee_quat", "arm_qpos",
                "object_xy", "goal_xy", "qpos", "step", "time"}
    assert expected.issubset(obs.keys())

  def test_initial_qpos_is_applied(self, env):
    cfg = _make_simple_config()
    # Pick a non-zero home pose so we can detect it was applied.
    cfg.robot_initial_qpos = [0.1, -0.2, 0.3, -0.1, 0.2, -0.3]
    obs, _ = env.reset(scene=cfg)
    assert obs["arm_qpos"] == pytest.approx(cfg.robot_initial_qpos, abs=1e-3)


class TestController:
  """Tests for PushingEnv.Controller — in particular the IK solver."""

  def test_ik_converges_to_current_ee_position(self, env):
    """IK for the current EE position should return a qpos ≈ the current qpos
    (zero residual)."""
    cfg = _make_simple_config()
    obs, _ = env.reset(scene=cfg)
    current_ee = obs["ee_pos"].astype(np.float64)
    q = env.controller.ik_for_ee_pos(current_ee, obs["qpos"])
    assert np.all(np.isfinite(q))
    assert q.shape == obs["arm_qpos"].shape

  def test_ik_moves_toward_target(self, env):
    """IK for a nearby reachable target should return a qpos that, when applied,
    brings the EE much closer to the target."""
    cfg = _make_simple_config()
    obs, _ = env.reset(scene=cfg)
    current_ee = obs["ee_pos"].astype(np.float64)
    target = current_ee + np.array([0.02, 0.0, 0.0])
    q_new = env.controller.ik_for_ee_pos(target, obs["qpos"])

    # Apply the solution and check EE is much closer to the target.
    env.controller.reset_initial_qpos(env.data, q_new)
    new_ee = env.controller.get_ee_pos(env.data).astype(np.float64)
    assert np.linalg.norm(new_ee - target) < np.linalg.norm(current_ee - target)

  def test_ik_raises_when_diverged(self, env):
    """A wildly-unreachable target should either raise RuntimeError or return
    something finite (the cap on dq makes true divergence rare)."""
    cfg = _make_simple_config()
    obs, _ = env.reset(scene=cfg)
    unreachable = np.array([100.0, 100.0, 100.0])
    with pytest.raises(RuntimeError):
      env.controller.ik_for_ee_pos(unreachable, obs["qpos"])


class _FakeViewerWithGeoms:
  """Minimal viewer stand-in for insert_hints — exposes a user_scn with a
  fixed-size geoms buffer."""

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
    cfg.goal_pos = [0.20, 0.00]
    p = RandomPushPolicy()
    p.reset(_StubController(), cfg)
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
    viewer.user_scn.ngeom = 2  # already full
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
  def _make_fake_episode(self, T: int = 5, H: int = 64, W: int = 64, n_joints: int = 6):
    rng = np.random.default_rng(42)
    return {
        "image":      rng.integers(0, 256, (T, H, W, 3), dtype=np.uint8),
        "action":     rng.uniform(-0.02, 0.02, (T, n_joints)).astype(np.float32),
        "reward":     rng.uniform(-1.0, 0.0, (T,)).astype(np.float32),
        "timestamp":  (np.arange(T) * 0.1).astype(np.float32),
        "joint_qpos": rng.uniform(-1.0, 1.0, (T, n_joints)).astype(np.float32),
        "ee_pos":     rng.uniform(-0.5, 0.5, (T, 3)).astype(np.float32),
        "ee_quat":    np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 1)),
        "object_xy":  rng.uniform(-0.3, 0.3, (T, 2)).astype(np.float32),
    }

  def test_write_and_read_back(self, tmp_path):
    writer = EpisodeWriter(str(tmp_path), format="hdf5")
    episode = self._make_fake_episode(T=5, H=64, W=64, n_joints=6)
    cfg = _make_simple_config()
    path = writer.write_episode(
        episode,
        metadata={
            "config":  cfg,
            "seed":    42,
            "source":  "sim",
            "outcome": "done",
            "goal_xy": cfg.goal_pos,
        })

    assert path.exists()
    with h5py.File(path, "r") as f:
      assert f["images"].shape     == (5, 64, 64, 3)
      assert f["images"].dtype     == np.uint8
      assert f["actions"].shape    == (5, 6)
      assert f["rewards"].shape    == (5,)
      assert f["timestamps"].shape == (5,)
      assert f["joint_qpos"].shape == (5, 6)
      assert f["ee_pos"].shape     == (5, 3)
      assert f["ee_quat"].shape    == (5, 4)
      assert f["object_xy"].shape  == (5, 2)
      meta = f["metadata"]
      assert int(meta["seed"][()])        == 42
      assert "sim" in str(meta["source"][()])
      assert "done" in str(meta["outcome"][()])
      assert meta["goal_xy"].shape == (2,)

  def test_config_is_valid_json(self, tmp_path):
    writer = EpisodeWriter(str(tmp_path))
    episode = self._make_fake_episode(T=3)
    cfg = _make_simple_config()
    path = writer.write_episode(
        episode,
        metadata={
            "config": cfg,
            "seed": 0,
            "source": "sim"})
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
            "goal_xy": cfg.goal_pos,
        })
    assert path.exists()
    assert path.suffix == ".rrd"
    assert path.stat().st_size > 0

  def test_rerun_counter_increments(self, tmp_path):
    pytest.importorskip("rerun")
    writer = EpisodeWriter(str(tmp_path), format="rerun")
    ep = self._make_fake_episode(T=2)
    p1 = writer.write_episode(ep)
    p2 = writer.write_episode(ep)
    assert p1.name == "episode_00000.rrd"
    assert p2.name == "episode_00001.rrd"


# ---------------------------------------------------------------------------
# RandomPushPolicy tests
# ---------------------------------------------------------------------------

class _StubController:
  """Minimal controller stand-in: IK returns the same qpos it was given."""

  def __init__(self, arm_qpos_adr=None):
    self.arm_qpos_adr = np.asarray(arm_qpos_adr if arm_qpos_adr is not None else [0, 1, 2, 3, 4, 5])

  def ik_for_ee_pos(self, target_xyz, qpos):
    # Return a (6,) arm qpos — Action.from_qpos asserts shape (6,).
    q = np.asarray(qpos, dtype=np.float32)
    if q.shape != (6,):
      q = q[self.arm_qpos_adr] if q.ndim == 1 and q.size >= 6 else np.zeros(6, dtype=np.float32)
    return q.copy()


def _policy_obs(ee_pos, qpos=None, n_joints: int = 6):
  qpos_arr = (np.zeros(n_joints, dtype=np.float32)
              if qpos is None else np.asarray(qpos, dtype=np.float32))
  return {
      "ee_pos":   np.asarray(ee_pos, dtype=np.float32),
      "arm_qpos": qpos_arr,
      "qpos":     qpos_arr,
  }


def _fresh_policy():
  cfg = _make_simple_config()
  # Put goal clearly separated from target to keep push_dir well-defined.
  cfg.target.pos = [0.05, 0.00, 0.03]
  cfg.goal_pos = [0.20, 0.00]
  p = RandomPushPolicy()
  p.reset(_StubController(), cfg)
  return p


class TestRandomPushPolicy:
  """Exercises the state-machine logic. Controller is stubbed so no MuJoCo is needed."""

  def test_initial_state(self):
    p = _fresh_policy()
    assert p.state == "initial"

  def test_approach_xy_offsets_behind_object(self):
    p = _fresh_policy()
    obj = np.array(p.scene.target.pos[:2])
    # Approach point lies behind the object (opposite side from the goal).
    # Since goal is +x of the object, approach is at -x of the object.
    assert p.approach_xy[0] < obj[0]
    # Distance equals the standoff.
    d = float(np.linalg.norm(p.approach_xy - obj))
    assert d == pytest.approx(RandomPushPolicy._STANDOFF, rel=1e-3)

  def test_approach_xyz_uses_push_z(self):
    p = _fresh_policy()
    assert p.approach_xyz[2] == pytest.approx(RandomPushPolicy._PUSH_Z)

  def test_goal_xyz_uses_push_z(self):
    p = _fresh_policy()
    assert p.goal_xyz[2] == pytest.approx(RandomPushPolicy._PUSH_Z)

  def test_transition_initial_to_ready(self):
    p = _fresh_policy()
    # EE at ready position → initial should advance to ready.
    obs = _policy_obs(np.append(p.ready_xy, RandomPushPolicy._PUSH_Z))
    _, prev = p.act(obs)
    assert prev == "initial"
    assert p.state == "ready"

  def test_ready_is_sticky_without_external_signal(self):
    """ready → approach is a manual transition; policy itself does not self-advance."""
    p = _fresh_policy()
    p.state = "ready"
    obs = _policy_obs(np.append(p.ready_xy, RandomPushPolicy._PUSH_Z))
    _, prev = p.act(obs)
    assert prev is None
    assert p.state == "ready"

  def test_transition_approach_to_push(self):
    p = _fresh_policy()
    p.state = "approach"
    obs = _policy_obs(p.approach_xyz)
    _, prev = p.act(obs)
    assert prev == "approach"
    assert p.state == "push"

  def test_transition_push_to_done(self):
    p = _fresh_policy()
    p.state = "push"
    obs = _policy_obs(p.goal_xyz)
    _, prev = p.act(obs)
    assert prev == "push"
    assert p.state == "done"

  def test_done_is_terminal(self):
    p = _fresh_policy()
    p.state = "done"
    obs = _policy_obs(p.goal_xyz)
    _, prev = p.act(obs)
    assert prev is None
    assert p.state == "done"

  def test_act_returns_action_and_prev_state_tuple(self):
    p = _fresh_policy()
    obs = _policy_obs(np.array([0.0, 0.0, 0.1]))
    action, prev = p.act(obs)
    assert isinstance(action, Action)
    # prev_state is None when no transition happened, else the prior state.
    assert prev is None or isinstance(prev, str)


# ---------------------------------------------------------------------------
# ScenePlayer tests (fake env + fake policy — no MuJoCo)
# ---------------------------------------------------------------------------

def _fake_obs(step: int, n_joints: int = 6):
  return {
      "image":    np.full((4, 4, 3), step % 256, dtype=np.uint8),
      "ee_pos":   np.array([0.0, 0.0, 0.1], dtype=np.float32),
      "ee_quat":  np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
      "arm_qpos": np.zeros(n_joints, dtype=np.float32),
      "object_xy": np.array([0.05, 0.0], dtype=np.float32),
      "goal_xy":  np.array([0.2, 0.0], dtype=np.float32),
      "qpos":     np.zeros(n_joints, dtype=np.float32),
      "time":     float(step) * 0.1,
      "step":     step,
  }


class _FakeEnv:
  """Minimal PushingEnv stand-in. Tracks step count, honors control over done flags."""

  def __init__(self, terminate_at: int | None = None, crash_at: int | None = None, n_joints: int = 6):
    self.terminate_at = terminate_at
    self.crash_at     = crash_at
    self.n_joints     = n_joints
    self._step        = 0
    self.reset_calls  = 0
    self.controller   = _StubController()

  def generate_scene(self):
    return _make_simple_config()

  def reset(self, *, scene=None, seed=None):
    self.reset_calls += 1
    self._step = 0
    return _fake_obs(0, self.n_joints), {}

  def step(self, action):
    self._step += 1
    if self.crash_at is not None and self._step >= self.crash_at:
      raise RuntimeError("simulated IK failure")
    terminated = self.terminate_at is not None and self._step >= self.terminate_at
    return _fake_obs(self._step, self.n_joints), -0.1, terminated, False, {}

  def _get_obs(self):
    return _fake_obs(self._step, self.n_joints)


class _FakePolicy:
  """Policy stub with scripted state progression."""

  def __init__(self, states: list[str], n_joints: int = 6):
    self._states = list(states)
    self.state = self._states[0] if self._states else "initial"
    self._idx = 0
    self.n_joints = n_joints
    self.act_calls = 0
    self.insert_hints_calls = 0
    self.reset_calls = 0

  def reset(self, controller, scene):
    self.reset_calls += 1
    self.state = self._states[0] if self._states else "initial"
    self._idx = 0

  def act(self, obs):
    prev = None
    if self._idx + 1 < len(self._states):
      next_state = self._states[self._idx + 1]
      if next_state != self.state:
        prev = self.state
        self.state = next_state
      self._idx += 1
    self.act_calls += 1
    return Action.from_qpos(np.zeros(self.n_joints, dtype=np.float32)), prev

  def insert_hints(self, viewer):
    self.insert_hints_calls += 1


class _FakeViewer:
  def __init__(self, max_frames: int):
    self._frames = max_frames
    self.user_scn = type("UserScn", (), {"ngeom": 0, "maxgeom": 100})()
    self.sync_calls = 0

  def is_running(self):
    return self._frames > 0

  def sync(self):
    self.sync_calls += 1
    self._frames -= 1


class TestScenePlayerHeadless:
  def _make_player(self, env, policy):
    cfg = _make_simple_config()
    cfg.max_steps = 20
    player = ScenePlayer(cfg, env, policy)
    player.reset()
    return player

  def test_auto_advances_ready_to_approach(self):
    env    = _FakeEnv(terminate_at=None)
    policy = _FakePolicy(["ready", "ready", "approach", "push", "done"])
    player = self._make_player(env, policy)

    episode, outcome = player.run_headless(max_steps=10)
    assert outcome == "done"
    # The player flipped ready → approach before calling act().
    assert policy.state == "done"
    assert episode is not None
    assert episode["action"].shape[0] >= 1

  def test_episode_has_all_required_fields(self):
    env    = _FakeEnv()
    policy = _FakePolicy(["approach", "done"])
    player = self._make_player(env, policy)

    episode, outcome = player.run_headless(max_steps=5)
    assert outcome == "done"
    assert episode is not None
    expected_keys = {
        "image", "action", "reward",
        "timestamp", "joint_qpos", "ee_pos", "ee_quat", "object_xy",
    }
    assert expected_keys.issubset(episode.keys())
    T = episode["action"].shape[0]
    assert episode["action"].dtype == np.float32
    assert episode["ee_quat"].shape == (T, 4)
    assert episode["object_xy"].shape == (T, 2)

  def test_outcome_timeout(self):
    # Policy never reaches "done"; env never terminates.
    env    = _FakeEnv()
    policy = _FakePolicy(["approach", "approach", "approach", "approach", "approach"])
    player = self._make_player(env, policy)

    episode, outcome = player.run_headless(max_steps=3)
    assert outcome == "timeout"
    assert episode is not None
    assert episode["action"].shape[0] == 3

  def test_outcome_terminated(self):
    # Env terminates on step 2; policy never reaches "done".
    env    = _FakeEnv(terminate_at=2)
    policy = _FakePolicy(["approach", "approach", "approach", "approach"])
    player = self._make_player(env, policy)

    episode, outcome = player.run_headless(max_steps=10)
    assert outcome == "terminated"
    assert episode is not None
    assert episode["action"].shape[0] == 2

  def test_outcome_crashed_returns_partial_data(self):
    env    = _FakeEnv(crash_at=3)
    policy = _FakePolicy(["approach"] * 10)
    player = self._make_player(env, policy)

    episode, outcome = player.run_headless(max_steps=10)
    assert outcome == "crashed"
    # Two successful steps happened before the crash on step 3.
    assert episode is not None
    assert episode["action"].shape[0] == 2


class TestScenePlayerInteractive:
  def _make_player(self, env, policy):
    cfg = _make_simple_config()
    return ScenePlayer(cfg, env, policy)

  def test_shows_hints_only_while_ready(self):
    env    = _FakeEnv()
    policy = _FakePolicy(["ready", "ready", "approach", "approach"])
    player = self._make_player(env, policy)
    viewer = _FakeViewer(max_frames=3)

    player.run_interactive(viewer, step_delay=0.0)
    # Hints inserted while state was "ready" (initial frame only — state advances each act).
    assert policy.insert_hints_calls >= 1
    assert viewer.sync_calls == 3

  def test_stops_on_done(self):
    env    = _FakeEnv()
    policy = _FakePolicy(["approach", "done"])
    player = self._make_player(env, policy)
    viewer = _FakeViewer(max_frames=10)

    player.run_interactive(viewer, step_delay=0.0)
    # One act call advances to done, then the loop exits.
    assert policy.state == "done"
    assert viewer.sync_calls == 1

  def test_stops_on_env_terminated(self):
    env    = _FakeEnv(terminate_at=2)
    policy = _FakePolicy(["approach"] * 10)
    player = self._make_player(env, policy)
    viewer = _FakeViewer(max_frames=10)

    player.run_interactive(viewer, step_delay=0.0)
    assert viewer.sync_calls == 2
