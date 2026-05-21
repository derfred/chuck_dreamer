"""Gym-compatible pushing environment wrapping SceneBuilder."""

from __future__ import annotations

from typing import Any, Literal, cast

import gymnasium as gym
import mujoco  # type: ignore[import-untyped]
import numpy as np
from gymnasium import spaces

from ..reward import GoalDistanceReward
from ..training.observation import (
  EE,
  IMAGE,
  JOINTS,
  OBJECT_UV,
  OBJECT_XY,
  PROPRIO,
  ObsMode,
  normalize_obs_mode,
)
from .observation_image import ObservationImage
from .scene_builder import SceneBuilder
from .scene_generator import SceneGenerator
from .scene_config import SceneConfig
from .step_info import StepInfo

Observation = dict[str, Any]

ActMode = Literal["joint", "ee"]


# Conservative world-space EE position bounds. Used for advertising the EE
# action space; actual reachability is enforced by IK convergence.
_EE_POS_LOW = np.array([-1.0, -1.0, 0.0], dtype=np.float32)
_EE_POS_HIGH = np.array([1.0, 1.0, 1.0], dtype=np.float32)


class Controller:
  def reset(self, model, config: SceneConfig) -> None:
    self.model   = model
    self.config  = config
    self.ik_data = mujoco.MjData(model)

    self.ee_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in config.joint_names
    ]
    self.arm_qpos_adr = np.array([model.jnt_qposadr[jid] for jid in joint_ids])
    self.arm_dof_idx  = np.array([model.jnt_dofadr[jid]  for jid in joint_ids])

    self.arm_ctrl_idx = np.array([
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
      for n in config.actuator_names
    ])
    self.arm_ctrl_range = model.actuator_ctrlrange[self.arm_ctrl_idx].copy()  # (n_joints, 2)

  def get_ee_pos(self, data) -> np.ndarray:
    return cast(np.ndarray, data.site_xpos[self.ee_sid].copy().astype(np.float32))

  def get_ee_quat(self, data) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.site_xmat[self.ee_sid])
    return cast(np.ndarray, quat.astype(np.float32))

  def get_arm_qpos(self, data) -> np.ndarray:
    return cast(np.ndarray, data.qpos[self.arm_qpos_adr].copy().astype(np.float32))

  def reset_initial_qpos(self, data, qpos) -> None:
    """Set arm joint positions and matching control setpoints so the
    position actuators hold the initial pose instead of driving back to zero."""
    qpos = np.asarray(qpos, dtype=np.float64)
    data.qpos[self.arm_qpos_adr] = qpos
    data.qvel[self.arm_dof_idx]  = 0.0
    data.ctrl[self.arm_ctrl_idx] = qpos
    mujoco.mj_forward(self.model, data)

  def ik_for_pose(
      self,
      data,
      target_pos: np.ndarray,
      target_quat: np.ndarray | None = None,
  ) -> np.ndarray:
    """Solve IK for an EE pose, warm-started from ``data.qpos[arm_qpos_adr]``.

    If ``target_quat`` is None, only position IK is solved (3-DOF). With a
    target quaternion, a 6-DOF damped least-squares step is taken with
    per-block weights to balance position (m) and orientation (rad) scales.
    On non-convergence with orientation, falls back to position-only IK
    using the current quat as target before raising.
    """
    seed = data.qpos[self.arm_qpos_adr].copy()
    if target_quat is None:
      return self._ik_position_only(data.qpos, target_pos, seed)

    try:
      return self._ik_pose(data.qpos, target_pos, np.asarray(target_quat, dtype=np.float64), seed)
    except RuntimeError:
      return self._ik_position_only(data.qpos, target_pos, seed)

  def _ik_position_only(
      self,
      seed_full_qpos: np.ndarray,
      target_xyz: np.ndarray,
      q_seed: np.ndarray,
  ) -> np.ndarray:
    d = self.ik_data
    d.qpos[:] = seed_full_qpos
    q = q_seed.copy()

    lam_sq    = 0.05 ** 2
    max_dq    = 0.3
    _jac_pos  = np.zeros((3, self.model.nv))
    _eye3     = np.eye(3)
    converged = False
    err = np.array([np.inf, np.inf, np.inf])

    for _ in range(30):
        mujoco.mj_forward(self.model, d)
        ee = d.site_xpos[self.ee_sid]
        if not np.all(np.isfinite(ee)):
            raise RuntimeError(f"IK diverged (non-finite EE pos), last q={q}")
        err = target_xyz - ee
        if np.linalg.norm(err) < 1e-3:
            converged = True
            break
        mujoco.mj_jacSite(self.model, d, _jac_pos, None, self.ee_sid)
        J = _jac_pos[:, self.arm_dof_idx]
        dq = J.T @ np.linalg.solve(J @ J.T + lam_sq * _eye3, err)

        dq_norm = np.linalg.norm(dq)
        if dq_norm > max_dq:
            dq *= max_dq / dq_norm

        q = q + dq
        d.qpos[self.arm_qpos_adr] = q

    if not converged:
      raise RuntimeError(f"IK did not converge: err={err}, ||err||={np.linalg.norm(err):.4f}")

    return cast(np.ndarray, q)

  def _ik_pose(
      self,
      seed_full_qpos: np.ndarray,
      target_pos: np.ndarray,
      target_quat: np.ndarray,
      q_seed: np.ndarray,
  ) -> np.ndarray:
    """6-DOF IK using stacked position + orientation Jacobians.

    Orientation error is computed via mju_subQuat-equivalent (quaternion
    difference → axis-angle 3-vec) which avoids quaternion sign ambiguity.
    Per-block weights balance the meters/radians scale mismatch.
    """
    d = self.ik_data
    d.qpos[:] = seed_full_qpos
    q = q_seed.copy()

    lam_sq    = 0.05 ** 2
    max_dq    = 0.15
    w_pos     = 1.0
    w_rot     = 0.5
    _jac_pos  = np.zeros((3, self.model.nv))
    _jac_rot  = np.zeros((3, self.model.nv))
    _eye6     = np.eye(6)
    converged = False
    err6 = np.full(6, np.inf)

    cur_quat = np.zeros(4, dtype=np.float64)
    qcur_inv = np.zeros(4, dtype=np.float64)
    qdiff    = np.zeros(4, dtype=np.float64)
    orient_err = np.zeros(3, dtype=np.float64)

    target_quat = target_quat / max(np.linalg.norm(target_quat), 1e-12)

    for _ in range(40):
        mujoco.mj_forward(self.model, d)
        ee = d.site_xpos[self.ee_sid]
        if not np.all(np.isfinite(ee)):
            raise RuntimeError(f"IK diverged (non-finite EE pos), last q={q}")

        pos_err = target_pos - ee

        mujoco.mju_mat2Quat(cur_quat, d.site_xmat[self.ee_sid])
        mujoco.mju_negQuat(qcur_inv, cur_quat)
        mujoco.mju_mulQuat(qdiff, target_quat, qcur_inv)
        mujoco.mju_quat2Vel(orient_err, qdiff, 1.0)

        err6[:3] = w_pos * pos_err
        err6[3:] = w_rot * orient_err

        if np.linalg.norm(pos_err) < 1e-3 and np.linalg.norm(orient_err) < 1e-2:
            converged = True
            break

        mujoco.mj_jacSite(self.model, d, _jac_pos, _jac_rot, self.ee_sid)
        J = np.vstack([
          w_pos * _jac_pos[:, self.arm_dof_idx],
          w_rot * _jac_rot[:, self.arm_dof_idx],
        ])
        dq = J.T @ np.linalg.solve(J @ J.T + lam_sq * _eye6, err6)

        dq_norm = np.linalg.norm(dq)
        if dq_norm > max_dq:
            dq *= max_dq / dq_norm

        q = q + dq
        d.qpos[self.arm_qpos_adr] = q

    if not converged:
      raise RuntimeError(
        f"6-DOF IK did not converge: pos_err={np.linalg.norm(err6[:3]/w_pos):.4f}, "
        f"rot_err={np.linalg.norm(err6[3:]/w_rot):.4f}"
      )

    return cast(np.ndarray, q)

  def update_arm(self, data, action: np.ndarray, act_mode: ActMode) -> None:
    """Apply ``action`` to the arm actuators, dispatching on ``act_mode``.

    - ``"joint"``: action is ``(n_joints,)`` qpos written directly to
      actuator setpoints (with clipping).
    - ``"ee"``: action is ``(7,)`` ``[x, y, z, qw, qx, qy, qz]``. IK is
      run with the current ``data.qpos`` as warm-start, then the
      resulting joint qpos is written.
    """
    if act_mode == "joint":
      qpos_target = np.asarray(action, dtype=np.float64)
    else:
      action = np.asarray(action, dtype=np.float64)
      if action.shape != (7,):
        raise ValueError(f"EE action must be shape (7,), got {action.shape}")
      qpos_target = self.ik_for_pose(data, action[:3], action[3:])

    ctrl = np.clip(
        qpos_target,
        self.arm_ctrl_range[:, 0],
        self.arm_ctrl_range[:, 1],
    )
    data.ctrl[self.arm_ctrl_idx] = ctrl


class PushingEnv(gym.Env):
  """
  MuJoCo pushing environment.

  Observation and action shapes are determined by ``cfg.env.obs_mode`` and
  ``cfg.env.act_mode`` (see ``ObsMode`` / ``ActMode``).
  """

  metadata = {"render_modes": ["rgb_array"]}

  def __init__(self, config) -> None:
    super().__init__()
    self.config    = config
    self.builder   = SceneBuilder()
    self.generator = SceneGenerator(config)

    env_cfg = config.get("env", {}) if hasattr(config, "get") else {}
    self.obs_mode: ObsMode = normalize_obs_mode(env_cfg.get("obs_mode", "joints"))
    self.act_mode: ActMode = cast(ActMode, env_cfg.get("act_mode", "ee"))

    self.model: mujoco.MjModel | None = None
    self.data: mujoco.MjData | None = None
    self.renderer: mujoco.Renderer | None = None
    self.scene: SceneConfig | None = None
    self.controller: Controller = Controller()
    self.step_count: int = 0
    # The env records reward under the default reward function so that
    # downstream tools (analysis, plotting) see consistent values. The
    # buffer recomputes reward at sample time using whatever ``RewardFn``
    # the trainer is configured with.
    self._default_reward_fn = GoalDistanceReward()

  @property
  def render_size(self) -> tuple[int, int]:
    """Parse a 'WxH' string and return (render_h, render_w)."""
    w, h = self.config.env.render_size.lower().split("x")
    return int(h), int(w)

  @property
  def n_joints(self) -> int:
    if self.scene is not None:
      return len(self.scene.joint_names)
    return self.generator.n_joints

  def _component_dims(self) -> dict[str, tuple[int, ...]]:
    """Per-component trailing shapes for the current ``obs_mode``.

    Vector components are computed from the configured scene (number of
    joints, etc.); the image component reports the render size — the
    model-side shape is overridden in :attr:`model_obs_shape` to match
    the loader's resize.
    """
    H, W = self.render_size
    dims: dict[str, tuple[int, ...]] = {}
    for c in self.obs_mode:
      if c == IMAGE:
        dims[c] = (H, W, 3)
      elif c == JOINTS:
        dims[c] = (self.n_joints,)                  # joint_qpos
      elif c == PROPRIO:
        dims[c] = (self.n_joints,)                  # joint_qpos
      elif c == EE:
        dims[c] = (3 + 4,)                          # ee_pos ‖ ee_quat
      elif c == OBJECT_XY:
        dims[c] = (2,)                              # object_xy
      elif c == OBJECT_UV:
        dims[c] = (2,)                              # object pixel coords (u, v)
      else:
        raise ValueError(f"unknown obs_mode component {c!r}")
    return dims

  @property
  def observation_space(self) -> spaces.Space:  # type: ignore[override]
    """Gym observation space as a dict over the configured components.

    Always a ``spaces.Dict`` keyed by component name — uniform whether
    one or many components are listed. Image dims are reported at the
    raw render size; the loader resizes them before they reach the model.
    """
    dims = self._component_dims()
    parts: dict[str, spaces.Space] = {}
    for c, shape in dims.items():
      if c == IMAGE:
        parts[c] = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
      else:
        parts[c] = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
    return spaces.Dict(parts)

  @property
  def model_obs_shape(self) -> dict[str, tuple[int, ...]]:
    """Per-component trailing shapes the model encoder expects.

    Mirrors :attr:`observation_space` but overrides the image shape with
    ``config.model.encoder.image_size`` — the loader resizes captured
    frames to that side length before they reach the model.
    """
    dims = self._component_dims()
    if IMAGE in dims:
      image_size = int(self.config.model.encoder.image_size)
      dims[IMAGE] = (image_size, image_size, 3)
    return dims

  @property
  def action_space(self) -> spaces.Box:  # type: ignore[override]
    low:  np.ndarray
    high: np.ndarray
    if self.act_mode == "joint":
      if self.scene is None:
        n_joints = 6
        low  = np.full((n_joints,), -np.inf, dtype=np.float32)
        high = np.full((n_joints,), np.inf, dtype=np.float32)
      else:
        low  = self.controller.arm_ctrl_range[:, 0].astype(np.float32)
        high = self.controller.arm_ctrl_range[:, 1].astype(np.float32)
      return spaces.Box(low=low, high=high, dtype=np.float32)

    low  = np.concatenate([_EE_POS_LOW, np.full(4, -1.0, dtype=np.float32)])
    high = np.concatenate([_EE_POS_HIGH, np.full(4, 1.0, dtype=np.float32)])
    return spaces.Box(low=low, high=high, dtype=np.float32)

  def policy_obs(self, full_obs: Observation) -> dict[str, np.ndarray]:
    """Project the full obs dict to the per-component dict the model consumes.

    Always a dict keyed by component name (matches the buffer schema
    and what :meth:`model_obs_shape` advertises), independent of how
    many components ``obs_mode`` lists.
    """
    out: dict[str, np.ndarray] = {}
    for c in self.obs_mode:
      if c == IMAGE:
        image_obs: ObservationImage = full_obs["image"]
        out[c] = np.asarray(image_obs.image, dtype=np.uint8)
      elif c == JOINTS:
        out[c] = np.asarray(full_obs["arm_qpos"], dtype=np.float32)
      elif c == PROPRIO:
        out[c] = np.asarray(full_obs["arm_qpos"], dtype=np.float32)
      elif c == EE:
        out[c] = np.concatenate([
          np.asarray(full_obs["ee_pos"],  dtype=np.float32),
          np.asarray(full_obs["ee_quat"], dtype=np.float32),
        ])
      elif c == OBJECT_XY:
        out[c] = np.asarray(full_obs["object_xy"], dtype=np.float32)
      elif c == OBJECT_UV:
        if "object_uv" not in full_obs:
          raise KeyError(
            "policy_obs requires 'object_uv' in the env obs for obs_mode "
            "component 'object_uv'; the sim env does not yet compute it."
          )
        out[c] = np.asarray(full_obs["object_uv"], dtype=np.float32)
      else:
        raise ValueError(f"unknown obs_mode component {c!r}")
    return out

  def generate_scene(self) -> SceneConfig:
    """Generate a new random scene config."""
    return self.generator.sample()

  # ------------------------------------------------------------------
  # Gym API
  # ------------------------------------------------------------------

  def reset(  # type: ignore[override]
      self,
      *,
      scene: SceneConfig | None = None,
      seed: int | None = None
  ) -> tuple[Observation, dict[str, Any]]:
    assert scene is not None, "Must provide a SceneConfig to reset()."
    super().reset(seed=seed)

    self.scene = scene
    self.model = self.builder.build(scene, render_size=self.render_size)
    self.data  = mujoco.MjData(self.model)
    self.controller.reset(self.model, scene)

    if self.renderer is not None:
      self.renderer.close()
    self.renderer = mujoco.Renderer(self.model, self.render_size[0], self.render_size[1])
    self._setup_seg_ids()

    mujoco.mj_forward(self.model, self.data)

    initial_qpos = scene.joint_initial_qpos
    if initial_qpos is not None:
      self.controller.reset_initial_qpos(self.data, initial_qpos)

    self.step_count = 0
    return self._get_obs(), {}

  def step(self, action: np.ndarray) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
    assert (
        self.model is not None
        and self.data is not None
        and self.scene is not None
    ), "Call reset() before step()."

    self.controller.update_arm(self.data, np.asarray(action), self.act_mode)

    n_substeps = max(1, int(round(self.scene.control_dt / self.model.opt.timestep)))
    for _ in range(n_substeps):
        mujoco.mj_step(self.model, self.data)

    self.step_count += 1

    obs        = self._get_obs()
    step_info  = self._build_step_info()
    reward     = self._compute_reward(step_info)
    terminated, truncated = self._check_done(step_info)
    info: dict[str, Any] = {"step_info": step_info}
    return obs, reward, terminated, truncated, info

  def render(self) -> ObservationImage | None:  # type: ignore[override]
    """Render the current scene as an :class:`ObservationImage`.

    Captures the RGB frame and the segmentation pass back-to-back so the
    returned object carries both the image and per-geom masks aligned to
    the same camera state. Returns ``None`` before :meth:`reset` has been
    called.
    """
    if self.renderer is None or self.data is None:
      return None

    # Use the scene's main_camera (set up by SceneBuilder) so per-episode
    # camera randomisation actually takes effect. Without an explicit camera
    # argument mujoco.Renderer falls back to its free orbit camera, which
    # ignores config.camera entirely.
    self.renderer.update_scene(self.data, camera="main_camera")
    image = np.asarray(self.renderer.render())

    self.renderer.enable_segmentation_rendering()
    self.renderer.update_scene(self.data, camera="main_camera")
    seg = self.renderer.render()   # (H, W, 2)
    self.renderer.disable_segmentation_rendering()
    self.renderer.update_scene(self.data, camera="main_camera")

    return ObservationImage.from_seg(image, self._segmentation_masks(seg))

  def close(self) -> None:
    if self.renderer is not None:
      self.renderer.close()
      self.renderer = None

  # ------------------------------------------------------------------
  # Internal helpers
  # ------------------------------------------------------------------

  # Arm geoms are anonymous in the MJCF, so we identify them by the
  # bodies they belong to. Every geom under one of these bodies is
  # bundled into a single ``arm`` segmentation mask.
  _ARM_BODY_NAMES = (
      "Base",
      "Rotation_Pitch",
      "Upper_Arm",
      "Lower_Arm",
      "Wrist_Pitch_Roll",
      "Fixed_Jaw",
      "Moving_Jaw",
  )

  def _setup_seg_ids(self):
    """Resolve the geoms we care about to their IDs, once."""
    # goal_marker may be absent when the scene draws its own goal visual
    # (e.g. the "real" preset's ring of clutter boxes); skip it if missing.
    names = ['target_object_geom']
    if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'goal_marker') != -1:
        names.insert(0, 'goal_marker')

    self._seg_geom_ids = {
        name: self.model.geom(name).id
        for name in names
    }

    arm_body_ids = {
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in self._ARM_BODY_NAMES
    }
    arm_body_ids.discard(-1)
    self._seg_arm_geom_ids = np.array(
        [g for g in range(self.model.ngeom) if int(self.model.geom_bodyid[g]) in arm_body_ids],
        dtype=np.int32,
    )

  def _segmentation_masks(self, seg):
    """seg: (H, W, 2) array from the segmentation renderer.
    Returns dict[str, (H, W) bool array]."""
    ids     = seg[..., 0]
    types   = seg[..., 1]
    is_geom = types == mujoco.mjtObj.mjOBJ_GEOM

    masks   = {"background": ids == -1}
    covered = np.zeros(ids.shape, dtype=bool)
    for name, gid in self._seg_geom_ids.items():
      m = is_geom & (ids == gid)
      masks[name] = m
      covered |= m

    # Downstream observation builders expect a ``goal_marker`` mask even when
    # the scene doesn't render an explicit one (e.g. the "real" preset draws
    # its goal as part of the clutter ring).
    if "goal_marker" not in masks:
      masks["goal_marker"] = np.zeros(ids.shape, dtype=bool)

    arm_mask = is_geom & np.isin(ids, self._seg_arm_geom_ids)
    masks["arm"] = arm_mask
    covered |= arm_mask

    masks["background"] |= ~covered
    return masks

  def _get_obs(self) -> Observation:
    assert (
        self.renderer is not None
        and self.data is not None
        and self.scene is not None
    )
    image_obs = self.render()
    assert image_obs is not None
    return {
        "image": image_obs,
        "ee_pos": self.controller.get_ee_pos(self.data),
        "ee_quat": self.controller.get_ee_quat(self.data),
        "arm_qpos": self.controller.get_arm_qpos(self.data),
        "object_xy": self._get_object_pos(),
        "goal_xy": np.array(self.scene.goal_xy, dtype=np.float32),
    }

  def _build_step_info(self) -> StepInfo:
    assert self.scene is not None and self.data is not None
    return StepInfo(
      object_xy=self._get_object_pos(),
      ee_pos=self.controller.get_ee_pos(self.data),
      ee_quat=self.controller.get_ee_quat(self.data),
      goal_xy=np.array(self.scene.goal_xy, dtype=np.float32),
      step=self.step_count,
      time=float(self.data.time),
    )

  def _compute_reward(self, info: StepInfo) -> float:
    return self._default_reward_fn(info)

  def _check_done(self, info: StepInfo) -> tuple[bool, bool]:
    """Return ``(terminated, truncated)`` per gym convention.

    ``terminated``: goal reached (MDP-terminal).
    ``truncated``: max-steps timeout (not MDP-terminal — the agent might
    still be making progress).
    """
    assert self.scene is not None
    distance = float(np.linalg.norm(info.object_xy - info.goal_xy))
    at_goal = distance < self.scene.goal_tolerance
    timeout = self.step_count >= self.scene.max_steps
    return at_goal, (timeout and not at_goal)

  def _get_object_pos(self) -> np.ndarray:
    assert self.model is not None and self.data is not None
    body_id = mujoco.mj_name2id(
        self.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "target_object")
    return np.asarray(
        self.data.xpos[body_id][:2].copy(), dtype=np.float32)
