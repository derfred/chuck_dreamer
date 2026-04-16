"""Gym-compatible pushing environment wrapping SceneBuilder."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .scene_builder import SceneBuilder
from .scene_config import SceneConfig

# Action clip limits: [dx_lo, dx_hi, dy_lo, dy_hi, dz_lo, dz_hi]
_ACTION_CLIP = np.array(
    [-0.02, 0.02, -0.02, 0.02, -0.01, 0.01], dtype=np.float32)

Observation = dict[str, np.ndarray]


class PushingEnv(gym.Env):
    """
    MuJoCo pushing environment.

    Action space: Box(3,) — end-effector position deltas [dx, dy, dz].
    Observation space: Dict with keys 'image', 'ee_pos', 'object_pos'.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        builder: SceneBuilder,
        render_size: tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        self.builder = builder
        self.render_size = render_size

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.renderer: mujoco.Renderer | None = None
        self.config: SceneConfig | None = None
        self.step_count: int = 0

        H, W = render_size
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8),
            "ee_pos": spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32),
            "object_pos": spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array([-0.02, -0.02, -0.01], dtype=np.float32),
            high=np.array([0.02, 0.02, 0.01], dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(  # type: ignore[override]
        self,
        config: SceneConfig | None = None,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        if config is None:
            raise ValueError(
                "PushingEnv.reset() requires a SceneConfig argument.")
        self.config = config
        self.model = self.builder.build(config, render_size=self.render_size)
        self.data = mujoco.MjData(self.model)
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = mujoco.Renderer(
            self.model, self.render_size[0], self.render_size[1])
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        assert self.model is not None and self.data is not None and self.config is not None, (
            "Call reset() before step().")
        # Clip action
        action = np.clip(
            action.astype(np.float64),
            _ACTION_CLIP[[0, 2, 4]],
            _ACTION_CLIP[[1, 3, 5]],
        )

        # Update mocap target
        self.data.mocap_pos[0] = self.data.mocap_pos[0] + action

        # Step physics
        n_substeps = max(
            1, int(
                round(
                    self.config.control_dt / self.model.opt.timestep)))
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_done()
        truncated = False
        info: dict[str, Any] = {
            "object_pos": self._get_object_pos(),
            "ee_pos": self.data.mocap_pos[0].copy(),
            "goal_pos": np.array(self.config.goal_pos, dtype=np.float32),
            "step": self.step_count,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:  # type: ignore[override]
        if self.renderer is None or self.data is None:
            return None
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> Observation:
        assert self.renderer is not None and self.data is not None
        self.renderer.update_scene(self.data)
        image = self.renderer.render()
        return {
            "image": image,
            "ee_pos": self.data.mocap_pos[0].copy().astype(np.float32),
            "object_pos": self._get_object_pos(),
        }

    def _compute_reward(self) -> float:
        assert self.config is not None
        obj_pos = self._get_object_pos()
        goal_pos = np.array(self.config.goal_pos, dtype=np.float32)
        distance = float(np.linalg.norm(obj_pos - goal_pos))
        return -distance

    def _check_done(self) -> bool:
        assert self.config is not None
        obj_pos = self._get_object_pos()
        goal_pos = np.array(self.config.goal_pos, dtype=np.float32)
        distance = float(np.linalg.norm(obj_pos - goal_pos))
        at_goal = distance < self.config.goal_tolerance
        timeout = self.step_count >= self.config.max_steps
        return at_goal or timeout

    def _get_object_pos(self) -> np.ndarray:
        assert self.model is not None and self.data is not None
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target_object")
        return self.data.xpos[body_id][:2].copy().astype(np.float32)
