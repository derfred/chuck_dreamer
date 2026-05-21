#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from chuck_dreamer.config import load_config
from chuck_dreamer.dreamer import build_model, CEMPolicy
from chuck_dreamer.sim import PushingEnv


# =========================
# Hard-coded smoke test config
# =========================

CHECKPOINT_PATH = Path("checkpoints/default/latest.safetensors")

NUM_EPISODES = 1

# CEM parameters: keep small for smoke test.
CEM_HORIZON = 3
CEM_NUM_SAMPLES = 64
CEM_NUM_ELITES = 8
CEM_NUM_ITERATIONS = 2
CEM_INIT_STD = 0.5
CEM_MIN_STD = 0.05


def load_config_from_checkpoint_if_possible(path: Path):
    """
    Prefer the config embedded inside the checkpoint.
    Fall back to configs/default.yaml if no embedded config is found.
    """
    if path.exists():
        # Try Torch checkpoint metadata loader.
        try:
            from chuck_dreamer.dreamer.torch_model import (
                load_config_from_checkpoint as load_torch_config,
            )

            cfg = load_torch_config(str(path))
            if cfg is not None:
                print(f"Loaded config from Torch checkpoint metadata: {path}")
                return cfg
        except Exception as e:
            print(f"Torch checkpoint config load skipped: {e}")

        # Try MLX checkpoint metadata loader.
        try:
            from chuck_dreamer.dreamer.mlx_model import (
                load_config_from_checkpoint as load_mlx_config,
            )

            cfg = load_mlx_config(str(path))
            if cfg is not None:
                print(f"Loaded config from MLX checkpoint metadata: {path}")
                return cfg
        except Exception as e:
            print(f"MLX checkpoint config load skipped: {e}")

    print("Using default config.")
    return load_config()


def run_one_episode(env: PushingEnv, model, episode_idx: int):
    scene = env.generate_scene()
    obs, _ = env.reset(scene=scene)

    action_space = env.action_space

    policy = CEMPolicy(
        model,
        horizon=CEM_HORIZON,
        num_samples=CEM_NUM_SAMPLES,
        num_elites=CEM_NUM_ELITES,
        num_iterations=CEM_NUM_ITERATIONS,
        init_std=CEM_INIT_STD,
        min_std=CEM_MIN_STD,
        action_low=action_space.low,
        action_high=action_space.high,
        obs_adapter=env.policy_obs,
    )

    policy.reset(scene)

    total_reward = 0.0
    outcome = "timeout"

    max_steps = scene.max_steps
    if max_steps is None:
        max_steps = int(env.config.env.max_steps or 100)

    for step in range(max_steps):
        action = policy.act(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        step_info = info["step_info"]
        object_xy = step_info.object_xy
        goal_xy = step_info.goal_xy
        dist = float(np.linalg.norm(object_xy - goal_xy))

        print(
            f"[episode {episode_idx} step {step:03d}] "
            f"reward={float(reward): .4f} "
            f"dist={dist: .4f} "
            f"action_norm={float(np.linalg.norm(action)): .4f}"
        )

        if terminated:
            outcome = "done"
            break

        if truncated:
            outcome = "timeout"
            break

    print(
        f"[episode {episode_idx}] "
        f"outcome={outcome}, total_reward={total_reward:.4f}"
    )

    return outcome, total_reward


def main():
    cfg = load_config_from_checkpoint_if_possible(CHECKPOINT_PATH)

    env = PushingEnv(cfg)

    # Build env once before model so model_obs_shape/action_dim reflect this env.
    scene = env.generate_scene()
    env.reset(scene=scene)

    obs_shape = env.model_obs_shape
    action_dim = int(env.action_space.shape[0])

    print("obs_mode:", cfg.env.obs_mode)
    print("act_mode:", cfg.env.act_mode)
    print("obs_shape:", obs_shape)
    print("action_dim:", action_dim)
    print("action_low:", env.action_space.low)
    print("action_high:", env.action_space.high)

    model = build_model(cfg, obs_shape=obs_shape, action_dim=action_dim)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Set CHECKPOINT_PATH in scripts/run_cem_sim.py."
        )

    extra = model.load(str(CHECKPOINT_PATH))
    print("Loaded checkpoint:", CHECKPOINT_PATH)
    print("Checkpoint metadata:", extra)

    results = []

    for episode_idx in range(NUM_EPISODES):
        outcome, total_reward = run_one_episode(env, model, episode_idx)
        results.append((outcome, total_reward))

    env.close()

    print("\nSummary:")
    for i, (outcome, total_reward) in enumerate(results):
        print(f"  episode {i}: outcome={outcome}, total_reward={total_reward:.4f}")


if __name__ == "__main__":
    main()