"""Tests for the reward functions and replay-buffer reward recompute."""

from __future__ import annotations

import numpy as np
import pytest

from chuck_dreamer.reward import GoalDistanceReward, build_reward_fn
from chuck_dreamer.sim.step_info import StepInfo
from chuck_dreamer.training.replay_buffer import ReplayBuffer


def _step_info(distance: float = 0.5) -> StepInfo:
  return StepInfo(
    object_xy=np.array([distance, 0.0], dtype=np.float32),
    ee_pos=np.zeros(3, dtype=np.float32),
    ee_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    goal_xy=np.zeros(2, dtype=np.float32),
    step=0,
    time=0.0,
  )


class TestGoalDistanceReward:
  def test_negative_distance(self):
    fn = GoalDistanceReward()
    r = fn(_step_info(distance=0.3))
    assert r == pytest.approx(-0.3, abs=1e-6)

  def test_zero_at_goal(self):
    fn = GoalDistanceReward()
    r = fn(_step_info(distance=0.0))
    assert r == pytest.approx(0.0, abs=1e-6)


class TestBuildRewardFn:
  def test_goal_distance(self):
    fn = build_reward_fn("goal_distance")
    assert isinstance(fn, GoalDistanceReward)

  def test_unknown_kind_raises(self):
    with pytest.raises(ValueError):
      build_reward_fn("definitely-not-a-real-reward")


def _make_episode_with_step_info(
  ep_id: int,
  T: int = 30,
  obs_dim: int = 4,
  action_dim: int = 2,
  goal_xy: tuple[float, float] = (1.0, 0.0),
) -> dict:
  """Episode carrying step_info + goal_xy so reward recompute is possible.

  Stored reward is constant 0.0 — recomputed reward should differ.
  """
  obs = np.zeros((T + 1, obs_dim), dtype=np.float32)
  for t in range(T + 1):
    obs[t, 0] = ep_id * 1000 + t
  action = np.full((T, action_dim), float(ep_id), dtype=np.float32)
  reward = np.zeros((T,), dtype=np.float32)  # placeholder
  done = np.zeros((T,), dtype=bool)
  done[-1] = True

  # object_xy moves linearly from (0, 0) to goal_xy over T steps
  goal_arr = np.asarray(goal_xy, dtype=np.float32)
  object_xy = np.linspace(np.zeros(2, dtype=np.float32), goal_arr, T).astype(np.float32)
  step_info = {
    "object_xy": object_xy,
    "ee_pos":    np.zeros((T, 3), dtype=np.float32),
    "ee_quat":   np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 1)),
    "time":      np.arange(T, dtype=np.float32) * 0.1,
  }

  return {
    "obs": {"state": obs},
    "action": action,
    "reward": reward,
    "done": done,
    "step_info": step_info,
    "goal_xy": goal_arr,
  }


class TestReplayBufferRewardRecompute:
  def test_no_reward_fn_returns_stored_reward(self):
    buf = ReplayBuffer(capacity_steps=10_000, min_episode_len=10, seed=0)
    buf.add_episode(_make_episode_with_step_info(0, T=30))

    batch = buf.sample(batch_size=4, seq_len=10)
    # Stored reward is all zeros.
    np.testing.assert_array_equal(np.asarray(batch["reward"]), np.zeros((4, 10), dtype=np.float32))

  def test_reward_fn_overrides_stored_reward(self):
    """With reward_fn=GoalDistanceReward, sampled reward equals -‖obj-goal‖."""
    buf = ReplayBuffer(
      capacity_steps=10_000,
      min_episode_len=10,
      reward_fn=GoalDistanceReward(),
      seed=0,
    )
    buf.add_episode(_make_episode_with_step_info(0, T=30, goal_xy=(1.0, 0.0)))

    batch = buf.sample(batch_size=4, seq_len=10)
    rewards = np.asarray(batch["reward"])
    # All non-zero (object never sits exactly on the goal except at t=T-1
    # which we may or may not hit).
    assert (rewards != 0.0).any()
    # All non-positive (negative distance).
    assert (rewards <= 1e-6).all()

  def test_reward_swap_changes_output(self):
    """Two buffers with different reward_fns yield different reward batches for the same data."""
    ep = _make_episode_with_step_info(0, T=30)

    buf_a = ReplayBuffer(
      capacity_steps=10_000, min_episode_len=10,
      reward_fn=GoalDistanceReward(), seed=0,
    )
    buf_a.add_episode(ep)

    class _ConstReward:
      def __call__(self, info):
        return 7.0

    buf_b = ReplayBuffer(
      capacity_steps=10_000, min_episode_len=10,
      reward_fn=_ConstReward(), seed=0,
    )
    buf_b.add_episode(ep)

    batch_a = buf_a.sample(batch_size=8, seq_len=10)
    batch_b = buf_b.sample(batch_size=8, seq_len=10)

    ra = np.asarray(batch_a["reward"])
    rb = np.asarray(batch_b["reward"])
    np.testing.assert_array_equal(rb, np.full(rb.shape, 7.0, dtype=np.float32))
    assert not np.array_equal(ra, rb)

  def test_episode_without_goal_xy_falls_back_to_stored_reward(self):
    """Episodes lacking goal_xy can't be recomputed (it's episode-scalar
    metadata, not in step_info), so sample() returns the stored reward
    even when reward_fn is set."""
    buf = ReplayBuffer(
      capacity_steps=10_000, min_episode_len=10,
      reward_fn=GoalDistanceReward(), seed=0,
    )
    ep = _make_episode_with_step_info(0, T=30)
    del ep["goal_xy"]
    ep["reward"] = np.full((30,), 42.0, dtype=np.float32)
    buf.add_episode(ep)

    batch = buf.sample(batch_size=4, seq_len=10)
    np.testing.assert_array_equal(
      np.asarray(batch["reward"]),
      np.full((4, 10), 42.0, dtype=np.float32),
    )
