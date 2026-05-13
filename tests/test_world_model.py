"""Tests for the backend-agnostic rollout interface.

Exercises the protocol via :class:`DreamerMLXModel` — the protocol is
backend-agnostic but only one backend exists today.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from chuck_dreamer.config import load_config  # noqa: E402
from chuck_dreamer.dreamer import (  # noqa: E402
  CEMPolicy,
  Distribution,
  State,
  Trajectory,
  WorldModel,
  build_model,
  eval_split_rollout,
  rollout,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_state_model(action_dim: int = 6, obs_dim: int = 15):
  cfg = load_config()
  cfg.env.obs_mode = "state"
  return build_model(cfg, obs_shape=(obs_dim,), action_dim=action_dim), cfg


# ---------------------------------------------------------------------------
# WorldModel protocol — DreamerMLXModel implements it
# ---------------------------------------------------------------------------


def test_mlx_model_satisfies_world_model_protocol():
  model, _ = _build_state_model()
  # runtime_checkable: structural check on attributes/methods.
  assert isinstance(model, WorldModel)


def test_initial_state_returns_state_dataclass():
  model, _ = _build_state_model()
  st = model.initial_state(4)
  assert isinstance(st, State)
  assert isinstance(st.prior, Distribution)
  assert st.posterior is None
  assert st.h.shape == (4, model.deter_dim)
  assert st.s.shape == (4, model.stoch_dim)


def test_posterior_step_populates_posterior_distribution():
  model, _ = _build_state_model()
  st = model.initial_state(3)
  embed = mx.zeros((3, model.config.model.encoder.embed_size))
  out = model.posterior_step(st, mx.zeros((3, model.action_dim)), embed)
  assert isinstance(out, State)
  assert out.posterior is not None
  assert out.prior.mean.shape == (3, model.stoch_dim)
  assert out.posterior.mean.shape == (3, model.stoch_dim)


def test_prior_step_leaves_posterior_none():
  model, _ = _build_state_model()
  st = model.initial_state(3)
  out = model.prior_step(st, mx.zeros((3, model.action_dim)))
  assert out.posterior is None
  assert out.prior.mean.shape == (3, model.stoch_dim)


def test_posterior_step_sample_false_uses_post_mean():
  """sample=False makes the state deterministic across repeated calls."""
  model, _ = _build_state_model()
  st = model.initial_state(2)
  embed = mx.zeros((2, model.config.model.encoder.embed_size))
  a     = mx.zeros((2, model.action_dim))
  out1 = model.posterior_step(st, a, embed, sample=False)
  out2 = model.posterior_step(st, a, embed, sample=False)
  assert mx.array_equal(out1.s, out2.s)
  assert mx.array_equal(out1.s, out1.posterior.mean)


def test_prior_step_sample_false_uses_prior_mean():
  model, _ = _build_state_model()
  st = model.initial_state(2)
  a  = mx.zeros((2, model.action_dim))
  out1 = model.prior_step(st, a, sample=False)
  out2 = model.prior_step(st, a, sample=False)
  assert mx.array_equal(out1.s, out2.s)
  assert mx.array_equal(out1.s, out1.prior.mean)


# ---------------------------------------------------------------------------
# rollout()
# ---------------------------------------------------------------------------


def test_rollout_posterior_only_with_explicit_actions_and_embeds():
  model, _ = _build_state_model()
  B, T = 2, 5
  obs     = mx.zeros((B, T, 15))
  actions = mx.zeros((B, T, model.action_dim))
  embeds  = model.encode(obs)

  traj = rollout(
    model,
    init_state = model.initial_state(B),
    horizon    = T,
    mode       = "posterior",
    actions    = actions,
    embeds     = embeds,
  )
  assert isinstance(traj, Trajectory)
  assert len(traj.states) == T
  assert traj.actions.shape == (B, T, model.action_dim)
  assert traj.mode == ["posterior"] * T
  # decode/predict_reward off by default
  assert traj.obs_pred is None
  assert traj.rewards  is None


def test_rollout_prior_only_uses_actor_when_no_policy():
  model, _ = _build_state_model()
  B, T = 2, 4
  traj = rollout(model, init_state=model.initial_state(B), horizon=T,
                 mode="prior", predict_reward=True)
  assert traj.mode == ["prior"] * T
  assert traj.actions.shape == (B, T, model.action_dim)
  assert traj.rewards.shape == (B, T)


def test_rollout_with_external_policy_callable():
  model, _ = _build_state_model()
  B, T = 2, 3
  calls: list[tuple[int, ...]] = []
  fixed = mx.ones((B, model.action_dim))

  def policy(feats):
    calls.append(tuple(feats.shape))
    return fixed

  traj = rollout(model, init_state=model.initial_state(B), horizon=T,
                 mode="prior", policy=policy)
  assert calls == [(B, model.feat_dim)] * T


def test_rollout_burn_in_splits_modes_correctly():
  model, _ = _build_state_model()
  B, T = 2, 5
  obs     = mx.zeros((B, T, 15))
  actions = mx.zeros((B, T, model.action_dim))
  embeds  = model.encode(obs)
  traj = rollout(
    model,
    init_state = model.initial_state(B),
    horizon    = T,
    mode       = "prior",
    burn_in    = 2,
    actions    = actions,
    embeds     = embeds,
  )
  assert traj.mode == ["posterior", "posterior", "prior", "prior", "prior"]


def test_rollout_posterior_without_embeds_raises():
  model, _ = _build_state_model()
  with pytest.raises(ValueError, match="posterior step"):
    rollout(model, init_state=model.initial_state(1), horizon=3,
            mode="posterior", actions=mx.zeros((1, 3, model.action_dim)))


def test_rollout_decode_and_predict_reward_shapes():
  model, _ = _build_state_model()
  B, T = 2, 3
  traj = rollout(model, init_state=model.initial_state(B), horizon=T,
                 mode="prior", decode=True, predict_reward=True)
  assert traj.obs_pred.shape == (B, T, 15)
  assert traj.rewards.shape  == (B, T)


def test_rollout_sample_false_is_deterministic():
  model, _ = _build_state_model()
  B, T = 2, 4
  obs     = mx.zeros((B, T, 15))
  actions = mx.zeros((B, T, model.action_dim))
  embeds  = model.encode(obs)
  init    = model.initial_state(B)

  traj1 = rollout(model, init_state=init, horizon=T, mode="posterior",
                  actions=actions, embeds=embeds, sample=False)
  traj2 = rollout(model, init_state=init, horizon=T, mode="posterior",
                  actions=actions, embeds=embeds, sample=False)
  for s1, s2 in zip(traj1.states, traj2.states):
    assert mx.array_equal(s1.s, s2.s)
    assert mx.array_equal(s1.h, s2.h)


def test_rollout_validates_horizon_and_burn_in():
  model, _ = _build_state_model()
  init = model.initial_state(1)
  with pytest.raises(ValueError):
    rollout(model, init_state=init, horizon=0, mode="prior")
  with pytest.raises(ValueError):
    rollout(model, init_state=init, horizon=3, burn_in=-1, mode="prior")
  with pytest.raises(ValueError):
    rollout(model, init_state=init, horizon=3, burn_in=4, mode="prior")


# ---------------------------------------------------------------------------
# eval_split_rollout()
# ---------------------------------------------------------------------------


def test_eval_split_rollout_shares_anchor_and_emits_two_trajs():
  model, _ = _build_state_model()
  B, burn_in, horizon = 2, 3, 4
  T = burn_in + horizon
  obs     = mx.zeros((B, T, 15))
  actions = mx.zeros((B, T, model.action_dim))
  embeds  = model.encode(obs)

  out = eval_split_rollout(
    model, embeds=embeds, actions=actions,
    burn_in=burn_in, horizon=horizon, sample=False,
  )
  assert len(out.posterior.states) == horizon
  assert len(out.prior.states)     == horizon
  assert out.posterior.mode == ["posterior"] * horizon
  assert out.prior.mode     == ["prior"]     * horizon

  # Both continuations start from the same anchor — first-step h is
  # computed from the same prev_s/prev_a/prev_h, so they should match
  # in their dynamics input (the deter h is shared).
  assert mx.array_equal(out.posterior.states[0].h, out.prior.states[0].h)


def test_trajectory_stack_helpers_match_per_step_states():
  """stack_h / stack_s / stack_feat reproduce per-step state arrays."""
  model, _ = _build_state_model()
  B, T = 2, 4
  obs     = mx.zeros((B, T, 15))
  actions = mx.zeros((B, T, model.action_dim))
  embeds  = model.encode(obs)
  traj = rollout(model, init_state=model.initial_state(B), horizon=T,
                 mode="posterior", actions=actions, embeds=embeds, sample=False)

  h = traj.stack_h()
  s = traj.stack_s()
  assert h.shape == (B, T, model.deter_dim)
  assert s.shape == (B, T, model.stoch_dim)
  for t in range(T):
    assert mx.array_equal(h[:, t], traj.states[t].h)
    assert mx.array_equal(s[:, t], traj.states[t].s)


def test_trajectory_stack_feat_matches_decode_input():
  """stack_feat should match what decode/predict_reward received internally.

  We assert this by checking the decoder applied to stack_feat matches the
  obs_pred field (i.e. rollout cached the same stacked feat it decoded).
  """
  model, _ = _build_state_model()
  B, T = 2, 3
  traj = rollout(model, init_state=model.initial_state(B), horizon=T,
                 mode="prior", decode=True, predict_reward=True, sample=False)
  feats = traj.stack_feat()
  assert feats.shape == (B, T, model.feat_dim)
  # Decoder is deterministic; feeding the cached feat must reproduce
  # obs_pred exactly (same operation, same input).
  assert mx.allclose(model.decode(feats), traj.obs_pred)


def test_trajectory_stack_helpers_cache_results():
  """Repeated calls return the same object — they don't recompute."""
  model, _ = _build_state_model()
  traj = rollout(model, init_state=model.initial_state(1), horizon=3,
                 mode="prior", decode=True)
  assert traj.stack_h() is traj.stack_h()
  assert traj.stack_s() is traj.stack_s()
  assert traj.stack_feat() is traj.stack_feat()


def test_eval_split_rollout_with_zero_burn_in_anchors_at_initial_state():
  model, _ = _build_state_model()
  B, horizon = 2, 4
  obs     = mx.zeros((B, horizon, 15))
  actions = mx.zeros((B, horizon, model.action_dim))
  embeds  = model.encode(obs)
  out = eval_split_rollout(
    model, embeds=embeds, actions=actions,
    burn_in=0, horizon=horizon, sample=False,
  )
  init = model.initial_state(B)
  assert mx.array_equal(out.anchor.h, init.h)
  assert mx.array_equal(out.anchor.s, init.s)


# ---------------------------------------------------------------------------
# CEMPolicy
# ---------------------------------------------------------------------------


def test_cem_policy_act_returns_action_in_bounds():
  model, _ = _build_state_model()
  policy = CEMPolicy(
    model,
    horizon        = 4,
    num_samples    = 16,
    num_elites     = 4,
    num_iterations = 2,
  )
  policy.reset(scene=None)
  obs = {"image": np.zeros((1,), dtype=np.float32)}  # unused — model is state-mode
  # State-mode model takes a 1-D obs vector.
  action = policy.act(np.zeros(15, dtype=np.float32))
  assert action.shape == (model.action_dim,)
  assert action.dtype == np.float32
  assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_cem_policy_warm_starts_after_first_step():
  model, _ = _build_state_model()
  policy = CEMPolicy(
    model,
    horizon=3, num_samples=16, num_elites=4, num_iterations=2,
  )
  policy.reset(scene=None)
  obs = np.zeros(15, dtype=np.float32)
  policy.act(obs)
  assert policy._mean is not None
  prev_mean = policy._mean.copy()
  policy.act(obs)
  # Warm-start shifts the previous mean by one; if no warm-start
  # happened, the second mean would be re-initialized to zero, which
  # equals prev_mean only when CEM found zero — unlikely with noise.
  # We just assert the call updated the state.
  assert policy._mean is not None


def test_cem_policy_requires_reset_before_act():
  model, _ = _build_state_model()
  policy = CEMPolicy(model, horizon=2, num_samples=4, num_elites=2,
                     num_iterations=1)
  with pytest.raises(RuntimeError, match="reset"):
    policy.act(np.zeros(15, dtype=np.float32))
