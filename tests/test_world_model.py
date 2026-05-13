"""Tests for the backend-agnostic rollout interface.

Parametrized across the MLX and PyTorch backends. Most tests run on
both; CEMPolicy tests are MLX-only for now because CEMPolicy hard-codes
``mx.array`` and ``mx.broadcast_to``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pytest

from chuck_dreamer.config import load_config
from chuck_dreamer.dreamer import (
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
# Backend abstraction for the parametrized tests
# ---------------------------------------------------------------------------


@dataclass
class Backend:
  """Tiny shim for cross-backend array ops the tests use.

  The protocol/rollout code is backend-agnostic, but the tests construct
  inputs and inspect outputs — those need backend-specific primitives.
  """

  name:     str         # config.hardware.device value
  zeros:    Callable[[tuple[int, ...]], Any]
  ones:     Callable[[tuple[int, ...]], Any]
  equal:    Callable[[Any, Any], bool]
  allclose: Callable[[Any, Any], bool]


def _mlx_backend() -> Backend | None:
  try:
    import mlx.core as mx  # noqa: PLC0415
  except ImportError:
    return None
  return Backend(
    name     = "mlx",
    zeros    = lambda shape: mx.zeros(shape),
    ones     = lambda shape: mx.ones(shape),
    equal    = lambda a, b: bool(mx.array_equal(a, b)),
    allclose = lambda a, b: bool(mx.allclose(a, b)),
  )


def _torch_backend() -> Backend | None:
  try:
    import torch  # noqa: PLC0415
  except ImportError:
    return None
  return Backend(
    name     = "cpu",  # routes to DreamerTorchModel
    zeros    = lambda shape: torch.zeros(*shape),
    ones     = lambda shape: torch.ones(*shape),
    equal    = lambda a, b: bool(torch.equal(a, b)),
    allclose = lambda a, b: bool(torch.allclose(a, b)),
  )


_ALL_BACKENDS = [b for b in (_mlx_backend(), _torch_backend()) if b is not None]


@pytest.fixture(params=_ALL_BACKENDS, ids=lambda b: b.name)
def backend(request) -> Backend:
  return request.param


def _build_state_model(backend: Backend, action_dim: int = 6, obs_dim: int = 15):
  cfg = load_config()
  cfg.env.obs_mode = "state"
  cfg.hardware.device = backend.name
  return build_model(cfg, obs_shape=(obs_dim,), action_dim=action_dim), cfg


# ---------------------------------------------------------------------------
# WorldModel protocol
# ---------------------------------------------------------------------------


def test_model_satisfies_world_model_protocol(backend: Backend):
  model, _ = _build_state_model(backend)
  assert isinstance(model, WorldModel)


def test_initial_state_returns_state_dataclass(backend: Backend):
  model, _ = _build_state_model(backend)
  st = model.initial_state(4)
  assert isinstance(st, State)
  assert isinstance(st.prior, Distribution)
  assert st.posterior is None
  assert tuple(st.h.shape) == (4, model.deter_dim)
  assert tuple(st.s.shape) == (4, model.stoch_dim)


def test_posterior_step_populates_posterior_distribution(backend: Backend):
  model, _ = _build_state_model(backend)
  st = model.initial_state(3)
  embed = backend.zeros((3, model.config.model.encoder.embed_size))
  out = model.posterior_step(st, backend.zeros((3, model.action_dim)), embed)
  assert isinstance(out, State)
  assert out.posterior is not None
  assert tuple(out.prior.mean.shape)     == (3, model.stoch_dim)
  assert tuple(out.posterior.mean.shape) == (3, model.stoch_dim)


def test_prior_step_leaves_posterior_none(backend: Backend):
  model, _ = _build_state_model(backend)
  st = model.initial_state(3)
  out = model.prior_step(st, backend.zeros((3, model.action_dim)))
  assert out.posterior is None
  assert tuple(out.prior.mean.shape) == (3, model.stoch_dim)


def test_posterior_step_sample_false_uses_post_mean(backend: Backend):
  model, _ = _build_state_model(backend)
  st = model.initial_state(2)
  embed = backend.zeros((2, model.config.model.encoder.embed_size))
  a     = backend.zeros((2, model.action_dim))
  out1 = model.posterior_step(st, a, embed, sample=False)
  out2 = model.posterior_step(st, a, embed, sample=False)
  assert backend.equal(out1.s, out2.s)
  assert backend.equal(out1.s, out1.posterior.mean)


def test_prior_step_sample_false_uses_prior_mean(backend: Backend):
  model, _ = _build_state_model(backend)
  st = model.initial_state(2)
  a  = backend.zeros((2, model.action_dim))
  out1 = model.prior_step(st, a, sample=False)
  out2 = model.prior_step(st, a, sample=False)
  assert backend.equal(out1.s, out2.s)
  assert backend.equal(out1.s, out1.prior.mean)


# ---------------------------------------------------------------------------
# rollout()
# ---------------------------------------------------------------------------


def test_rollout_posterior_only_with_explicit_actions_and_embeds(backend: Backend):
  model, _ = _build_state_model(backend)
  B, T = 2, 5
  obs     = backend.zeros((B, T, 15))
  actions = backend.zeros((B, T, model.action_dim))
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
  assert tuple(traj.actions.shape) == (B, T, model.action_dim)
  assert traj.mode == ["posterior"] * T
  assert traj.obs_pred is None
  assert traj.rewards  is None


def test_rollout_prior_only_uses_actor_when_no_policy(backend: Backend):
  model, _ = _build_state_model(backend)
  B, T = 2, 4
  traj = rollout(model, init_state=model.initial_state(B), horizon=T,
                 mode="prior", predict_reward=True)
  assert traj.mode == ["prior"] * T
  assert tuple(traj.actions.shape) == (B, T, model.action_dim)
  assert tuple(traj.rewards.shape) == (B, T)


def test_rollout_with_external_policy_callable(backend: Backend):
  model, _ = _build_state_model(backend)
  B, T = 2, 3
  calls: list[tuple[int, ...]] = []
  fixed = backend.ones((B, model.action_dim))

  def policy(feats):
    calls.append(tuple(feats.shape))
    return fixed

  rollout(model, init_state=model.initial_state(B), horizon=T,
          mode="prior", policy=policy)
  assert calls == [(B, model.feat_dim)] * T


def test_rollout_burn_in_splits_modes_correctly(backend: Backend):
  model, _ = _build_state_model(backend)
  B, T = 2, 5
  obs     = backend.zeros((B, T, 15))
  actions = backend.zeros((B, T, model.action_dim))
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


def test_rollout_posterior_without_embeds_raises(backend: Backend):
  model, _ = _build_state_model(backend)
  with pytest.raises(ValueError, match="posterior step"):
    rollout(model, init_state=model.initial_state(1), horizon=3,
            mode="posterior", actions=backend.zeros((1, 3, model.action_dim)))


def test_rollout_decode_and_predict_reward_shapes(backend: Backend):
  model, _ = _build_state_model(backend)
  B, T = 2, 3
  traj = rollout(model, init_state=model.initial_state(B), horizon=T,
                 mode="prior", decode=True, predict_reward=True)
  assert tuple(traj.obs_pred.shape) == (B, T, 15)
  assert tuple(traj.rewards.shape)  == (B, T)


def test_rollout_sample_false_is_deterministic(backend: Backend):
  model, _ = _build_state_model(backend)
  B, T = 2, 4
  obs     = backend.zeros((B, T, 15))
  actions = backend.zeros((B, T, model.action_dim))
  embeds  = model.encode(obs)
  init    = model.initial_state(B)

  traj1 = rollout(model, init_state=init, horizon=T, mode="posterior",
                  actions=actions, embeds=embeds, sample=False)
  traj2 = rollout(model, init_state=init, horizon=T, mode="posterior",
                  actions=actions, embeds=embeds, sample=False)
  for s1, s2 in zip(traj1.states, traj2.states):
    assert backend.equal(s1.s, s2.s)
    assert backend.equal(s1.h, s2.h)


def test_rollout_validates_horizon_and_burn_in(backend: Backend):
  model, _ = _build_state_model(backend)
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


def test_eval_split_rollout_shares_anchor_and_emits_two_trajs(backend: Backend):
  model, _ = _build_state_model(backend)
  B, burn_in, horizon = 2, 3, 4
  T = burn_in + horizon
  obs     = backend.zeros((B, T, 15))
  actions = backend.zeros((B, T, model.action_dim))
  embeds  = model.encode(obs)

  out = eval_split_rollout(
    model, embeds=embeds, actions=actions,
    burn_in=burn_in, horizon=horizon, sample=False,
  )
  assert len(out.posterior.states) == horizon
  assert len(out.prior.states)     == horizon
  assert out.posterior.mode == ["posterior"] * horizon
  assert out.prior.mode     == ["prior"]     * horizon
  assert backend.equal(out.posterior.states[0].h, out.prior.states[0].h)


def test_trajectory_stack_helpers_match_per_step_states(backend: Backend):
  model, _ = _build_state_model(backend)
  B, T = 2, 4
  obs     = backend.zeros((B, T, 15))
  actions = backend.zeros((B, T, model.action_dim))
  embeds  = model.encode(obs)
  traj = rollout(model, init_state=model.initial_state(B), horizon=T,
                 mode="posterior", actions=actions, embeds=embeds, sample=False)

  h = traj.stack_h()
  s = traj.stack_s()
  assert tuple(h.shape) == (B, T, model.deter_dim)
  assert tuple(s.shape) == (B, T, model.stoch_dim)
  for t in range(T):
    assert backend.equal(h[:, t], traj.states[t].h)
    assert backend.equal(s[:, t], traj.states[t].s)


def test_trajectory_stack_feat_matches_decode_input(backend: Backend):
  model, _ = _build_state_model(backend)
  B, T = 2, 3
  traj = rollout(model, init_state=model.initial_state(B), horizon=T,
                 mode="prior", decode=True, predict_reward=True, sample=False)
  feats = traj.stack_feat()
  assert tuple(feats.shape) == (B, T, model.feat_dim)
  assert backend.allclose(model.decode(feats), traj.obs_pred)


def test_trajectory_stack_helpers_cache_results(backend: Backend):
  model, _ = _build_state_model(backend)
  traj = rollout(model, init_state=model.initial_state(1), horizon=3,
                 mode="prior", decode=True)
  assert traj.stack_h() is traj.stack_h()
  assert traj.stack_s() is traj.stack_s()
  assert traj.stack_feat() is traj.stack_feat()


def test_eval_split_rollout_with_zero_burn_in_anchors_at_initial_state(backend: Backend):
  model, _ = _build_state_model(backend)
  B, horizon = 2, 4
  obs     = backend.zeros((B, horizon, 15))
  actions = backend.zeros((B, horizon, model.action_dim))
  embeds  = model.encode(obs)
  out = eval_split_rollout(
    model, embeds=embeds, actions=actions,
    burn_in=0, horizon=horizon, sample=False,
  )
  init = model.initial_state(B)
  assert backend.equal(out.anchor.h, init.h)
  assert backend.equal(out.anchor.s, init.s)


# ---------------------------------------------------------------------------
# CEMPolicy — MLX-only for now (the policy hard-codes mx.array)
# ---------------------------------------------------------------------------


def _build_mlx_state_model():
  mx = pytest.importorskip("mlx.core")  # noqa: F841
  cfg = load_config()
  cfg.env.obs_mode = "state"
  cfg.hardware.device = "mlx"
  return build_model(cfg, obs_shape=(15,), action_dim=6)


def test_cem_policy_act_returns_action_in_bounds():
  model = _build_mlx_state_model()
  policy = CEMPolicy(model, horizon=4, num_samples=16, num_elites=4, num_iterations=2)
  policy.reset(scene=None)
  action = policy.act(np.zeros(15, dtype=np.float32))
  assert action.shape == (model.action_dim,)
  assert action.dtype == np.float32
  assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_cem_policy_warm_starts_after_first_step():
  model = _build_mlx_state_model()
  policy = CEMPolicy(model, horizon=3, num_samples=16, num_elites=4, num_iterations=2)
  policy.reset(scene=None)
  obs = np.zeros(15, dtype=np.float32)
  policy.act(obs)
  assert policy._mean is not None
  policy.act(obs)
  assert policy._mean is not None


def test_cem_policy_requires_reset_before_act():
  model = _build_mlx_state_model()
  policy = CEMPolicy(model, horizon=2, num_samples=4, num_elites=2, num_iterations=1)
  with pytest.raises(RuntimeError, match="reset"):
    policy.act(np.zeros(15, dtype=np.float32))
