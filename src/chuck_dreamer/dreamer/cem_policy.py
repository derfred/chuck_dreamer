"""Cross-Entropy Method (CEM) planning policy.

Plans by sampling action sequences from a diagonal-Gaussian over the
horizon, scoring each by predicted return under the world model, and
refitting the Gaussian to the top-k. The next action played is the first
step of the refitted mean.

Implements :class:`~chuck_dreamer.policy.Policy` so it's a drop-in
replacement for ``DreamerPolicy`` / ``ScriptedPolicy`` in the collector.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from .world_model import State, WorldModel, rollout


class CEMPolicy:
  """Online planner: at each env step, run CEM in latent space.

  At reset, we initialize a posterior tracker from zero. Each :meth:`act`
  call advances the tracker one posterior step using the previous action
  and the new observation, then plans a fresh action sequence from that
  state. The planned action's first step is what we play.

  Args:
    model:           the world model (any :class:`WorldModel`).
    horizon:         planning horizon (steps to imagine).
    num_samples:     action sequences sampled per CEM iteration.
    num_elites:      top-k retained for refitting.
    num_iterations:  CEM refitting iterations per env step.
    init_std:        std for the initial action distribution.
    min_std:         floor on the refitted std (avoids collapse).
    action_low:      per-dim lower bound for sampled actions.
    action_high:     per-dim upper bound. Actions get clipped before
                     being passed into the model.
    warm_start:      if True, the refitted mean from the previous call
                     seeds this call's mean (shifted one step).
  """

  def __init__(
    self,
    model:          WorldModel,
    *,
    horizon:        int = 12,
    num_samples:    int = 256,
    num_elites:     int = 32,
    num_iterations: int = 4,
    init_std:       float = 1.0,
    min_std:        float = 0.1,
    action_low:     float | np.ndarray = -1.0,
    action_high:    float | np.ndarray =  1.0,
    warm_start:     bool = True,
  ) -> None:
    self.model          = model
    self.horizon        = horizon
    self.num_samples    = num_samples
    self.num_elites     = num_elites
    self.num_iterations = num_iterations
    self.init_std       = init_std
    self.min_std        = min_std
    self.action_low     = np.asarray(action_low,  dtype=np.float32)
    self.action_high    = np.asarray(action_high, dtype=np.float32)
    self.warm_start     = warm_start

    self._tracker_state: State | None = None
    self._prev_action:   np.ndarray | None = None
    self._mean:          np.ndarray | None = None  # (H, action_dim)

  # ------------------------------------------------------------------
  # Policy protocol
  # ------------------------------------------------------------------

  def reset(self, scene) -> None:  # noqa: ARG002 — scene unused for now
    self._tracker_state = self.model.initial_state(1)
    self._prev_action   = np.zeros((self.model.action_dim,), dtype=np.float32)
    self._mean          = None

  def act(self, obs: Any) -> np.ndarray:
    if self._tracker_state is None or self._prev_action is None:
      raise RuntimeError("CEMPolicy.act() called before reset()")

    # 1. Advance the posterior tracker with the previous action and the
    #    new observation. Batch dim is 1 throughout.
    obs_b   = self._batchify(obs)
    embed   = self.model.encode(obs_b)
    act_in  = mx.array(self._prev_action[None])  # (1, A)
    self._tracker_state = self.model.posterior_step(
      self._tracker_state, act_in, embed, sample=False,
    )

    # 2. Plan from the current state. The plan is batched across CEM
    #    samples; we broadcast the tracker state across the sample dim.
    plan_state = self._broadcast_state(self._tracker_state, self.num_samples)
    mean = self._initial_mean()
    std  = np.full_like(mean, self.init_std)

    for _ in range(self.num_iterations):
      samples = self._sample_actions(mean, std)              # (N, H, A) np
      returns = self._score(plan_state, samples)             # (N,)     np
      elites  = samples[np.argsort(-returns)[:self.num_elites]]
      mean    = elites.mean(axis=0)
      std     = np.maximum(elites.std(axis=0), self.min_std)

    # 3. Play the first step of the refitted mean. Cache for warm-start.
    action = np.clip(mean[0], self.action_low, self.action_high).astype(np.float32)
    self._mean        = mean
    self._prev_action = action
    return action

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  def _batchify(self, obs):
    """Add a leading batch axis. Mirrors the notebook's ``_batch_obs``."""
    if isinstance(obs, dict):
      return {k: mx.array(np.asarray(v)[None]) for k, v in obs.items()}
    return mx.array(np.asarray(obs)[None])

  def _broadcast_state(self, state: State, n: int) -> State:
    """Repeat a batch-size-1 state to batch size ``n`` for CEM scoring."""
    from .world_model import Distribution
    def rep(x):
      return mx.broadcast_to(x, (n,) + x.shape[1:])
    return State(
      h         = rep(state.h),
      s         = rep(state.s),
      prior     = Distribution(mean=rep(state.prior.mean), std=rep(state.prior.std)),
      posterior = None if state.posterior is None else Distribution(
        mean=rep(state.posterior.mean), std=rep(state.posterior.std),
      ),
    )

  def _initial_mean(self) -> np.ndarray:
    """First-iteration mean: warm-started from the previous plan, or zero."""
    H, A = self.horizon, self.model.action_dim
    if self.warm_start and self._mean is not None:
      shifted = np.zeros((H, A), dtype=np.float32)
      shifted[:-1] = self._mean[1:]
      return shifted
    return np.zeros((H, A), dtype=np.float32)

  def _sample_actions(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Sample ``(N, H, A)`` from a diagonal Gaussian, clipped to bounds."""
    noise = np.random.randn(self.num_samples, *mean.shape).astype(np.float32)
    samples = mean[None] + std[None] * noise
    return np.clip(samples, self.action_low, self.action_high)

  def _score(self, plan_state: State, samples: np.ndarray) -> np.ndarray:
    """Return ``(N,)`` sums of predicted reward over the horizon."""
    actions_mx = mx.array(samples)  # (N, H, A)
    traj = rollout(
      self.model,
      init_state     = plan_state,
      horizon        = self.horizon,
      mode           = "prior",
      actions        = actions_mx,
      sample         = False,
      predict_reward = True,
    )
    rewards = np.asarray(traj.rewards)  # (N, H)
    return rewards.sum(axis=-1)
