"""Cross-Entropy Method (CEM) planning policy.

Plans by sampling action sequences from a diagonal-Gaussian over the
horizon, scoring each by predicted return under the world model, and
refitting the Gaussian to the top-k. The next action played is the first
step of the refitted mean.

Implements :class:`~chuck_dreamer.policy.Policy` so it's a drop-in
replacement for ``DreamerPolicy`` / ``ScriptedPolicy`` in the collector.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np

from ..training.observation import Observation
from .world_model import State, WorldModel, as_numpy, rollout


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
    discount:        per-step discount used when summing predicted
                     rewards. ``1.0`` is the unbiased sum; ``<1`` damps
                     speculative late-horizon reward.
    action_low:      per-dim lower bound for sampled actions. Pass
                     ``env.action_space.low`` so samples stay inside the
                     env's actual action box (the in-tree default ±1
                     was wrong for both ``ee`` and ``joint`` modes).
    action_high:     per-dim upper bound. Actions are clipped to this
                     box before being fed to the model and before being
                     returned to the env.
    warm_start:      if True, the refitted mean from the previous call
                     seeds this call's mean (shifted one step).
    ee_max_delta:    if set and action_dim == 7, project sampled EE pose
                     actions so each imagined step moves at most this
                     many meters per xyz coordinate from the previous
                     imagined pose.
    ee_freeze_quat:  if True in EE projection mode, keep quaternion fixed
                     to the last executed/seeded EE orientation.
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
    discount:       float = 1.0,
    action_low:     float | np.ndarray = -1.0,
    action_high:    float | np.ndarray = 1.0,
    warm_start:     bool = True,
    ee_max_delta:   float | None = None,
    ee_freeze_quat: bool = True,
  ) -> None:
    if horizon < 1:
      raise ValueError(f"horizon must be >= 1, got {horizon}")
    if num_samples < 1:
      raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if num_elites < 1:
      raise ValueError(f"num_elites must be >= 1, got {num_elites}")
    if num_elites > num_samples:
      raise ValueError(
        f"num_elites must be <= num_samples, got {num_elites} > {num_samples}"
      )
    if num_iterations < 1:
      raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
    if init_std <= 0:
      raise ValueError(f"init_std must be > 0, got {init_std}")
    if min_std < 0:
      raise ValueError(f"min_std must be >= 0, got {min_std}")
    if not (0.0 < discount <= 1.0):
      raise ValueError(f"discount must be in (0, 1], got {discount}")
    if ee_max_delta is not None and ee_max_delta <= 0:
      raise ValueError(f"ee_max_delta must be > 0 when set, got {ee_max_delta}")

    self.model          = model
    self.horizon        = horizon
    self.num_samples    = num_samples
    self.num_elites     = num_elites
    self.num_iterations = num_iterations
    self.init_std       = init_std
    self.min_std        = min_std
    self.discount       = discount
    self.warm_start     = warm_start
    self.ee_max_delta   = ee_max_delta
    self.ee_freeze_quat = ee_freeze_quat

    low  = np.broadcast_to(
      np.asarray(action_low,  dtype=np.float32),
      (self.model.action_dim,),
    ).astype(np.float32, copy=False)
    high = np.broadcast_to(
      np.asarray(action_high, dtype=np.float32),
      (self.model.action_dim,),
    ).astype(np.float32, copy=False)
    if np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)):
      raise ValueError(
        f"CEMPolicy requires finite action bounds; got low={low}, high={high}"
      )
    if np.any(low >= high):
      raise ValueError(
        f"Each action_low must be strictly less than action_high; got low={low}, high={high}"
      )
    self.action_low  = low
    self.action_high = high

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

  def seed_action(self, action: np.ndarray) -> None:
    """Seed CEM with a known valid action.

    Useful for absolute action spaces such as EE pose actions, where a
    zero action is not a valid or reachable pose. Call this after reset().
    """
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (self.model.action_dim,):
      raise ValueError(
        f"seed action must have shape {(self.model.action_dim,)}, got {action.shape}"
      )

    action = np.clip(action, self.action_low, self.action_high).astype(np.float32)
    self._prev_action = action
    self._mean = np.tile(action[None, :], (self.horizon, 1)).astype(np.float32)

  def act(self, obs: Any) -> np.ndarray:
    if self._tracker_state is None or self._prev_action is None:
      raise RuntimeError("CEMPolicy.act() called before reset()")

    # 1. Advance the posterior tracker with the previous action and the
    #    new observation. Batch dim is 1 throughout. Inputs stay numpy and
    #    the model lifts them via its native ``coerce`` (mlx → mx.array,
    #    torch → tensor on-device). Wrapped in no_grad on the torch
    #    backend — CEM is inference-only and gradient graphs are a
    #    measurable cost per step.
    obs_b = self._batchify(obs)
    with self._inference_context():
      embed  = self.model.encode(obs_b)
      act_in = self.model.coerce(self._prev_action[None])  # (1, A)
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
    action: np.ndarray = np.clip(mean[0], self.action_low, self.action_high).astype(np.float32)
    self._mean        = mean
    self._prev_action = action
    return action

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  def _inference_context(self):
    """``torch.no_grad()`` on torch; ``nullcontext`` on mlx / no-torch.

    CEM never backprops. Building gradient graphs through ``encode``,
    ``posterior_step``, and ``rollout`` on every step is pure overhead
    when the model params have ``requires_grad=True`` (which they do —
    the trainer leaves them on after load).
    """
    try:
      import torch  # type: ignore[import-not-found]
    except ImportError:
      return nullcontext()
    return torch.no_grad()

  def _batchify(self, obs):
    """One model-side obs → (1, ...) batched model-side obs.

    Resizes the image to the model's expected input size when present —
    the trainer's loader does the same resize via
    :meth:`Observation.resize_image`, so single-step inference must
    match or the encoder gets the wrong spatial shape.
    """
    mode = getattr(self.model, "obs_mode", None)
    if mode is None:
      # Backend lacks an obs_mode attribute — fall back to a literal
      # dict-walk so older test models still work.
      if isinstance(obs, dict):
        return {k: np.asarray(v)[None] for k, v in obs.items()}
      return np.asarray(obs)[None]
    wrapped = Observation.from_buffer_value(obs, mode)
    image_size = getattr(self.model, "image_size", None)
    if image_size is not None:
      wrapped = wrapped.resize_image(int(image_size))
    return wrapped.add_batch_axis().to_buffer_value()

  def _broadcast_state(self, state: State, n: int) -> State:
    """Repeat a batch-size-1 state to batch size ``n`` for CEM scoring.

    Uses the backend's ``broadcast_to`` so the operation stays in the
    model's native tensor type (mlx.array or torch.Tensor).
    """
    from .world_model import Distribution

    def rep(x):
      return self.model.broadcast_to(x, (n,) + tuple(x.shape[1:]))
    return State(
      h=rep(state.h),
      s=rep(state.s),
      prior=Distribution(mean=rep(state.prior.mean), std=rep(state.prior.std)),
      posterior=None if state.posterior is None else Distribution(
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

  def _project_ee_samples(self, samples: np.ndarray) -> np.ndarray:
    """Project sampled absolute EE poses into small reachable steps.

    EE actions are absolute poses:
      [x, y, z, qw, qx, qy, qz]

    Direct Gaussian sampling in this 7D box can produce poses that are
    inside action bounds but unreachable by IK. This projection keeps
    each imagined xyz step close to the previous imagined pose, and
    optionally freezes the quaternion to the last executed/seeded pose.
    """
    if self.ee_max_delta is None:
      return samples
    if self.model.action_dim != 7:
      return samples
    if self._prev_action is None:
      return samples

    samples = samples.copy()

    max_delta = float(self.ee_max_delta)
    prev_pos = np.broadcast_to(
      self._prev_action[:3].astype(np.float32),
      (samples.shape[0], 3),
    ).copy()

    for t in range(samples.shape[1]):
      raw_pos = samples[:, t, :3]
      delta = np.clip(raw_pos - prev_pos, -max_delta, max_delta)
      safe_pos = prev_pos + delta
      samples[:, t, :3] = safe_pos
      prev_pos = safe_pos

    if self.ee_freeze_quat:
      samples[:, :, 3:7] = self._prev_action[3:7].astype(np.float32)
    else:
      quat = samples[:, :, 3:7]
      norm = np.linalg.norm(quat, axis=-1, keepdims=True)
      samples[:, :, 3:7] = quat / np.maximum(norm, 1e-6)

    return np.clip(
      samples,
      self.action_low[None, None, :],
      self.action_high[None, None, :],
    ).astype(np.float32, copy=False)

  def _sample_actions(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Sample ``(N, H, A)`` from a diagonal Gaussian, clipped to bounds."""
    noise = np.random.randn(self.num_samples, *mean.shape).astype(np.float32)
    samples = mean[None] + std[None] * noise
    samples = np.clip(
      samples,
      self.action_low[None, None, :],
      self.action_high[None, None, :],
    ).astype(np.float32, copy=False)
    return self._project_ee_samples(samples)

  def _score(self, plan_state: State, samples: np.ndarray) -> np.ndarray:
    """Return ``(N,)`` discounted predicted returns over the horizon.

    Numpy ``samples`` are passed straight through — :func:`rollout`
    coerces them into the model's native tensor type via ``model.coerce``.
    """
    with self._inference_context():
      traj = rollout(
        self.model,
        init_state=plan_state,
        horizon=self.horizon,
        mode="prior",
        actions=samples,  # (N, H, A) numpy — rollout() coerces it
        sample=False,
        predict_reward=True,
      )
    if traj.rewards is None:
      raise RuntimeError(
        "CEMPolicy expects rollout(..., predict_reward=True) to return rewards"
      )
    # ``as_numpy`` handles both mlx.array (numpy-compatible) and
    # torch.Tensor (needs ``detach().cpu()`` since the model params have
    # ``requires_grad=True``).
    rewards = as_numpy(traj.rewards).astype(np.float32, copy=False)  # (N, H)
    if self.discount == 1.0:
      return rewards.sum(axis=-1)
    discounts = (self.discount ** np.arange(self.horizon, dtype=np.float32))[None, :]
    return (rewards * discounts).sum(axis=-1)
