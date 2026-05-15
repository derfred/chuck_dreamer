"""Backend-agnostic world-model interface.

The Dreamer world model exposes three primitives — encode an observation,
advance the latent state under the posterior (teacher-forced) or the prior
(open-loop) — and three things you build on top of them: eval rollouts that
split posterior vs prior after a burn-in, on-policy imagination for actor
training, and planning algorithms like CEM.

This module defines:
  - :class:`Distribution`        — a diagonal-Gaussian record (mean, std).
  - :class:`State`               — the RSSM latent state at one timestep.
  - :class:`Trajectory`          — the result of a rollout.
  - :class:`WorldModel` protocol — what every backend (MLX, torch, …) must
                                   implement to be a drop-in.
  - :func:`rollout`              — the one composable driver that covers
                                   all three downstream use cases.

Nothing here imports a backend. The protocol carries arrays as ``Any`` and
each backend documents its native array type. Callers that need numpy do
the conversion at the trajectory level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclass
class Distribution:
  """A diagonal Gaussian over the stochastic latent ``s``.

  Both fields are backend tensors of shape ``(..., stoch_dim)``. They are
  the parameters *before* the squash/min-std post-processing the backend
  applies internally; the backend's :meth:`WorldModel.sample` and
  :meth:`WorldModel.mode` honor that post-processing.
  """

  mean: Any
  std:  Any


@dataclass
class State:
  """RSSM latent state at one timestep.

  ``h`` is the deterministic GRU state; ``s`` is the stochastic sample
  drawn from either the posterior (when an observation was available) or
  the prior (in imagination). ``prior`` is always populated. ``posterior``
  is populated only on posterior steps and is ``None`` after a prior step.

  Backends may stash extra fields under :attr:`extras` (e.g. cached
  embeddings) without breaking the interface.
  """

  h: Any
  s: Any
  prior:     Distribution
  posterior: Distribution | None = None
  extras:    dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
  """The output of :func:`rollout`.

  ``states`` has length ``horizon`` (one per advanced step). ``actions``
  has shape ``(B, horizon, action_dim)``; ``embeds`` is populated when the
  rollout was teacher-forced (posterior). ``obs_pred`` and ``rewards`` are
  populated only when the corresponding flags were set on :func:`rollout`.

  The fields are backend tensors batched along axis 0. ``mode`` records
  which dynamics were used at each timestep so eval code can find the
  posterior/prior boundary without recomputing it.

  ``stacker`` is the callable :func:`rollout` used to stack lists of
  per-step tensors along a new time axis (typically the backend's
  :meth:`stack_along_time`). It powers the lazy :meth:`stack_h`,
  :meth:`stack_s`, :meth:`stack_feat` helpers, which avoid recomputing the
  same stacks twice when post-processing.
  """

  states:   list[State]
  actions:  Any
  mode:     list[Literal["posterior", "prior"]]
  embeds:   Any | None = None
  obs_pred: Any | None = None
  rewards:  Any | None = None
  stacker:  Callable[[list[Any]], Any] | None = None

  _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

  def stack_h(self) -> Any:
    """``(B, T, deter_dim)`` — the deterministic state across the rollout."""
    if "h" not in self._cache:
      self._cache["h"] = self._stack([s.h for s in self.states])
    return self._cache["h"]

  def stack_s(self) -> Any:
    """``(B, T, stoch_dim)`` — the stochastic state across the rollout."""
    if "s" not in self._cache:
      self._cache["s"] = self._stack([s.s for s in self.states])
    return self._cache["s"]

  def stack_feat(self) -> Any:
    """``(B, T, feat_dim)`` — concatenated ``[h, s]`` across the rollout."""
    if "feat" not in self._cache:
      def _feat(state: State) -> Any:
        # Concatenate via numpy so mlx/torch tensors round-trip cleanly
        # when the trajectory is post-processed on CPU.
        return np.concatenate([np.asarray(state.h), np.asarray(state.s)], axis=-1)
      self._cache["feat"] = self._stack([_feat(s) for s in self.states])
    return self._cache["feat"]

  def _stack(self, xs: list[Any]) -> Any:
    if self.stacker is not None:
      return self.stacker(xs)
    return _default_stack(xs)

  # ---- numpy projections for post-processing on CPU ----

  def obs_pred_numpy(self) -> Any:
    """``obs_pred`` converted to numpy. ``None`` if decoding was skipped.

    Preserves the layout of the decoder output — dict obs (e.g.
    image_proprio) come back as a dict of numpy arrays.
    """
    return None if self.obs_pred is None else as_numpy(self.obs_pred)

  def h_numpy(self) -> Any:
    """``(B, T, deter_dim)`` deterministic-state trajectory as numpy."""
    return as_numpy(self.stack_h())

  def s_numpy(self) -> Any:
    """``(B, T, stoch_dim)`` stochastic-state trajectory as numpy."""
    return as_numpy(self.stack_s())


def as_numpy(x: Any) -> Any:
  """Convert a backend tensor (or dict of tensors) to numpy.

  Handles ``torch.Tensor`` (via ``detach().cpu().numpy()``) and anything
  else numpy can ingest (mlx arrays support ``__array__``). Recursive on
  dicts so dict-shaped obs_pred / inputs round-trip cleanly.
  """
  if isinstance(x, dict):
    return {k: as_numpy(v) for k, v in x.items()}
  if hasattr(x, "detach"):  # torch.Tensor
    return x.detach().cpu().numpy()
  return np.asarray(x)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class WorldModel(Protocol):
  """The minimum surface a backend must implement to drive :func:`rollout`.

  All array-typed args/returns are the backend's native tensor type
  (``mlx.core.array``, ``torch.Tensor``, …). Shapes are documented in
  comments; the protocol does not enforce them.

  The protocol is intentionally small. Trainer-specific concerns
  (optimizer state, save/load, loss computation) live on the concrete
  backend class, not here — they don't compose into rollouts.
  """

  action_dim: int
  stoch_dim:  int
  deter_dim:  int

  def initial_state(self, batch_size: int) -> State:
    """Zero-initialized state, broadcast to ``batch_size``."""
    ...

  def encode(self, obs: Any) -> Any:
    """Embed observations. ``obs`` is ``(B, ...)`` or ``(B, T, ...)``;
    the return matches that with the trailing dims replaced by
    ``(embed_dim,)``. Dict observations are passed through as-is.

    Implementations must accept numpy arrays in addition to their native
    tensor type, coercing them to the backend's native representation
    before running the encoder. This is the boundary at which loaders
    (eval data, replay-buffer dicts) hand off to backend ops without
    callers having to dispatch on ``hardware.device``.
    """
    ...

  def posterior_step(
    self,
    state:  State,
    action: Any,            # (B, action_dim)
    embed:  Any,            # (B, embed_dim) — embedding of obs AFTER action
    *,
    sample: bool = True,
  ) -> State:
    """Advance one step using the posterior. With ``sample=False``,
    ``state.s`` is the posterior mean; otherwise it's a reparameterized
    sample."""
    ...

  def prior_step(
    self,
    state:  State,
    action: Any,            # (B, action_dim)
    *,
    sample: bool = True,
  ) -> State:
    """Advance one step using the prior only (no observation). With
    ``sample=False``, ``state.s`` is the prior mean."""
    ...

  def feat(self, state: State) -> Any:
    """Concatenate ``[h, s]`` — the feature consumed by decoder / reward /
    actor / critic. Shape ``(B, deter_dim + stoch_dim)``."""
    ...

  # ---- optional decode-side heads. Implementations always provide them;
  # callers may skip them via the rollout() flags to save compute. ----

  def decode(self, feats: Any) -> Any:
    """Reconstruct observations from features. Input ``(B, T, feat_dim)``
    or ``(B, feat_dim)``; output mirrors the encoder's input layout."""
    ...

  def predict_reward(self, feats: Any) -> Any:
    """Predict scalar reward from features. Input ``(..., feat_dim)`` →
    output ``(...,)``."""
    ...

  def actor_action(self, feats: Any, *, sample: bool = True) -> Any:
    """Action from the model's actor. With ``sample=False`` returns the
    deterministic mode (post-squash)."""
    ...

  # ---- backend coercion / shape ops used by backend-agnostic callers
  # (CEM, evaluator, rollout). Optional — :func:`rollout` only requires
  # ``coerce`` when callers pass numpy inputs.

  def coerce(self, x: Any) -> Any:
    """Lift numpy (or already-native tensors) into the backend's native
    tensor type. Recursive on dicts."""
    ...

  def broadcast_to(self, x: Any, shape: tuple[int, ...]) -> Any:
    """Broadcast a tensor to ``shape``. Used by CEM to expand a B=1 state
    across CEM samples without leaving the backend's tensor type."""
    ...


# A policy is anything that turns a feature batch into an action batch.
# The trainer's neural actor satisfies this via ``actor_action``; CEM
# satisfies it by running a nested ``rollout`` of its own.
PolicyFn = Callable[[Any], Any]


# ---------------------------------------------------------------------------
# rollout()
# ---------------------------------------------------------------------------


Mode = Literal["posterior", "prior"]


def rollout(
  model:          WorldModel,
  *,
  init_state:     State,
  horizon:        int,
  mode:           Mode = "prior",
  burn_in:        int = 0,
  actions:        Any | None = None,
  embeds:         Any | None = None,
  policy:         PolicyFn | None = None,
  sample:         bool = True,
  decode:         bool = False,
  predict_reward: bool = False,
) -> Trajectory:
  """Run one rollout under the world model.

  Step ``t`` (0-indexed) is "posterior" when ``t < burn_in`` or ``mode ==
  "posterior"``; "prior" otherwise. Burn-in lets eval rollouts share an
  init_state with the buffered embeds for the first few steps, then
  continue with the prior (open-loop) — call this function twice from the
  same anchor state to compare both continuations against ground truth.

  Action sources, in priority order:
    1. ``actions``: explicit ``(B, horizon, action_dim)`` tensor. The
       caller drives action selection (eval replay, CEM action samples).
    2. ``policy``: a ``(B, feat_dim) -> (B, action_dim)`` callable invoked
       on the current state's feature each step.
    3. The model's actor (``model.actor_action``).

  Embeddings are required for every posterior step:
    - If ``embeds`` is given (``(B, horizon, embed_dim)``), step ``t`` uses
      ``embeds[:, t]`` whenever step ``t`` is a posterior step.
    - Otherwise, a posterior step is illegal — we raise.

  ``sample`` propagates to both posterior and prior steps and to the
  policy (when the actor is used). With ``sample=False`` the rollout is
  deterministic given the inputs.

  ``decode`` and ``predict_reward`` are off by default — they're the
  expensive heads and most callers don't need them every step. When on,
  they run once after the dynamics loop on the stacked feature trajectory.
  """
  if horizon < 1:
    raise ValueError(f"horizon must be >= 1; got {horizon}")
  if burn_in < 0:
    raise ValueError(f"burn_in must be >= 0; got {burn_in}")
  if burn_in > horizon:
    raise ValueError(f"burn_in ({burn_in}) cannot exceed horizon ({horizon})")

  # Allow numpy inputs at the rollout boundary — callers like the
  # Evaluator hand us numpy arrays from disk. Backends expose ``coerce``
  # to lift them into their native tensor type; missing means "already
  # native" (the trainer path).
  coerce = getattr(model, "coerce", None)
  if coerce is not None:
    if actions is not None:
      actions = coerce(actions)
    if embeds is not None:
      embeds = coerce(embeds)

  state  = init_state
  states: list[State] = []
  modes:  list[Mode]  = []
  acts:   list[Any]   = []

  for t in range(horizon):
    step_mode: Mode = "posterior" if (t < burn_in or mode == "posterior") else "prior"

    if actions is not None:
      action_t = actions[:, t]
    elif policy is not None:
      action_t = policy(model.feat(state))
    else:
      action_t = model.actor_action(model.feat(state), sample=sample)

    if step_mode == "posterior":
      if embeds is None:
        raise ValueError(
          f"posterior step at t={t} requires `embeds`; pass embeds=... or "
          f"use mode='prior' / burn_in=0."
        )
      state = model.posterior_step(state, action_t, embeds[:, t], sample=sample)
    else:
      state = model.prior_step(state, action_t, sample=sample)

    states.append(state)
    modes.append(step_mode)
    acts.append(action_t)

  # Stack actions back into (B, T, A). Backends use different stack
  # primitives; defer to the model.
  stacker     = getattr(model, "stack_along_time", _default_stack)
  actions_out = stacker(acts)

  obs_pred = None
  rewards  = None
  feats_cached = None
  if decode or predict_reward:
    feats_cached = stacker([model.feat(s) for s in states])
    if decode:
      obs_pred = model.decode(feats_cached)
    if predict_reward:
      rewards = model.predict_reward(feats_cached)

  traj = Trajectory(
    states   = states,
    actions  = actions_out,
    mode     = modes,
    embeds   = embeds,
    obs_pred = obs_pred,
    rewards  = rewards,
    stacker  = stacker,
  )
  # If we already computed the stacked feature for decoding, cache it on
  # the trajectory so post-processing doesn't pay for it again.
  if feats_cached is not None:
    traj._cache["feat"] = feats_cached
  return traj


def _default_stack(xs: list[Any]) -> Any:
  """Fallback stack used when the backend doesn't provide one.

  Falls back to numpy. Backends should provide :meth:`stack_along_time`
  to keep everything in their native tensor type.
  """
  return np.stack([np.asarray(x) for x in xs], axis=1)


# ---------------------------------------------------------------------------
# Eval helper: burn-in posterior, then split into posterior/prior tails.
# ---------------------------------------------------------------------------


@dataclass
class EvalRollout:
  """Two continuations from a shared posterior burn-in.

  ``anchor`` is the latent state after the last burn-in step. ``posterior``
  continues teacher-forced using the same actions and embeddings;
  ``prior`` continues open-loop. Compare them against ground-truth obs to
  measure how fast the prior drifts from the posterior under the same
  action sequence.
  """

  anchor:    State
  posterior: Trajectory
  prior:     Trajectory


def eval_split_rollout(
  model:          WorldModel,
  *,
  embeds:         Any,        # (B, burn_in + horizon, embed_dim)
  actions:        Any,        # (B, burn_in + horizon, action_dim)
  burn_in:        int,
  horizon:        int,
  init_state:     State | None = None,
  sample:         bool = False,
  decode:         bool = True,
  predict_reward: bool = True,
) -> EvalRollout:
  """Run the eval pattern: burn-in posterior, then split into posterior /
  prior continuations.

  The two continuations share the anchor state produced by the burn-in.
  They consume the same action sequence (``actions[:, burn_in:]``); the
  posterior tail additionally uses ``embeds[:, burn_in:]`` while the prior
  tail ignores them.

  ``sample`` defaults to ``False`` — eval is normally deterministic so
  repeated runs are comparable.

  ``init_state`` defaults to the model's zero state for batch size
  inferred from ``embeds``. Pass an explicit state when chaining (e.g.
  when burn-in spans the entire episode and you anchor elsewhere).
  """
  if burn_in < 0 or horizon < 1:
    raise ValueError(f"burn_in must be >= 0, horizon >= 1; got {burn_in}, {horizon}")

  # Pre-flight: shape check. Both eyes need the same (B, T_total).
  if actions is None or embeds is None:
    raise ValueError("eval_split_rollout requires both `embeds` and `actions`")

  total = burn_in + horizon
  # Defer detailed shape checks to the backend — different array libs
  # report shape mismatches differently and we don't want to dictate one.

  if init_state is None:
    # Infer batch size from embeds.
    batch_size = int(embeds.shape[0])
    init_state = model.initial_state(batch_size)

  # Step 1: burn-in posterior to produce the anchor. We just run a
  # posterior-only rollout for `burn_in` steps and take the last state.
  if burn_in > 0:
    burn = rollout(
      model,
      init_state = init_state,
      horizon    = burn_in,
      mode       = "posterior",
      actions    = actions[:, :burn_in],
      embeds     = embeds[:, :burn_in],
      sample     = sample,
      decode     = False,
      predict_reward = False,
    )
    anchor = burn.states[-1]
  else:
    anchor = init_state

  tail_actions = actions[:, burn_in:burn_in + horizon]
  tail_embeds  = embeds [:, burn_in:burn_in + horizon]

  posterior = rollout(
    model,
    init_state = anchor,
    horizon    = horizon,
    mode       = "posterior",
    actions    = tail_actions,
    embeds     = tail_embeds,
    sample     = sample,
    decode     = decode,
    predict_reward = predict_reward,
  )
  prior = rollout(
    model,
    init_state = anchor,
    horizon    = horizon,
    mode       = "prior",
    actions    = tail_actions,
    sample     = sample,
    decode     = decode,
    predict_reward = predict_reward,
  )
  return EvalRollout(anchor=anchor, posterior=posterior, prior=prior)
