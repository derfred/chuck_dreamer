"""Policy protocol + the gated-startup wrapper.

:class:`GatedPolicy` adapts a planner / actor (CEM, Dreamer) that has
no notion of a "ready / hover" phase to interactive use, where the
operator wants to inspect the scene (or position the real arm) before
the policy starts driving the system. While held, ``act()`` emits an
action supplied by the caller (current EE pose in sim, current joint
qpos on the real arm) — whatever keeps the system still without
advancing the inner policy's internal state. On :meth:`release` the
inner policy is reset against the current scene/obs and every
subsequent call is forwarded to it.
"""

from typing import TYPE_CHECKING, Any, Callable, Protocol

import numpy as np

if TYPE_CHECKING:
  from chuck_dreamer.sim.scene_config import SceneConfig


class Policy(Protocol):
  def reset(self, scene: "SceneConfig") -> None: ...
  def act(self, obs: Any) -> np.ndarray: ...


class GatedPolicy:
  """Hold-then-delegate wrapper.

  Caller hands in the env's full obs every step. While held, this
  wrapper packs a hold action from that obs via ``hold_action``. After
  :meth:`release`, the obs is run through ``project`` and forwarded to
  the inner policy — that's where the env's flat raw obs is collapsed
  into the per-component dict the model encoder consumes (see
  :meth:`PushingEnv.policy_obs`).

  The inner policy's ``reset`` is deferred until release, so its
  internal state (CEM tracker, Dreamer posterior) starts from the obs
  the operator actually handed over to it — not the boot-time obs that
  might be from a still-being-positioned arm.

  Args:
    inner:        the underlying policy (e.g. :class:`CEMPolicy`,
                  :class:`DreamerPolicy`). Its ``reset`` runs at
                  :meth:`release`; its ``act`` only after.
    hold_action:  ``obs -> action`` callable that produces the
                  no-op-equivalent action while held. Sim builds one
                  from ``env.act_mode`` (current EE pose or arm qpos);
                  the real arm uses current joint qpos.
    project:      optional ``full_obs -> model_obs`` callable applied
                  before forwarding to the inner policy. Typically
                  ``env.policy_obs``. Default is identity.
  """

  def __init__(
    self,
    inner:       Policy,
    *,
    hold_action: Callable[[Any], np.ndarray],
    project:     Callable[[Any], Any] | None = None,
  ) -> None:
    self.inner       = inner
    self.hold_action = hold_action
    self.project     = project or (lambda obs: obs)
    self.released    = False
    self._reset_arg: Any = None
    self._has_reset_arg  = False

  def reset(self, scene_or_obs: Any) -> None:
    """Stash the reset argument; defer the inner reset until release.

    Accepts whatever the inner policy expects — a :class:`SceneConfig`
    for sim policies, an initial-obs dict for real-arm adapters.
    """
    self.released       = False
    self._reset_arg     = scene_or_obs
    self._has_reset_arg = True

  def release(self) -> None:
    if self.released:
      return
    if not self._has_reset_arg:
      raise RuntimeError("GatedPolicy.release() called before reset()")
    self.inner.reset(self._reset_arg)
    self.released = True

  def act(self, obs: Any) -> np.ndarray:
    if self.released:
      return self.inner.act(self.project(obs))
    return self.hold_action(obs)
