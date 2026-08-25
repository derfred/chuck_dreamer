"""Policy protocol, the action a policy returns, and the gated-startup wrapper.

:class:`Action` is what every policy hands back. It carries the setpoint, the
observation it was computed from, and a unique ``seq`` -- the observation
because the runtime's trajectory planner measures the policy's own interval
from ``action.obs.t``, and the ``seq`` because a consumer must distinguish "a
new command" from "the same one re-read" without comparing float arrays.

An action names its target in exactly one space: joint positions, or a
Cartesian point in the arm-base or table/world frame (optionally with an
orientation). Resolving a Cartesian target to joints is inverse kinematics and
happens at the consumer, behind :meth:`Action.resolve`.

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

import itertools
import threading
from typing import TYPE_CHECKING, Any, Callable, Protocol

import numpy as np

from chuck_dreamer.common.tracks import Frame

if TYPE_CHECKING:
  from chuck_dreamer.sim.scene_config import SceneConfig


class ActionSpaceError(ValueError):
  """An action named no target space, or more than one."""


# Process-wide monotonic action ids. The counter is shared rather than per-policy
# so two producers can never mint the same seq, and `itertools.count` is atomic
# under CPython's GIL -- the lock makes that independent of the interpreter.
_seq_counter = itertools.count(1)
_seq_lock    = threading.Lock()


def _next_seq() -> int:
  with _seq_lock:
    return next(_seq_counter)


class Action:
  """One setpoint from a policy, tagged with its observation and a unique id.

  Exactly one target must be given::

      Action(obs, q=np.zeros(6))              # joint space, rad
      Action(obs, arm_pos=[0.2, 0.0, 0.15])   # arm-base frame, m
      Action(obs, world_pos=[0.1, 0.3, 0.0])  # table/world frame, m

  ``seq`` is assigned at construction and never repeats, so the control loop
  compares actions by identity instead of by value: two actions with the same
  joint vector are still distinct commands. Actions are immutable.

  ``obs`` is whatever the producing loop hands its policy -- a
  ``RuntimeObservation`` in the runtime, a component dict in sim -- and is kept
  only so a consumer can relate the action back to what it answers (the
  trajectory planner reads ``obs.t``). It is deliberately not narrowed here.
  """

  __slots__ = ("obs", "seq", "_q", "_pos", "_quat", "_frame")

  def __init__(
    self,
    obs: Any,
    *,
    q=None,
    arm_pos=None,
    world_pos=None,
    quat=None,
    seq: int | None = None,
  ) -> None:
    given = [(name, value) for name, value in
             (("q", q), ("arm_pos", arm_pos), ("world_pos", world_pos))
             if value is not None]
    if not given:
      raise ActionSpaceError("an Action needs one of q=, arm_pos= or world_pos=")
    if len(given) > 1:
      raise ActionSpaceError(
        f"an Action names exactly one target space; got {', '.join(n for n, _ in given)}")

    name, value = given[0]
    vec         = np.asarray(value, dtype=np.float64).copy()
    if vec.ndim != 1:
      raise ActionSpaceError(f"{name} must be a 1-D vector; got shape {vec.shape}")
    if name != "q" and vec.shape != (3,):
      raise ActionSpaceError(f"{name} must be a 3-D point; got shape {vec.shape}")
    vec.flags.writeable = False

    if quat is not None and name == "q":
      raise ActionSpaceError("quat= only applies to a Cartesian target")
    if quat is None:
      orientation = None
    else:
      orientation = np.asarray(quat, dtype=np.float64).copy()
      if orientation.shape != (4,):
        raise ActionSpaceError(
          f"quat must be [w, x, y, z]; got shape {orientation.shape}")
      orientation.flags.writeable = False

    self.obs    = obs
    self.seq    = _next_seq() if seq is None else int(seq)
    self._q     = vec if name == "q" else None
    self._pos   = None if name == "q" else vec
    self._quat  = orientation
    self._frame = (Frame.NONE if name == "q" else Frame.ARM if name == "arm_pos" else Frame.TABLE)

  # -- target ----------------------------------------------------------------

  @property
  def frame(self) -> Frame:
    """The frame the target is expressed in; ``Frame.NONE`` for joint space."""
    return self._frame

  @property
  def is_joint_space(self) -> bool:
    return self._q is not None

  @property
  def pos(self) -> np.ndarray | None:
    """The Cartesian target (m) in :attr:`frame`, or ``None`` in joint space."""
    return self._pos

  @property
  def quat(self) -> np.ndarray | None:
    """Target orientation ``[w, x, y, z]``, or ``None`` if unconstrained.

    Optional on a Cartesian action: a pushing task often cares only about
    where the tool goes, while the sim's ``act_mode="ee"`` commands a full
    6-DoF pose. ``None`` means "any orientation that reaches :attr:`pos`".
    """
    return self._quat

  @property
  def pose(self) -> np.ndarray | None:
    """``pos`` and ``quat`` concatenated into the sim's 7-vector EE action."""
    if self._pos is None or self._quat is None:
      return None
    return np.concatenate([self._pos, self._quat])

  @property
  def q(self) -> np.ndarray:
    """The joint target (rad).

    Available directly for a joint-space action. A Cartesian one must be
    resolved through IK first (:meth:`resolve`); reading ``q`` before that is a
    programming error, not a silent fallback -- the control loop commands
    joints and must never be handed a half-specified target.
    """
    if self._q is None:
      raise ActionSpaceError(
        f"this Action targets {self._frame.value}-frame {self._pos.tolist()}; "
        "resolve it through IK before reading q")
    return self._q

  def resolve(self, q: np.ndarray) -> "Action":
    """The joint-space equivalent of this action, keeping ``obs`` and ``seq``.

    The seam IK plugs into: a resolver computes the joint vector for
    :attr:`pos` and hands it back here. ``seq`` is carried over deliberately --
    resolving does not make it a *different* command, so the loop still sees one
    setpoint per policy step.
    """
    return Action(self.obs, q=q, seq=self.seq)

  # -- identity --------------------------------------------------------------

  def __eq__(self, other: Any) -> bool:
    """Actions are compared by id, never by setpoint value.

    Two actions carrying the same joint vector are still distinct commands (a
    policy re-publishing the same pose is issuing it again), and comparing the
    arrays elementwise would raise inside an ``if``.
    """
    return isinstance(other, Action) and self.seq == other.seq

  def __hash__(self) -> int:
    return hash(self.seq)

  def __repr__(self) -> str:
    if self._q is not None:
      target = f"q={self._q.tolist()}"
    else:
      target = f"{self._frame.value}_pos={self._pos.tolist()}"
      if self._quat is not None:
        target += f", quat={self._quat.tolist()}"
    return f"Action(seq={self.seq}, {target})"


class Policy(Protocol):
  """A producer of setpoints.

  ``act`` returns an :class:`Action` on every path -- the runtime's control
  loop and the sim env alike. The action names its own target space, so a
  consumer that commands joints and one that commands an EE pose read the same
  object rather than agreeing out-of-band on what a bare array meant.
  """
  def reset(self, scene: "SceneConfig") -> None: ...
  def act(self, obs: Any) -> "Action": ...


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
    to_action:    ``(obs, raw) -> Action`` adapter for inner policies that
                  return a bare vector rather than an :class:`Action` -- a
                  planner in a learned action space does not know whether its
                  output means joints or an EE pose, but the caller that built
                  the env does. Required only if such a policy is used.
  """

  def __init__(
    self,
    inner:       Policy,
    *,
    hold_action: Callable[[Any], Any],
    project:     Callable[[Any], Any] | None = None,
    to_action:   Callable[[Any, Any], "Action"] | None = None,
  ) -> None:
    self.inner       = inner
    self.hold_action = hold_action
    self.project     = project or (lambda obs: obs)
    self._to_action  = to_action
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

  def to_action(self, obs: Any, raw: Any) -> "Action":
    if self._to_action is None:
      raise TypeError(
        f"{type(self.inner).__name__} returned {type(raw).__name__}, not an "
        "Action. Pass to_action=(obs, raw) -> Action to GatedPolicy so the "
        "raw vector can be named in the space the env commands.")
    return self._to_action(obs, raw)

  def release(self) -> None:
    if self.released:
      return
    if not self._has_reset_arg:
      raise RuntimeError("GatedPolicy.release() called before reset()")
    self.inner.reset(self._reset_arg)
    self.released = True

  def act(self, obs: Any) -> "Action":
    """The inner policy's action once released, else a hold.

    ``to_action`` adapts whatever the inner policy returns: a policy that
    already speaks :class:`Action` passes through, while a planner working in a
    flat learned action space (CEM) hands back a bare vector that only the
    caller knows the layout of. Same for ``hold_action``.
    """
    raw = self.inner.act(self.project(obs)) if self.released else self.hold_action(obs)
    return raw if isinstance(raw, Action) else self.to_action(obs, raw)
