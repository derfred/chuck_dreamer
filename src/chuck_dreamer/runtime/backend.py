"""The ``RobotBackend`` protocol and the sim-less ``FakeBackend``.

The kernel and control loop talk to a ``RobotBackend``, never to hardware
directly (the keystone decision from the project plan): the whole runtime
is developed against ``FakeBackend`` / ``MujocoBackend`` with zero hardware
in the loop, and the real arm is a swap-in (``FeetechBackend``) at the
milestones that need it.

A backend owns joint-state I/O plus a lifecycle: ``start`` spins up
whatever the implementation needs (a physics thread, a serial bus) and
``stop`` tears it down. ``read_state`` / ``write_positions`` are the hot
path and must be cheap and thread-safe — the control thread calls both
every tick.
"""

from __future__ import annotations

import time
from typing import Any, ContextManager, Protocol, runtime_checkable

import numpy as np

from .control_state import ControlState


@runtime_checkable
class RobotBackend(Protocol):
  """Joint-position I/O behind which any arm (sim or real) hides.

  All position arrays are shape ``(n_joints,)`` float, in the joint order
  the backend was configured with. ``read``/``write`` are called by the
  control thread every tick and must be thread-safe with respect to any
  internal worker (e.g. a MuJoCo physics thread).
  """

  def start(self) -> None:
    """Bring the backend online (open bus / start physics thread)."""
    ...

  def stop(self) -> None:
    """Take the backend offline (close bus / join physics thread)."""
    ...

  def read_state(self, budget_s: float) -> ControlState:
    """Measure the arm within ``budget_s`` seconds, joint positions first.

    The control thread's per-tick entry point: it returns a
    :class:`~chuck_dreamer.runtime.control_state.ControlState` carrying the
    joint vector plus whatever diagnostics the remaining budget paid for. The *loop* supplies the budget (it knows the period and its own
    overhead); the *backend* decides which transactions fit (only it knows
    what they cost).

    ``budget_s`` is a soft bound. A transaction cannot be aborted mid-flight —
    a half-read half-duplex bus is corrupt — so the contract is "do not start
    work that is not expected to finish in time", not a hard deadline. Pass
    ``math.inf`` to read everything (priming at startup, tests).

    Must always return a state with a usable ``q``, even when the budget
    covers nothing: fall back to the cached reading and report the truth in
    ``q_age`` / :class:`~chuck_dreamer.runtime.control_state.FaultFlags`.
    Never block, and never raise on a transient bus error.
    """
    ...

  def last_positions(self) -> np.ndarray:
    """Freshest already-measured positions, shape ``(n_joints,)``, no I/O.

    Never touches the bus: returns the most recent reading taken by
    :meth:`read_state` (or at :meth:`start`). This is the observer path —
    the policy loop samples it so it can never contend with the control
    thread for the bus. For sim backends it reads straight from the model.
    """
    ...

  def write_positions(self, q: np.ndarray) -> None:
    """Command joint position setpoints, shape ``(n_joints,)``.

    Control thread only, like :meth:`read_state`.
    """
    ...

  def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
    """``(lower, upper)`` position bounds, each shape ``(n_joints,)``."""
    ...

  @property
  def home_qpos(self) -> np.ndarray:
    """The arm's rest / reference pose, shape ``(n_joints,)``."""
    ...

  @property
  def n_joints(self) -> int:
    """Number of controlled joints."""
    ...


@runtime_checkable
class ViewableBackend(Protocol):
  """A backend that can drive an interactive viewer (e.g. ``MujocoBackend``).

  Optional capability on top of :class:`RobotBackend`. The runtime narrows to
  this protocol before running the viewer loop; backends without it (Fake,
  Feetech) simply never enter that loop.
  """

  def open_viewer(self) -> ContextManager[Any]:
    """Context-managed viewer handle (must be driven on the main thread)."""
    ...

  def draw_overlays(self, viewer: Any, *, q_cmd: np.ndarray | None = None) -> None:
    """Draw per-frame overlays into ``viewer`` (markers, envelope, …)."""
    ...

  def physics_lock(self) -> Any:
    """Mutex serializing viewer ``sync()`` against the physics step."""
    ...


class InstantReadMixin:
  """``read_state`` for backends whose measurement is free (sim, in-process).

  ``FakeBackend`` and ``MujocoBackend`` read out of memory, so there is no
  budget to respect and nothing to prioritise: every call reads everything it
  has, immediately, with zero age. It is built on :meth:`last_positions` —
  for a sim backend "the cached reading" and "a fresh measurement" are the
  same thing, so there is no second accessor to justify.

  Velocity comes from ``read_velocities`` when the backend has a real one
  (MuJoCo's ``qvel``); the remaining diagnostic fields stay ``None`` because
  a simulated servo has no load, voltage or temperature to report. The
  state's optional fields express exactly that absence, so a consumer written
  against the real arm degrades cleanly instead of reading fabricated numbers.
  """

  def read_state(self, budget_s: float = float("inf")) -> ControlState:
    del budget_s  # in-memory read: nothing to schedule
    now        = time.monotonic()
    velocities = getattr(self, "read_velocities", None)
    t0         = time.perf_counter()
    q          = self.last_positions()
    qd         = velocities() if callable(velocities) else None
    return ControlState(
      now=now, q=q, q_age=0.0, qd=qd, read_s=time.perf_counter() - t0)


class FakeBackend(InstantReadMixin):
  """No-physics backend whose arm perfectly and instantly tracks commands.

  ``write_positions`` stores the command clamped to the backend's own
  joint limits; ``read_state`` returns it back (an *echo* model). The
  arm therefore tracks the commanded setpoint exactly — motion is shaped
  entirely by the control loop's trajectory and safety limiting, which is
  what M0/M1 tests want to assert against. No threads, no clock:
  ``start``/``stop`` are no-ops.
  """

  def __init__(
    self,
    n_joints: int,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    q_init: np.ndarray | None = None,
  ) -> None:
    self._n = int(n_joints)
    self._lower = (
      np.full(self._n, -np.inf) if lower is None
      else np.asarray(lower, dtype=np.float64).copy()
    )
    self._upper = (
      np.full(self._n, np.inf) if upper is None
      else np.asarray(upper, dtype=np.float64).copy()
    )
    if self._lower.shape != (self._n,) or self._upper.shape != (self._n,):
      raise ValueError(
        f"limit shapes must be ({self._n},); got lower={self._lower.shape}, "
        f"upper={self._upper.shape}")
    q0 = self._default_home() if q_init is None else np.asarray(q_init, dtype=np.float64)
    if q0.shape != (self._n,):
      raise ValueError(f"q_init must have shape ({self._n},); got {q0.shape}")
    self._home = np.clip(q0, self._lower, self._upper)
    self._q    = self._home.copy()

  def _default_home(self) -> np.ndarray:
    """Centre of the joint box; zeros on any axis that is unbounded.

    Bounded axes only: an unbounded axis has no centre, and averaging its
    infinities is an invalid operation, so it is masked out before the add.
    """
    bounded      = np.isfinite(self._lower) & np.isfinite(self._upper)
    home         = np.zeros(self._n)
    home[bounded] = 0.5 * (self._lower[bounded] + self._upper[bounded])
    return home

  def start(self) -> None:
    pass

  def stop(self) -> None:
    pass

  def last_positions(self) -> np.ndarray:
    return self._q.copy()

  def write_positions(self, q: np.ndarray) -> None:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (self._n,):
      raise ValueError(f"write_positions expects ({self._n},); got {q.shape}")
    self._q = np.clip(q, self._lower, self._upper)

  def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
    return self._lower.copy(), self._upper.copy()

  @property
  def home_qpos(self) -> np.ndarray:
    return self._home.copy()

  @property
  def n_joints(self) -> int:
    return self._n
