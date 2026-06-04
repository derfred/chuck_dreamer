"""The ``RobotBackend`` protocol and the sim-less ``FakeBackend``.

The kernel and control loop talk to a ``RobotBackend``, never to hardware
directly (the keystone decision from the project plan): the whole runtime
is developed against ``FakeBackend`` / ``MujocoBackend`` with zero hardware
in the loop, and the real arm is a swap-in (``FeetechBackend``) at the
milestones that need it.

A backend owns joint-position I/O plus a lifecycle: ``start`` spins up
whatever the implementation needs (a physics thread, a serial bus) and
``stop`` tears it down. ``read_positions`` / ``write_positions`` are the
hot path and must be cheap and thread-safe — the control thread calls
both every tick.
"""

from __future__ import annotations

from typing import Any, ContextManager, Protocol, runtime_checkable

import numpy as np


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

  def read_positions(self) -> np.ndarray:
    """Latest measured joint positions, shape ``(n_joints,)``."""
    ...

  def write_positions(self, q: np.ndarray) -> None:
    """Command joint position setpoints, shape ``(n_joints,)``."""
    ...

  def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
    """``(lower, upper)`` position bounds, each shape ``(n_joints,)``."""
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


class FakeBackend:
  """No-physics backend whose arm perfectly and instantly tracks commands.

  ``write_positions`` stores the command clamped to the backend's own
  joint limits; ``read_positions`` returns it back (an *echo* model). The
  arm therefore tracks the kernel's commanded setpoint exactly — motion is
  shaped entirely by the kernel's slew/clamp, which is what M0/M1 tests
  want to assert against. No threads, no clock: ``start``/``stop`` are
  no-ops.
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
    q0 = np.zeros(self._n) if q_init is None else np.asarray(q_init, dtype=np.float64)
    self._q = np.clip(q0, self._lower, self._upper)

  def start(self) -> None:
    pass

  def stop(self) -> None:
    pass

  def read_positions(self) -> np.ndarray:
    return self._q.copy()

  def write_positions(self, q: np.ndarray) -> None:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (self._n,):
      raise ValueError(f"write_positions expects ({self._n},); got {q.shape}")
    self._q = np.clip(q, self._lower, self._upper)

  def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
    return self._lower.copy(), self._upper.copy()

  @property
  def n_joints(self) -> int:
    return self._n
