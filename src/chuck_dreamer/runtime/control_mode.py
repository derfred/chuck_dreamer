"""The control loop's mode state machine.

The trajectory-era replacement for the old ``ControlKernel`` mode enum: a
small, self-contained machine the control loop advances once per tick. It
owns *what mode the loop is in* and nothing else — no trajectory, no backend,
no clock — so the transition table is testable without threads.

Two request paths lead out of NORMAL:

* ``request_estop()`` — an immediate latched freeze. The loop stops commanding
  motion at once; nothing leaves ESTOP until ``release()``.
* ``request_brake()`` — a graceful stop. The loop plans a coast-to-stop
  segment (BRAKING) and reports BRAKED once that segment has played out.

Both are latched: they persist until explicitly released, so a producer that
has not noticed the request cannot drive the arm out of a stopped state.

::

    NORMAL   --request_estop--> ESTOP    --release--> NORMAL
    NORMAL   --request_brake--> BRAKING  --braked---> BRAKED --release--> NORMAL
    BRAKING  --request_estop--> ESTOP           (escalation always wins)
    BRAKED   --request_estop--> ESTOP

``wait_for`` lets a caller block until the loop reaches a mode — it is how the
loop's owner observes that a requested stop actually landed, rather than
sleeping for a guessed duration.
"""

from __future__ import annotations

import threading
from enum import Enum


class ControlMode(Enum):
  """What the control loop is currently doing."""
  NORMAL  = "normal"    # consuming actions, tracking the trajectory
  BRAKING = "braking"   # coast-to-stop segment in flight
  BRAKED  = "braked"    # coast-down finished; holding position
  ESTOP   = "estop"     # latched freeze; no motion commanded


# Requests are latched, so the machine only ever moves along these edges.
_TRANSITIONS: dict[ControlMode, frozenset[ControlMode]] = {
  ControlMode.NORMAL:  frozenset({ControlMode.BRAKING, ControlMode.ESTOP}),
  ControlMode.BRAKING: frozenset({ControlMode.BRAKED, ControlMode.ESTOP}),
  ControlMode.BRAKED:  frozenset({ControlMode.NORMAL, ControlMode.ESTOP}),
  ControlMode.ESTOP:   frozenset({ControlMode.NORMAL}),
}


class InvalidTransition(RuntimeError):
  """An illegal mode change was attempted (e.g. NORMAL straight to BRAKED)."""


class ControlModeMachine:
  """Thread-safe mode state for the control loop.

  Requests come from any thread; the transitions that *advance* the machine
  (``braked``) are driven by the control thread as its trajectory progresses.
  Every mutation is under one lock and notifies waiters, so ``wait_for`` never
  misses an edge that happened between the check and the wait.
  """

  def __init__(self, mode: ControlMode = ControlMode.NORMAL) -> None:
    self._mode = mode
    self._cond = threading.Condition(threading.Lock())

  @property
  def mode(self) -> ControlMode:
    with self._cond:
      return self._mode

  # -- transitions ----------------------------------------------------------

  def _to(self, target: ControlMode) -> ControlMode:
    """Move to ``target``; a no-op if already there. Returns the new mode."""
    with self._cond:
      if self._mode is target:
        return self._mode
      if target not in _TRANSITIONS[self._mode]:
        raise InvalidTransition(f"cannot go from {self._mode.value} to {target.value}")
      self._mode = target
      self._cond.notify_all()
      return self._mode

  def request_estop(self) -> ControlMode:
    """Latch an immediate freeze. Legal from every mode; never refused —
    escalating to a hard stop must not depend on what the loop was doing."""
    return self._to(ControlMode.ESTOP)

  def request_brake(self) -> ControlMode:
    """Ask for a graceful coast-to-stop.

    Ignored once the arm is already stopping or stopped (BRAKING/BRAKED are
    idempotent) and refused under ESTOP: a latched freeze outranks a graceful
    stop, and silently downgrading it would command motion on an e-stopped arm.
    """
    with self._cond:
      if self._mode in (ControlMode.BRAKING, ControlMode.BRAKED, ControlMode.ESTOP):
        return self._mode
    return self._to(ControlMode.BRAKING)

  def braked(self) -> ControlMode:
    """Control thread: the coast-to-stop segment has finished.

    Only meaningful while BRAKING; ignored otherwise so a late call cannot
    pull an e-stopped arm back into a merely-braked state.
    """
    with self._cond:
      if self._mode is not ControlMode.BRAKING:
        return self._mode
    return self._to(ControlMode.BRAKED)

  def release(self) -> ControlMode:
    """Clear a latched stop and resume normal operation.

    Refused while BRAKING — the arm is still moving, and resuming mid-coast
    would hand a partly-stopped trajectory back to the policy. Wait for BRAKED.
    """
    with self._cond:
      if self._mode is ControlMode.NORMAL:
        return self._mode
      if self._mode is ControlMode.BRAKING:
        raise InvalidTransition("cannot release while braking; wait for the coast-down to finish")
    return self._to(ControlMode.NORMAL)

  # -- observation ----------------------------------------------------------

  def wait_for(self, mode: ControlMode, timeout: float | None = None) -> bool:
    """Block until the machine is in ``mode``; return whether it got there.

    Returns immediately when already in ``mode``. A ``False`` return means the
    timeout elapsed first — the caller decides whether that is fatal.
    """
    with self._cond:
      return bool(self._cond.wait_for(lambda: self._mode is mode, timeout))
