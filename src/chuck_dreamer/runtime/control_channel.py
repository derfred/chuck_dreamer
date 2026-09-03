from __future__ import annotations

import threading

from ..policy import Action
from .control_state import ControlState


class ControlChannel:
  """Thread-safe latest-wins exchange: actions up, control state down."""

  def __init__(self) -> None:
    self._lock                        = threading.Lock()
    self._action: Action | None       = None
    self._state:  ControlState | None = None

  # -- policy -> control -----------------------------------------------------

  def publish(self, action: Action) -> None:
    """Overwrite the action slot with ``action`` (policy thread)."""
    with self._lock:
      self._action = action

  def get(self) -> Action | None:
    """Most recent action, or ``None`` before any publish (control thread)."""
    with self._lock:
      return self._action

  @property
  def seq(self) -> int:
    """``seq`` of the pending action; ``0`` before any publish."""
    with self._lock:
      return 0 if self._action is None else self._action.seq

  # -- control -> policy -----------------------------------------------------

  def publish_state(self, state: ControlState) -> None:
    with self._lock:
      self._state = state

  def state(self) -> ControlState | None:
    with self._lock:
      return self._state
