from __future__ import annotations

import threading

from ..policy import Action


class SetpointChannel:
  """Thread-safe single-slot latest-wins channel of policy actions."""

  def __init__(self) -> None:
    self._lock                  = threading.Lock()
    self._action: Action | None = None

  def publish(self, action: Action) -> None:
    """Overwrite the slot with ``action``."""
    if not isinstance(action, Action):
      raise TypeError(
        f"SetpointChannel carries Action objects; got {type(action).__name__}. "
        "A policy returns Action(obs, q=...) rather than a bare joint vector.")
    with self._lock:
      self._action = action

  def get(self) -> Action | None:
    """Most recent action, or ``None`` before any publish."""
    with self._lock:
      return self._action

  @property
  def seq(self) -> int:
    """``seq`` of the pending action; ``0`` before any publish."""
    with self._lock:
      return 0 if self._action is None else self._action.seq
