"""Controller wrapper around a real-arm policy.

The controller owns the lifecycle of a :class:`RealPolicy`:

  * ``"idle"``    — policy is instantiated but inactive. ``step()`` is a
                    no-op (the arm holds its current pose). This is the
                    state we boot in so the operator can verify the
                    camera feed before motion starts.
  * ``"running"`` — policy is producing actions; ``step()`` queries it
                    and returns the commanded action.
  * ``"stopped"`` — terminal; the runner shuts down on next tick.

State transitions are driven by :meth:`start` and :meth:`stop`; the
runner calls them from the cv2 main thread in response to key presses.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .policies import RealPolicy


State = Literal["idle", "running", "stopped"]


class Controller:
  """Owns a :class:`RealPolicy`; exposes ``step()`` and lifecycle methods."""

  def __init__(self, policy: RealPolicy) -> None:
    self.policy = policy
    self.state:  State = "idle"
    self._started = False

  def start(self, initial_obs: dict) -> None:
    """Start the policy. The first call resets the underlying policy.

    Subsequent ``start()`` calls while running are no-ops so a stray
    keypress doesn't reset latent state mid-run.
    """
    if self.state == "running":
      return
    if not self._started:
      self.policy.reset(initial_obs)
      self._started = True
    self.state = "running"

  def stop(self) -> None:
    self.state = "stopped"

  def step(self, obs: dict) -> np.ndarray | None:
    """Run one policy step.

    Returns the commanded action when the policy is running. While idle
    or stopped, returns ``None`` so the runner can hold the arm in place.
    """
    if self.state == "running":
      return np.asarray(self.policy.act(obs), dtype=np.float32)
    return None
