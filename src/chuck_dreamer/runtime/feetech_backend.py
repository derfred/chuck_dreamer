"""``FeetechBackend`` — the real SO-101 arm. STUB; hardware deferred.

The real arm is 7 STS3215 servos on one TTL bus at 115200 baud. The control
thread owns the bus exclusively: a sync write every tick (~2.5 ms) and a
decimated/diagnostic sync read (~7 ms). That driver, and its hardware
validation (the M1 *hardware* exit criteria — bus-timing budget, watchdog
tuned to real Jetson jitter), are out of scope for the current task.

This stub records its construction parameters but does not open the bus;
every I/O method raises :class:`NotImplementedError`. No hardware libraries
(``scservo_sdk`` / lerobot bus) are imported, so importing this module is
always safe. M1-hardware fills the method bodies against the docstringed
budget above.
"""

from __future__ import annotations

import numpy as np

_NOT_IMPLEMENTED = (
  "FeetechBackend is a stub: the real STS3215 bus driver is deferred to the "
  "M1 hardware milestone (sync write ~2.5 ms/tick, decimated read ~7 ms, bus "
  "owned exclusively by the control thread). Use FakeBackend or MujocoBackend."
)


class FeetechBackend:
  """Placeholder for the real SO-101 follower. Raises on any bus operation."""

  def __init__(
    self,
    *,
    port: str,
    baud: int = 115200,
    joint_names: list[str] | None = None,
  ) -> None:
    self._port = port
    self._baud = baud
    self._joint_names = list(joint_names) if joint_names is not None else None

  @property
  def n_joints(self) -> int:
    if self._joint_names is None:
      raise NotImplementedError(_NOT_IMPLEMENTED)
    return len(self._joint_names)

  def start(self) -> None:
    raise NotImplementedError(_NOT_IMPLEMENTED)

  def stop(self) -> None:
    # Safe to call during shutdown even if never started.
    pass

  def read_positions(self) -> np.ndarray:
    raise NotImplementedError(_NOT_IMPLEMENTED)

  def write_positions(self, q: np.ndarray) -> None:
    raise NotImplementedError(_NOT_IMPLEMENTED)

  def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError(_NOT_IMPLEMENTED)
