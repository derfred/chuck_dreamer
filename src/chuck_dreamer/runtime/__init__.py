"""SO-101 pushing runtime: control kernel, backends, loops, telemetry.

Boots from the project's OmegaConf session config (the ``runtime:`` block
of ``configs/default.yaml``), spins a policy loop and a control loop against
a :class:`RobotBackend`, and shuts down cleanly. The control kernel talks to
a backend protocol, never to hardware directly, so the runtime runs entirely
in sim (``FakeBackend`` / ``MujocoBackend``) with the real arm as a swap-in.

``FeetechBackend`` is intentionally *not* re-exported here: importing it
pulls hardware libraries, so callers import it explicitly by path. The real
teleop ``LerobotLeaderReader`` is likewise left out (reached by config path
only); ``FakeLeaderReader`` and the pure mapping are import-safe and exported.
"""

from __future__ import annotations

from .backend import FakeBackend, RobotBackend
from .kernel import ControlKernel, KernelLimits, KernelOutput, Mode
from .observation import RuntimeObservation
from .setpoint_channel import SetpointChannel
from .sources import GoToPose, ManualPolicy, SineSweep
from .teleop import FakeLeaderReader, LeaderReader, leader_action_to_follower_qpos

__all__ = [
  "RobotBackend",
  "FakeBackend",
  "ControlKernel",
  "KernelLimits",
  "KernelOutput",
  "Mode",
  "SetpointChannel",
  "RuntimeObservation",
  "GoToPose",
  "SineSweep",
  "ManualPolicy",
  "LeaderReader",
  "FakeLeaderReader",
  "leader_action_to_follower_qpos",
]
