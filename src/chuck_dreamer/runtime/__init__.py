"""SO-101 pushing runtime: control loop, backends, loops, telemetry.

Boots from the project's OmegaConf session config (the ``runtime:`` block
of ``configs/default.yaml``), spins a policy loop and a control loop against
a :class:`RobotBackend`, and shuts down cleanly. The control loop talks to
a backend protocol, never to hardware directly, so the runtime runs entirely
in sim (``FakeBackend`` / ``MujocoBackend``) with the real arm as a swap-in.

``FeetechBackend`` is intentionally *not* re-exported here: importing it
pulls hardware libraries, so callers import it explicitly by path. The real
teleop ``LerobotLeaderReader`` is likewise left out (reached by config path
only); ``FakeLeaderReader`` and the pure mapping are import-safe and exported.

The M3 observation half (``Sensor`` / ``SimCameraSensor``, the perception
pipeline, the ``RerunSink``) is exported. The heavy/optional pieces are not:
``WebcamSensor`` (lazy ``cv2``) and ``ObjectLocalizerModule`` (SAM2 / torch)
are reached by config path only, exactly like the hardware backends.
"""

from __future__ import annotations

from .backend import FakeBackend, RobotBackend
from .modalities import (
  ModalityError,
  RuntimeObservation,
  check_required,
  compose_modalities,
)
from .perception import EePoseModule, PerceptionModule, PerceptionPipeline
from .rerun_sink import RerunSink
from .sensors import Sensor, SimCameraSensor
from .setpoint_channel import SetpointChannel
from .sources import GoToPose, ManualPolicy, SineSweep
from .teleop import FakeLeaderReader, LeaderReader, leader_action_to_follower_qpos

__all__ = [
  "RobotBackend",
  "FakeBackend",
  "SetpointChannel",
  "RuntimeObservation",
  "ModalityError",
  "compose_modalities",
  "check_required",
  "Sensor",
  "SimCameraSensor",
  "PerceptionModule",
  "PerceptionPipeline",
  "EePoseModule",
  "RerunSink",
  "GoToPose",
  "SineSweep",
  "ManualPolicy",
  "LeaderReader",
  "FakeLeaderReader",
  "leader_action_to_follower_qpos",
]
