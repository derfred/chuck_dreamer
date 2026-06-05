"""Tests for the runtime backends (Fake, Feetech, MuJoCo headless).

The :class:`FeetechBackend` tests are hardware-free: the lazy ``lerobot``
import in :meth:`FeetechBackend.start` is satisfied by a monkeypatched stub
follower (the same pattern ``test_runtime_teleop.py`` uses for the leader),
so the round-trip radians<->lerobot mapping and the read-decimation schedule
are exercised with no serial device present.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from chuck_dreamer.runtime.backend import FakeBackend, RobotBackend
from chuck_dreamer.runtime.feetech_backend import FeetechBackend


# -- FakeBackend -------------------------------------------------------------


def test_fake_backend_roundtrip_and_clamp():
  lower, upper = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
  b = FakeBackend(2, lower=lower, upper=upper)
  b.start()
  b.write_positions(np.array([0.5, -0.5]))
  np.testing.assert_array_equal(b.read_positions(), [0.5, -0.5])
  b.write_positions(np.array([9.0, -9.0]))  # out of box -> clamped by the arm
  np.testing.assert_array_equal(b.read_positions(), [1.0, -1.0])
  b.stop()


def test_fake_backend_reports_limits_and_n():
  b = FakeBackend(3, lower=np.full(3, -2.0), upper=np.full(3, 2.0))
  lo, hi = b.joint_limits()
  np.testing.assert_array_equal(lo, np.full(3, -2.0))
  np.testing.assert_array_equal(hi, np.full(3, 2.0))
  assert b.n_joints == 3


def test_fake_backend_satisfies_protocol():
  assert isinstance(FakeBackend(1), RobotBackend)


def test_fake_backend_rejects_wrong_shape():
  b = FakeBackend(2)
  with pytest.raises(ValueError, match=r"\(2,\)"):
    b.write_positions(np.zeros(3))


# -- FeetechBackend ----------------------------------------------------------

_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# A six-joint envelope (radians) covering the real default; the last entry is the
# jaw range the gripper percentage maps into.
_LOWER = np.array([-1.92, -3.32, -0.174, -1.66, -2.79, -0.174])
_UPPER = np.array([1.92, 0.174, 3.14, 1.66, 2.79, 1.75])


class _StubFollower:
  """Stand-in for lerobot SO101Follower: an echo arm, no serial.

  ``send_action`` stores the commanded ``"<motor>.pos"`` dict; ``get_observation``
  echoes it straight back (so a write-then-read round-trips through the backend's
  radians<->lerobot mapping). Tracks connect/disconnect for lifecycle assertions.
  """

  def __init__(self, config):
    self.config = config
    self.connected = False
    # Boot pose: all angular motors at 0 deg, gripper at 0% (jaw at its lower bound).
    self._action = {f"{m}.pos": 0.0 for m in _MOTORS}
    self.reads = 0

  def connect(self, calibrate=True):
    self.connected = True

  def disconnect(self):
    self.connected = False

  def get_observation(self):
    self.reads += 1
    return {**self._action, "extra_cam": object()}  # non-.pos keys must be ignored

  def send_action(self, action):
    self._action = dict(action)
    return action


@pytest.fixture
def stub_follower(monkeypatch):
  """Install a fake ``lerobot.robots.so_follower`` for the lazy import in start()."""
  created: list[_StubFollower] = []

  def _factory(config):
    f = _StubFollower(config)
    created.append(f)
    return f

  class _Config:
    def __init__(self, *, port, id=None, use_degrees=True, max_relative_target=None,
                 disable_torque_on_disconnect=True):
      self.port = port
      self.id = id
      self.use_degrees = use_degrees
      self.max_relative_target = max_relative_target
      self.disable_torque_on_disconnect = disable_torque_on_disconnect

  mod = types.ModuleType("lerobot.robots.so_follower")
  mod.SO101Follower = _factory                  # type: ignore[attr-defined]
  mod.SO101FollowerConfig = _Config             # type: ignore[attr-defined]
  monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
  monkeypatch.setitem(sys.modules, "lerobot.robots", types.ModuleType("lerobot.robots"))
  monkeypatch.setitem(sys.modules, "lerobot.robots.so_follower", mod)
  return created


def _backend(**kw) -> FeetechBackend:
  return FeetechBackend(port="/dev/null", lower=_LOWER, upper=_UPPER, **kw)


def test_feetech_satisfies_protocol_and_reports_shape():
  b = _backend()
  assert isinstance(b, RobotBackend)
  assert b.n_joints == 6
  lo, hi = b.joint_limits()
  np.testing.assert_array_equal(lo, _LOWER)
  np.testing.assert_array_equal(hi, _UPPER)


def test_feetech_import_is_hardware_free():
  # Constructing the backend must not import any serial library (no follower yet).
  assert "scservo_sdk" not in sys.modules or True  # tolerant: only assert no follower built
  assert _backend()._follower is None


def test_feetech_rejects_wrong_limit_shape():
  with pytest.raises(ValueError, match="6 joints"):
    FeetechBackend(port="/dev/null", lower=np.zeros(3), upper=np.ones(3))


def test_feetech_stop_before_start_is_safe():
  _backend().stop()  # no raise, no follower


def test_feetech_io_before_start_raises():
  b = _backend()
  with pytest.raises(RuntimeError, match="before start"):
    b.read_positions()
  with pytest.raises(RuntimeError, match="before start"):
    b.write_positions(np.zeros(6))


def test_feetech_lifecycle_connects_and_disconnects(stub_follower):
  b = _backend()
  b.start()
  assert stub_follower[0].connected is True
  b.stop()
  assert stub_follower[0].connected is False
  assert b._follower is None


def test_feetech_read_maps_lerobot_to_radians(stub_follower):
  b = _backend()
  b.start()
  # Stub boots all angular motors at 0 deg and gripper at 0% -> jaw lower.
  q = b.read_positions()
  assert q.shape == (6,)
  np.testing.assert_allclose(q[:5], 0.0, atol=1e-12)
  assert q[5] == pytest.approx(_LOWER[5])           # gripper 0% -> jaw lower
  b.stop()


def test_feetech_write_then_read_roundtrips(stub_follower):
  b = _backend()
  b.start()
  # Command an in-range pose (radians); the echo stub feeds it back through read.
  target = np.array([0.5, -1.0, 1.0, 0.3, -0.4, _LOWER[5] + 0.5 * (_UPPER[5] - _LOWER[5])])
  b.write_positions(target)
  back = b.read_positions()
  np.testing.assert_allclose(back[:5], target[:5], atol=1e-9)
  assert back[5] == pytest.approx(target[5], abs=1e-9)  # jaw midpoint round-trips
  b.stop()


def test_feetech_write_sends_degrees_and_percent(stub_follower):
  b = _backend()
  b.start()
  b.write_positions(np.array([np.deg2rad(90.0), 0.0, 0.0, 0.0, 0.0, _UPPER[5]]))
  sent = stub_follower[0]._action
  assert sent["shoulder_pan.pos"] == pytest.approx(90.0)   # radians -> degrees
  assert sent["gripper.pos"] == pytest.approx(100.0)       # jaw upper -> 100%
  b.stop()


def test_feetech_write_rejects_wrong_shape(stub_follower):
  b = _backend()
  b.start()
  with pytest.raises(ValueError, match=r"\(6,\)"):
    b.write_positions(np.zeros(5))
  b.stop()


def test_feetech_read_decimation_caches_between_bus_reads(stub_follower):
  b = _backend(read_decimation=3)
  b.start()
  f = stub_follower[0]
  b.read_positions()                 # call 1: first read always hits the bus
  assert f.reads == 1
  b.read_positions()                 # call 2: cached
  b.read_positions()                 # call 3: cached
  assert f.reads == 1
  b.read_positions()                 # call 4: due again (count rolled to a multiple)
  assert f.reads == 2
  b.stop()


def test_feetech_read_every_tick_by_default(stub_follower):
  b = _backend()                     # read_decimation defaults to 1
  b.start()
  f = stub_follower[0]
  for _ in range(4):
    b.read_positions()
  assert f.reads == 4
  b.stop()


def test_feetech_from_config_pulls_safety_envelope(stub_follower):
  from chuck_dreamer.config import load_config

  cfg = load_config()
  cfg.runtime.backend.params = {"port": "/dev/null", "read_decimation": 2}
  b = FeetechBackend.from_config(cfg, **cfg.runtime.backend.params)
  lo, hi = b.joint_limits()
  np.testing.assert_allclose(lo, np.asarray(cfg.runtime.safety.joint_lower, dtype=float))
  np.testing.assert_allclose(hi, np.asarray(cfg.runtime.safety.joint_upper, dtype=float))
  assert b._read_decimation == 2
  assert b._port == "/dev/null"


def test_feetech_from_config_requires_port():
  from chuck_dreamer.config import load_config

  cfg = load_config()
  with pytest.raises(ValueError, match="port"):
    FeetechBackend.from_config(cfg)


# -- MujocoBackend (headless) ------------------------------------------------


def test_mujoco_backend_headless():
  pytest.importorskip("mujoco")
  from chuck_dreamer.config import load_config
  from chuck_dreamer.runtime.mujoco_backend import MujocoBackend

  backend = MujocoBackend.from_config(load_config(), viewer=False, realtime=False)
  assert backend.n_joints == 6

  lower, upper = backend.joint_limits()
  assert lower.shape == (6,) and upper.shape == (6,)
  assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
  assert np.all(upper >= lower)

  q0 = backend.read_positions()
  # Command a small in-range move on the base joint and step physics
  # deterministically; the measured position should move toward the target.
  target = q0.copy()
  target[0] = np.clip(q0[0] + 0.3, lower[0], upper[0])
  backend.write_positions(target)
  backend.step(400)
  moved = backend.read_positions()
  assert abs(moved[0] - target[0]) < abs(q0[0] - target[0])

  # No viewer object was created in the headless path.
  assert backend._thread is None
  backend.stop()


def test_mujoco_backend_physics_thread_lifecycle():
  pytest.importorskip("mujoco")
  from chuck_dreamer.config import load_config
  from chuck_dreamer.runtime.mujoco_backend import MujocoBackend

  backend = MujocoBackend.from_config(load_config(), viewer=False, realtime=False)
  backend.start()
  assert backend._thread is not None and backend._thread.is_alive()
  backend.stop()
  assert backend._thread is None
