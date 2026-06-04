"""Tests for the runtime backends (Fake, Feetech stub, MuJoCo headless)."""

from __future__ import annotations

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


# -- FeetechBackend stub -----------------------------------------------------


def test_feetech_stub_raises_on_start():
  b = FeetechBackend(port="/dev/ttyACM0")
  with pytest.raises(NotImplementedError, match="stub"):
    b.start()


def test_feetech_stub_raises_on_io():
  b = FeetechBackend(port="/dev/null")
  with pytest.raises(NotImplementedError):
    b.read_positions()
  with pytest.raises(NotImplementedError):
    b.write_positions(np.zeros(6))
  with pytest.raises(NotImplementedError):
    b.joint_limits()


def test_feetech_stub_stop_is_safe():
  FeetechBackend(port="/dev/null").stop()  # no raise


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
