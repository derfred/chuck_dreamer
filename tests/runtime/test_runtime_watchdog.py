"""Tests for :class:`chuck_dreamer.runtime.control_loop.Watchdog`.

The watchdog is synchronous: every check runs inside ``observe``, so these
tests drive it by hand and assert immediately after. The only real time
involved is the sustained-breach duration ``T``.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from chuck_dreamer.runtime.control_loop import FollowingErrorLimits, Watchdog
from chuck_dreamer.runtime.kernel import ControlKernel, KernelLimits, Mode

_ZERO3 = np.zeros(3)


def test_following_error_trips_after_sustained_breach():
  # theta_0=0.05, k=0 -> fixed 0.05 rad threshold; T=0.05s sustained duration.
  fe = FollowingErrorLimits.build(theta_0=0.05, k=0.0, T=0.05, n_joints=3)
  count = {"n": 0}
  wd = Watchdog(
    fe_limits=fe, on_trip=lambda: count.__setitem__("n", count["n"] + 1),
  )
  q_meas = np.zeros(3)
  q_cmd = np.array([0.2, 0.0, 0.0])  # joint 0 breaches theta_0 immediately
  end = time.monotonic() + 0.3
  while time.monotonic() < end and not wd.tripped:
    wd.observe(q_cmd, q_meas, _ZERO3)
    time.sleep(0.01)
  assert wd.tripped
  assert count["n"] == 1
  assert "following_error" in wd.trip_reason
  assert "joint=0" in wd.trip_reason


def test_following_error_does_not_trip_on_transient_breach():
  # A breach shorter than T must not trip: clear it before T elapses.
  fe = FollowingErrorLimits.build(theta_0=0.05, k=0.0, T=1.0, n_joints=3)
  wd = Watchdog(fe_limits=fe, on_trip=lambda: None)
  q_meas = np.zeros(3)
  breach = np.array([0.2, 0.0, 0.0])
  wd.observe(breach, q_meas, _ZERO3)
  time.sleep(0.02)
  wd.observe(q_meas, q_meas, _ZERO3)  # error clears well before T
  time.sleep(0.05)
  assert not wd.tripped


def test_following_error_threshold_scales_with_velocity():
  # theta_0=0.0, k=1.0 -> threshold equals |v_cmd|; an error under the
  # velocity-scaled threshold must not trip even though it exceeds theta_0.
  fe = FollowingErrorLimits.build(theta_0=0.0, k=1.0, T=0.02, n_joints=1)
  wd = Watchdog(fe_limits=fe, on_trip=lambda: None)
  q_meas = np.array([0.0])
  q_cmd = np.array([0.5])
  v_cmd = np.array([1.0])  # theta = 0.0 + 1.0*1.0 = 1.0 > |e|=0.5
  end = time.monotonic() + 0.1
  while time.monotonic() < end:
    wd.observe(q_cmd, q_meas, v_cmd)
    time.sleep(0.01)
  assert not wd.tripped


def test_following_error_trip_latches_kernel_at_measured_position():
  limits = KernelLimits.build(np.full(3, -1.0), np.full(3, 1.0), 1.0)
  kernel = ControlKernel(limits, np.zeros(3), mode=Mode.NORMAL)
  fe = FollowingErrorLimits.build(theta_0=0.05, k=0.0, T=0.02, n_joints=3)

  wd_holder: dict[str, Watchdog] = {}

  def on_trip():
    q = wd_holder["wd"].trip_q_meas
    kernel.request_estop_at(q)

  wd = Watchdog(fe_limits=fe, on_trip=on_trip)
  wd_holder["wd"] = wd
  q_meas = np.array([0.3, 0.0, 0.0])  # arm physically stuck here
  q_cmd = np.array([0.9, 0.0, 0.0])   # commanded target keeps drifting away
  end = time.monotonic() + 0.2
  while time.monotonic() < end and kernel.mode is not Mode.ESTOP:
    wd.observe(q_cmd, q_meas, _ZERO3)
    time.sleep(0.01)
  assert kernel.mode is Mode.ESTOP
  np.testing.assert_allclose(kernel.q_cmd, q_meas)


def test_following_error_rejects_bad_limits():
  with pytest.raises(ValueError, match="theta_0"):
    FollowingErrorLimits.build(theta_0=-0.1, k=0.0, T=0.1, n_joints=3)
  with pytest.raises(ValueError, match="T must be positive"):
    FollowingErrorLimits.build(theta_0=0.05, k=0.0, T=0.0, n_joints=3)
  with pytest.raises(ValueError, match="k must be"):
    FollowingErrorLimits.build(theta_0=0.05, k=-1.0, T=0.1, n_joints=3)
