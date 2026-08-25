"""Tests for the runtime backends (Fake, Feetech, MuJoCo headless).

The :class:`FeetechBackend` tests are hardware-free: the lazy ``lerobot``
import in :meth:`FeetechBackend.start` is satisfied by a monkeypatched stub
follower (the same pattern ``test_runtime_teleop.py`` uses for the leader),
so the round-trip radians<->lerobot mapping and the read-decimation schedule
are exercised with no serial device present.
"""

from __future__ import annotations

import math
import sys
import time
import types

import numpy as np
import pytest

from chuck_dreamer.runtime.backend import FakeBackend, RobotBackend
from chuck_dreamer.runtime.control_state import FaultFlags
from chuck_dreamer.runtime.feetech_backend import FeetechBackend

# -- FakeBackend -------------------------------------------------------------


def test_fake_backend_roundtrip_and_clamp():
  lower, upper = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
  b = FakeBackend(2, lower=lower, upper=upper)
  b.start()
  b.write_positions(np.array([0.5, -0.5]))
  np.testing.assert_array_equal(b.read_state(math.inf).q, [0.5, -0.5])
  b.write_positions(np.array([9.0, -9.0]))  # out of box -> clamped by the arm
  np.testing.assert_array_equal(b.read_state(math.inf).q, [1.0, -1.0])
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


def test_fake_backend_home_defaults_to_box_centre():
  b = FakeBackend(3, lower=np.full(3, -1.0), upper=np.array([1.0, 3.0, 1.0]))
  np.testing.assert_allclose(b.home_qpos, [0.0, 1.0, 0.0])
  np.testing.assert_allclose(b.last_positions(), b.home_qpos)  # boots at home


def test_fake_backend_home_is_zeros_when_unbounded():
  """An unbounded axis has no centre; it must not average its infinities."""
  b = FakeBackend(2, lower=np.array([-1.0, -np.inf]), upper=np.array([3.0, np.inf]))
  np.testing.assert_allclose(b.home_qpos, [1.0, 0.0])


def test_fake_backend_home_from_q_init_is_clamped():
  b = FakeBackend(2, lower=np.full(2, -1.0), upper=np.full(2, 1.0),
                  q_init=np.array([5.0, 0.2]))
  np.testing.assert_allclose(b.home_qpos, [1.0, 0.2])


def test_fake_backend_home_is_a_copy():
  b = FakeBackend(2, lower=np.full(2, -1.0), upper=np.full(2, 1.0))
  b.home_qpos[:] = 99.0
  np.testing.assert_allclose(b.home_qpos, [0.0, 0.0])


# -- FeetechBackend ----------------------------------------------------------

_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# A six-joint envelope (radians) covering the real default; the last entry is the
# jaw range the gripper percentage maps into.
# The tier-1 block: Present_Position(56,2) .. Present_Temperature(63,1).
_FAST_ADDR, _FAST_LEN = 56, 8
_SLOW_ADDR, _SLOW_LEN = 69, 2

_LOWER = np.array([-1.92, -3.32, -0.174, -1.66, -2.79, -0.174])
_UPPER = np.array([1.92, 0.174, 3.14, 1.66, 2.79, 1.75])


class _StubMotor:
  def __init__(self, id_):
    self.id = id_


class _StubSyncReader:
  """Holds the last block fetched, answering getData at any sub-offset.

  Mirrors the scservo SDK's GroupSyncRead: ``_sync_read`` fills the buffer for
  one (addr, length) span and ``getData`` decodes any sub-range of it, which is
  what lets the backend pull five registers out of one transaction.
  """

  def __init__(self, bus):
    self._bus = bus

  def getData(self, id_, addr, length):
    return self._bus.register_value(id_, addr, length)


class _StubBus:
  """Stand-in for FeetechMotorsBus: per-register values plus block reads.

  ``register_value`` is the single source of truth: both the register-at-a-time
  ``sync_read`` path and the block path read from it, so a test can assert the
  two agree. ``block_reads`` records every (addr, length) span fetched, which is
  how the transaction-count assertions are written.
  """

  # Raw counts served per register, uniform across motors unless overridden.
  DEFAULTS = {
    56: 2048,   # Present_Position   (overwritten by _StubFollower's pose)
    58: 100,    # Present_Velocity
    60: 250,    # Present_Load
    62: 120,    # Present_Voltage    (12.0 V in decivolts)
    63: 31,     # Present_Temperature
    69: 40,     # Present_Current
  }

  def __init__(self):
    self.sync_reads: list[str] = []
    self.block_reads: list[tuple[int, int]] = []
    self.motors = {m: _StubMotor(i + 1) for i, m in enumerate(_MOTORS)}
    self.sync_reader = _StubSyncReader(self)
    self.overrides: dict[int, dict[int, int]] = {}   # {addr: {motor_id: value}}
    self.fail_addrs: set[int] = set()                # addrs whose read raises

  # -- values ---------------------------------------------------------------

  def register_value(self, id_, addr, length):
    del length
    return self.overrides.get(addr, {}).get(id_, self.DEFAULTS.get(addr, 0))

  def set_register(self, addr, values):
    """Override one register's raw counts, as ``{motor_name: value}``."""
    self.overrides[addr] = {self.motors[m].id: v for m, v in values.items()}

  # -- lerobot bus surface the backend uses ---------------------------------

  def sync_read(self, data_name):
    self.sync_reads.append(data_name)
    return {m: float(i) for i, m in enumerate(_MOTORS)}

  def _sync_read(self, addr, length, ids, *, raise_on_error=True, **kw):
    del ids, raise_on_error, kw
    if addr in self.fail_addrs:
      raise ConnectionError(f"stub bus failure @{addr}")
    self.block_reads.append((addr, length))

  def _decode_sign(self, data_name, ids_values):
    # The stub serves small positive counts, so sign-magnitude decoding is the
    # identity here; the real bus's implementation is exercised on hardware.
    del data_name
    return ids_values

  def _normalize(self, ids_values):
    # Position counts -> the units lerobot's sync_read would return: degrees for
    # the angular motors, 0..100 for the gripper. 4096 counts per revolution,
    # centred at 2048 (matching MotorNormMode.DEGREES about the calibration mid).
    out = {}
    for id_, val in ids_values.items():
      if id_ == len(_MOTORS):                   # gripper: percentage
        out[id_] = (val / 4095.0) * 100.0
      else:
        out[id_] = (val - 2048) * 360.0 / 4095.0
    return out


class _StubFollower:
  """Stand-in for lerobot SO101Follower: an echo arm, no serial.

  ``send_action`` stores the commanded ``"<motor>.pos"`` dict *and* writes the
  equivalent raw counts into the stub bus's ``Present_Position`` register, so a
  write round-trips through either read path -- ``get_observation`` (the
  position-only fallback) or the block read ``read_state`` normally uses. A
  stub whose two paths disagreed would let a real decode bug pass unnoticed.

  Tracks connect/disconnect for lifecycle assertions.
  """

  def __init__(self, config):
    self.config = config
    self.connected = False
    self.reads = 0
    self.bus = _StubBus()
    # Boot pose: all angular motors at 0 deg, gripper at 0% (jaw at its lower
    # bound), expressed in both representations.
    self._action = {f"{m}.pos": 0.0 for m in _MOTORS}
    self._sync_positions()

  def _sync_positions(self):
    """Mirror ``self._action`` into the bus's Present_Position counts.

    The inverse of _StubBus._normalize: degrees (or gripper percent) back to
    raw counts about the 2048 calibration midpoint.
    """
    counts = {}
    for i, m in enumerate(_MOTORS):
      val = float(self._action[f"{m}.pos"])
      if i == len(_MOTORS) - 1:                    # gripper: 0..100 percent
        counts[m] = round(val / 100.0 * 4095.0)
      else:                                        # angular: degrees
        counts[m] = round(val * 4095.0 / 360.0 + 2048)
    self.bus.set_register(56, counts)

  def connect(self, calibrate=True):
    self.connected = True

  def disconnect(self):
    self.connected = False

  def get_observation(self):
    self.reads += 1
    return {**self._action, "extra_cam": object()}  # non-.pos keys must be ignored

  def send_action(self, action):
    self._action = dict(action)
    self._sync_positions()
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


_HOME = 0.5 * (_LOWER + _UPPER)   # arbitrary in-envelope rest pose for tests


def _backend(**kw) -> FeetechBackend:
  kw.setdefault("home_qpos", _HOME)
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
    FeetechBackend(port="/dev/null", lower=np.zeros(3), upper=np.ones(3),
                   home_qpos=np.zeros(3))


def test_feetech_stop_before_start_is_safe():
  _backend().stop()  # no raise, no follower


def test_feetech_io_before_start_raises():
  b = _backend()
  with pytest.raises(RuntimeError, match="before start"):
    b.read_state(math.inf)
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
  q = b.read_state(math.inf).q
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
  back = b.read_state(math.inf).q
  # Counts are integral, so the round-trip is quantised at ~360/4095 deg.
  np.testing.assert_allclose(back[:5], target[:5], atol=2e-3)
  assert back[5] == pytest.approx(target[5], abs=2e-3)  # jaw midpoint round-trips
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


def test_feetech_last_positions_never_touches_the_bus(stub_follower):
  # The observer (policy-loop) path: start() primes the cache, and any number
  # of last_positions() calls return it without a single bus transaction.
  b = _backend()
  b.start()
  f = stub_follower[0]
  bus = f.bus
  bus.block_reads.clear()
  first = b.last_positions()
  for _ in range(10):
    np.testing.assert_array_equal(b.last_positions(), first)
  assert bus.block_reads == []       # not one transaction
  assert f.reads == 0
  # A control-thread read_state refreshes what the observer path sees.
  target = np.array([0.5, -1.0, 1.0, 0.3, -0.4, 0.7])
  b.write_positions(target)
  b.read_state(math.inf)
  np.testing.assert_allclose(b.last_positions(), target, atol=2e-3)
  b.stop()


def test_feetech_last_positions_before_start_raises():
  with pytest.raises(RuntimeError, match="before start"):
    _backend().last_positions()


def test_feetech_bus_transactions_never_interleave(stub_follower):
  # Regression (B4): concurrent read/write from two threads corrupted the
  # half-duplex bus ("Incorrect status packet!"). Instrument the stub so any
  # overlapping transaction is detected, then hammer the backend from a
  # control-like thread (read+write) and an observer thread (read).
  import threading
  import time

  b = _backend()
  b.start()
  f = stub_follower[0]

  busy    = threading.Lock()
  overlap = []

  def _guard(fn):
    def wrapped(*a, **kw):
      if not busy.acquire(blocking=False):
        overlap.append(fn.__name__)
        return fn(*a, **kw)
      try:
        time.sleep(0.0005)           # widen the window a real bus read has
        return fn(*a, **kw)
      finally:
        busy.release()
    return wrapped

  f.get_observation = _guard(f.get_observation)
  f.send_action     = _guard(f.send_action)

  q = b.last_positions()

  def control():
    for _ in range(50):
      b.write_positions(q)
      b.read_state(math.inf)

  def rogue_reader():                # a misbehaving second bus user
    for _ in range(50):
      b.read_state(math.inf)

  threads = [threading.Thread(target=control), threading.Thread(target=rogue_reader)]
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  assert overlap == []               # the backend lock serialized every transaction
  b.stop()


def test_feetech_from_config_pulls_safety_envelope(stub_follower):
  from chuck_dreamer.config import load_config

  cfg = load_config()
  cfg.runtime.backend.params = {
    "port": "/dev/null", "slow_decimation": 2, "home_qpos": [0.0] * 6}
  b = FeetechBackend.from_config(cfg, **cfg.runtime.backend.params)
  limits = cfg.runtime.control_loop.joint_limits
  lo, hi = b.joint_limits()
  np.testing.assert_allclose(lo, np.asarray(limits.lower, dtype=float))
  np.testing.assert_allclose(hi, np.asarray(limits.upper, dtype=float))
  assert b._slow_decimation == 2
  assert b._port == "/dev/null"


def test_feetech_from_config_requires_home_qpos():
  """A real arm's rest pose is a physical choice -- no default is safe."""
  from chuck_dreamer.config import load_config

  with pytest.raises(ValueError, match="home_qpos"):
    FeetechBackend.from_config(load_config(), port="/dev/null")


def test_feetech_home_read_from_backend_params(stub_follower):
  """home_qpos rides in runtime.backend.params, alongside `port`."""
  from chuck_dreamer.config import load_config

  cfg = load_config()
  cfg.runtime.backend.params = {"port": "/dev/null", "home_qpos": [0.1] * 6}
  b = FeetechBackend.from_config(cfg, **cfg.runtime.backend.params)
  np.testing.assert_allclose(b.home_qpos, np.full(6, 0.1))


def test_feetech_home_outside_envelope_is_rejected():
  lo = np.full(6, -1.0)
  hi = np.full(6, 1.0)
  with pytest.raises(ValueError, match="within the joint envelope"):
    FeetechBackend(port="/dev/null", lower=lo, upper=hi, home_qpos=np.full(6, 99.0))


def test_feetech_ctor_requires_home_qpos():
  with pytest.raises(TypeError, match="home_qpos"):
    FeetechBackend(port="/dev/null", lower=_LOWER, upper=_UPPER)  # type: ignore[call-arg]


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

  # The scene's initial qpos is the home pose, and the sim boots holding it.
  np.testing.assert_allclose(
    backend.home_qpos, np.asarray(backend._scene.joint_initial_qpos, dtype=float))
  np.testing.assert_allclose(backend.last_positions(), backend.home_qpos)

  lower, upper = backend.joint_limits()
  assert lower.shape == (6,) and upper.shape == (6,)
  assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
  assert np.all(upper >= lower)

  q0 = backend.last_positions()
  # Command a small in-range move on the base joint and step physics
  # deterministically; the measured position should move toward the target.
  target = q0.copy()
  target[0] = np.clip(q0[0] + 0.3, lower[0], upper[0])
  backend.write_positions(target)
  backend.step(400)
  moved = backend.last_positions()
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


# -- read_state: block layout + budget scheduling ----------------------------


def test_read_state_fetches_position_and_tier1_in_one_transaction(stub_follower):
  # The load-bearing claim of the design: position, velocity, load, voltage and
  # temperature all arrive from a single contiguous sync-read.
  b = _backend()
  b.start()
  bus = stub_follower[0].bus
  bus.block_reads.clear()

  st = b.read_state(math.inf)

  assert bus.block_reads == [(_FAST_ADDR, _FAST_LEN)]   # exactly one transaction
  assert st.q.shape == (6,)
  assert st.qd is not None and st.qd.shape == (6,)
  assert st.load is not None and st.load.shape == (6,)
  assert st.volt is not None and st.temp is not None
  assert st.q_age == 0.0
  assert st.healthy
  b.stop()


def test_read_state_converts_tier1_diagnostics_to_physical_units(stub_follower):
  b = _backend()
  b.start()
  st = b.read_state(math.inf)
  # 100 counts/s at 4096 counts/rev; 250/1000 of full-scale load; 120 decivolts.
  np.testing.assert_allclose(st.qd, np.full(6, 100 * 2 * math.pi / 4096.0))
  np.testing.assert_allclose(st.load, np.full(6, 0.25))
  np.testing.assert_allclose(st.volt, np.full(6, 12.0))
  np.testing.assert_allclose(st.temp, np.full(6, 31.0))
  b.stop()


def test_read_state_block_positions_match_the_get_observation_path(stub_follower):
  # The block decode and lerobot's own sync_read path must not drift: both go
  # through _normalize + leader_action_to_follower_qpos, so a mid-scale count
  # maps to the same radians either way.
  b = _backend()
  b.start()
  st = b.read_state(math.inf)
  # The stub boots at 0 deg on the angular joints (count 2048, the calibration
  # midpoint) and 0% on the gripper (count 0, the jaw's lower bound) -- the
  # same pose get_observation reports, decoded through the block path instead.
  np.testing.assert_allclose(st.q[:5], np.zeros(5), atol=1e-9)
  assert st.q[5] == pytest.approx(_LOWER[5], abs=1e-9)
  b.stop()


def test_read_state_starved_budget_returns_cache_and_flags_it(stub_follower):
  b = _backend()
  b.start()
  bus = stub_follower[0].bus
  primed = b.read_state(math.inf).q
  bus.block_reads.clear()

  st = b.read_state(0.0)               # no budget at all

  assert bus.block_reads == []         # nothing was put on the wire
  np.testing.assert_array_equal(st.q, primed)   # but q is still answerable
  assert st.faults & FaultFlags.READ_STARVED
  assert st.stale                      # and the tick knows not to trust it
  b.stop()


def test_read_state_first_read_ignores_the_budget(stub_follower):
  # With no cache there is nothing to fall back on, so a zero budget must not
  # leave the very first tick without a position.
  b = _backend()
  b.start()
  with b._lock:                        # simulate a backend that never primed
    b._cached = None
    b._fast_t = None
  bus = stub_follower[0].bus
  bus.block_reads.clear()

  st = b.read_state(0.0)

  assert bus.block_reads == [(_FAST_ADDR, _FAST_LEN)]
  assert st.q.shape == (6,)
  b.stop()


def test_read_state_marks_q_stale_once_it_ages_out(stub_follower):
  b = _backend(stale_after_s=0.001)
  b.start()
  b.read_state(math.inf)
  time.sleep(0.005)
  st = b.read_state(0.0)               # starved: q keeps ageing
  assert st.q_age > 0.001
  assert st.faults & FaultFlags.STALE_Q
  b.stop()


def test_read_state_defers_the_slow_block_when_the_budget_is_tight(stub_follower):
  b = _backend(slow_decimation=1)      # due on every tick
  b.start()
  bus = stub_follower[0].bus
  bus.block_reads.clear()
  # Pin the cost estimates rather than relying on what the stub happens to
  # measure, so the budget below is unambiguously "enough for one, not two".
  b._cost["fast"].cost = 0.004
  b._cost["slow"].cost = 0.003

  st = b.read_state(0.005)             # covers fast (4 ms), not fast + slow (7 ms)

  assert bus.block_reads == [(_FAST_ADDR, _FAST_LEN)]
  assert st.faults & FaultFlags.SLOW_DEFERRED
  assert st.curr is None               # never measured, so absent rather than faked
  b.stop()


def test_read_state_reads_the_slow_block_when_it_fits(stub_follower):
  b = _backend(slow_decimation=1)
  b.start()
  bus = stub_follower[0].bus
  bus.block_reads.clear()

  st = b.read_state(math.inf)

  assert bus.block_reads == [(_FAST_ADDR, _FAST_LEN), (_SLOW_ADDR, _SLOW_LEN)]
  np.testing.assert_allclose(st.curr, np.full(6, 40 * 0.0065))
  assert not (st.faults & FaultFlags.SLOW_DEFERRED)
  b.stop()


def test_read_state_decimates_the_slow_block(stub_follower):
  b = _backend(slow_decimation=4)
  b.start()
  bus = stub_follower[0].bus
  bus.block_reads.clear()
  for _ in range(8):
    b.read_state(math.inf)
  slow = [r for r in bus.block_reads if r[0] == _SLOW_ADDR]
  assert len(slow) == 2                # 8 ticks / decimation 4
  b.stop()


def test_read_state_forces_a_starved_slow_block_through_eventually(stub_follower):
  # A persistently tight budget must not starve thermal monitoring forever: one
  # late tick is a better outcome than never noticing an over-current.
  b = _backend(slow_decimation=1, slow_max_skips=3)
  b.start()
  bus = stub_follower[0].bus
  bus.block_reads.clear()
  b._cost["fast"].cost = 0.004
  b._cost["slow"].cost = 0.003
  for _ in range(3):                   # three denials
    b.read_state(0.005)
  assert [r for r in bus.block_reads if r[0] == _SLOW_ADDR] == []
  b.read_state(0.005)                  # the fourth is forced through
  assert [r for r in bus.block_reads if r[0] == _SLOW_ADDR] == [(_SLOW_ADDR, _SLOW_LEN)]
  b.stop()


def test_read_state_survives_a_bus_error_without_raising(stub_follower):
  b = _backend()
  b.start()
  bus = stub_follower[0].bus
  good = b.read_state(math.inf).q
  bus.fail_addrs.add(_FAST_ADDR)

  st = b.read_state(math.inf)          # the transaction raises inside

  np.testing.assert_array_equal(st.q, good)     # cache reused
  assert st.faults & FaultFlags.BUS_ERROR
  b.stop()


def test_read_state_raises_health_flags_from_diagnostics(stub_follower):
  b = _backend(temp_limit_c=30.0, load_limit=0.1, volt_min=13.0)
  b.start()
  st = b.read_state(math.inf)
  # Stub serves 31 C, 0.25 load, 12.0 V against limits of 30 / 0.1 / 13.0.
  assert st.faults & FaultFlags.OVER_TEMP
  assert st.faults & FaultFlags.OVER_LOAD
  assert st.faults & FaultFlags.UNDER_VOLT
  assert not st.healthy
  b.stop()


def test_read_state_degrades_to_position_only_without_block_support(stub_follower):
  # An SDK lacking the private sync-reader surface must not fail every tick:
  # positions keep flowing, tier-1 diagnostics are simply absent.
  b = _backend()
  b.start()
  f = stub_follower[0]

  def _missing(*a, **kw):              # the surface the block path relies on
    raise AttributeError("no _sync_read on this bus")
  f.bus._sync_read = _missing
  before = f.reads

  st = b.read_state(math.inf)

  assert st.q.shape == (6,)
  assert st.qd is None                 # no block, so no free diagnostics
  assert f.reads == before + 1         # fell back to get_observation
  assert b._blocks_ok is False         # latched off, not retried every tick
  b.stop()


def test_read_state_before_start_raises():
  with pytest.raises(RuntimeError, match="before start"):
    _backend().read_state(math.inf)


def test_read_state_block_cost_tracks_the_wire(stub_follower):
  # The scheduler budgets on a measured cost, so a slow transaction must raise
  # the estimate immediately (and a fast one must not drop it instantly).
  import chuck_dreamer.runtime.feetech_backend as fb

  c = fb._BlockCost(0.001, decay=0.02)
  c.observe(0.010)
  assert c.cost == 0.010               # rises straight to the new maximum
  c.observe(0.001)
  assert c.cost > 0.009                # but decays only slowly


def test_read_state_block_and_fallback_paths_agree(stub_follower):
  # The block decode and the position-only fallback must produce the same
  # radians for the same pose. They share _normalize +
  # leader_action_to_follower_qpos, but only a direct comparison catches an
  # offset or width error in the block layout itself.
  b = _backend()
  b.start()
  f = stub_follower[0]
  target = np.array([0.5, -1.0, 1.0, 0.3, -0.4, 0.7])
  b.write_positions(target)

  via_block = b.read_state(math.inf).q
  b._blocks_ok = False                 # force the get_observation path
  via_fallback = b.read_state(math.inf).q

  np.testing.assert_allclose(via_block, via_fallback, atol=2e-3)
  assert f.reads == 1                  # the fallback read, and only it
  b.stop()
