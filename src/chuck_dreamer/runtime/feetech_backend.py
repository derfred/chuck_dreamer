"""The real SO-101 follower behind the backend protocol, with a budgeted read.

The interesting part of this module is :meth:`FeetechBackend.read_state` and
the block layout it rests on. The STS3215 control table puts position and four
diagnostics in one *contiguous* span:

===== ==== ==== ==== ==== ====
addr    56   58   60   62   63
reg    Pos  Vel Load Volt Temp
len      2    2    2    1    1
===== ==== ==== ==== ==== ====

A sync-read's cost is dominated by round-trips, not payload: every register
read separately pays a fresh instruction packet plus the USB-serial adapter's
turnaround latency, while widening one read from 2 to 8 bytes adds only a few
dozen microseconds of wire time. So position, velocity, load, voltage and
temperature together cost *one* transaction — barely more than position alone
— and only ``Present_Current`` (addr 69, outside the span) needs a second.

That collapses the prioritisation problem to scheduling two blocks:

``_BLOCK_FAST``
  Position + tier-1 diagnostics. Attempted every tick; there is never a reason
  to read position by itself.

``_BLOCK_SLOW``
  Current. Decimated hard — and skipped entirely when the tick's budget will
  not cover it, since it is the one read the loop can genuinely do without.

Both blocks measure their own cost on the wire and keep a high-water estimate,
so the schedule self-calibrates to whatever adapter and baud rate is actually
plugged in rather than trusting hardcoded milliseconds.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

import numpy as np

from .control_state import ControlState, FaultFlags
from .teleop import (
  _FOLLOWER_JOINTS,
  _LEADER_MOTORS,
  follower_qpos_to_action,
  leader_action_to_follower_qpos,
)

logger = logging.getLogger(__name__)

_N_JOINTS = len(_LEADER_MOTORS)


class _Block:
  """One contiguous span of the control table, fetched in a single sync-read.

  ``fields`` maps each register name carried by the block to its offset and
  width *within* the span, which is what lets one ``_sync_read`` decode into
  several registers: the SDK's ``getData(id, addr, length)`` accepts any
  sub-range of the block it just fetched.

  The layout is a shared, immutable description of the span. Its measured
  *cost* is deliberately not stored here: two backends on different serial
  adapters have genuinely different transaction costs, so a shared estimate
  would let one bus mis-size the other's budget. Costs live per-instance in
  :class:`_BlockCost`.
  """

  __slots__ = ("addr", "fields", "length", "name", "seed_cost")

  def __init__(self, name, addr, fields, *, seed_cost: float):
    self.name      = name
    self.addr      = addr
    self.fields    = fields                   # {register: (offset, width)}
    self.length    = max(off + width for off, width in fields.values())
    self.seed_cost = seed_cost


class _BlockCost:
  """Per-backend estimate of one block's transaction cost.

  A high-water estimate in the spirit of a p95: it jumps straight to any new
  maximum and decays slowly, so a single slow transaction immediately makes
  the scheduler cautious while a run of fast ones only gradually restores
  optimism. Budgeting on the mean would overrun on roughly half of all ticks.
  """

  __slots__ = ("_decay", "cost")

  def __init__(self, seed: float, decay: float = 0.02):
    self.cost   = seed
    self._decay = decay

  def observe(self, elapsed: float) -> None:
    if elapsed > self.cost:
      self.cost = elapsed                     # rise instantly
    else:
      self.cost += self._decay * (elapsed - self.cost)   # decay slowly


def _block(name, addr, regs, *, seed_cost):
  """Build a :class:`_Block` from ``(register, addr, width)`` triples."""
  return _Block(
    name, addr, {r: (a - addr, w) for r, a, w in regs}, seed_cost=seed_cost)


# Position + everything that shares its span (addresses from the STS3215
# control table; see lerobot.motors.feetech.tables.STS_SMS_SERIES_CONTROL_TABLE).
_BLOCK_FAST = _block("fast", 56, (
  ("Present_Position",    56, 2),
  ("Present_Velocity",    58, 2),
  ("Present_Load",        60, 2),
  ("Present_Voltage",     62, 1),
  ("Present_Temperature", 63, 1),
), seed_cost=0.004)

# Current sits outside the span, so it costs its own transaction.
_BLOCK_SLOW = _block("slow", 69, (
  ("Present_Current", 69, 2),
), seed_cost=0.003)

# Raw-unit conversions (lerobot.motors.feetech.tables documents the encodings).
_STEPS_PER_REV   = 4096.0                     # sts3215 resolution
_VEL_RAD_PER_CNT = 2.0 * math.pi / _STEPS_PER_REV   # counts/s -> rad/s
_LOAD_FULL_SCALE = 1000.0                     # sign-magnitude counts -> [-1, 1]
_VOLT_PER_CNT    = 0.1                        # decivolts -> volts
_CURR_PER_CNT    = 0.0065                     # 6.5 mA per count -> amps


class FeetechBackend:
  """Real SO-101 follower over lerobot, behind the backend protocol.

  ``lower`` / ``upper`` are the configured per-joint position bounds (radians,
  follower order) reported verbatim by :meth:`joint_limits`; their last entry is
  the jaw range the gripper percentage maps into.

  ``home_qpos`` is the arm's rest pose (radians, follower order) and is
  **required**: a real follower has no model to derive one from, and the rest
  pose of a physical arm is a deliberate choice about where it should sit —
  there is no default that is right by accident. It must lie within the
  envelope.
  """

  def __init__(
    self,
    *,
    port: str,
    lower: np.ndarray,
    upper: np.ndarray,
    home_qpos: np.ndarray,
    calibration_id: str | None = None,
    use_degrees: bool = True,
    slow_decimation: int = 25,
    slow_max_skips: int = 200,
    stale_after_s: float = 0.100,
    temp_limit_c: float = 65.0,
    load_limit: float = 0.90,
    volt_min: float = 9.0,
    calibrate: bool = True,
    max_relative_target: float | None = None,
    disable_torque_on_disconnect: bool = True,
  ) -> None:
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.shape != (_N_JOINTS,) or hi.shape != (_N_JOINTS,):
      raise ValueError(
        f"FeetechBackend drives {_N_JOINTS} joints ({', '.join(_FOLLOWER_JOINTS)}); "
        f"got limit shapes lower={lo.shape}, upper={hi.shape}")
    if np.any(hi < lo):
      raise ValueError("every upper bound must be >= its lower bound")
    home = np.asarray(home_qpos, dtype=np.float64)
    if home.shape != (_N_JOINTS,):
      raise ValueError(
        f"home_qpos must have shape ({_N_JOINTS},); got {home.shape}")
    if np.any(home < lo) or np.any(home > hi):
      raise ValueError(
        "home_qpos must lie within the joint envelope; got "
        f"{np.asarray(home).tolist()} outside [{lo.tolist()}, {hi.tolist()}]")
    if int(slow_decimation) < 1:
      raise ValueError("slow_decimation must be >= 1")
    if int(slow_max_skips) < 1:
      raise ValueError("slow_max_skips must be >= 1")
    if float(stale_after_s) <= 0:
      raise ValueError("stale_after_s must be positive")

    self._port                            = port
    self._lower                           = lo.copy()
    self._upper                           = hi.copy()
    self._home                            = home.copy()
    self._jaw_lower                       = float(lo[-1])
    self._jaw_upper                       = float(hi[-1])
    self._calibration_id                  = calibration_id
    self._use_degrees                     = bool(use_degrees)
    self._slow_decimation                 = int(slow_decimation)
    self._slow_max_skips                  = int(slow_max_skips)
    self._stale_after_s                   = float(stale_after_s)
    self._temp_limit_c                    = float(temp_limit_c)
    self._load_limit                      = float(load_limit)
    self._volt_min                        = float(volt_min)
    self._calibrate                       = bool(calibrate)
    self._max_relative_target             = max_relative_target
    self._disable_torque_on_disconnect    = bool(disable_torque_on_disconnect)

    # Bus state. The control thread owns the bus (read_state /
    # write_positions); other threads observe via last_positions (cache only).
    # _lock is held across every bus transaction so that even a misbehaving
    # extra caller cannot interleave packets on the half-duplex bus — two
    # concurrent sync-reads corrupt each other's status packets.
    self._lock                            = threading.Lock()
    self._cached: np.ndarray | None       = None
    self._follower: Any = None  # lerobot SO101Follower (untyped), built in start()

    # read_state scheduling. _fast_t / _slow_t are the monotonic stamps of the
    # last successful block read, and are what the state's ages are measured
    # from; _slow_skips counts consecutive budget-denied slow reads so a
    # persistently tight budget cannot starve thermal monitoring forever.
    self._fast_t: float | None            = None
    self._slow_t: float | None            = None
    self._slow_count                      = 0
    self._slow_skips                      = 0
    self._blocks_ok                       = True   # cleared if block reads are unsupported
    # Transaction costs are per-instance: two backends on different adapters
    # measure genuinely different timings, and a shared estimate would let one
    # bus mis-size the other's budget.
    self._cost                            = {
      _BLOCK_FAST.name: _BlockCost(_BLOCK_FAST.seed_cost),
      _BLOCK_SLOW.name: _BlockCost(_BLOCK_SLOW.seed_cost),
    }
    self._tier1: dict[str, np.ndarray]    = {}
    self._tier2: dict[str, np.ndarray]    = {}

  @classmethod
  def from_config(
    cls, cfg, *, port: str | None = None, home_qpos=None, **params,
  ) -> FeetechBackend:
    """Build from ``cfg.runtime`` (the harness backend convention).

    The harness calls ``backend.from_config(cfg, **runtime.backend.params)``; the
    configured ``params`` (``port``, ``home_qpos``, ``calibration_id``,
    ``slow_decimation``, …) flow through as keyword arguments. The joint-space
    envelope and jaw bounds are pulled from
    ``runtime.control_loop.joint_limits`` so the backend reports exactly the
    bounds the control loop limits against.

    ``port`` and ``home_qpos`` are the two settings that describe *this arm*
    rather than the control policy, so both live in ``runtime.backend.params``
    and both are required — there is no sane default for either.
    """
    from omegaconf import OmegaConf

    limits = cfg.runtime.control_loop.joint_limits
    if limits.get("lower") is None or limits.get("upper") is None:
      raise ValueError(
        "FeetechBackend needs explicit runtime.control_loop.joint_limits."
        "lower/upper (radians) — there is no model to read joint limits from")

    def _arr(node) -> np.ndarray:
      return np.asarray(
        OmegaConf.to_container(node, resolve=True)
        if OmegaConf.is_config(node) else node, dtype=np.float64)

    resolved_port = port if port is not None else params.pop("port", None)
    if resolved_port is None:
      raise ValueError("FeetechBackend requires a serial `port` (set runtime.backend.params.port)")
    resolved_home = home_qpos if home_qpos is not None else params.pop("home_qpos", None)
    if resolved_home is None:
      raise ValueError(
        "FeetechBackend requires an explicit `home_qpos` (radians, follower "
        "order) — set runtime.backend.params.home_qpos. A real arm's rest pose "
        "is a physical choice; there is no safe default")
    return cls(
      port=str(resolved_port),
      lower=_arr(limits.lower),
      upper=_arr(limits.upper),
      home_qpos=_arr(resolved_home),
      **params,
    )

  # -- backend protocol -----------------------------------------------------

  @property
  def n_joints(self) -> int:
    return _N_JOINTS

  def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
    return self._lower.copy(), self._upper.copy()

  @property
  def home_qpos(self) -> np.ndarray:
    return self._home.copy()

  def start(self) -> None:
    # Lazy import: keeps module import hardware-free (no serial libs loaded
    # unless a real follower is actually constructed). Mirrors the leader reader.
    from lerobot.robots.so_follower import (  # type: ignore[import-untyped]
      SO101Follower,
      SO101FollowerConfig,
    )

    cfg = SO101FollowerConfig(
      port=self._port,
      id=self._calibration_id,
      use_degrees=self._use_degrees,
      max_relative_target=self._max_relative_target,
      disable_torque_on_disconnect=self._disable_torque_on_disconnect,
    )
    self._follower = SO101Follower(cfg)
    self._follower.connect(calibrate=self._calibrate)
    # Prime the cache with one bus read on the (single) starting thread:
    # last_positions() is then always answerable without touching the bus,
    # regardless of which loop thread runs first.
    with self._lock:
      self._cached     = None
      self._fast_t     = None
      self._slow_t     = None
      self._slow_count = 0
      self._slow_skips = 0
      self._tier1      = {}
      self._tier2      = {}
      # Prime through the *block* path so the cache, the tier-1/tier-2 stores
      # and the freshness stamps are all populated before the control thread
      # starts: read_state must never see a half-initialised backend. Falls
      # back to the plain position read if the block path is unavailable (an
      # SDK without the private sync-reader surface, or a stubbed bus).
      self._blocks_ok = True
      self._fast_read_locked(time.monotonic())
      if self._cached is None:
        self._bus_read_locked()
        self._fast_t = time.monotonic()

  def stop(self) -> None:
    # Safe to call during shutdown even if never started; never raise.
    follower = self._follower
    self._follower = None
    if follower is not None:
      try:
        follower.disconnect()
      except Exception:
        logger.exception("follower disconnect failed")

  def _bus_read_locked(self) -> np.ndarray:
    """One sync-read transaction; caller must hold ``self._lock``."""
    obs = self._follower.get_observation()
    q = leader_action_to_follower_qpos(
      {k: v for k, v in obs.items() if k.endswith(".pos")},
      jaw_lower=self._jaw_lower, jaw_upper=self._jaw_upper)
    self._cached = q
    return q.copy()

  # -- budgeted read_state --------------------------------------------------

  def _block_read_locked(self, block: _Block) -> dict[str, np.ndarray]:
    """Fetch one contiguous span and decode every register it carries.

    A single ``_sync_read(addr, length, ids)`` puts the whole span in the SDK's
    receive buffer; ``getData`` then decodes any sub-range of it, so N registers
    cost one transaction. Sign decoding is delegated to lerobot's
    ``_decode_sign`` (keyed by register name) rather than reimplemented here —
    ``Present_Load`` is sign-magnitude on bit 10 and ``Present_Velocity`` on
    bit 15, which is exactly the kind of detail worth not duplicating.

    Caller must hold ``self._lock``. Timing is measured around the transaction
    only, so the cost estimate tracks the bus and not the decode.
    """
    bus = self._follower.bus
    ids = [bus.motors[m].id for m in _LEADER_MOTORS]

    t0 = time.perf_counter()
    bus._sync_read(block.addr, block.length, ids, raise_on_error=True)
    self._cost[block.name].observe(time.perf_counter() - t0)

    out: dict[str, np.ndarray] = {}
    for reg, (offset, width) in block.fields.items():
      raw = {i: bus.sync_reader.getData(i, block.addr + offset, width) for i in ids}
      decoded = bus._decode_sign(reg, raw)
      out[reg] = np.asarray([float(decoded[i]) for i in ids], dtype=np.float64)
    return out

  def _positions_from_block(self, counts: np.ndarray) -> np.ndarray:
    """Raw ``Present_Position`` counts -> follower-ordered radians.

    Routed through the same normalisation lerobot's ``sync_read`` would apply
    (calibration ranges, drive mode, per-motor norm mode) and then the same
    degrees/percentage mapping :func:`leader_action_to_follower_qpos` uses, so
    the block path and the ``get_observation`` path cannot drift apart: joints
    0..4 are degrees, joint 5 is a 0..100 jaw percentage.
    """
    bus  = self._follower.bus
    ids  = [bus.motors[m].id for m in _LEADER_MOTORS]
    norm = bus._normalize({i: int(c) for i, c in zip(ids, counts)})
    return leader_action_to_follower_qpos(
      {f"{m}.pos": float(norm[i]) for m, i in zip(_LEADER_MOTORS, ids)},
      jaw_lower=self._jaw_lower, jaw_upper=self._jaw_upper)

  def _health_flags(self) -> FaultFlags:
    """Threshold the latest diagnostics into health flags.

    Only fields actually measured are checked — an absent tier-2 block means
    no thermal verdict, not a healthy one.
    """
    flags = FaultFlags.NONE
    temp  = self._tier2.get("temp")
    load  = self._tier1.get("load")
    volt  = self._tier2.get("volt")
    if temp is not None and np.any(temp > self._temp_limit_c):
      flags |= FaultFlags.OVER_TEMP
    if load is not None and np.any(np.abs(load) > self._load_limit):
      flags |= FaultFlags.OVER_LOAD
    if volt is not None and np.any(volt < self._volt_min):
      flags |= FaultFlags.UNDER_VOLT
    return flags

  def _fast_read_locked(self, now: float) -> FaultFlags:
    """Read the position block, refreshing q and the tier-1 diagnostics.

    On a bus error the cached values are kept and ``BUS_ERROR`` is raised as a
    flag rather than an exception: one corrupt status packet must not kill the
    control thread, and the caller still gets a usable (if ageing) state.
    """
    if not self._blocks_ok:
      # No block support on this SDK/bus: fall back to the plain position
      # sync-read. Positions stay correct; the tier-1 diagnostics are simply
      # absent, which the optional ControlState fields already express.
      try:
        self._bus_read_locked()
      except Exception:
        logger.warning("position read failed; reusing cache", exc_info=True)
        return FaultFlags.BUS_ERROR
      # Drop any diagnostics captured before the latch: this path cannot
      # refresh them, and a stale value reported as current is worse than an
      # absent one — the whole point of the optional fields.
      self._tier1.clear()
      self._tier2.pop("volt", None)
      self._tier2.pop("temp", None)
      self._fast_t = now
      if self._slow_t is None:
        self._slow_t = now
      return FaultFlags.NONE

    try:
      raw = self._block_read_locked(_BLOCK_FAST)
    except (AttributeError, TypeError):
      # The private sync-reader surface this path relies on is missing. That is
      # a permanent property of the bus, not a transient failure, so latch it
      # off and retry as a position read rather than failing every tick.
      logger.warning(
        "block read unsupported; read_state degrades to position-only",
        exc_info=True)
      self._blocks_ok = False
      return self._fast_read_locked(now)
    except Exception:
      logger.warning("fast block read failed; reusing cache", exc_info=True)
      return FaultFlags.BUS_ERROR

    self._cached = self._positions_from_block(raw["Present_Position"])
    self._tier1  = {
      "qd":   raw["Present_Velocity"] * _VEL_RAD_PER_CNT,
      "load": raw["Present_Load"] / _LOAD_FULL_SCALE,
    }
    # Voltage and temperature ride the fast block for free, so they land in
    # tier 2 (the "slow diagnostics" the state exposes) despite being fresh.
    self._tier2.update({
      "volt": raw["Present_Voltage"] * _VOLT_PER_CNT,
      "temp": raw["Present_Temperature"],
    })
    self._fast_t = now
    if self._slow_t is None:
      self._slow_t = now
    return FaultFlags.NONE

  def _slow_read_locked(self, now: float) -> FaultFlags:
    """Read the tier-2 (current) block."""
    if not self._blocks_ok:
      return FaultFlags.NONE
    try:
      raw = self._block_read_locked(_BLOCK_SLOW)
    except Exception:
      logger.warning("slow block read failed; reusing cache", exc_info=True)
      return FaultFlags.BUS_ERROR

    self._tier2["curr"] = raw["Present_Current"] * _CURR_PER_CNT
    self._slow_t        = now
    self._slow_skips    = 0
    return FaultFlags.NONE

  def read_state(self, budget_s: float = math.inf) -> ControlState:
    """Measure the arm within ``budget_s``, joint positions first.

    The schedule, in priority order:

    1. The fast block (position + velocity + load + voltage + temperature) is
       read whenever its estimated cost fits the budget. There is no
       position-only path — the diagnostics in that span are free.
    2. If the budget will not cover even that, the cached position is returned
       with a truthful ``q_age`` and ``READ_STARVED``. The tick still runs; it
       just knows it is flying on old data.
    3. The tier-2 block (current) is read only when its decimation counter is
       due *and* the slack left after the fast block covers it. Deferrals are
       counted, and after ``slow_max_skips`` consecutive denials one is forced
       through regardless — a missed thermal fault is a worse outcome than a
       single late tick.

    Never raises on a bus error (it becomes a flag), and never blocks.
    """
    if self._follower is None:
      raise RuntimeError("FeetechBackend.read_state before start()")

    now    = time.monotonic()
    t0     = time.perf_counter()
    budget = float(budget_s)
    flags  = FaultFlags.NONE

    with self._lock:
      # -- tier 1 ------------------------------------------------------------
      if budget >= self._cost[_BLOCK_FAST.name].cost or self._cached is None:
        # `self._cached is None` forces the very first read through regardless
        # of budget: there is no cache to fall back on, so a starved first tick
        # would have no position at all.
        flags |= self._fast_read_locked(now)
      else:
        flags |= FaultFlags.READ_STARVED

      if self._cached is None:
        raise RuntimeError("FeetechBackend.read_state could not obtain any position")

      q_age = 0.0 if self._fast_t is None else max(now - self._fast_t, 0.0)
      if q_age > self._stale_after_s:
        flags |= FaultFlags.STALE_Q

      # -- tier 2 ------------------------------------------------------------
      self._slow_count += 1
      if self._slow_count >= self._slow_decimation:
        # Charge the fast block at its *estimate*, not at what it happened to
        # cost this time: the decision is about whether both blocks fit a
        # typical tick, and one unusually quick transaction should not talk the
        # scheduler into work the next tick cannot afford.
        reserved = self._cost[_BLOCK_FAST.name].cost
        forced   = self._slow_skips >= self._slow_max_skips
        if forced or budget - reserved >= self._cost[_BLOCK_SLOW.name].cost:
          self._slow_count = 0
          flags |= self._slow_read_locked(now)
        else:
          self._slow_skips += 1
          flags |= FaultFlags.SLOW_DEFERRED

      q        = self._cached.copy()
      tier1    = dict(self._tier1)
      tier2    = dict(self._tier2)
      slow_age = math.inf if self._slow_t is None else max(now - self._slow_t, 0.0)
      flags   |= self._health_flags()

    return ControlState(
      now=now, q=q, q_age=q_age,
      qd=tier1.get("qd"), load=tier1.get("load"),
      volt=tier2.get("volt"), temp=tier2.get("temp"), curr=tier2.get("curr"),
      slow_age=slow_age, read_s=time.perf_counter() - t0, faults=flags)

  def last_positions(self) -> np.ndarray:
    """Freshest cached reading, no bus I/O — the observer (policy-loop) path.

    ``start()`` primes the cache, so this is always answerable after start.
    """
    if self._follower is None:
      raise RuntimeError("FeetechBackend.last_positions before start()")
    with self._lock:
      assert self._cached is not None, "start() must prime the cache"
      return self._cached.copy()

  def write_positions(self, q: np.ndarray) -> None:
    """Command joint setpoints (follower-ordered radians) to the servos."""
    if self._follower is None:
      raise RuntimeError("FeetechBackend.write_positions before start()")
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (_N_JOINTS,):
      raise ValueError(f"write_positions expects ({_N_JOINTS},); got {q.shape}")
    action = follower_qpos_to_action(
      q, jaw_lower=self._jaw_lower, jaw_upper=self._jaw_upper)
    with self._lock:
      self._follower.send_action(action)
