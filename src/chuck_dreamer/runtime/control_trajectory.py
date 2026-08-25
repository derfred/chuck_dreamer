import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ControlTrajectoryConfig:
  v_max: np.ndarray             # rad/s,   per joint
  a_max: np.ndarray             # rad/s^2, per joint
  P_nominal: float              # initial policy-period estimate [s]

  beta: float = 0.5             # terminal-velocity damping; 0.0 disables entirely
  kappa_inv: float = 2.0        # nominal horizon T = kappa_inv * P

  P_min: float = 0.020          # clamp on a single measured policy interval
  P_max: float = 0.300
  P_alpha: float = 0.1          # EMA weight for the smoothed period estimate

  t_brake: float = 0.080        # coast-to-stop duration [s]
  v_eps_frac: float = 1e-6      # "At rest" threshold, as a fraction of v_max
  dt_max: float = 0.050         # clamp on one control tick, guards GC pauses
  horizon_iters: int = 4        # limit-driven horizon refinement passes
  horizon_samples: int = 17     # sample count for the limit check

  @classmethod
  def build(cls, cfg, n_joints) -> "ControlTrajectoryConfig":
    """Construct from a DictConfig, broadcasting scalars to per-joint."""
    v_max = np.broadcast_to(np.asarray(cfg.control_loop.safety.max_velocity, dtype=float), (n_joints,))
    a_max = np.broadcast_to(np.asarray(cfg.control_loop.safety.max_acceleration, dtype=float), (n_joints,))

    params: dict[str, Any] = {}
    if hasattr(cfg.control_loop.safety, "coast_to_stop_time"):
      params["t_brake"] = float(cfg.control_loop.safety.coast_to_stop_time)

    return cls(v_max=v_max, a_max=a_max, P_nominal=1/float(cfg.policy_rate_hz), **params)

  def __post_init__(self):
    self.v_max = np.asarray(self.v_max, dtype=float)
    self.a_max = np.asarray(self.a_max, dtype=float)

  @property
  def v_eps(self) -> np.ndarray:
    """Per-joint velocity below which the reference counts as at rest."""
    return self.v_eps_frac * self.v_max


def _coeffs(q0, v0, a0, q1, v1, T):
  """Coefficients over normalised time s = t/T in [0, 1].

  Boundary conditions: (q0, v0, a0) at s=0 and (q1, v1, 0) at s=1.
  Terminal acceleration is always zero -- there is no sensible estimator for
  it, and zero makes the hand-off into a braking segment continuous.
  """
  w0 = v0 * T
  w1 = v1 * T
  z0 = a0 * T * T
  d  = q1 - q0

  c = np.empty(q0.shape + (6,), dtype=float)
  c[..., 0] = q0
  c[..., 1] = w0
  c[..., 2] = 0.5 * z0
  c[..., 3] =  10.0 * d - 6.0 * w0 - 4.0 * w1 - 1.5 * z0
  c[..., 4] = -15.0 * d + 8.0 * w0 + 7.0 * w1 + 1.5 * z0
  c[..., 5] =   6.0 * d - 3.0 * w0 - 3.0 * w1 - 0.5 * z0
  return c


def _basis(s):
  """Position / velocity / acceleration basis vectors in normalised time."""
  s = np.asarray(s, dtype=float)
  o = np.ones_like(s)
  z = np.zeros_like(s)
  p = np.stack([o, s, s**2, s**3, s**4, s**5])
  v = np.stack([z, o, 2.0 * s, 3.0 * s**2, 4.0 * s**3, 5.0 * s**4])
  a = np.stack([z, z, 2.0 * o, 6.0 * s, 12.0 * s**2, 20.0 * s**3])
  return p, v, a


def _violation_scale(c, T, cfg):
  """How much T must grow to satisfy the velocity / acceleration limits.

  Returns <= 1.0 if the segment is already within limits.  Velocity scales as
  1/T and acceleration as 1/T^2, hence the sqrt on the acceleration term.  The
  coefficients themselves depend on T, so this is iterated by the caller.
  """
  ss = np.linspace(0.0, 1.0, cfg.horizon_samples)
  _, sv, sa = _basis(ss)

  qd  = (c @ sv) / T
  qdd = (c @ sa) / (T * T)

  rv = np.max(np.abs(qd)  / cfg.v_max[:, None])
  ra = np.max(np.abs(qdd) / cfg.a_max[:, None])
  return float(max(rv, math.sqrt(max(ra, 0.0))))


def _select_horizon(q0, v0, a0, q1, v1, cfg, P_est):
  """Nominal horizon, extended (never shortened) to respect joint limits.

  The floor at kappa_inv * P_est is the important half: without it a small
  commanded motion produces a very short segment that finishes long before the
  next action arrives, and the arm stutters to a halt on every cycle.
  """
  t_floor = cfg.kappa_inv * P_est
  T = t_floor
  for _ in range(cfg.horizon_iters):
    scale = _violation_scale(_coeffs(q0, v0, a0, q1, v1, T), T, cfg)
    if scale <= 1.0:
      break
    T *= min(scale, 2.0)
  return max(T, t_floor)


class ControlTrajectory:
  """One planned quintic segment.

  Immutable except for `s`, which `tick` advances. `update` and `brake` return new instances.
  """

  __slots__ = ("c", "T", "s", "q_end", "qd_end", "q_prev_end", "t_target", "P_est", "cfg")

  def __init__(self, c, T, q_end, qd_end, q_prev_end, t_target, P_est, cfg):
    self.c          = c
    self.T          = T
    self.s          = 0.0
    self.q_end      = q_end
    self.qd_end     = qd_end
    self.q_prev_end = q_prev_end
    self.t_target   = t_target
    self.P_est      = P_est
    self.cfg        = cfg

  # -- Execution

  def eval(self, s = None):
    """Reference (q, qd, qdd) at normalised time s, defaulting to now."""
    s          = self.s if s is None else s
    s          = min(max(float(s), 0.0), 1.0)
    sp, sv, sa = _basis(s)
    q          = self.c @ sp
    qd         = (self.c @ sv) / self.T
    qdd        = (self.c @ sa) / (self.T * self.T)
    return q, qd, qdd

  @property
  def expired(self):
    return self.s >= 1.0

  @property
  def stationary(self):
    """Whether this trajectory has stopped commanding motion.

    Asks the reference itself, at its current position on the segment: a hold
    is stationary from the start, a coast-down only once it has played out.

    Distinct from `expired`, which is about the segment running out of time: a
    hold has T = inf and never expires, yet is stationary throughout.
    """
    return bool(np.all(np.abs(self.eval()[1]) <= self.cfg.v_eps))

  def tick(self, dt):
    """Advance and return the next joint command, or None if the segment ran out."""
    dt      = min(float(dt), self.cfg.dt_max)
    self.s += dt / self.T
    if self.s >= 1.0:
      return None
    return self.eval(self.s)

  # -- Planning
  @classmethod
  def hold(cls, q: np.ndarray, cfg: ControlTrajectoryConfig) -> "ControlTrajectory":
    """Construct a HOLD trajectory (zero velocity, zero acceleration)."""
    q         = np.asarray(q, dtype=float)
    c         = np.zeros(q.shape + (6,), dtype=float)
    c[..., 0] = q
    return cls(c=c, T=math.inf, q_end=q, qd_end=np.zeros_like(q), q_prev_end=None, t_target=None, P_est=cfg.P_nominal, cfg=cfg)

  def update(self, action, state, q_ref_actual=None) -> "ControlTrajectory":
    """Plan a new segment toward `action.q`."""
    cfg = self.cfg
    q_t = np.asarray(action.q, dtype=float)

    if self.t_target is None:
      # first action after a reset: seed from measurement, no velocity history
      q0    = np.asarray(state.q, dtype=float)
      v0    = np.zeros_like(q0)
      a0    = np.zeros_like(q0)
      v1    = np.zeros_like(q0)
      P_est = cfg.P_nominal
    else:
      q0, v0, a0 = self.eval()

      if q_ref_actual is not None:
        q0 = q_ref_actual  # incase the control loop safety-limited the last command

      # instantaneous interval for the finite difference; smoothed one for the
      # horizon, so a single slow inference does not stretch the next segment
      dt_policy = float(np.clip(action.obs.t - self.t_target, cfg.P_min, cfg.P_max))
      P_est     = (1.0 - cfg.P_alpha) * self.P_est + cfg.P_alpha * dt_policy

      dq = q_t - self.q_end
      v1 = cfg.beta * dq / dt_policy

      # do not carry velocity through a direction reversal, or an oscillating
      # policy makes the arm ring as it overshoots on every half cycle
      if self.q_prev_end is not None:
        dq_prev = self.q_end - self.q_prev_end
        v1      = np.where(np.sign(dq) * np.sign(dq_prev) < 0.0, 0.0, v1)

      v1 = np.clip(v1, -cfg.v_max, cfg.v_max)

    T = _select_horizon(q0, v0, a0, q_t, v1, cfg, P_est)
    c = _coeffs(q0, v0, a0, q_t, v1, T)

    return ControlTrajectory(c=c, T=T, q_end=q_t, qd_end=v1, q_prev_end=self.q_end, t_target=action.obs.t, P_est=P_est, cfg=cfg)

  def stop(self, q=None) -> "ControlTrajectory":
    """Hold at ``q``, defaulting to the reference *right now*."""
    if q is None:
      q, _, _ = self.eval()
    return ControlTrajectory.hold(q, self.cfg)

  def brake(self) -> "ControlTrajectory":
    """Smooth coast-to-stop from where the reference is *now*."""
    q0, v0, a0 = self.eval()
    if not np.any(v0):
      return self.stop(q0)

    tb = self.cfg.t_brake
    q1 = q0 + v0 * tb / 2.0                      # clamp to joint limits here
    z  = np.zeros_like(q1)
    c  = _coeffs(q0, v0, a0, q1, z, tb)

    return ControlTrajectory(c=c, T=tb, q_end=q1, qd_end=z, q_prev_end=None, t_target=None, P_est=self.P_est, cfg=self.cfg)
