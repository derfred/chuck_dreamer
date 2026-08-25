"""Cartesian workspace envelope: a box the end-effector may not leave.

The control loop's other safety layer (``ControlLoop._safety_limit``) is
joint-space: it clamps each joint into its own range. That says nothing about
where the *tip* ends up, and a joint-legal pose can still put the gripper
through the mat, off the table edge, or into the camera rig. This module adds
the Cartesian half:

* a **box**, ``lower``/``upper`` in metres, in the arm-base frame — the frame
  FK natively lands in;
* a **slowdown margin** (``workspace_margin_m``): inside the box but within
  that distance of a wall, the
  commanded Cartesian step is scaled down, reaching exactly zero *at* the
  wall, so the arm decelerates into the boundary instead of being clipped at
  it. The scale is a single scalar applied to the whole joint step, so the
  motion's direction is preserved -- clamping per-axis would bend the path
  sideways along the wall, which is a different (and surprising) motion than
  the policy asked for;
* a **breach check**: if the *measured* tip is outside the box the envelope is
  already violated -- nothing the limiter does to the next command can undo
  that -- so it is reported as a breach and the loop latches an e-stop that
  only an explicit ``release()`` clears.

Only the outward component of a step is limited. A command that moves *away*
from a near wall is never slowed, so the arm can always retreat from (and be
released out of) a boundary it is pressed against.

The tip position comes from the same Pinocchio FK the offline pipeline uses
(:mod:`chuck_dreamer.common.fk`), so "where the arm is" means the same thing
online and in the recorded trajectories. FK covers the five positioning
joints; the jaw does not move the tip and is passed through untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

__all__ = ["WorkspaceBox", "WorkspaceLimiter", "WorkspaceLimit", "build_workspace_limiter"]

# FK positions the tip from the first five joints; the jaw (joint 6) does not.
N_POSITIONING_JOINTS = 5


@dataclass(frozen=True)
class WorkspaceBox:
  """An axis-aligned box in the arm-base frame, metres.

  Held as the per-axis minimum and maximum. Config states the box as the two
  opposite corners that span it, which :meth:`from_corners` sorts into this
  form -- a corner pair is the natural way to write a box down, but every
  reader here wants "which way is out on this axis".
  """

  lower: np.ndarray   # (3,) metres, arm frame; per-axis minimum
  upper: np.ndarray   # (3,) metres, arm frame; per-axis maximum

  def __post_init__(self) -> None:
    object.__setattr__(self, "lower", np.asarray(self.lower, dtype=np.float64).reshape(3))
    object.__setattr__(self, "upper", np.asarray(self.upper, dtype=np.float64).reshape(3))
    if np.any(self.upper <= self.lower):
      raise ValueError(f"workspace box is empty on some axis: lower={self.lower.tolist()} upper={self.upper.tolist()}")

  @classmethod
  def from_corners(cls, a, b) -> "WorkspaceBox":
    """The box spanned by two opposite corners, given in either order."""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    return cls(lower=np.minimum(a, b), upper=np.maximum(a, b))

  def contains(self, p: np.ndarray, *, tol: float = 0.0) -> bool:
    """Whether ``p`` (arm frame, metres) is inside, allowing ``tol`` slop."""
    p = np.asarray(p, dtype=np.float64)
    return bool(np.all(p >= self.lower - tol) and np.all(p <= self.upper + tol))

  def outside_by(self, p: np.ndarray) -> float:
    """How far outside the box ``p`` is (metres); ``0.0`` when inside."""
    p = np.asarray(p, dtype=np.float64)
    return float(np.max(np.maximum(np.maximum(self.lower - p, p - self.upper), 0.0)))

  def slack(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from ``p`` to the lower and upper wall on each axis.

    Both are clamped at zero: a point already outside has no remaining travel
    in that direction, which is what the limiter needs (negative slack would
    read as "may still move outward").
    """
    p = np.asarray(p, dtype=np.float64)
    return np.maximum(p - self.lower, 0.0), np.maximum(self.upper - p, 0.0)


@dataclass(frozen=True)
class WorkspaceLimit:
  """What the limiter did to one tick's command, for telemetry."""

  scale: float          # 1.0 = untouched, 0.0 = fully held
  breach_m: float       # how far the *measured* tip is outside the box
  ee_pos: np.ndarray    # (3,) measured tip, arm frame, metres

  @property
  def limited(self) -> bool:
    return self.scale < 1.0

  @property
  def breached(self) -> bool:
    return self.breach_m > 0.0


class WorkspaceLimiter:
  """Scales joint commands so the tip decelerates to a stop at the box wall.

  One instance per control loop; ``limit`` is called on the control thread
  every tick and must stay cheap (it is two Pinocchio calls plus arithmetic).
  Not thread-safe -- :class:`~chuck_dreamer.common.fk.FK` holds mutable
  Pinocchio ``Data``, and only the control thread touches this.
  """

  def __init__(
    self,
    fk: Any,
    box: WorkspaceBox,
    *,
    margin_m: float = 0.05,
    dq: np.ndarray | None = None,
    verify_iters: int = 8,
  ) -> None:
    if margin_m <= 0.0:
      raise ValueError(f"workspace margin_m must be positive; got {margin_m}")
    self._fk       = fk
    self._box      = box
    self._margin   = float(margin_m)
    self._dq       = None if dq is None else np.asarray(dq, dtype=np.float64).reshape(N_POSITIONING_JOINTS)
    self._verify_iters = int(verify_iters)

  @property
  def box(self) -> WorkspaceBox:
    return self._box

  @property
  def margin_m(self) -> float:
    """Width of the slowdown ramp inside each wall, metres."""
    return self._margin

  def ee_pos(self, q: np.ndarray) -> np.ndarray:
    """Tip position (arm frame, metres) for a full joint vector."""
    return np.asarray(self._fk(self._positioning(q), self._dq), dtype=np.float64).reshape(3)

  def _positioning(self, q: np.ndarray) -> np.ndarray:
    """The five FK joints of ``q``, zero-padded if the arm reports fewer."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size >= N_POSITIONING_JOINTS:
      return q[:N_POSITIONING_JOINTS]
    padded                = np.zeros(N_POSITIONING_JOINTS, dtype=np.float64)
    padded[:q.size]       = q
    return padded

  def limit(self, q_meas: np.ndarray, q_cmd: np.ndarray) -> tuple[np.ndarray, WorkspaceLimit]:
    """Constrain ``q_cmd`` against the box, given where the arm actually is.

    Returns the (possibly scaled-back) command and a :class:`WorkspaceLimit`
    describing the tick. The measured pose is the anchor: the step is
    ``q_cmd - q_meas``, its Cartesian effect is read off the Jacobian *at the
    measurement*, and the scale that keeps that effect inside the box is
    applied to the whole step. Scaling the step (rather than editing the
    endpoint) is what makes the constraint velocity-shaped -- the closer the
    tip is to a wall, the smaller the step it is allowed to take, so the
    commanded speed tapers to zero exactly at the boundary.

    A breach is reported, never "fixed": the returned command holds position,
    because the only correct response to being outside the envelope is to stop
    and let a human decide.
    """
    q_meas = np.asarray(q_meas, dtype=np.float64)
    q_cmd  = np.asarray(q_cmd,  dtype=np.float64)

    p        = self.ee_pos(q_meas)
    breach_m = self._box.outside_by(p)
    if breach_m > 0.0:
      return q_meas.copy(), WorkspaceLimit(scale=0.0, breach_m=breach_m, ee_pos=p)

    step = q_cmd - q_meas
    if not np.any(step):
      return q_cmd, WorkspaceLimit(scale=1.0, breach_m=0.0, ee_pos=p)

    # Cartesian displacement this step would produce, to first order.
    dp    = self._fk.jacobian(self._positioning(q_meas), self._dq) @ self._positioning(step)
    scale = self._verified(q_meas, step, self._scale_for(p, dp))
    return q_meas + scale * step, WorkspaceLimit(scale=scale, breach_m=0.0, ee_pos=p)

  def _verified(self, q_meas: np.ndarray, step: np.ndarray, scale: float) -> float:
    """``scale``, backed off until the scaled command's *actual* FK is inside.

    :meth:`_scale_for` works off the Jacobian, which is exact only in the
    limit: over a large joint step the tip's true path curves away from the
    linear prediction and can end up outside a box the first-order estimate
    said it would stay inside. At the rates this loop runs, one tick's step is
    small enough that the error is microscopic -- but "small enough" is a
    property of the configured velocity limit, not of the envelope, and a
    safety bound should not quietly depend on a limit set elsewhere.

    So the scaled endpoint is checked against real FK and halved until it
    holds. The loop terminates: ``scale = 0`` is the measured pose, which is
    inside by the breach check above, so the worst case is a hold.
    """
    for _ in range(self._verify_iters):
      if scale <= 0.0 or self._box.contains(self.ee_pos(q_meas + scale * step)):
        return scale
      scale *= 0.5
    return 0.0

  def _scale_for(self, p: np.ndarray, dp: np.ndarray) -> float:
    """Largest ``s`` in [0, 1] with ``p + s*dp`` inside the ramped envelope.

    Per axis and per wall: the outward part of the step may cover at most the
    remaining distance to that wall, damped by ``d / margin`` once inside the
    margin. The damping is what makes the limit a *velocity* constraint rather
    than a position clip -- at ``d = margin`` the step passes untouched, and it
    tapers linearly to zero as ``d -> 0``, so the tip converges on the wall
    without ever crossing it. The binding axis wins, and the single scalar is
    applied to every joint so the path keeps its direction.
    """
    to_lower, to_upper = self._box.slack(p)
    scale              = 1.0
    for axis in range(3):
      move = float(dp[axis])
      if move == 0.0:
        continue
      # Distance to the wall this axis is moving toward.
      d = float(to_upper[axis] if move > 0.0 else to_lower[axis])
      # Allowed outward travel: the full step until the margin, then damped.
      budget = d * min(d / self._margin, 1.0)
      if abs(move) > budget:
        scale = min(scale, budget / abs(move))
    return max(0.0, min(1.0, scale))


# -- construction from config -------------------------------------------------


def _corners(value) -> tuple[np.ndarray, np.ndarray]:
  """The two ``[x, y, z]`` corners of a ``workspace`` config value."""
  raw = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
  arr = np.asarray(raw, dtype=np.float64)
  if arr.shape != (2, 3):
    raise ValueError(
      f"runtime.control_loop.safety.workspace must be the two opposite corners of the "
      f"box, [[x, y, z], [x, y, z]] in metres; got shape {arr.shape}")
  return arr[0], arr[1]


def build_workspace_limiter(cfg: DictConfig) -> WorkspaceLimiter | None:
  """Build the limiter from ``runtime.control_loop.safety.workspace``, or ``None``.

  No box means no Cartesian envelope, which is the default: the box is rig
  geometry and there is no safe value to guess. Stating one turns the envelope
  on -- there is no separate enable flag, because a box that is configured but
  switched off is a trap, reading as protection that is not there.

  The corners are arm-frame. A table-frame box would have to be rotated through
  ``table_to_arm`` first, and how that transform reaches the runtime is not
  settled; convert the bounds yourself for now.
  """
  ws = OmegaConf.select(cfg, "control_loop.safety.workspace")
  if ws is None:
    return None

  box = WorkspaceBox.from_corners(*_corners(ws))

  from chuck_dreamer.common import FK_MODEL_PATH
  from chuck_dreamer.common.fk import FK, load_fk_dq

  fk_cfg = OmegaConf.select(cfg, "fk") or {}
  dq_dir = fk_cfg.get("dq_dir")
  return WorkspaceLimiter(
    FK(fk_cfg.get("model") or FK_MODEL_PATH), box,
    margin_m=float(OmegaConf.select(cfg, "control_loop.safety.workspace_margin_m") or 0.05),
    dq=None if dq_dir is None else load_fk_dq(str(dq_dir)),
  )
