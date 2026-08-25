"""Ground-truth forward kinematics for the SO-101 arm via Pinocchio.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pinocchio as pin

# Canonical real-robot joint names, in the order the arm reports them. The
# public `q` argument is in this order, and it matches the URDF's own joint
# ordering (Pinocchio assigns idx_q 0..4 to exactly these).
CANONICAL_JOINT_NAMES = [
  "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
]

# URDF frame for the gripper tip. `gripper_frame_link` is the fixed TCP frame
# hanging off `gripper_link`; it is the URDF's counterpart to the MJCF's
# `ee_tip` site, and the two agree to <1 mm once the models are aligned.
EE_FRAME_NAME = "gripper_frame_link"

# Nominal home pose: the arm fully extended along +x, which is the URDF's
# zero. Unlike the MJCF this needs no half-turn entries — the URDF's encoder
# zero is the real arm's.
Q_HOME = np.zeros(5, dtype=np.float64)


def load_fk_dq(cache_dir: str | Path) -> np.ndarray:
  """Load per-joint zero offsets ``dq`` from ``<cache_dir>/fk_dq.json``.

  Returns the 5-vector added to a real-arm joint reading before FK:
  ``p_arm = fk(q_real, dq=load_fk_dq(...))``. Corrects residual per-arm
  encoder-zero error on top of the URDF's calibration.

  Returns zeros if the file doesn't exist — i.e. uncalibrated arms still
  work, they just inherit the URDF's nominal zeros.
  """
  import json

  p = Path(cache_dir) / "fk_dq.json"
  if not p.exists():
    return np.zeros(5, dtype=np.float64)
  blob = json.loads(p.read_text())
  dq = np.asarray(blob["dq"], dtype=np.float64)
  if dq.shape != (5,):
    raise ValueError(f"{p}: dq must be shape (5,), got {dq.shape}")
  return dq


class FK:
  """Stateful FK evaluator. Not thread-safe; that is acceptable."""

  def __init__(self, model_path: str | Path,
               ee_frame_name: str = EE_FRAME_NAME):
    model_path = Path(model_path)
    if not model_path.exists():
      raise FileNotFoundError(f"URDF not found: {model_path}")
    self.model = pin.buildModelFromUrdf(str(model_path))
    self.data = self.model.createData()
    self.ee_frame_name = ee_frame_name
    if not self.model.existFrame(ee_frame_name):
      raise ValueError(f"frame '{ee_frame_name}' not found in model")
    self.ee_frame_id = self.model.getFrameId(ee_frame_name)
    missing = [n for n in CANONICAL_JOINT_NAMES
               if not self.model.existJointName(n)]
    if missing:
      raise ValueError(f"joints not found in model: {missing}")
    self.joint_ids = [self.model.getJointId(n) for n in CANONICAL_JOINT_NAMES]
    # Configuration-vector slot per canonical joint. Read from the model
    # rather than assumed, so a URDF reordering can't silently transpose q.
    self.qpos_idx = np.array(
      [self.model.joints[j].idx_q for j in self.joint_ids])

  def _configuration(self, q, dq=None) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (5,):
      raise ValueError(f"q must be shape (5,), got {q.shape}")
    if dq is None:
      dq = np.zeros(5)
    else:
      dq = np.asarray(dq, dtype=np.float64)
      if dq.shape != (5,):
        raise ValueError(f"dq must be shape (5,), got {dq.shape}")
    # Unlisted joints (the jaw) stay at zero: they do not move the tip.
    full = np.zeros(self.model.nq, dtype=np.float64)
    full[self.qpos_idx] = q + dq
    return full

  def __call__(self, q, dq=None) -> np.ndarray:
    pin.forwardKinematics(self.model, self.data, self._configuration(q, dq))
    pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
    return np.asarray(self.data.oMf[self.ee_frame_id].translation.copy())

  def jacobian(self, q, dq=None) -> np.ndarray:
    """``(3, 5)`` translational Jacobian of the EE frame, world-aligned.

    Row ``i``, column ``j`` is d(tip position along world axis i)/d(q_j) for
    the canonical joints, in the same order as :attr:`CANONICAL_JOINT_NAMES`.
    ``LOCAL_WORLD_ALIGNED`` gives the derivative in *base* axes (the frame the
    workspace box is expressed in) rather than the tip's own rotating axes.
    """
    J = pin.computeFrameJacobian(
      self.model, self.data, self._configuration(q, dq), self.ee_frame_id,
      pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return np.asarray(J[:3, self.qpos_idx])

  def fk_chain(self, q, dq=None) -> np.ndarray:
    """World positions of each joint frame along the arm, plus the EE."""
    pin.forwardKinematics(self.model, self.data, self._configuration(q, dq))
    pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
    pts = [self.data.oMi[j].translation.copy() for j in self.joint_ids]
    pts.append(self.data.oMf[self.ee_frame_id].translation.copy())
    return np.asarray(pts)


# ---------------------------------------------------------------------------
# Verify CLI
# ---------------------------------------------------------------------------

def _verify(model_path: str, delta: float) -> None:
  fk = FK(model_path)

  # Check 1: the chain's link lengths, which are frame-independent and so
  # comparable against the datasheet by eye.
  chain = fk.fk_chain(Q_HOME)
  seg   = np.linalg.norm(np.diff(chain, axis=0), axis=1)
  print("[check 1] link lengths at home pose (m):")
  names = CANONICAL_JOINT_NAMES + [fk.ee_frame_name]
  for a, b, d in zip(names, names[1:], seg):
    print(f"           {a:>15s} -> {b:<20s} {d:.5f}")
  print()

  # Check 2: joint sign sanity.
  p0 = fk(Q_HOME)
  print("[check 2] joint-sign sanity (delta = "
        f"{delta:+.3f} rad on each joint, others at Q_HOME):")
  print(f"           {'joint':>15s}  {'disp (m)':<30s}  dominant")
  for i, name in enumerate(CANONICAL_JOINT_NAMES):
    q1       = Q_HOME.copy()
    q1[i]   += delta
    disp     = fk(q1) - p0
    axis_i   = int(np.argmax(np.abs(disp)))
    axis     = ["x", "y", "z"][axis_i]
    sign     = "+" if disp[axis_i] >= 0 else "-"
    disp_str = f"[{disp[0]:+.4f} {disp[1]:+.4f} {disp[2]:+.4f}]"
    print(f"           {name:>15s}  {disp_str:<30s}  {sign}{axis}")
  print()

  # Check 3: home pose sanity.
  print("[check 3] home pose:")
  print(f"           q_home = {Q_HOME.tolist()}")
  print(f"           p_home = {fk(Q_HOME).tolist()}")


def main() -> None:
  p   = argparse.ArgumentParser(description="SO-101 forward kinematics utility.")
  sub = p.add_subparsers(dest="cmd", required=True)
  v   = sub.add_parser("verify", help="Run the three verification checks.")
  v.add_argument("--model", required=True, help="Path to the SO-101 URDF.")
  v.add_argument("--delta", type=float, default=0.1,
                 help="Per-joint perturbation for the sign-sanity check (rad).")
  args = p.parse_args()
  if args.cmd == "verify":
    _verify(args.model, args.delta)


if __name__ == "__main__":
  main()
