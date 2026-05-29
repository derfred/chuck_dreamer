"""Arm-to-world calibration: 4-touch Umeyama alignment.

The operator drives the gripper to known mat fiducials. We record the
joint vector at each touch, run FK to get the EE position in arm frame,
then solve a rigid transform ``T_world_arm`` that maps arm-frame EE
positions into mat/world-frame positions (millimeters, mat plane at z=0).

This is a sub-step of `annotate-live`. It assumes camera extrinsics and
the world-frame definition are already established by the rest of the
flow. See `docs/calibrate_live_arm_brief.md` for the full contract.
"""
from __future__ import annotations

import datetime as _dt
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np

from chuck_dreamer.real.fk_calibration.fk import FK

# Re-exported so callers don't need a second import.
__all__ = [
  "DEFAULT_TOUCH_SEQUENCE", "TouchRecord", "world_positions",
  "umeyama", "solve_arm_to_world", "run_arm_touch_calibration",
]

# Default 4-touch sequence. The first three points are collinear (all on
# the mat x-axis at y=0 / on a single line), which leaves Umeyama
# rank-deficient in rotation about that line; the fourth, off-line touch
# breaks the degeneracy.
DEFAULT_TOUCH_SEQUENCE = ("left-middle", "origin", "right-middle", "left-near")

# Per-positioning-joint spread threshold (rad). Beyond this the touch
# wasn't really at rest and the median is suspect.
SPREAD_WARN_RAD = 0.05

# Window over which we sample joints once the user signals "touching".
RECORD_WINDOW_SEC  = 0.6
RECORD_POLL_HZ     = 60

# Quality gates on the per-touch alignment residuals (millimeters).
RES_ACCEPT_MM = 5.0
RES_WARN_MM   = 10.0


def world_positions(L_mm: float, D_mm: float) -> dict[str, np.ndarray]:
  """Canonical mat-frame positions of the 7 named touch points (mm).

  Convention (operator view, robot at the near edge of the mat):
    - Origin = circle center, directly in front of the robot.
    - **x** = left/right. +x = right, −x = left. Lines extend ±L/2 in x.
    - **y** = near/far. −y = near (closer to robot), +y = far. The two
      painted lines sit at y = ±D/2.
    - **middle** is the y-midpoint between the two lines (y = 0).
    - z = 0 (mat plane).

  The seven points form a rectangle centered on the origin: ±L/2 in x,
  ±D/2 in y. ``left-middle`` and ``right-middle`` sit between the lines
  at the rectangle's left and right edges.
  """
  return {
    "origin":       np.array([0.0,         0.0, 0.0]),
    "right-near":   np.array([+L_mm / 2.0, -D_mm / 2.0, 0.0]),
    "right-middle": np.array([+L_mm / 2.0,         0.0, 0.0]),
    "right-far":    np.array([+L_mm / 2.0, +D_mm / 2.0, 0.0]),
    "left-near":    np.array([-L_mm / 2.0, -D_mm / 2.0, 0.0]),
    "left-middle":  np.array([-L_mm / 2.0,         0.0, 0.0]),
    "left-far":     np.array([-L_mm / 2.0, +D_mm / 2.0, 0.0]),
  }


@dataclass
class TouchRecord:
  name:      str
  q:         np.ndarray   # (5,) median positioning joints (rad)
  q_spread:  np.ndarray   # (5,) max-min per joint (rad)
  p_arm:     np.ndarray   # (3,) EE in arm frame (mm)
  p_world:   np.ndarray   # (3,) target in world frame (mm)


def umeyama(P_arm: np.ndarray, P_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Closed-form rigid alignment. Returns ``(R, t)`` s.t.
  ``p_world ≈ R @ p_arm + t``. The reflection correction (``d``) is
  required; without it SVD can return a mirror rather than a rotation.
  """
  P_arm = np.asarray(P_arm, dtype=np.float64)
  P_world = np.asarray(P_world, dtype=np.float64)
  c_a = P_arm.mean(axis=0)
  c_w = P_world.mean(axis=0)
  H = (P_arm - c_a).T @ (P_world - c_w)
  U, _, Vt = np.linalg.svd(H)
  d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
  R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
  t = c_w - R @ c_a
  return R, t


# Relative threshold for declaring the touch points collinear (and hence
# the 3D Umeyama solve rank-deficient). Compares the smallest singular
# value of the centered point cloud to the largest — well-separated
# points are O(1); a degenerate line has ratio ~0.
COLLINEARITY_RATIO = 1e-3


def is_collinear(P: np.ndarray, ratio: float = COLLINEARITY_RATIO) -> bool:
  """True if ``P`` (N×3) lies on (or near) a single line."""
  P = np.asarray(P, dtype=np.float64)
  if P.shape[0] < 3:
    return True
  centered = P - P.mean(axis=0)
  s = np.linalg.svd(centered, compute_uv=False)
  if s[0] < 1e-9:
    return True
  return float(s[1] / s[0]) < ratio


def umeyama_planar(P_arm: np.ndarray, P_world: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
  """Planar (SE(2)) alignment lifted back to a 3D rigid transform.

  Assumes the mat plane and the arm-base xy-plane are parallel (the mat
  sits on the table the arm is bolted to). The roll about the world z
  is unobservable from points on a line, so we pin it: R rotates only
  about z, and the z-translation is the mean per-touch z-offset.

  Use this when :func:`is_collinear` flags the 3D solve as
  rank-deficient. For 3 collinear-in-xy points (e.g. mat y-axis), this
  still solves cleanly because the planar problem has 3 DOF (yaw + tx +
  ty) and 3 points give 6 scalar constraints.
  """
  P_arm = np.asarray(P_arm, dtype=np.float64)
  P_world = np.asarray(P_world, dtype=np.float64)
  A = P_arm[:, :2]
  W = P_world[:, :2]
  c_a = A.mean(axis=0)
  c_w = W.mean(axis=0)
  H = (A - c_a).T @ (W - c_w)                       # (2, 2)
  U, _, Vt = np.linalg.svd(H)
  d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
  R2 = Vt.T @ np.diag([1.0, d]) @ U.T                # (2, 2)
  t2 = c_w - R2 @ c_a                                # (2,)

  # Lift to 3D: yaw rotation about z, z-translation = mean z offset.
  R = np.eye(3, dtype=np.float64)
  R[:2, :2] = R2
  tz = float((P_world[:, 2] - P_arm[:, 2]).mean())
  t = np.array([t2[0], t2[1], tz], dtype=np.float64)
  return R, t


def solve_arm_to_world(
  touches:        list[TouchRecord],
  *,
  fk_dq:          np.ndarray | None = None,
  fk_model_path:  str | Path | None = None,
  touch_sequence: tuple[str, ...] | list[str] | None = None,
  session_id:     str | None = None,
  L_mm:           float | None = None,
  D_mm:           float | None = None,
  extra_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, np.ndarray, float, float]:
  """Solve Umeyama from a list of ``TouchRecord`` and build the JSON block.

  Returns ``(block_or_None, residuals_mm, max_residual_mm, rms_residual_mm)``.
  ``block_or_None`` is ``None`` iff the result fails the rejection gate
  (``max_residual >= RES_WARN_MM``); residuals are returned in either
  case so callers can log/report. The block is the same shape the live
  calibration emits — see ``docs/calibrate_live_arm_brief.md``.
  """
  if len(touches) < 3:
    raise ValueError(f"need ≥3 touches for Umeyama, got {len(touches)}")
  if fk_dq is None:
    fk_dq = np.zeros(5, dtype=np.float64)
  fk_dq = np.asarray(fk_dq, dtype=np.float64)

  P_arm   = np.stack([t.p_arm   for t in touches], axis=0)
  P_world = np.stack([t.p_world for t in touches], axis=0)

  # If the touch points are collinear (e.g. the three_touch prototype:
  # all three on the mat y-axis), the 3D Umeyama is rank-deficient in
  # rotation about that line. Fall back to a planar SE(2) solve that
  # assumes the mat plane and arm xy-plane are parallel — same
  # assumption the rest of the runtime makes when it commands hover
  # poses by adding a z-offset to a mat-plane target.
  solver = "umeyama_3d"
  if is_collinear(P_arm) or is_collinear(P_world):
    R, t = umeyama_planar(P_arm, P_world)
    solver = "umeyama_planar"
  else:
    R, t = umeyama(P_arm, P_world)

  residuals = np.linalg.norm((P_arm @ R.T + t) - P_world, axis=1)
  max_res = float(residuals.max())
  rms_res = float(np.sqrt(np.mean(residuals ** 2)))

  if max_res >= RES_WARN_MM:
    return None, residuals, max_res, rms_res

  block: dict[str, Any] = {
    "T_world_arm": {
      "R": R.tolist(),
      "t": t.tolist(),
    },
    "diagnostics": {
      "touches": [
        {
          "name":     tr.name,
          "q":        tr.q.tolist(),
          "q_spread": tr.q_spread.tolist(),
          "p_arm":    tr.p_arm.tolist(),
          "p_world":  tr.p_world.tolist(),
          "residual": float(r),
        }
        for tr, r in zip(touches, residuals)
      ],
      "max_residual_mm": max_res,
      "rms_residual_mm": rms_res,
      "n_touches":       len(touches),
    },
    "metadata": {
      "timestamp":                 _dt.datetime.now().isoformat(timespec="seconds"),
      "fk_dq":                     fk_dq.tolist(),
      "fk_model_path":             str(fk_model_path) if fk_model_path else "",
      "calibrate_live_session_id": session_id or uuid.uuid4().hex,
      "mat_line_length_mm":        float(L_mm) if L_mm is not None else None,
      "mat_line_separation_mm":    float(D_mm) if D_mm is not None else None,
      "touch_sequence":            list(touch_sequence) if touch_sequence else
                                   [t.name for t in touches],
      "solver":                    solver,
      # XY of the world-origin touch in the arm's base frame (mm). The
      # arm's base sits at some unknown offset/rotation from the mat
      # origin; this is the FK-derived (x, y) of the gripper when the
      # operator placed it on the circle center, so downstream code can
      # express targets as "Δ from origin in arm-XY" without needing the
      # full R/t transform.
      "origin_p_arm_xy_mm":        next(
        (tr.p_arm[:2].tolist() for tr in touches if tr.name == "origin"),
        None,
      ),
    },
  }
  if extra_metadata:
    block["metadata"].update(extra_metadata)
  return block, residuals, max_res, rms_res


def _record_touch(arm: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
  """Sample joints for RECORD_WINDOW_SEC at RECORD_POLL_HZ, return
  ``(median_q5, spread_q5)`` for the 5 positioning joints (rad).
  """
  dt = 1.0 / float(RECORD_POLL_HZ)
  n_samples = max(2, int(round(RECORD_WINDOW_SEC * RECORD_POLL_HZ)))
  samples = np.empty((n_samples, 5), dtype=np.float64)
  for i in range(n_samples):
    qpos6 = arm.read_joint_qpos()
    samples[i] = qpos6[:5]
    if i < n_samples - 1:
      time.sleep(dt)
  q_median = np.median(samples, axis=0)
  q_spread = samples.max(axis=0) - samples.min(axis=0)
  return q_median, q_spread


def run_arm_touch_calibration(
  arm:          Any,
  fk:           FK,
  L_mm:         float,
  D_mm:         float,
  *,
  fk_dq:        np.ndarray | None = None,
  fk_model_path: str | Path | None = None,
  touch_sequence: tuple[str, ...] = DEFAULT_TOUCH_SEQUENCE,
  session_id:   str | None = None,
) -> dict[str, Any] | None:
  """Drive the operator through the 4-touch sequence, solve T_world_arm.

  Returns the JSON-ready dict to merge into the calibration bundle, or
  ``None`` if the operator aborted or the alignment was rejected.

  ``fk`` returns EE position in *meters* in the arm base frame; we
  convert to millimeters here so all world-frame quantities share units.
  """
  if fk_dq is None:
    fk_dq = np.zeros(5, dtype=np.float64)
  fk_dq = np.asarray(fk_dq, dtype=np.float64)

  targets = world_positions(L_mm, D_mm)
  missing = [n for n in touch_sequence if n not in targets]
  if missing:
    raise click.ClickException(
      f"unknown touch labels in sequence: {missing}. "
      f"valid: {sorted(targets)}")

  click.echo("")
  click.echo(click.style(
    "── arm-to-world calibration ─────────────────────────────", fg="cyan"))
  click.echo(f"sequence ({len(touch_sequence)} touches): "
             f"{', '.join(touch_sequence)}")
  click.echo("for each touch: position the gripper tip on the named point,")
  click.echo("hold steady, then press ENTER to sample. Ctrl-C aborts.")
  click.echo("")

  touches: list[TouchRecord] = []
  for name in touch_sequence:
    p_world = targets[name]
    prompt = (f"  touch {len(touches) + 1}/{len(touch_sequence)}  "
              f"{name:>13s}  target_world=[{p_world[0]:+7.1f}, "
              f"{p_world[1]:+7.1f}, {p_world[2]:+5.1f}] mm  "
              f"→ ENTER when steady")
    try:
      click.prompt(prompt, default="", show_default=False, prompt_suffix="")
    except (click.Abort, KeyboardInterrupt):
      click.echo("\naborted by user; arm calibration skipped.", err=True)
      return None

    q_median, q_spread = _record_touch(arm, name)
    p_arm_m = fk(q_median, dq=fk_dq)
    p_arm_mm = p_arm_m * 1000.0

    flag = ""
    if np.any(q_spread > SPREAD_WARN_RAD):
      flag = click.style("  [SPREAD>0.05rad]", fg="yellow")
    click.echo(f"    q={np.round(q_median, 3).tolist()}  "
               f"spread_max={float(q_spread.max()):.3f}rad  "
               f"p_arm={np.round(p_arm_mm, 1).tolist()} mm{flag}")

    touches.append(TouchRecord(
      name=name, q=q_median, q_spread=q_spread,
      p_arm=p_arm_mm, p_world=p_world,
    ))

  block, residuals, max_res, rms_res = solve_arm_to_world(
    touches,
    fk_dq=fk_dq,
    fk_model_path=fk_model_path,
    touch_sequence=touch_sequence,
    session_id=session_id,
    L_mm=L_mm,
    D_mm=D_mm,
  )

  click.echo("")
  click.echo("  per-touch residuals (mm):")
  for tr, r in zip(touches, residuals):
    click.echo(f"    {tr.name:>13s}: {r:6.2f}")
  click.echo(f"  max_residual = {max_res:.2f} mm   "
             f"rms_residual = {rms_res:.2f} mm")

  if block is None:
    click.echo(click.style(
      f"  REJECT: max_residual {max_res:.2f}mm ≥ {RES_WARN_MM:.0f}mm. "
      "Likely a bad touch, FK joint-offset error, or wrong mat "
      "dimensions in config. Transform NOT persisted.", fg="red"))
    # Dump every touch's raw inputs + the would-be transform so the
    # operator can diagnose which point misbehaved without re-running.
    P_arm   = np.stack([t.p_arm   for t in touches], axis=0)
    P_world = np.stack([t.p_world for t in touches], axis=0)
    if is_collinear(P_arm) or is_collinear(P_world):
      R_dbg, t_dbg = umeyama_planar(P_arm, P_world)
      solver_dbg = "umeyama_planar"
    else:
      R_dbg, t_dbg = umeyama(P_arm, P_world)
      solver_dbg = "umeyama_3d"
    pred = P_arm @ R_dbg.T + t_dbg

    click.echo("")
    click.echo(click.style("  ── rejection details ────────────────────────", fg="red"))
    click.echo(f"  solver: {solver_dbg}")
    click.echo("  per-touch:")
    for tr, p_pred, r in zip(touches, pred, residuals):
      spread_flag = " [SPREAD>0.05rad]" if np.any(tr.q_spread > SPREAD_WARN_RAD) else ""
      click.echo(f"    {tr.name:>13s}{spread_flag}")
      click.echo(f"      q (rad)     = {np.round(tr.q, 4).tolist()}")
      click.echo(f"      q_spread    = {np.round(tr.q_spread, 4).tolist()}")
      click.echo(f"      p_arm  (mm) = [{tr.p_arm[0]:+8.2f}, "
                 f"{tr.p_arm[1]:+8.2f}, {tr.p_arm[2]:+8.2f}]")
      click.echo(f"      p_world(mm) = [{tr.p_world[0]:+8.2f}, "
                 f"{tr.p_world[1]:+8.2f}, {tr.p_world[2]:+8.2f}]")
      click.echo(f"      predicted   = [{p_pred[0]:+8.2f}, {p_pred[1]:+8.2f}, "
                 f"{p_pred[2]:+8.2f}]  (residual = {r:6.2f} mm)")
    click.echo("  rejected transform (for inspection only):")
    click.echo(f"    R = {np.round(R_dbg, 4).tolist()}")
    click.echo(f"    t = {np.round(t_dbg, 2).tolist()} mm")
    return None
  if max_res >= RES_ACCEPT_MM:
    click.echo(click.style(
      f"  WARN: max_residual {max_res:.2f}mm in [{RES_ACCEPT_MM:.0f}, "
      f"{RES_WARN_MM:.0f}) mm. Accepted, but consider redoing if "
      "precision matters.", fg="yellow"))
  else:
    click.echo(click.style(f"  OK ({max_res:.2f}mm < {RES_ACCEPT_MM:.0f}mm)",
                           fg="green"))
  return block
