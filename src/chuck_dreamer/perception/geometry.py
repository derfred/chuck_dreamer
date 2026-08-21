"""World-frame geometry for the painted mat fiducials.

The mat sits at ``Z = 0``. Origin is at the **midpoint of the two
lines**, so each line spans ``x = -L/2 .. +L/2``. The circle is centred
at ``(+L/2, 0)`` — the "near" end of the lines, which is where the
origin used to sit. +Z points up out of the mat; +Y by right-hand rule.
With the camera looking down at the mat the way the spec assumes, +Y
projects toward the bottom of the image, so "line 1 (bottom)" sits at
``y = +D/2`` and "line 2 (top)" sits at ``y = -D/2``.

Note the historical convention placed the origin at the circle centre;
everything expressed in this frame (stored extrinsics, touchpoint
targets, mat-extent bounds) shifts by ``L/2`` under the change.
"""
from __future__ import annotations


import numpy as np


def mat_circle_center_x_mm(L_mm: float) -> float:
  """x of the circle centre in the world frame.

  The circle sits at the "near" end of the line pair; with the origin at
  the line midpoint that is ``+L/2``. Single source of truth so the solve
  and the overlay renderers cannot drift apart."""
  return L_mm / 2.0


def line_endpoints_world(L_mm: float, D_mm: float) -> dict[str, np.ndarray]:
  """4 known line endpoints, each a ``(3,)`` world-frame point in mm."""
  return {
    "line1_near": np.array([+L_mm / 2.0, +D_mm / 2.0, 0.0]),
    "line1_far":  np.array([-L_mm / 2.0, +D_mm / 2.0, 0.0]),
    "line2_near": np.array([+L_mm / 2.0, -D_mm / 2.0, 0.0]),
    "line2_far":  np.array([-L_mm / 2.0, -D_mm / 2.0, 0.0]),
  }


def circle_samples_world(r_mm: float, n: int, phase_offset: int = 0,
                         reverse: bool = False,
                         center_x_mm: float = 0.0) -> np.ndarray:
  """``(N, 3)`` samples on the world-frame circle at ``Z = 0``.

  ``center_x_mm`` is the circle centre along x; with the origin at the
  line midpoint the circle sits at ``+L/2``, so callers pass that.

  ``phase_offset`` is an integer index into the canonical
  ``theta_i = 2 pi i / n`` sequence; ``reverse`` flips the traversal
  direction. Both knobs exist for the phase-alignment step in
  ``annotate-mat`` — the image-side ellipse parametrization starts at
  cv2's major axis and may run either way, so we sweep both knobs and
  pick the lowest-error pair.
  """
  i = np.arange(n)
  if reverse:
    i = -i
  thetas = 2.0 * np.pi * (i + phase_offset) / float(n)
  return np.stack([
    center_x_mm + r_mm * np.cos(thetas),
    r_mm * np.sin(thetas),
    np.zeros(n),
  ], axis=1)


def line_world_points(L_mm: float, D_mm: float) -> np.ndarray:
  """``(4, 3)`` array in the canonical order used by the annotation UI:
  ``(line1_near, line1_far, line2_near, line2_far)``.
  """
  e = line_endpoints_world(L_mm, D_mm)
  return np.stack([e["line1_near"], e["line1_far"],
                   e["line2_near"], e["line2_far"]], axis=0)


def axes_3d(scale_mm: float = 100.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Origin + three axis-tip world points for the verification triad."""
  return (
    np.array([0.0, 0.0, 0.0]),
    np.array([scale_mm, 0.0, 0.0]),
    np.array([0.0, scale_mm, 0.0]),
    np.array([0.0, 0.0, scale_mm]),
  )
