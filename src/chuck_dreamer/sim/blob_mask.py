"""Derive per-frame target masks from raw video via blob detection.

LeRobot teleop datasets ship without segmentation masks, but the
reconstruction-loss focus path expects a ``segmentation_target`` mask.
For our pushing setup the manipulated object is a round white marker —
the same one ``real/blob.py`` detects for hardware scene registration —
so we can recover a usable target mask straight from the camera images.

Pipeline per episode:
  1. Run the blob detector on every frame, collect ``(cx, cy, r)`` and a
     ``detected`` flag.
  2. Smooth the trajectory with a constant-velocity Kalman RTS smoother.
     Frames where detection failed are imputed from the smoother's
     prediction; frames where it succeeded are corrected by the
     measurement. The radius is smoothed independently with a 1D
     constant-position Kalman.
  3. Rasterise the smoothed circle into a ``(H, W)`` boolean mask per
     frame and stack into ``(T, H, W)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .blob import detect_blob


@dataclass
class _Trajectory:
  """Per-frame raw detections from a video, shape ``(T,)``."""

  cx:       np.ndarray          # (T,) float, NaN where not detected
  cy:       np.ndarray          # (T,)
  r:        np.ndarray          # (T,)
  detected: np.ndarray          # (T,) bool


def _detect_trajectory(frames_rgb: np.ndarray) -> _Trajectory:
  """Run the blob detector on every frame of an ``(T, H, W, 3)`` RGB stack."""
  T = frames_rgb.shape[0]
  cx = np.full((T,), np.nan, dtype=np.float64)
  cy = np.full((T,), np.nan, dtype=np.float64)
  r  = np.full((T,), np.nan, dtype=np.float64)
  detected = np.zeros((T,), dtype=bool)
  for t in range(T):
    bgr = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2BGR)
    fit = detect_blob(bgr)
    if not fit.detected:
      continue
    cx[t], cy[t] = fit.centroid_xy
    r[t]         = fit.radius_px
    detected[t]  = True
  return _Trajectory(cx=cx, cy=cy, r=r, detected=detected)


def _reject_radius_outliers(traj: _Trajectory, n_std: float = 1.0) -> _Trajectory:
  """Drop detections whose radius is more than ``n_std`` std from the mean.

  The marker is a rigid disk; under a fixed camera its on-screen radius
  should be near-constant for the episode. Detections that drift far in
  radius are usually mis-segmentations (e.g. a stray bright patch picked
  up because the real marker was occluded) — marking them not-detected
  lets the Kalman smoother bridge across them instead of being pulled
  off-course.
  """
  if not traj.detected.any():
    return traj
  r_valid = traj.r[traj.detected]
  if r_valid.size < 2:
    return traj
  mu = float(r_valid.mean())
  sigma = float(r_valid.std())
  if sigma <= 0:
    return traj
  keep = traj.detected & (np.abs(traj.r - mu) <= n_std * sigma)
  rejected = int(traj.detected.sum() - keep.sum())
  if rejected == 0:
    return traj
  cx = np.where(keep, traj.cx, np.nan)
  cy = np.where(keep, traj.cy, np.nan)
  r  = np.where(keep, traj.r,  np.nan)
  return _Trajectory(cx=cx, cy=cy, r=r, detected=keep)


def _kalman_smooth_cv(
  z:   np.ndarray,            # (T,) measurement, NaN where missing
  *,
  q_pos: float,               # process noise on position (px^2 per step)
  r_pos: float,               # measurement noise (px^2)
  init_var: float = 1e4,      # initial state covariance
  gate_chi2: float | None = 9.0,  # reject |y|^2/S > gate_chi2 (None = off)
) -> np.ndarray:
  """1D constant-velocity Kalman RTS smoother for a single coordinate.

  Returns a length-T array with the smoothed position. Missing
  measurements (NaN in ``z``) are skipped at the update step, so the
  filter just propagates the prediction across the gap and the smoother
  ties both ends together.

  ``gate_chi2`` is a Mahalanobis-distance gate on the innovation: a
  measurement whose squared normalised residual ``y^2 / S`` exceeds the
  threshold is treated as an outlier and skipped (the filter only
  propagates its prediction). 9.0 ≈ 3σ for the 1-DoF chi-squared; this
  is what makes large frame-to-frame jumps stay flagged as outliers
  rather than dragging the state along.
  """
  T = z.shape[0]

  # State: [pos, vel]. Constant-velocity transition.
  F = np.array([[1.0, 1.0],
                [0.0, 1.0]], dtype=np.float64)
  H = np.array([[1.0, 0.0]], dtype=np.float64)
  # Discrete white-noise acceleration process covariance.
  Q = q_pos * np.array([[1/4, 1/2],
                        [1/2, 1.0]], dtype=np.float64)
  R = np.array([[r_pos]], dtype=np.float64)

  # Forward pass.
  x_pred = np.zeros((T, 2), dtype=np.float64)
  P_pred = np.zeros((T, 2, 2), dtype=np.float64)
  x_filt = np.zeros((T, 2), dtype=np.float64)
  P_filt = np.zeros((T, 2, 2), dtype=np.float64)

  # Seed from the first valid measurement (or zero if none).
  first_valid = next((t for t in range(T) if np.isfinite(z[t])), None)
  if first_valid is None:
    return np.zeros((T,), dtype=np.float64)
  x = np.array([z[first_valid], 0.0], dtype=np.float64)
  P = np.eye(2, dtype=np.float64) * init_var

  for t in range(T):
    # Predict.
    x = F @ x
    P = F @ P @ F.T + Q
    x_pred[t] = x
    P_pred[t] = P
    # Update if we have a measurement that passes the chi-squared gate.
    if np.isfinite(z[t]):
      y = z[t] - (H @ x)[0]
      S = (H @ P @ H.T + R)[0, 0]
      if gate_chi2 is None or (y * y) <= gate_chi2 * S:
        K = (P @ H.T).ravel() / S
        x = x + K * y
        P = P - np.outer(K, H @ P)
    x_filt[t] = x
    P_filt[t] = P

  # RTS backward pass.
  x_smooth = x_filt.copy()
  P_smooth = P_filt.copy()
  for t in range(T - 2, -1, -1):
    # P_pred[t+1] is the predicted covariance for step t+1 given filter
    # state at t. C = P_filt[t] @ F.T @ inv(P_pred[t+1]).
    try:
      C = P_filt[t] @ F.T @ np.linalg.inv(P_pred[t + 1])
    except np.linalg.LinAlgError:
      continue
    x_smooth[t] = x_filt[t] + C @ (x_smooth[t + 1] - x_pred[t + 1])
    P_smooth[t] = P_filt[t] + C @ (P_smooth[t + 1] - P_pred[t + 1]) @ C.T

  return x_smooth[:, 0]


def _kalman_smooth_cp(
  z:   np.ndarray,
  *,
  q:   float,                 # process noise (px^2 per step)
  r:   float,                 # measurement noise (px^2)
  init_var: float = 1e4,
) -> np.ndarray:
  """1D constant-position Kalman RTS smoother (used for the radius).

  The marker doesn't change size, so a random-walk model for ``r`` is
  the right fit. Same NaN-tolerant interface as ``_kalman_smooth_cv``.
  """
  T = z.shape[0]
  first_valid = next((t for t in range(T) if np.isfinite(z[t])), None)
  if first_valid is None:
    return np.zeros((T,), dtype=np.float64)

  x_pred = np.zeros((T,), dtype=np.float64)
  P_pred = np.zeros((T,), dtype=np.float64)
  x_filt = np.zeros((T,), dtype=np.float64)
  P_filt = np.zeros((T,), dtype=np.float64)

  x = float(z[first_valid])
  P = init_var
  for t in range(T):
    P = P + q
    x_pred[t] = x
    P_pred[t] = P
    if np.isfinite(z[t]):
      S = P + r
      K = P / S
      x = x + K * (z[t] - x)
      P = (1.0 - K) * P
    x_filt[t] = x
    P_filt[t] = P

  x_smooth = x_filt.copy()
  for t in range(T - 2, -1, -1):
    if P_pred[t + 1] <= 0:
      continue
    C = P_filt[t] / P_pred[t + 1]
    x_smooth[t] = x_filt[t] + C * (x_smooth[t + 1] - x_pred[t + 1])
  return x_smooth


def _rasterize_circles(
  cx: np.ndarray, cy: np.ndarray, r: np.ndarray,
  H: int, W: int,
) -> np.ndarray:
  """Draw filled disks ``(cx_t, cy_t, r_t)`` into a ``(T, H, W)`` bool array."""
  T = cx.shape[0]
  out = np.zeros((T, H, W), dtype=bool)
  for t in range(T):
    if not (np.isfinite(cx[t]) and np.isfinite(cy[t]) and np.isfinite(r[t])):
      continue
    radius = max(int(round(float(r[t]))), 1)
    canvas = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(canvas,
               (int(round(float(cx[t]))), int(round(float(cy[t])))),
               radius, color=255, thickness=-1)
    out[t] = canvas > 0
  return out


def derive_target_mask(
  frames_rgb: np.ndarray,
  *,
  q_pos:           float = 0.25,    # tight: penalise jumps, expect ~0.5 px/step accel
  r_pos:           float = 9.0,     # ~3 px detection jitter
  gate_chi2:       float = 9.0,     # 3σ innovation gate (1-DoF chi²)
  q_radius:        float = 0.05,
  r_radius:        float = 4.0,
  radius_n_std:    float = 1.0,     # drop detections > N stds from per-episode mean
) -> np.ndarray:
  """Derive a ``(T, H, W)`` bool ``segmentation_target`` mask from RGB frames.

  Pipeline:
    1. Per-frame blob detection.
    2. Discard detections whose radius is > ``radius_n_std`` stds from
       the episode mean — these are almost always mis-segmentations.
    3. Smooth (cx, cy) with a constant-velocity Kalman RTS smoother
       that uses a chi-squared innovation gate to reject jumpy
       measurements, and the radius with a constant-position smoother.
    4. Rasterise the smoothed circles back into binary masks.

  The combination of a small ``q_pos`` and the innovation gate makes
  the smoother strongly biased toward smooth trajectories: a single
  detection that disagrees with the constant-velocity prediction by
  more than ~3σ is treated as an outlier and ignored. Frames where
  detection failed (or was rejected) are filled from the smoother, so
  the mask sequence has no gaps as long as at least one frame had a
  good detection.

  Returns an all-False mask if no frame had a detection.
  """
  if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
    raise ValueError(f"frames_rgb must be (T, H, W, 3), got {frames_rgb.shape}")
  T, H, W, _ = frames_rgb.shape

  traj = _detect_trajectory(frames_rgb)
  if not traj.detected.any():
    return np.zeros((T, H, W), dtype=bool)
  traj = _reject_radius_outliers(traj, n_std=radius_n_std)
  if not traj.detected.any():
    return np.zeros((T, H, W), dtype=bool)

  cx_smooth = _kalman_smooth_cv(traj.cx, q_pos=q_pos, r_pos=r_pos, gate_chi2=gate_chi2)
  cy_smooth = _kalman_smooth_cv(traj.cy, q_pos=q_pos, r_pos=r_pos, gate_chi2=gate_chi2)
  r_smooth  = _kalman_smooth_cp(traj.r,  q=q_radius, r=r_radius)
  return _rasterize_circles(cx_smooth, cy_smooth, r_smooth, H, W)
