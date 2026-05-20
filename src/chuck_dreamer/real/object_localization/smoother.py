"""Per-episode RTS smoother over per-frame ``ObjectPose`` observations.

Deliverable 4 of ``object_localization_spec.md``. Runs a 2-D
constant-velocity-with-friction Kalman filter forward, then a
Rauch-Tung-Striebel backward pass, then flags long observation gaps.

The smoother exists because per-frame pose estimation drops out during
arm occlusion and occasionally produces confident-looking but wrong
observations. The downstream model needs a gap-free ``(x, y)``
trajectory per episode plus a boolean per frame saying "trust this".
The backward RTS pass is what lets the gap interior be anchored from
*both* ends, instead of extrapolated forward from one side as a
real-time filter would have to.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, overload

import numpy as np

from .estimator import ObjectPose


@dataclass
class SmoothedFrame:
  xy_mm: np.ndarray            # (2,) world-frame (x, y), always present
  covariance_mm2: np.ndarray   # (2, 2) position covariance from smoother
  visually_observed: bool      # True iff the underlying ObjectPose was used
  gap_too_long: bool           # True iff this frame sits in a long occlusion run


@dataclass
class _SmootherDiagnostics:
  innovations: np.ndarray      # (T, 2)
  mahalanobis: np.ndarray      # (T,)
  gated: np.ndarray            # (T,) bool — True iff observation rejected
  observation_noise: np.ndarray  # (T, 2, 2)


class SmoothedTrajectoryEstimator:
  """RTS smoother over a per-episode list of ``ObjectPose | None``."""

  def __init__(self, config: dict[str, Any] | None) -> None:
    config = dict(config or {})
    self.fps = float(config.get("fps", 30.0))
    if self.fps <= 0:
      raise ValueError(f"smoother.fps must be > 0, got {self.fps}")
    self.dt  = 1.0 / self.fps
    self.alpha = float(config.get("friction_alpha", 0.92))

    pn = dict(config.get("process_noise") or {})
    self.q_pos_mm    = float(pn.get("pos_mm",    0.5))
    self.q_vel_mm_s  = float(pn.get("vel_mm_s",  200.0))

    obs = dict(config.get("observation_noise") or {})
    self.base_sigma_mm     = float(obs.get("base_sigma_mm",     1.0))
    self.confidence_floor  = float(obs.get("confidence_floor",  0.05))
    self.residual_scale    = float(obs.get("residual_scale",    2.0))

    gating = dict(config.get("gating") or {})
    self.mahalanobis_threshold = float(gating.get("mahalanobis_threshold", 9.21))

    init = dict(config.get("initial") or {})
    self.init_pos_sigma_mm   = float(init.get("pos_sigma_mm",   2.0))
    self.init_vel_sigma_mm_s = float(init.get("vel_sigma_mm_s", 500.0))

    self.gap_too_long_frames = int(config.get("gap_too_long_frames", 30))

  # -------------------------------------------------------------------------
  # Public API
  # -------------------------------------------------------------------------
  @overload
  def smooth_episode(
    self, poses: list[ObjectPose | None],
  ) -> list[SmoothedFrame]: ...
  @overload
  def smooth_episode(
    self, poses: list[ObjectPose | None], *, return_diagnostics: bool,
  ) -> tuple[list[SmoothedFrame], _SmootherDiagnostics]: ...

  def smooth_episode(
    self,
    poses: list[ObjectPose | None],
    *,
    return_diagnostics: bool = False,
  ):
    if not poses:
      empty = []
      if return_diagnostics:
        return empty, _SmootherDiagnostics(
          innovations       = np.zeros((0, 2), dtype=np.float64),
          mahalanobis       = np.zeros((0,),    dtype=np.float64),
          gated             = np.zeros((0,),    dtype=bool),
          observation_noise = np.zeros((0, 2, 2), dtype=np.float64),
        )
      return empty

    T = len(poses)
    first_idx = next((i for i, p in enumerate(poses) if p is not None), None)
    if first_idx is None or first_idx != 0:
      raise ValueError(
        "smooth_episode requires the first frame to have a non-None pose; "
        "the spec guarantees the object is clearly visible at t=0.")

    F = self._transition_matrix()
    Q = self._process_noise()
    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

    x0 = poses[0]
    assert x0 is not None
    x_post = np.array([x0.xy_mm[0], x0.xy_mm[1], 0.0, 0.0], dtype=np.float64)
    P_post = np.diag([
      self.init_pos_sigma_mm   ** 2, self.init_pos_sigma_mm   ** 2,
      self.init_vel_sigma_mm_s ** 2, self.init_vel_sigma_mm_s ** 2,
    ]).astype(np.float64)

    x_pred_seq: list[np.ndarray] = [None] * T   # type: ignore[list-item]
    P_pred_seq: list[np.ndarray] = [None] * T   # type: ignore[list-item]
    x_post_seq: list[np.ndarray] = [None] * T   # type: ignore[list-item]
    P_post_seq: list[np.ndarray] = [None] * T   # type: ignore[list-item]

    innovations = np.zeros((T, 2), dtype=np.float64)
    mahalanobis = np.zeros((T,),    dtype=np.float64)
    gated       = np.zeros((T,),    dtype=bool)
    R_log       = np.zeros((T, 2, 2), dtype=np.float64)
    visually_observed = np.zeros((T,), dtype=bool)

    # t = 0: the initial state IS the posterior; the "prediction" at t=0
    # is recorded as the same so the RTS recursion has a well-defined
    # P_{t+1|t} for t = -1 (which we never actually consume).
    x_pred_seq[0] = x_post.copy()
    P_pred_seq[0] = P_post.copy()
    x_post_seq[0] = x_post.copy()
    P_post_seq[0] = P_post.copy()
    visually_observed[0] = True
    R_log[0] = self._observation_noise(poses[0])

    for t in range(1, T):
      x_pred = F @ x_post
      P_pred = F @ P_post @ F.T + Q
      x_pred_seq[t] = x_pred
      P_pred_seq[t] = P_pred

      pose = poses[t]
      if pose is None:
        x_post = x_pred
        P_post = P_pred
        x_post_seq[t] = x_post
        P_post_seq[t] = P_post
        continue

      R_t = self._observation_noise(pose)
      R_log[t] = R_t
      z  = np.asarray(pose.xy_mm, dtype=np.float64).reshape(2)
      nu = z - H @ x_pred
      S  = H @ P_pred @ H.T + R_t
      try:
        S_inv = np.linalg.inv(S)
      except np.linalg.LinAlgError:
        x_post = x_pred
        P_post = P_pred
        x_post_seq[t] = x_post
        P_post_seq[t] = P_post
        gated[t] = True
        continue
      d2 = float(nu @ S_inv @ nu)
      innovations[t] = nu
      mahalanobis[t] = d2
      if d2 > self.mahalanobis_threshold:
        gated[t] = True
        x_post = x_pred
        P_post = P_pred
      else:
        K = P_pred @ H.T @ S_inv
        x_post = x_pred + K @ nu
        # Joseph form would be more numerically stable; the simple form
        # is fine for a 4-state filter at frame rates measured in tens.
        P_post = (np.eye(4) - K @ H) @ P_pred
        visually_observed[t] = True

      x_post_seq[t] = x_post
      P_post_seq[t] = P_post

    x_smooth = [None] * T   # type: ignore[var-annotated]
    P_smooth = [None] * T   # type: ignore[var-annotated]
    x_smooth[T - 1] = x_post_seq[T - 1]
    P_smooth[T - 1] = P_post_seq[T - 1]
    for t in range(T - 2, -1, -1):
      try:
        P_pred_next_inv = np.linalg.inv(P_pred_seq[t + 1])
      except np.linalg.LinAlgError:
        x_smooth[t] = x_post_seq[t]
        P_smooth[t] = P_post_seq[t]
        continue
      C = P_post_seq[t] @ F.T @ P_pred_next_inv
      x_smooth[t] = x_post_seq[t] + C @ (x_smooth[t + 1] - x_pred_seq[t + 1])
      P_smooth[t] = P_post_seq[t] + C @ (P_smooth[t + 1] - P_pred_seq[t + 1]) @ C.T

    gap_too_long = self._gap_too_long_mask(visually_observed)

    frames: list[SmoothedFrame] = []
    for t in range(T):
      xs = x_smooth[t]
      Ps = P_smooth[t]
      frames.append(SmoothedFrame(
        xy_mm             = np.asarray(xs[:2], dtype=np.float64),
        covariance_mm2    = np.asarray(Ps[:2, :2], dtype=np.float64),
        visually_observed = bool(visually_observed[t]),
        gap_too_long      = bool(gap_too_long[t]),
      ))

    if return_diagnostics:
      diag = _SmootherDiagnostics(
        innovations       = innovations,
        mahalanobis       = mahalanobis,
        gated             = gated,
        observation_noise = R_log,
      )
      return frames, diag
    return frames

  # -------------------------------------------------------------------------
  # Internals
  # -------------------------------------------------------------------------
  def _transition_matrix(self) -> np.ndarray:
    dt    = self.dt
    alpha = self.alpha
    return np.array([
      [1.0, 0.0, dt,    0.0  ],
      [0.0, 1.0, 0.0,   dt   ],
      [0.0, 0.0, alpha, 0.0  ],
      [0.0, 0.0, 0.0,   alpha],
    ], dtype=np.float64)

  def _process_noise(self) -> np.ndarray:
    # Diagonal piecewise-constant white-acceleration form, with the
    # diagonal magnitudes coming directly from the config. The cross
    # terms are zero — fine here because the friction term in F already
    # couples position and velocity, and the spec's process-noise
    # parameters are interpreted as marginal stds rather than a joint
    # acceleration noise density.
    return np.diag([
      self.q_pos_mm     ** 2,
      self.q_pos_mm     ** 2,
      (self.q_vel_mm_s * self.dt) ** 2,
      (self.q_vel_mm_s * self.dt) ** 2,
    ]).astype(np.float64)

  def _observation_noise(self, pose: ObjectPose) -> np.ndarray:
    conf = max(float(pose.confidence), self.confidence_floor)
    resid = float(pose.reprojection_error_px) if np.isfinite(pose.reprojection_error_px) else 0.0
    sigma = self.base_sigma_mm / conf + self.residual_scale * resid
    return (sigma ** 2) * np.eye(2, dtype=np.float64)

  def _gap_too_long_mask(self, visually_observed: np.ndarray) -> np.ndarray:
    """For each contiguous run of False in ``visually_observed``,
    mark the whole run as gap_too_long iff its length exceeds the
    threshold. Runs at the start / end of the episode count too.
    """
    T = len(visually_observed)
    out = np.zeros(T, dtype=bool)
    i = 0
    while i < T:
      if visually_observed[i]:
        i += 1
        continue
      j = i
      while j < T and not visually_observed[j]:
        j += 1
      if (j - i) > self.gap_too_long_frames:
        out[i:j] = True
      i = j
    return out
