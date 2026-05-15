"""Data types shared between scene registration and storage backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class CalibrationCapture:
  """One checkerboard observation kept during phase A.1."""

  image_path: Path
  description: str
  corners_px: np.ndarray   # (N, 2) float32 — refined corner pixel coords


@dataclass
class CameraIntrinsics:
  """Intrinsics solved for one camera from its calibration captures.

  ``K`` is the 3×3 camera matrix; ``dist`` is OpenCV's distortion vector
  (usually 5 values for the rational+radial model). ``reproj_error`` is
  the RMS pixel error reported by ``cv2.calibrateCamera``.
  """

  K: np.ndarray                  # (3, 3)
  dist: np.ndarray               # (k,) — OpenCV distortion coefficients
  reproj_error: float
  image_size: tuple[int, int]    # (width, height)
  captures: list[CalibrationCapture] = field(default_factory=list)


@dataclass
class MarkerClip:
  """One marker-location clip captured during phase A.2.

  Stored as a uint8 ``(T, H, W, 3)`` array plus a per-frame ``BlobFit``
  summary; both are persisted by the storage backend.
  """

  loc_name: str
  frames: np.ndarray             # (T, H, W, 3) uint8
  blob_centroids_px: np.ndarray  # (T, 2) float32 — NaN where not detected
  blob_radii_px: np.ndarray      # (T,)   float32 — NaN where not detected
  blob_detected: np.ndarray      # (T,)   bool
  fps: int
  duration_s: float

  @property
  def detected_fraction(self) -> float:
    if self.blob_detected.size == 0:
      return 0.0
    return float(self.blob_detected.mean())


@dataclass
class SceneRegistration:
  """The full result of phase A: per-camera intrinsics + marker clips."""

  intrinsics: dict[str, CameraIntrinsics] = field(default_factory=dict)
  markers: list[MarkerClip] = field(default_factory=list)
