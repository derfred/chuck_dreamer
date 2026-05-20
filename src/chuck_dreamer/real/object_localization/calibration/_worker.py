"""ProcessPool worker for pooled intrinsics calibration.

Lives in its own module (rather than inside ``intrinsics.py``) so it
pickles cleanly across process boundaries and doesn't drag the
rest of the calibration module's imports into every child process.

Each worker builds its own ``LeRobotDataset`` lazily on first touch,
caches it for the lifetime of the worker, and runs corner detection
on the requested frame. We swallow per-frame errors as MISS to keep
the main process's parallel loop running even when one file in one
dataset is corrupt.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Per-worker dataset cache. Process-local; reset across the pool's
# lifetime, but stable within one worker.
_ds_cache: dict[str, Any] = {}


def detect_corners_worker(
  dataset_id: str, frame_idx: int, camera_key: str,
  ck_size: tuple[int, int],
) -> tuple[bool, Any]:
  """Run one (dataset, frame) corner detection.

  Returns ``(ok, payload)`` where on success ``payload`` is
  ``(corners, (W, H))`` and on failure ``(False, None)``.
  """
  try:
    ds = _get_dataset(dataset_id)
    rgb = _get_frame(ds, frame_idx, camera_key)
    corners = _detect_corners(rgb, ck_size)
    if corners is None:
      return False, None
    H, W = rgb.shape[:2]
    return True, (corners, (W, H))
  except Exception:
    # The main loop counts these as MISS and keeps going.
    return False, None


def _get_dataset(dataset_id: str) -> Any:
  if dataset_id in _ds_cache:
    return _ds_cache[dataset_id]
  from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
  ds = LeRobotDataset(dataset_id)
  _ds_cache[dataset_id] = ds
  return ds


def _get_frame(ds: Any, frame_idx: int, camera_key: str) -> np.ndarray:
  # Imported lazily so the child process only pays for the import path
  # it actually uses (avoids hauling the rest of the package in).
  from ..dataset import get_frame
  return get_frame(ds, frame_idx, camera_key)


def _detect_corners(rgb: np.ndarray, size: tuple[int, int]):
  import cv2
  gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
  flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
  found, corners = cv2.findChessboardCornersSB(gray, size, flags=flags)
  if not found or corners is None:
    return None
  refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
  corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (11, 11), (-1, -1), refine_criteria)
  return corners
