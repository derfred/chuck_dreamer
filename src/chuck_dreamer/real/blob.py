"""Round-white-marker blob detection.

Lifted from the prototyping script ``blob-corners.py`` at the repo root.
Only the blob + circle-fit primitives are kept here; corner detection is
unused for marker registration.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BlobFit:
  """One frame's blob detection result.

  ``detected`` is True iff both the segmentation and the circle fit
  succeeded. Centroid/radius are NaN when ``detected`` is False so they
  can be stored in a fixed-shape array.
  """

  detected: bool
  centroid_xy: tuple[float, float]   # NaN when not detected
  radius_px: float                    # NaN when not detected
  mask: np.ndarray | None             # uint8 0/255 or None when not detected


def find_white_blob(bgr: np.ndarray) -> np.ndarray | None:
  """Return a binary mask of the largest bright, low-saturation, roundish
  blob, or None if no plausible blob exists.

  Loose thresholds — the blob doesn't need to be perfectly segmented.
  Internal holes (specular highlights) are filled.
  """
  hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv, (0, 0, 140), (180, 90, 255))

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

  num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
  if num <= 1:
    return None

  best_idx = -1
  best_score = -1.0
  for i in range(1, num):
    area = stats[i, cv2.CC_STAT_AREA]
    if area < 300:
      continue
    component = (labels == i).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    if not contours:
      continue
    c = contours[0]
    peri = cv2.arcLength(c, closed=True)
    if peri < 1:
      continue
    circularity = 4 * np.pi * area / (peri * peri)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    solidity = area / max(hull_area, 1.0)
    score = area * circularity * solidity
    if score > best_score:
      best_score = score
      best_idx = i

  if best_idx < 0:
    return None

  blob = (labels == best_idx).astype(np.uint8) * 255
  blob_filled = blob.copy()
  contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
  cv2.drawContours(blob_filled, contours, -1, 255, thickness=cv2.FILLED)
  return blob_filled


def fit_circle_to_blob(blob_mask: np.ndarray) -> tuple[float, float, float] | None:
  """Fit ``(cx, cy, r)`` to the blob silhouette.

  Uses ``cv2.minEnclosingCircle`` on the largest external contour.
  Returns None when the mask has no contour.
  """
  contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
  if not contours:
    return None
  c = max(contours, key=cv2.contourArea)
  (cx, cy), r = cv2.minEnclosingCircle(c)
  return float(cx), float(cy), float(r)


def detect_blob(bgr: np.ndarray) -> BlobFit:
  """High-level wrapper: segment + fit in one call.

  Always returns a :class:`BlobFit`; check ``.detected`` to branch.
  """
  mask = find_white_blob(bgr)
  if mask is None:
    return BlobFit(False, (float("nan"), float("nan")), float("nan"), None)
  fit = fit_circle_to_blob(mask)
  if fit is None:
    return BlobFit(False, (float("nan"), float("nan")), float("nan"), None)
  cx, cy, r = fit
  return BlobFit(True, (cx, cy), r, mask)


def render_blob_overlay(bgr: np.ndarray, fit: BlobFit) -> np.ndarray:
  """Return ``bgr`` with the blob contour + fit circle drawn on it."""
  out = bgr.copy()
  if not fit.detected or fit.mask is None:
    return out
  contours, _ = cv2.findContours(fit.mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
  cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
  cx, cy = fit.centroid_xy
  cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(fit.radius_px)),
              (0, 255, 255), 2)
  cv2.circle(out, (int(round(cx)), int(round(cy))), 3, (0, 255, 255), -1)
  return out
