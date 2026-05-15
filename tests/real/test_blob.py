"""Unit tests for :mod:`chuck_dreamer.real.blob`."""

from __future__ import annotations

import cv2
import numpy as np

from chuck_dreamer.real.blob import (
  detect_blob,
  find_white_blob,
  fit_circle_to_blob,
  render_blob_overlay,
)


def _make_disk_image(
  cx: int, cy: int, r: int,
  size: tuple[int, int] = (240, 320),   # (H, W)
  bg: int = 30,
) -> np.ndarray:
  """A white disk on a dark background, BGR uint8."""
  H, W = size
  img = np.full((H, W, 3), bg, dtype=np.uint8)
  cv2.circle(img, (cx, cy), r, (240, 240, 240), thickness=-1)
  return img


def test_find_white_blob_finds_disk():
  img = _make_disk_image(160, 120, 40)
  mask = find_white_blob(img)
  assert mask is not None
  assert mask.dtype == np.uint8
  # Mask should cover roughly the disk area.
  area_actual = int((mask > 0).sum())
  area_expected = int(np.pi * 40 * 40)
  assert 0.7 * area_expected < area_actual < 1.3 * area_expected


def test_find_white_blob_returns_none_on_empty():
  img = np.zeros((120, 160, 3), dtype=np.uint8)
  assert find_white_blob(img) is None


def test_find_white_blob_rejects_tiny_blob():
  # A 5px-radius disk has area ~78, below the 300px minimum.
  img = _make_disk_image(80, 60, 5, size=(120, 160))
  assert find_white_blob(img) is None


def test_fit_circle_recovers_center_and_radius():
  img = _make_disk_image(200, 130, 35)
  mask = find_white_blob(img)
  assert mask is not None
  fit = fit_circle_to_blob(mask)
  assert fit is not None
  cx, cy, r = fit
  assert abs(cx - 200) < 2
  assert abs(cy - 130) < 2
  assert abs(r - 35) < 3


def test_detect_blob_high_level_wrapper():
  img = _make_disk_image(100, 80, 25, size=(160, 200))
  fit = detect_blob(img)
  assert fit.detected
  assert fit.mask is not None
  assert not np.isnan(fit.radius_px)
  assert abs(fit.centroid_xy[0] - 100) < 2
  assert abs(fit.centroid_xy[1] - 80) < 2


def test_detect_blob_no_detection_returns_nan():
  img = np.zeros((120, 160, 3), dtype=np.uint8)
  fit = detect_blob(img)
  assert not fit.detected
  assert np.isnan(fit.radius_px)
  assert fit.mask is None


def test_render_blob_overlay_preserves_shape():
  img = _make_disk_image(120, 100, 30)
  fit = detect_blob(img)
  out = render_blob_overlay(img, fit)
  assert out.shape == img.shape
  assert out.dtype == img.dtype


def test_render_blob_overlay_passthrough_when_no_detection():
  img = np.zeros((120, 160, 3), dtype=np.uint8)
  fit = detect_blob(img)
  out = render_blob_overlay(img, fit)
  # When nothing is detected we just copy the input.
  assert np.array_equal(out, img)
