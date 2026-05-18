"""Integration test for the interactive calibration loop.

Drives :func:`calibrate_cameras` with a stub camera that renders a
projected checkerboard and a stub UI that scripts the SPACE/ESC events.
The recovered intrinsics should match the ground-truth K within a tight
tolerance.
"""

from __future__ import annotations

import cv2
import numpy as np

from chuck_dreamer.real.scene_registration import (
  CheckerboardSpec,
  calibrate_cameras,
)

from ._stubs import StubFrameSource, StubUi


def _render_checkerboard(
  K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
  spec: CheckerboardSpec, image_size: tuple[int, int],
) -> np.ndarray:
  """Render a chequer image (RGB uint8) of the given pose."""
  W, H = image_size
  img = np.full((H, W, 3), 220, dtype=np.uint8)
  # Build square centres + paint each square in alternating colours.
  cols, rows = spec.cols + 1, spec.rows + 1
  half = spec.square_size_m
  # Top-left corner of the (0,0) square in board coords (so the inner
  # corner grid matches the spec's object_points).
  for r in range(rows):
    for c in range(cols):
      if (r + c) % 2 == 0:
        continue
      corners3d = np.array([
        [c * half,       r * half,       0],
        [(c + 1) * half, r * half,       0],
        [(c + 1) * half, (r + 1) * half, 0],
        [c * half,       (r + 1) * half, 0],
      ], dtype=np.float32)
      # Shift so the inner corners line up with object_points
      corners3d[:, :2] -= half
      pts, _ = cv2.projectPoints(corners3d, rvec, tvec, K, distCoeffs=np.zeros(5))
      pts = pts.reshape(-1, 2).astype(np.int32)
      cv2.fillConvexPoly(img, pts, (10, 10, 10))
  return img


def test_calibrate_cameras_recovers_K(tmp_path):
  spec = CheckerboardSpec(cols=9, rows=6, square_size_m=0.02)
  W, H = 640, 480
  K_true = np.array([
    [550.0, 0.0, 320.0],
    [0.0, 550.0, 240.0],
    [0.0, 0.0, 1.0],
  ])

  # Choose 8 plausible poses (board ~0.4 m away, modest tilts).
  rng = np.random.default_rng(0)
  poses = []
  for _ in range(8):
    rvec = rng.uniform(-0.3, 0.3, size=3)
    tvec = np.array([
      rng.uniform(-0.04, 0.04),
      rng.uniform(-0.03, 0.03),
      rng.uniform(0.35, 0.55),
    ])
    poses.append((rvec, tvec))

  # We need one preview frame per capture (the loop reads, then maybe
  # captures). Build a script that returns the right pose-image for each
  # preview tick. Index by `read_count`.
  rendered = [
    _render_checkerboard(K_true, rvec, tvec, spec, (W, H)) for rvec, tvec in poses]

  reads: dict[str, int] = {"n": 0}

  def factory(_i: int) -> np.ndarray:
    img = rendered[min(reads["n"], len(rendered) - 1)]
    reads["n"] += 1
    return img

  src = StubFrameSource(
    name="top", fps=30, image_size=(W, H), frame_factory=factory)

  # 8 SPACE presses, then ESC. Prompts: empty descriptions.
  ui = StubUi(
    key_script=["space"] * 8 + ["esc"],
    prompt_script=[""] * 8,
  )

  out = calibrate_cameras(
    {"top": src}, spec, tmp_path / "scene",
    min_captures=6, ui=ui)

  assert "top" in out
  intr = out["top"]
  # Tolerances are looser than the synthetic-points test in
  # ``test_calibration.py`` because corner detection on raw rendered
  # squares adds a fraction of a pixel of error.
  assert intr.reproj_error < 1.5
  assert abs(intr.K[0, 0] - K_true[0, 0]) / K_true[0, 0] < 0.05
  assert abs(intr.K[1, 1] - K_true[1, 1]) / K_true[1, 1] < 0.05
  assert abs(intr.K[0, 2] - K_true[0, 2]) < 10
  assert abs(intr.K[1, 2] - K_true[1, 2]) < 10
  assert intr.image_size == (W, H)
  # Some rendered poses may fail corner detection; require ≥6 captures
  # so cv2.calibrateCamera still has plenty of constraints.
  assert len(intr.captures) >= 6

  # The capture PNGs should be on disk under the per-camera dir.
  cap_dir = tmp_path / "scene" / "intrinsics" / "top" / "captures"
  pngs = sorted(p.name for p in cap_dir.glob("*.png"))
  assert len(pngs) == len(intr.captures)
  assert pngs[0] == "capture-000.png"


def test_calibrate_returns_empty_when_too_few_captures(tmp_path):
  spec = CheckerboardSpec(cols=9, rows=6, square_size_m=0.02)
  W, H = 640, 480
  rng = np.random.default_rng(1)
  rvec = rng.uniform(-0.2, 0.2, size=3)
  tvec = np.array([0.0, 0.0, 0.45])
  img = _render_checkerboard(
    np.array([[550, 0, 320], [0, 550, 240], [0, 0, 1]], float),
    rvec, tvec, spec, (W, H))

  src = StubFrameSource(
    name="top", fps=30, image_size=(W, H),
    frame_factory=lambda _i: img.copy())

  # Two SPACE presses then ESC, but min_captures=5 → no intrinsics solved.
  ui = StubUi(key_script=["space", "space", "esc"], prompt_script=["", ""])

  out = calibrate_cameras(
    {"top": src}, spec, tmp_path / "scene",
    min_captures=5, ui=ui)
  assert out == {}


def test_calibrate_skips_capture_when_no_board(tmp_path):
  spec = CheckerboardSpec(cols=9, rows=6, square_size_m=0.02)
  W, H = 320, 240
  src = StubFrameSource(
    name="top", fps=30, image_size=(W, H),
    frame_factory=lambda _i: np.full((H, W, 3), 30, dtype=np.uint8))
  # SPACE pressed but no board → must NOT count as a capture; then ESC.
  ui = StubUi(key_script=["space", "esc"])
  out = calibrate_cameras(
    {"top": src}, spec, tmp_path / "scene",
    min_captures=1, ui=ui)
  assert out == {}
