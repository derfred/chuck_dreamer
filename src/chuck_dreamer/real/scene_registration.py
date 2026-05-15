"""Phase A — scene registration.

Two sub-phases run in order:

  1. :func:`calibrate_cameras` — for each camera, show a live preview with
     checkerboard corner overlay; user presses SPACE to capture, ESC to
     move on. Estimates intrinsics with ``cv2.calibrateCamera``.
  2. :func:`register_markers` — primary camera is fixed; for each named
     marker location, capture a short video clip and validate that the
     blob detector sees the marker for enough frames (or, for the
     ``off_screen`` slot, that the marker is absent).

Both functions are written against two small abstractions so they can run
unattended in tests:

  * :class:`FrameSource` — a camera adapter (real impl wraps
    ``lerobot.cameras.opencv.OpenCVCamera``; test impl yields scripted
    frames).
  * :class:`Ui` — show a window and read keypresses. The real impl uses
    ``cv2.imshow``/``cv2.waitKey`` + ``input()``; the test impl replays a
    scripted key/description queue.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import cv2
import numpy as np

from .blob import detect_blob, render_blob_overlay
from .scene_types import (
  CalibrationCapture,
  CameraIntrinsics,
  MarkerClip,
  SceneRegistration,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstractions for camera + UI (so tests can inject fakes)
# ---------------------------------------------------------------------------


class FrameSource(Protocol):
  """A connected camera. Frames are RGB ``(H, W, 3)`` uint8."""

  name: str
  fps: int
  image_size: tuple[int, int]   # (width, height)

  def read(self) -> np.ndarray: ...
  def disconnect(self) -> None: ...


class Ui(Protocol):
  """Window + keyboard interaction.

  ``poll_key`` returns a single-character action token: ``"space"``,
  ``"esc"``, or ``""`` for "nothing pressed this tick". ``pump`` is a
  separate no-op-ish entrypoint used while recording a clip — it lets
  the real cv2 backend refresh its window but is not allowed to consume
  scripted keypresses in tests.
  """

  def show(self, window: str, image_bgr: np.ndarray) -> None: ...
  def poll_key(self, timeout_ms: int = 1) -> str: ...
  def pump(self, timeout_ms: int = 1) -> None: ...
  def prompt(self, message: str) -> str: ...
  def close(self, window: str) -> None: ...


# ---------------------------------------------------------------------------
# Real implementations
# ---------------------------------------------------------------------------


class OpenCVFrameSource:
  """Adapter from ``lerobot.cameras.opencv.OpenCVCamera`` to :class:`FrameSource`.

  Constructed from an already-built ``OpenCVCameraConfig`` so callers
  control auto-detect once at the cfg-builder layer.
  """

  def __init__(self, name: str, camera_cfg: Any) -> None:
    from lerobot.cameras.opencv import OpenCVCamera
    self.name = name
    self._cfg = camera_cfg
    self._cam = OpenCVCamera(camera_cfg)
    self._cam.connect()
    self.fps = int(camera_cfg.fps)
    self.image_size = (int(camera_cfg.width), int(camera_cfg.height))

  def read(self) -> np.ndarray:
    # lerobot returns RGB by default (per OpenCVCameraConfig.color_mode).
    return self._cam.read()

  def disconnect(self) -> None:
    if self._cam.is_connected:
      self._cam.disconnect()


class _Cv2Ui:
  """Real :class:`Ui` impl backed by ``cv2.imshow`` + ``cv2.waitKey``."""

  _KEY_MAP = {32: "space", 27: "esc"}

  def show(self, window: str, image_bgr: np.ndarray) -> None:
    cv2.imshow(window, image_bgr)

  def poll_key(self, timeout_ms: int = 1) -> str:
    k = cv2.waitKey(timeout_ms) & 0xFF
    if k == 255:
      return ""
    return self._KEY_MAP.get(k, "")

  def pump(self, timeout_ms: int = 1) -> None:
    # Drain whatever key happens to be pressed without acting on it; the
    # waitKey call is what allows the OpenCV window to repaint.
    cv2.waitKey(timeout_ms)

  def prompt(self, message: str) -> str:
    try:
      return input(message)
    except (EOFError, KeyboardInterrupt):
      return ""

  def close(self, window: str) -> None:
    try:
      cv2.destroyWindow(window)
    except cv2.error:
      pass


def default_ui() -> Ui:
  return _Cv2Ui()


# ---------------------------------------------------------------------------
# Phase A.1 — camera calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckerboardSpec:
  cols: int               # inner corners per row
  rows: int               # inner corners per col
  square_size_m: float

  @property
  def pattern_size(self) -> tuple[int, int]:
    return (self.cols, self.rows)

  def object_points(self) -> np.ndarray:
    """3D coordinates of the corners in the board's own frame, Z=0."""
    n = self.cols * self.rows
    objp = np.zeros((n, 3), dtype=np.float32)
    objp[:, :2] = np.indices((self.cols, self.rows)).T.reshape(-1, 2)
    objp[:, :2] *= self.square_size_m
    return objp


def _find_checkerboard(gray: np.ndarray, spec: CheckerboardSpec) -> np.ndarray | None:
  """Return refined ``(N, 2)`` corner coords, or None if not detected."""
  found, corners = cv2.findChessboardCornersSB(
    gray, spec.pattern_size,
    flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
  )
  if not found:
    return None
  # findChessboardCornersSB already returns sub-pixel-accurate corners.
  return corners.reshape(-1, 2).astype(np.float32)


def _render_calibration_overlay(
  frame_rgb: np.ndarray,
  spec: CheckerboardSpec,
  corners: np.ndarray | None,
  n_captures: int,
  cam_name: str,
  banner: str | None = None,
) -> np.ndarray:
  """Return the BGR frame the UI should show for one preview tick."""
  bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
  if corners is not None:
    pts = corners.reshape(-1, 1, 2)
    cv2.drawChessboardCorners(bgr, spec.pattern_size, pts, patternWasFound=True)
  status = f"camera={cam_name}  captures={n_captures}  " \
           f"(SPACE=capture, ESC=next camera)"
  cv2.putText(bgr, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
              (0, 255, 255), 2, cv2.LINE_AA)
  if banner:
    cv2.putText(bgr, banner, (10, bgr.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2, cv2.LINE_AA)
  return bgr


def _calibrate_one_camera(
  source: FrameSource,
  spec: CheckerboardSpec,
  captures_dir: Path,
  min_captures: int,
  ui: Ui,
) -> CameraIntrinsics | None:
  """Run the SPACE/ESC capture loop on one camera and solve intrinsics."""
  window = f"calibration: {source.name}"
  captures: list[CalibrationCapture] = []
  captures_dir.mkdir(parents=True, exist_ok=True)
  banner: str | None = None
  banner_until = 0.0
  last_corners: np.ndarray | None = None

  while True:
    frame_rgb = source.read()
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    last_corners = _find_checkerboard(gray, spec)
    now = time.monotonic()
    display = _render_calibration_overlay(
      frame_rgb, spec, last_corners, len(captures), source.name,
      banner=banner if now < banner_until else None,
    )
    ui.show(window, display)

    key = ui.poll_key(timeout_ms=1)
    if key == "esc":
      break
    if key == "space":
      if last_corners is None:
        banner = "checkerboard not detected"
        banner_until = now + 0.6
        print("\a", end="", flush=True)
        continue
      desc = ui.prompt(f"description for capture {len(captures)} (blank = none): ").strip()
      img_path = captures_dir / f"capture-{len(captures):03d}.png"
      cv2.imwrite(str(img_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
      captures.append(CalibrationCapture(
        image_path=img_path, description=desc, corners_px=last_corners.copy()))
      banner = f"captured #{len(captures) - 1}"
      banner_until = now + 0.6

  ui.close(window)

  if len(captures) < min_captures:
    logger.warning(
      "camera %s: only %d/%d captures — skipping intrinsics solve",
      source.name, len(captures), min_captures)
    return None

  return solve_intrinsics(captures, spec, source.image_size)


def solve_intrinsics(
  captures: list[CalibrationCapture],
  spec: CheckerboardSpec,
  image_size: tuple[int, int],
) -> CameraIntrinsics:
  """Run ``cv2.calibrateCamera`` over a set of captures."""
  objp = spec.object_points()
  object_points = [objp.copy() for _ in captures]
  image_points = [c.corners_px.astype(np.float32) for c in captures]
  rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
    object_points, image_points, image_size, None, None)
  return CameraIntrinsics(
    K=np.asarray(K, dtype=np.float64),
    dist=np.asarray(dist, dtype=np.float64).reshape(-1),
    reproj_error=float(rms),
    image_size=image_size,
    captures=list(captures),
  )


def calibrate_cameras(
  sources: dict[str, FrameSource],
  spec: CheckerboardSpec,
  scene_dir: Path,
  *,
  min_captures: int = 5,
  ui: Ui | None = None,
) -> dict[str, CameraIntrinsics]:
  """Drive phase A.1 across every camera in ``sources``."""
  ui = ui or default_ui()
  out: dict[str, CameraIntrinsics] = {}
  for name, src in sources.items():
    captures_dir = scene_dir / "intrinsics" / name / "captures"
    intr = _calibrate_one_camera(src, spec, captures_dir, min_captures, ui)
    if intr is not None:
      out[name] = intr
  return out


# ---------------------------------------------------------------------------
# Phase A.2 — marker clip capture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkerCaptureSpec:
  locations: tuple[str, ...]
  clip_duration_s: float
  min_blob_fraction: float        # for non-off_screen slots
  off_screen_slot: str = "off_screen"


def _validate_clip(
  clip: MarkerClip, spec: MarkerCaptureSpec,
) -> tuple[bool, str]:
  """Decide whether a captured clip is acceptable for its slot."""
  f = clip.detected_fraction
  if clip.loc_name == spec.off_screen_slot:
    ok = f < (1.0 - spec.min_blob_fraction)
    return ok, f"off_screen: detected_fraction={f:.2f} (must be < {1.0 - spec.min_blob_fraction:.2f})"
  ok = f >= spec.min_blob_fraction
  return ok, f"detected_fraction={f:.2f} (must be >= {spec.min_blob_fraction:.2f})"


def _render_marker_preview(
  frame_rgb: np.ndarray,
  loc_name: str,
  expect_blob: bool,
  banner: str | None,
  recording: bool,
  elapsed_s: float,
  duration_s: float,
) -> np.ndarray:
  fit = detect_blob(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
  bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
  bgr = render_blob_overlay(bgr, fit)
  status = f"location='{loc_name}'  expect={'blob' if expect_blob else 'no blob'}"
  cv2.putText(bgr, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
              (0, 255, 255), 2, cv2.LINE_AA)
  hint = "SPACE=start clip   ESC=abort" if not recording \
         else f"RECORDING  {elapsed_s:0.1f}/{duration_s:0.1f}s"
  cv2.putText(bgr, hint, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
              (0, 0, 255) if recording else (255, 255, 255), 2, cv2.LINE_AA)
  if banner:
    cv2.putText(bgr, banner, (10, bgr.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2, cv2.LINE_AA)
  if recording:
    cv2.circle(bgr, (bgr.shape[1] - 30, 30), 10, (0, 0, 255), -1)
  return bgr


def _capture_clip(
  source: FrameSource,
  loc_name: str,
  duration_s: float,
  ui: Ui,
  window: str,
  expect_blob: bool,
) -> MarkerClip:
  """Pull ``duration_s`` worth of frames at the camera's native fps.

  Returns the assembled :class:`MarkerClip` with per-frame blob results.
  """
  fps = source.fps
  n_target = max(1, int(round(fps * duration_s)))
  frames: list[np.ndarray] = []
  centroids: list[tuple[float, float]] = []
  radii: list[float] = []
  detected: list[bool] = []

  start = time.monotonic()
  while len(frames) < n_target:
    frame_rgb = source.read()
    fit = detect_blob(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    frames.append(frame_rgb)
    centroids.append(fit.centroid_xy)
    radii.append(fit.radius_px)
    detected.append(fit.detected)
    elapsed = time.monotonic() - start
    display = _render_marker_preview(
      frame_rgb, loc_name, expect_blob, banner=None,
      recording=True, elapsed_s=elapsed, duration_s=duration_s)
    ui.show(window, display)
    ui.pump(timeout_ms=1)

  return MarkerClip(
    loc_name=loc_name,
    frames=np.stack(frames, axis=0),
    blob_centroids_px=np.asarray(centroids, dtype=np.float32),
    blob_radii_px=np.asarray(radii, dtype=np.float32),
    blob_detected=np.asarray(detected, dtype=bool),
    fps=int(fps),
    duration_s=float(duration_s),
  )


def register_markers(
  source: FrameSource,
  spec: MarkerCaptureSpec,
  *,
  ui: Ui | None = None,
) -> list[MarkerClip]:
  """Drive phase A.2 on a single primary camera."""
  ui = ui or default_ui()
  window = f"markers: {source.name}"
  clips: list[MarkerClip] = []
  banner: str | None = None
  banner_until = 0.0

  for loc_name in spec.locations:
    expect_blob = loc_name != spec.off_screen_slot
    while True:                                  # retry loop for this slot
      # Live preview until the user presses SPACE or ESC.
      while True:
        frame_rgb = source.read()
        now = time.monotonic()
        display = _render_marker_preview(
          frame_rgb, loc_name, expect_blob,
          banner=banner if now < banner_until else None,
          recording=False, elapsed_s=0.0, duration_s=spec.clip_duration_s)
        ui.show(window, display)
        key = ui.poll_key(timeout_ms=1)
        if key == "esc":
          ui.close(window)
          raise SceneRegistrationAborted(
            f"aborted during marker registration at location {loc_name!r}")
        if key == "space":
          break

      clip = _capture_clip(
        source, loc_name, spec.clip_duration_s, ui, window, expect_blob)
      ok, reason = _validate_clip(clip, spec)
      if ok:
        clips.append(clip)
        banner = f"captured '{loc_name}' ({reason})"
        banner_until = time.monotonic() + 1.0
        break
      banner = f"rejected: {reason} — retry"
      banner_until = time.monotonic() + 2.0
      print("\a", end="", flush=True)

  ui.close(window)
  return clips


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class SceneRegistrationAborted(RuntimeError):
  """Raised when the user hits ESC during marker registration."""


def register_scene(
  sources: dict[str, FrameSource],
  primary_name: str,
  checker: CheckerboardSpec,
  markers: MarkerCaptureSpec,
  scene_dir: Path,
  *,
  min_calibration_captures: int = 5,
  skip_calibration: bool = False,
  skip_markers: bool = False,
  ui: Ui | None = None,
) -> SceneRegistration:
  """Top-level phase A driver. ``sources`` are already connected."""
  ui = ui or default_ui()
  intrinsics: dict[str, CameraIntrinsics] = {}
  if not skip_calibration:
    intrinsics = calibrate_cameras(
      sources, checker, scene_dir,
      min_captures=min_calibration_captures, ui=ui)

  marker_clips: list[MarkerClip] = []
  if not skip_markers:
    if primary_name not in sources:
      raise KeyError(f"primary camera {primary_name!r} not in sources {list(sources)}")
    marker_clips = register_markers(sources[primary_name], markers, ui=ui)

  return SceneRegistration(intrinsics=intrinsics, markers=marker_clips)


__all__ = [
  "CheckerboardSpec",
  "MarkerCaptureSpec",
  "FrameSource",
  "Ui",
  "OpenCVFrameSource",
  "SceneRegistrationAborted",
  "calibrate_cameras",
  "register_markers",
  "register_scene",
  "solve_intrinsics",
  "default_ui",
]


# Keep ``Iterable`` referenced for stub-friendly typing.
_ = Iterable
