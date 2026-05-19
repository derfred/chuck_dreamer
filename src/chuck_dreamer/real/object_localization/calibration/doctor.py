"""Diagnostics for LeRobot calibration datasets.

The doctor inspects each dataset without writing any calibration data
to disk. With ``image_dir`` set it also dumps per-episode diagnostic
PNGs (detected checkerboard corners, mat-marking overlay, sample
object-episode frames) so a user can eyeball whether the input data
looks sane before kicking off a full calibration.

Run via ``python main.py analyze-cameras --doctor --dataset <id>``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import constants
from ..dataset import episode_frame_indices, sample_episode_frames
from .intrinsics import _detect_corners
from .mat_detection import MatDetection, detect_mat_markings

logger = logging.getLogger(__name__)


_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _slugify(s: str) -> str:
  return _SLUG_RE.sub("__", s.strip()).strip("_")


@dataclass
class EpisodeReport:
  index:    int
  expected: str           # "empty" | "checkerboard" | "object_left" | ...
  found:    bool          # whether the dataset has this episode index
  length:   int = 0
  notes:    list[str] = field(default_factory=list)
  images:   list[Path] = field(default_factory=list)   # diagnostic PNGs


@dataclass
class DatasetReport:
  dataset_id:   str
  loaded:       bool
  num_episodes: int = 0
  image_shape:  tuple[int, int, int] | None = None   # (H, W, C)
  camera_key:   str = ""
  episodes:     list[EpisodeReport] = field(default_factory=list)
  warnings:     list[str] = field(default_factory=list)
  error:        str | None = None
  image_dir:    Path | None = None

  def format(self) -> str:
    lines: list[str] = [f"=== {self.dataset_id} ==="]
    if not self.loaded:
      lines.append(f"  FAILED to load: {self.error}")
      return "\n".join(lines)
    lines.append(f"  episodes={self.num_episodes}, camera_key={self.camera_key!r}")
    if self.image_shape is not None:
      H, W, C = self.image_shape
      lines.append(f"  image: {W}x{H}x{C}")
    if self.image_dir is not None:
      lines.append(f"  diagnostic images: {self.image_dir}")
    for ep in self.episodes:
      tag = "OK " if ep.found else "MISS"
      head = f"  [{tag}] ep{ep.index:>2} ({ep.expected}): length={ep.length}"
      if ep.notes:
        head += "  — " + "; ".join(ep.notes)
      lines.append(head)
      for img in ep.images:
        lines.append(f"        img: {img}")
    for w in self.warnings:
      lines.append(f"  ! {w}")
    return "\n".join(lines)


def _probe_checkerboard(ds, ep_idx: int, camera_key: str, n_samples: int = 8):
  """Return (detected_count, total_probed, first_detected_frame, first_corners).

  ``first_detected_frame`` and ``first_corners`` are ``None`` when no
  detection succeeds.
  """
  frames = sample_episode_frames(ds, ep_idx, camera_key, n_samples)
  if not frames:
    return 0, 0, None, None
  import cv2
  detected = 0
  first_frame = None
  first_corners = None
  for f in frames:
    gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    corners = _detect_corners(gray)
    if corners is not None:
      detected += 1
      if first_frame is None:
        first_frame = f
        first_corners = corners
  return detected, len(frames), first_frame, first_corners


def _probe_mat(ds, ep_idx: int, camera_key: str):
  """Return (detection_or_None, sample_frame, reason_if_failed)."""
  frames = sample_episode_frames(ds, ep_idx, camera_key, 1)
  if not frames:
    return None, None, "no frames"
  det = detect_mat_markings(frames[0])
  if det is None:
    return None, frames[0], "auto-detection failed (would need interactive annotation)"
  return det, frames[0], None


def _save_png(rgb: np.ndarray, out_path: Path) -> Path:
  """Write an HxWx3 RGB array as a PNG. Returns the path."""
  import cv2
  out_path.parent.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
  return out_path


def _render_checkerboard(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
  """Return an overlay showing detected checkerboard corners + grid order."""
  import cv2
  img = frame.copy()
  bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  cols, rows = constants.checkerboard_inner_corners()
  cv2.drawChessboardCorners(bgr, (cols, rows), corners, True)
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _render_mat_overlay(frame: np.ndarray, det: MatDetection) -> np.ndarray:
  """Draw detected lines, ellipse, and labeled endpoints on the frame."""
  import cv2
  img = frame.copy()
  bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  # Line 1 (bottom) green, Line 2 (top) red, circle orange.
  cv2.line(bgr, tuple(det.line1_near.astype(int)), tuple(det.line1_far.astype(int)),
           (0, 255, 0), 3)
  cv2.line(bgr, tuple(det.line2_near.astype(int)), tuple(det.line2_far.astype(int)),
           (0, 0, 255), 3)
  axes = (int(det.circle_axes[0] / 2), int(det.circle_axes[1] / 2))
  cv2.ellipse(bgr, tuple(det.circle_center.astype(int)), axes,
              float(det.circle_angle), 0, 360, (0, 165, 255), 2)
  for label, pt, color in (
    ("L1 near", det.line1_near, (0, 255, 0)),
    ("L1 far",  det.line1_far,  (0, 255, 0)),
    ("L2 near", det.line2_near, (0, 0, 255)),
    ("L2 far",  det.line2_far,  (0, 0, 255)),
  ):
    cv2.circle(bgr, tuple(pt.astype(int)), 8, color, -1)
    cv2.putText(bgr, label, tuple(pt.astype(int) + np.array([10, -10])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
  cv2.circle(bgr, tuple(det.circle_center.astype(int)), 6, (0, 165, 255), -1)
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _render_mat_failure(frame: np.ndarray) -> np.ndarray:
  """Annotate a frame the mat detector rejected, to help the user diagnose."""
  import cv2
  img = frame.copy()
  bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  H, W = bgr.shape[:2]
  banner = np.zeros((40, W, 3), dtype=np.uint8)
  cv2.putText(banner, "mat auto-detect failed — see overlay for raw frame",
              (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
  out = np.concatenate([banner, bgr], axis=0)
  return np.asarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


_DOCTOR_WINDOW = "doctor"
_DOCTOR_QUIT = {"v": False}


def _show_image(rgb: np.ndarray, title: str) -> None:
  """Show ``rgb`` in an OpenCV window; advance on any key (Esc/q to abort).

  Uses a single reused window so popups don't pile up. The displayed
  image is scaled to fit ``_MAX_WINDOW_*`` while preserving aspect
  ratio so very large frames stay on screen on a typical laptop.
  """
  import cv2
  if _DOCTOR_QUIT["v"]:
    return

  max_w, max_h = 1280, 760
  H, W = rgb.shape[:2]
  scale = min(max_w / W, max_h / H, 1.0)
  if scale < 1.0:
    disp = cv2.resize(rgb, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
  else:
    disp = rgb

  bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
  # Burn the title into the top of the image so the OS window-title bar
  # truncation does not hide it on small displays.
  H_disp, W_disp = bgr.shape[:2]
  banner = np.zeros((36, W_disp, 3), dtype=np.uint8)
  cv2.putText(banner, f"{title}    [any key: next  |  Esc/q: abort]",
              (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
  framed = np.concatenate([banner, bgr], axis=0)

  cv2.namedWindow(_DOCTOR_WINDOW, cv2.WINDOW_AUTOSIZE)
  cv2.imshow(_DOCTOR_WINDOW, framed)
  key = cv2.waitKey(0) & 0xFF
  if key in (27, ord("q"), ord("Q")):
    _DOCTOR_QUIT["v"] = True
    cv2.destroyAllWindows()


def _close_doctor_window() -> None:
  import cv2
  try:
    cv2.destroyAllWindows()
  except Exception:
    pass
  _DOCTOR_QUIT["v"] = False


def diagnose_dataset(
  dataset_id: str,
  image_dir:    Path | None = None,
  interactive:  bool = False,
) -> DatasetReport:
  """Inspect a single dataset and return a structured report.

  When ``image_dir`` is provided, per-episode diagnostic PNGs are
  written under ``image_dir/<slug>/`` and listed on the returned
  ``EpisodeReport.images``.

  When ``interactive`` is true, each overlay is also shown in a
  matplotlib popup as it is built (any key closes the window and moves
  on). Combine with ``image_dir`` to get popups + persisted PNGs.
  """
  report = DatasetReport(dataset_id=dataset_id, loaded=False)
  try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-untyped]
    ds = LeRobotDataset(dataset_id)
  except Exception as e:
    report.error = f"{type(e).__name__}: {e}"
    return report

  report.loaded = True
  report.num_episodes = int(getattr(ds, "num_episodes", 0))
  camera_key = constants.camera_key()
  report.camera_key = camera_key

  out_dir: Path | None = None
  if image_dir is not None:
    out_dir = Path(image_dir) / _slugify(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.image_dir = out_dir

  # Probe image dims via the camera key.
  try:
    sample = ds[0]
    if camera_key not in sample:
      report.warnings.append(
        f"camera key {camera_key!r} missing from frame 0 "
        f"(available: {[k for k in sample.keys() if 'image' in k.lower()] or list(sample.keys())[:8]})")
    else:
      t = sample[camera_key]
      arr = np.asarray(t.permute(1, 2, 0)) if hasattr(t, "permute") else np.asarray(t)
      if arr.ndim == 3:
        report.image_shape = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
  except Exception as e:
    report.warnings.append(f"could not sample frame 0: {e}")

  expected = [
    (constants.episode_empty(),        "empty"),
    (constants.episode_checkerboard(), "checkerboard"),
    (constants.episode_object_left(),  "object_left"),
    (constants.episode_object_mid(),   "object_mid"),
    (constants.episode_object_right(), "object_right"),
  ]
  for ep_idx, label in expected:
    ep_report = EpisodeReport(index=ep_idx, expected=label, found=False)
    if ep_idx >= report.num_episodes:
      ep_report.notes.append(
        f"dataset only has {report.num_episodes} episodes — index {ep_idx} not present")
      report.episodes.append(ep_report)
      continue
    ep_report.found = True
    try:
      rng = episode_frame_indices(ds, ep_idx)
      ep_report.length = len(rng)
    except Exception as e:
      ep_report.notes.append(f"frame-range lookup failed: {e}")
      report.episodes.append(ep_report)
      continue

    overlay_rgb: np.ndarray | None = None
    overlay_name: str | None = None
    overlay_title: str | None = None

    if label == "checkerboard":
      det, total, sample_frame, corners = _probe_checkerboard(ds, ep_idx, camera_key)
      ep_report.notes.append(
        f"checkerboard detected in {det}/{total} probed frames "
        f"({constants.checkerboard_inner_corners()} inner corners, "
        f"{constants.checkerboard_square_mm()}mm squares)")
      if det == 0:
        ep_report.notes.append("no checkerboard found — intrinsic calibration WILL FAIL")
      elif det < total // 2:
        ep_report.notes.append("low checkerboard detection rate — intrinsic RMS may be poor")
      if sample_frame is not None and corners is not None:
        overlay_rgb = _render_checkerboard(sample_frame, corners)
        overlay_name = f"ep{ep_idx:02d}_checkerboard_detected.png"
        overlay_title = f"{dataset_id} — ep{ep_idx} (checkerboard, detected)"
      else:
        frames = sample_episode_frames(ds, ep_idx, camera_key, 1)
        if frames:
          overlay_rgb = frames[0]
          overlay_name = f"ep{ep_idx:02d}_checkerboard_NOT_DETECTED.png"
          overlay_title = f"{dataset_id} — ep{ep_idx} (checkerboard, NOT detected)"

    elif label == "empty":
      mat_det, frame, why = _probe_mat(ds, ep_idx, camera_key)
      if mat_det is not None:
        ep_report.notes.append("mat markings auto-detected")
        if frame is not None:
          overlay_rgb = _render_mat_overlay(frame, mat_det)
          overlay_name = f"ep{ep_idx:02d}_mat_detected.png"
          overlay_title = f"{dataset_id} — ep{ep_idx} (mat, detected)"
      else:
        ep_report.notes.append(f"mat auto-detect: {why}")
        if frame is not None:
          overlay_rgb = _render_mat_failure(frame)
          overlay_name = f"ep{ep_idx:02d}_mat_FAILED.png"
          overlay_title = f"{dataset_id} — ep{ep_idx} (mat, FAILED)"

    else:  # object episodes
      ep_report.notes.append("would render a verify overlay using the fitted calibration")
      frames = sample_episode_frames(ds, ep_idx, camera_key, 1)
      if frames:
        overlay_rgb = frames[0]
        overlay_name = f"ep{ep_idx:02d}_{label}_sample.png"
        overlay_title = f"{dataset_id} — ep{ep_idx} ({label})"

    if overlay_rgb is not None:
      if out_dir is not None and overlay_name is not None:
        ep_report.images.append(_save_png(overlay_rgb, out_dir / overlay_name))
      if interactive and overlay_title is not None:
        _show_image(overlay_rgb, overlay_title)

    report.episodes.append(ep_report)

  return report


def diagnose_datasets(
  dataset_ids: list[str],
  image_dir:    Path | None = None,
  interactive:  bool = False,
) -> list[DatasetReport]:
  _DOCTOR_QUIT["v"] = False
  try:
    return [
      diagnose_dataset(did, image_dir=image_dir, interactive=interactive)
      for did in dataset_ids
    ]
  finally:
    if interactive:
      _close_doctor_window()
