"""Top-level orchestrator: per-dataset intrinsic + extrinsic calibration.

When ``interactive=True`` (the default), the pipeline opens matplotlib
review windows at each detection step:

  * Per checkerboard frame: accept the auto-detection, reject it, or
    click the 4 outer board corners.
  * For the mat: accept the auto-detection, drag individual control
    points, re-annotate from scratch, or rerun a different detector.
  * For the verify overlays: a final sign-off on the projected mat
    geometry; if rejected, no calibration is written.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .. import constants
from ..dataset import sample_episode_frames
from .annotate import annotate_mat
from .extrinsics import solve_extrinsics
from .intrinsics import calibrate_intrinsics
from .interactive import (
  review_checkerboard_frame,
  review_mat_detection,
  review_verify_overlays,
)
from .mat_detection import detect_mat_markings
from .serialization import (
  CameraCalibration,
  calibration_dir,
  load_calibration,
  save_calibration,
)
from .verify import render_overlay, save_overlay

logger = logging.getLogger(__name__)


def _laplacian_var(rgb: np.ndarray) -> float:
  """Variance of the Laplacian — used to pick a sharp empty-scene frame."""
  import cv2
  gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
  return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _pick_sharpest(frames: list[np.ndarray]) -> np.ndarray:
  scores = [_laplacian_var(f) for f in frames]
  i = int(np.argmax(scores))
  logger.info("picked frame %d/%d (sharpness=%.2f)", i, len(frames) - 1, scores[i])
  return frames[i]


def analyze_one(
  dataset_id: str,
  cache_dir:  Path,
  interactive: bool = True,
  force:       bool = False,
  intrinsic_samples: int = 60,
) -> CameraCalibration | None:
  """Compute and cache one calibration.

  Returns the :class:`CameraCalibration` written, or ``None`` if the
  user aborted at an interactive sign-off (no file is written in that
  case).
  """
  out_dir = calibration_dir(cache_dir, dataset_id)
  cal_json = out_dir / "calibration.json"
  if cal_json.exists() and not force:
    logger.info("[%s] cached calibration found at %s", dataset_id, cal_json)
    return load_calibration(cache_dir, dataset_id)

  from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-untyped]

  logger.info("[%s] loading dataset", dataset_id)
  ds = LeRobotDataset(dataset_id)

  camera_key = constants.camera_key()
  ep_checker = constants.episode_checkerboard()
  ep_empty   = constants.episode_empty()

  # Intrinsics — checkerboard episode.
  cb_frames = sample_episode_frames(ds, ep_checker, camera_key, intrinsic_samples)
  if not cb_frames:
    raise RuntimeError(f"[{dataset_id}] no frames in checkerboard episode {ep_checker}")
  reviewer = None
  if interactive:
    def reviewer(frame, corners, cols, rows, label):
      r = review_checkerboard_frame(frame, corners, cols, rows, label)
      return r.decision, r.corners
  intr = calibrate_intrinsics(cb_frames, reviewer=reviewer)

  # Extrinsics — empty mat episode.
  empty_frames = sample_episode_frames(ds, ep_empty, camera_key, 8)
  if not empty_frames:
    raise RuntimeError(f"[{dataset_id}] no frames in empty episode {ep_empty}")
  empty_frame = _pick_sharpest(empty_frames)

  detection = detect_mat_markings(empty_frame)
  mat_source = "auto" if detection is not None else "missing"
  if interactive:
    detection, mat_source = review_mat_detection(empty_frame, detection)
    if detection is None:
      logger.info("[%s] mat review aborted — no calibration written", dataset_id)
      return None
  elif detection is None:
    raise RuntimeError(
      f"[{dataset_id}] mat detection failed and interactive=False")

  extr = solve_extrinsics(detection, intr.K, intr.dist)

  cal = CameraCalibration(
    dataset_id=dataset_id,
    image_size=intr.image_size,
    K=intr.K,
    dist=intr.dist,
    R=extr.R,
    t=extr.t,
    intrinsic_rms_px=intr.rms_px,
    extrinsic_rms_px=extr.rms_px,
  )

  # Build verify overlays in memory (used both for the interactive sign-off
  # and for the on-disk artefacts).
  verify_overlays: list[tuple[str, np.ndarray]] = [
    ("empty scene", render_overlay(empty_frame, cal)),
  ]
  for ep_idx in constants.object_episodes():
    obj_frames = sample_episode_frames(ds, ep_idx, camera_key, 1)
    if obj_frames:
      verify_overlays.append((f"episode {ep_idx}", render_overlay(obj_frames[0], cal)))
    else:
      logger.warning("[%s] episode %d has no frames — skipping verify overlay",
                     dataset_id, ep_idx)

  if interactive:
    if not review_verify_overlays(verify_overlays):
      logger.info("[%s] verify sign-off rejected — no calibration written", dataset_id)
      return None

  # Persist everything.
  out_dir.mkdir(parents=True, exist_ok=True)
  save_calibration(cal, cache_dir)
  np.savez(out_dir / "intrinsic_corners.npz",
           **{f"frame{idx:03d}": pts for idx, pts in  # type: ignore[arg-type]
              zip(intr.used_frame_idxs, intr.used_corners)})
  (out_dir / "mat_annotations.json").write_text(json.dumps({
    "line1_near": detection.line1_near.tolist(),
    "line1_far":  detection.line1_far.tolist(),
    "line2_near": detection.line2_near.tolist(),
    "line2_far":  detection.line2_far.tolist(),
    "circle_center":     detection.circle_center.tolist(),
    "circle_samples_uv": detection.circle_samples_uv.tolist(),
    "_source":    mat_source,
  }, indent=2))
  verify_dir = out_dir / "verify"
  for title, overlay in verify_overlays:
    name = "empty_scene_reproj.png" if title == "empty scene" \
           else f"{title.replace(' ', '_')}_reproj.png"
    save_overlay_image(overlay, verify_dir / name)

  logger.info("[%s] done: intrinsic RMS %.3f px, extrinsic RMS %.2f px",
              dataset_id, cal.intrinsic_rms_px, cal.extrinsic_rms_px)
  return cal


def save_overlay_image(rgb: np.ndarray, out_path: Path) -> None:
  """Write an already-rendered overlay to PNG."""
  import cv2
  out_path.parent.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def analyze_datasets(
  dataset_ids: list[str],
  cache_dir:   Path,
  interactive: bool = True,
  force:       bool = False,
) -> dict[str, CameraCalibration]:
  """Per-dataset calibration. Idempotent; reuses ``cache_dir`` entries."""
  cache_dir = Path(cache_dir)
  cache_dir.mkdir(parents=True, exist_ok=True)
  out: dict[str, CameraCalibration] = {}
  for did in dataset_ids:
    try:
      result = analyze_one(did, cache_dir, interactive=interactive, force=force)
    except Exception as e:
      logger.error("[%s] calibration failed: %s", did, e)
      raise
    if result is not None:
      out[did] = result
  return out
