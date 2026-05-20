"""Top-level orchestrator: per-dataset intrinsic + extrinsic calibration.

Three review steps run in both interactive and non-interactive modes
(in interactive mode each step also pops up a cv2 window):

  1. **Mat detection** — auto-detect the painted lines + circle on the
     empty episode; in interactive mode the user can accept, drag the
     control points, re-annotate from scratch, or re-run the auto
     detector.
  2. **Checkerboard** — silently fit intrinsics from the checkerboard
     episode; in interactive mode show a single summary popup with
     detected corners overlaid on one frame plus the RMS.
  3. **Object highlights** — for each object episode, run the
     heuristic mask, draw a bbox + centroid, and back-project the
     centroid to the mat plane via the new extrinsics so the world
     ``(x, y)`` is shown on the overlay. In interactive mode each one
     pops up; in both modes they land on disk.

Verify-overlay sign-off (interactive only): the user gets a final
chance to accept or reject before ``calibration.json`` is written.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import constants
from ..dataset import sample_episode_frames
from ..geometry import ray_plane_intersection_z0
from .extrinsics import solve_extrinsics
from .intrinsics import calibrate_intrinsics
from .interactive import (
  click_to_seed_point,
  close_popup_window,
  render_checkerboard_overlay,
  render_object_highlight_overlay,
  reset_popup_state,
  review_mat_detection,
  review_verify_overlays,
  show_overlay,
)
from .mat_detection import detect_mat_markings
from .serialization import (
  CameraCalibration,
  calibration_dir,
  load_calibration,
  save_calibration,
)
from .verify import render_overlay

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


def _save_png(rgb: np.ndarray, out_path: Path) -> None:
  """Write an HxWx3 RGB array as a PNG."""
  import cv2
  out_path.parent.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _highlight_from_centroid(
  frame:    np.ndarray,
  cal:      CameraCalibration,
  label:    str,
  cx:       int,
  cy:       int,
  bbox:     tuple[int, int, int, int] | None,
) -> np.ndarray:
  """Render the bbox + centroid + back-projected (x, y) mm overlay."""
  world = ray_plane_intersection_z0(
    cal.K, cal.R, cal.t, np.array([[cx, cy]], dtype=np.float64), dist=cal.dist,
  )
  world_xy = (float(world[0, 0]), float(world[0, 1]))
  return render_object_highlight_overlay(
    frame, bbox_xyxy=bbox, centroid_uv=(cx, cy),
    world_xy_mm=world_xy, label=label,
  )


def _object_mask_from_click(
  frame: np.ndarray, mat_roi: np.ndarray, click_uv: tuple[int, int],
) -> np.ndarray | None:
  """Flood-fill–style mask around a user click, bounded to the mat ROI."""
  import cv2
  H, W = frame.shape[:2]
  cu, cv_ = click_uv
  hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
  seed = hsv[cv_, cu]
  diff = np.linalg.norm(hsv - seed[None, None, :], axis=-1)
  similar = (diff < 35) & mat_roi
  similar = cv2.morphologyEx(similar.astype(np.uint8) * 255,
                             cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
  num, labels, stats, _ = cv2.connectedComponentsWithStats(similar, connectivity=8)
  if num <= 1:
    return None
  # Pick the component containing the click.
  lab = int(labels[cv_, cu])
  if lab == 0:
    return None
  mask = labels == lab
  if int(mask.sum()) < 80:
    return None
  return np.asarray(mask)


def _filter_components_by_mat(
  mask: np.ndarray, mat_roi: np.ndarray,
  mat_detection_circle_uv: tuple[int, int] | None,
  mat_circle_radius_px:    float,
) -> np.ndarray | None:
  """Drop components that don't sit on the mat or that *are* the fiducial circle.

  The painted mat fiducial often has a strong non-mat color and gets
  picked by the heuristic. We reject any candidate whose centroid sits
  within ``2 × mat_circle_radius_px`` of the detected fiducial center,
  and keep the largest survivor.
  """
  import cv2
  m = (mask & mat_roi).astype(np.uint8)
  num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
  if num <= 1:
    return None
  best_label = -1
  best_area  = 0
  exclusion_radius_sq = (2.5 * max(mat_circle_radius_px, 8.0)) ** 2
  for i in range(1, num):
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < 80:
      continue
    cx_i, cy_i = float(centroids[i, 0]), float(centroids[i, 1])
    if mat_detection_circle_uv is not None:
      dx = cx_i - mat_detection_circle_uv[0]
      dy = cy_i - mat_detection_circle_uv[1]
      if dx * dx + dy * dy < exclusion_radius_sq:
        continue
    if area > best_area:
      best_area = area
      best_label = i
  if best_label < 0:
    return None
  return np.asarray(labels == best_label)


def _build_object_highlight(
  frame:        np.ndarray,
  cal:          CameraCalibration,
  label:        str,
  mat_detection_circle_uv: tuple[int, int] | None,
  mat_circle_radius_px:    float,
  interactive:  bool,
) -> np.ndarray:
  """Render an overlay highlighting the object in ``frame``.

  Steps:
    1. Estimate the mat ROI (Otsu on inverted grayscale).
    2. Run the color heuristic constrained to the ROI.
    3. Drop the painted mat fiducial circle (by exclusion radius around
       its centroid) and any too-small components.
    4. If nothing remains, in interactive mode pop a cv2 click-to-seed
       window; otherwise render the raw frame with a "(no segmentation)"
       label so the user knows nothing was found.
  """
  # Lazy import: pose/__init__ imports calibration, cycling at top scope.
  from ..pose.segmentation import heuristic_mask
  from .mat_detection import _estimate_mat_roi

  mat_roi = _estimate_mat_roi(frame)

  raw_mask = heuristic_mask(frame, mat_roi_mask=mat_roi)
  mask = None
  if raw_mask is not None:
    mask = _filter_components_by_mat(
      raw_mask, mat_roi, mat_detection_circle_uv, mat_circle_radius_px,
    )

  if mask is None and interactive:
    click = click_to_seed_point(
      frame, f"{label}: no object found — click on it (or skip)",
    )
    if click is not None:
      seeded = _object_mask_from_click(frame, mat_roi, click)
      if seeded is not None:
        mask = _filter_components_by_mat(
          seeded, mat_roi, mat_detection_circle_uv, mat_circle_radius_px,
        ) or seeded   # if filter strips it, keep the raw seeded mask

  if mask is None or not mask.any():
    return render_object_highlight_overlay(
      frame, bbox_xyxy=None, centroid_uv=None, world_xy_mm=None,
      label=f"{label} (no segmentation)",
    )

  ys, xs = np.where(mask)
  x0, x1 = int(xs.min()), int(xs.max())
  y0, y1 = int(ys.min()), int(ys.max())
  cx = int(xs.mean())
  cy = int(ys.mean())
  return _highlight_from_centroid(frame, cal, label, cx, cy, (x0, y0, x1, y1))


@dataclass
class _SharedIntrinsics:
  K:           np.ndarray
  dist:        np.ndarray
  image_size:  tuple[int, int]
  rms_px:      float
  n_views:     int
  source_datasets: list[str]
  preview_frame:   np.ndarray | None = None
  preview_corners: np.ndarray | None = None

  def to_json_dict(self) -> dict:
    return {
      "K":          self.K.tolist(),
      "dist":       self.dist.tolist(),
      "image_size": list(self.image_size),
      "intrinsic_rms_px": float(self.rms_px),
      "n_views":          int(self.n_views),
      "source_datasets":  list(self.source_datasets),
    }


def _load_or_fit_pooled_intrinsics(
  dataset_ids:       list[str],
  cache_dir:         Path,
  interactive:       bool,
  force:             bool,
  intrinsic_samples: int,
) -> _SharedIntrinsics | None:
  """Fit (or load) the shared intrinsics across every dataset's checkerboard.

  Returns ``None`` if the user aborts the interactive review (no
  intrinsics.json is written in that case).
  """
  shared_path = cache_dir / "intrinsics.json"
  cb_cols, cb_rows = constants.checkerboard_inner_corners()

  if shared_path.exists() and not force:
    logger.info("shared intrinsics found at %s — reusing", shared_path)
    blob = json.loads(shared_path.read_text())
    return _SharedIntrinsics(
      K          = np.asarray(blob["K"], dtype=np.float64),
      dist       = np.asarray(blob["dist"], dtype=np.float64).reshape(-1),
      image_size = (int(blob["image_size"][0]), int(blob["image_size"][1])),
      rms_px     = float(blob["intrinsic_rms_px"]),
      n_views    = int(blob["n_views"]),
      source_datasets = list(blob.get("source_datasets", [])),
    )

  # Pool checkerboard frames across all datasets.
  from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-untyped]
  camera_key = constants.camera_key()
  ep_checker = constants.episode_checkerboard()

  pooled_frames: list[np.ndarray] = []
  per_dataset_offsets: list[tuple[str, int, int]] = []   # (dataset_id, start, count)
  for did in dataset_ids:
    logger.info("[%s] loading checkerboard frames for pooled intrinsics", did)
    ds = LeRobotDataset(did)
    cb_frames = sample_episode_frames(ds, ep_checker, camera_key, intrinsic_samples)
    if not cb_frames:
      logger.warning("[%s] no frames in checkerboard episode %d — skipping", did, ep_checker)
      continue
    per_dataset_offsets.append((did, len(pooled_frames), len(cb_frames)))
    pooled_frames.extend(cb_frames)

  if not pooled_frames:
    raise RuntimeError("no checkerboard frames in any dataset")
  logger.info("pooled intrinsic fit: %d frames across %d dataset(s)",
              len(pooled_frames), len(per_dataset_offsets))

  intr = calibrate_intrinsics(pooled_frames)

  # Pick a representative used frame for the preview overlay.
  preview_frame: np.ndarray | None = None
  preview_corners: np.ndarray | None = None
  if intr.used_frame_idxs:
    idx_in_pool = intr.used_frame_idxs[len(intr.used_frame_idxs) // 2]
    if 0 <= idx_in_pool < len(pooled_frames):
      preview_frame = pooled_frames[idx_in_pool]
      preview_corners = intr.used_corners[
        intr.used_frame_idxs.index(idx_in_pool)
      ].reshape(-1, 1, 2)

  shared = _SharedIntrinsics(
    K          = intr.K,
    dist       = intr.dist,
    image_size = intr.image_size,
    rms_px     = intr.rms_px,
    n_views    = len(intr.used_frame_idxs),
    source_datasets = [did for did, _, _ in per_dataset_offsets],
    preview_frame   = preview_frame,
    preview_corners = preview_corners,
  )

  if interactive and preview_frame is not None and preview_corners is not None:
    overlay = render_checkerboard_overlay(preview_frame, preview_corners, cb_cols, cb_rows)
    title = (f"pooled checkerboard fit  —  RMS={intr.rms_px:.3f}px  "
             f"({len(intr.used_frame_idxs)}/{len(pooled_frames)} frames from "
             f"{len(per_dataset_offsets)} dataset(s))")
    if show_overlay(overlay, title) == "abort":
      logger.info("pooled intrinsic review aborted — no intrinsics written")
      return None

  cache_dir.mkdir(parents=True, exist_ok=True)
  shared_path.write_text(json.dumps(shared.to_json_dict(), indent=2))
  logger.info("wrote shared intrinsics to %s (RMS %.3f px from %d views)",
              shared_path, intr.rms_px, len(intr.used_frame_idxs))
  return shared


def analyze_one(
  dataset_id:  str,
  cache_dir:   Path,
  shared_intr: _SharedIntrinsics,
  interactive: bool = True,
  force:       bool = False,
  verify_dir_override: Path | None = None,
) -> CameraCalibration | None:
  """Compute and cache one calibration (extrinsic only — intrinsics are pre-fit).

  Returns the :class:`CameraCalibration` written, or ``None`` if the
  user aborted at an interactive sign-off.
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
  ep_empty   = constants.episode_empty()

  # ----- Step 1: mat detection ------------------------------------------- #
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

  # ----- Step 2: extrinsics (intrinsics are pooled, already fit) --------- #
  extr = solve_extrinsics(detection, shared_intr.K, shared_intr.dist)
  cal = CameraCalibration(
    dataset_id=dataset_id,
    image_size=shared_intr.image_size,
    K=shared_intr.K,
    dist=shared_intr.dist,
    R=extr.R,
    t=extr.t,
    intrinsic_rms_px=shared_intr.rms_px,
    extrinsic_rms_px=extr.rms_px,
  )

  # ----- Build verify + highlight overlays ------------------------------- #
  verify_overlays: list[tuple[str, np.ndarray]] = [
    ("empty scene", render_overlay(empty_frame, cal)),
  ]

  fiducial_uv = (
    float(detection.circle_center[0]),
    float(detection.circle_center[1]),
  )
  fiducial_radius_px = float(0.5 * (detection.circle_axes[0] + detection.circle_axes[1]) / 2.0)

  for ep_idx in constants.object_episodes():
    obj_frames = sample_episode_frames(ds, ep_idx, camera_key, 1)
    if not obj_frames:
      logger.warning("[%s] episode %d has no frames — skipping highlight",
                     dataset_id, ep_idx)
      continue
    highlight = _build_object_highlight(
      obj_frames[0], cal, f"episode {ep_idx}",
      mat_detection_circle_uv=(int(fiducial_uv[0]), int(fiducial_uv[1])),
      mat_circle_radius_px=fiducial_radius_px,
      interactive=interactive,
    )
    verify_overlays.append((f"episode {ep_idx} (object highlight)", highlight))

  # ----- Step 3: interactive sign-off ------------------------------------ #
  if interactive:
    if not review_verify_overlays(verify_overlays):
      logger.info("[%s] verify sign-off rejected — no calibration written", dataset_id)
      return None

  # ----- Persist --------------------------------------------------------- #
  out_dir.mkdir(parents=True, exist_ok=True)
  save_calibration(cal, cache_dir)
  (out_dir / "mat_annotations.json").write_text(json.dumps({
    "line1_near": detection.line1_near.tolist(),
    "line1_far":  detection.line1_far.tolist(),
    "line2_near": detection.line2_near.tolist(),
    "line2_far":  detection.line2_far.tolist(),
    "circle_center":     detection.circle_center.tolist(),
    "circle_samples_uv": detection.circle_samples_uv.tolist(),
    "_source":    mat_source,
  }, indent=2))
  verify_dir = verify_dir_override / dataset_id.replace("/", "__") if verify_dir_override else out_dir / "verify"
  for title, overlay in verify_overlays:
    if title == "empty scene":
      name = "empty_scene_reproj.png"
    elif "object highlight" in title:
      ep_idx = int(title.split()[1])
      name = f"episode_{ep_idx}_object_highlight.png"
    else:
      name = f"{title.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    _save_png(overlay, verify_dir / name)

  logger.info("[%s] done: intrinsic RMS %.3f px (shared), extrinsic RMS %.2f px",
              dataset_id, cal.intrinsic_rms_px, cal.extrinsic_rms_px)
  return cal


def analyze_datasets(
  dataset_ids: list[str],
  cache_dir:   Path,
  interactive: bool = True,
  force:       bool = False,
  intrinsic_samples: int = 60,
  intrinsics_only:   bool = False,
  verify_dir_override: Path | None = None,
) -> dict[str, CameraCalibration]:
  """Per-dataset calibration with shared (pooled) intrinsics.

  Intrinsic K, dist are fit once over the union of all datasets'
  checkerboard episodes and cached at ``<cache-dir>/intrinsics.json``.
  Each dataset then gets its own extrinsic (R, t) saved as
  ``<cache-dir>/<dataset>/calibration.json``. Idempotent; reuses cache
  entries unless ``force=True``.

  When ``intrinsics_only=True``, only the pooled intrinsic fit runs and
  ``intrinsics.json`` is written; no per-dataset extrinsic fitting
  happens. Useful for a two-phase workflow where intrinsics are
  computed once over many datasets and later runs pick them up.
  """
  cache_dir = Path(cache_dir)
  cache_dir.mkdir(parents=True, exist_ok=True)
  if interactive:
    reset_popup_state()

  out: dict[str, CameraCalibration] = {}
  try:
    shared_intr = _load_or_fit_pooled_intrinsics(
      list(dataset_ids), cache_dir,
      interactive=interactive, force=force,
      intrinsic_samples=intrinsic_samples,
    )
    if shared_intr is None:
      logger.info("pooled intrinsics unavailable — no per-dataset calibration run")
      return out
    if intrinsics_only:
      logger.info("intrinsics-only mode: skipping per-dataset extrinsic fitting")
      return out

    for did in dataset_ids:
      try:
        result = analyze_one(
          did, cache_dir, shared_intr,
          interactive=interactive, force=force,
          verify_dir_override=verify_dir_override,
        )
      except Exception as e:
        logger.error("[%s] calibration failed: %s", did, e)
        raise
      if result is not None:
        out[did] = result
  finally:
    if interactive:
      close_popup_window()
  return out
