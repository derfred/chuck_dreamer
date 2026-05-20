"""Pooled intrinsic calibration across multiple datasets, in parallel.

Algorithm overview (full spec in ``object_localization_spec.md``,
Deliverable 1, adapted for pooled multi-dataset input):

  1. Build a flat work queue of (dataset_id, frame_idx) items, interleaved
     across datasets so the early submissions cover a diverse mix of
     viewpoints.
  2. Fan the queue out across worker processes. Each worker rebuilds its
     own LeRobotDataset on first touch, then loads the frame and runs
     ``findChessboardCornersSB`` + a sub-pixel refinement.
  3. Main process accumulates results as they arrive; applies a shared
     centroid-bucket dedupe so near-duplicate viewpoints don't bloat the
     fit, and stops draining the queue once the bucket count hits target.
  4. ``cv2.calibrateCamera`` runs once on the deduped pool.
  5. The same intrinsics.json + verification artifacts are written into
     *each* input dataset's cache dir so the per-dataset loader contract
     used downstream keeps working.

The pool exists because the supplied recordings each fix the
checkerboard in a single pose for the entire episode; one episode does
not contain enough independent views to calibrate. Pooling across all
recording sessions from the same physical camera turns that into a
single well-conditioned fit.
"""
from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..dataset import (
  episode_bounds, episode_bounds_from_meta, get_frame, iter_episode_frame_indices,
)
from ..runtime import ObjectLocalizationConfig
from ..types import Intrinsics, dataset_cache_dir, write_intrinsics
from . import _worker  # noqa: F401 — registers the worker module path

logger = logging.getLogger(__name__)


@dataclass
class DatasetSpec:
  """One entry in the pooled-calibration input list.

  When ``frame_indices`` is None, the enumerator picks the candidate
  frames from the configured checkerboard episode the usual way.
  When provided, those exact global frame indices are used as the work
  queue for this dataset — no enumeration, no fallback, no LeRobotDataset
  open during the planning phase. This is the "resume from prior HITs"
  path: pass the indices that already detected last time.

  ``frame_bboxes`` (parallel to ``frame_indices``) optionally narrows
  the corner search to a per-frame ``(x0, y0, x1, y1)`` ROI. ``None``
  entries mean "search the whole frame". When set, this dramatically
  speeds up findChessboardCornersSB and bypasses background
  distractors, at the cost of relying on the box being right.
  """
  dataset_id: str
  frame_indices: list[int] | None = None
  frame_bboxes:  list[tuple[int, int, int, int] | None] | None = None

  def __post_init__(self) -> None:
    if (self.frame_indices is None) != (self.frame_bboxes is None) and self.frame_bboxes is not None:
      raise ValueError(
        f"DatasetSpec({self.dataset_id!r}): frame_bboxes requires frame_indices.")
    if (self.frame_indices is not None and self.frame_bboxes is not None
        and len(self.frame_indices) != len(self.frame_bboxes)):
      raise ValueError(
        f"DatasetSpec({self.dataset_id!r}): frame_indices ({len(self.frame_indices)}) "
        f"and frame_bboxes ({len(self.frame_bboxes)}) must have the same length.")


@dataclass
class DetectedFrame:
  dataset_id: str
  frame_idx: int            # global frame index inside that dataset
  corners: np.ndarray       # (N, 1, 2) float32 image coords
  bucket: tuple[int, int]   # centroid bucket the corners landed in


@dataclass
class PerDatasetCounts:
  attempts: int = 0
  hits: int = 0
  misses: int = 0
  dupes: int = 0


@dataclass
class PooledIntrinsicsResult:
  intrinsics: Intrinsics
  detections: list[DetectedFrame]
  per_view_rms: list[float]
  coverage_filled_cells: int
  coverage_total_cells: int
  per_dataset: dict[str, PerDatasetCounts] = field(default_factory=dict)


class IntrinsicsAbort(RuntimeError):
  """Raised when the pooled calibration can't be persisted."""


def calibrate_intrinsics_pooled(
  dataset_specs: list[DatasetSpec], ol_cfg: ObjectLocalizationConfig,
  cache_dir: Path, workers: int,
  no_gates: bool = False,
) -> PooledIntrinsicsResult:
  """Pooled + parallel checkerboard calibration.

  Submits one work item per candidate frame, drained `as_completed`. Stops
  submitting more work once we've filled enough distinct centroid buckets,
  but still drains any in-flight futures so their (rare) extra hits land.

  ``no_gates`` (set by the CLI's ``--no-gates`` flag) accepts every
  detected view: skips the centroid-bucket dedupe, skips the spatial
  coverage gate, and ignores ``intrinsics_n_frames_min``. Use when picks
  are hand-curated and any HIT should count toward the fit.
  """
  if not dataset_specs:
    raise IntrinsicsAbort("calibrate_intrinsics_pooled called with no datasets.")

  dataset_ids = [s.dataset_id for s in dataset_specs]
  ck_size = (int(ol_cfg.checkerboard_size[0]), int(ol_cfg.checkerboard_size[1]))
  ck_obj  = _checkerboard_object_points(ck_size, ol_cfg.checkerboard_square_mm)
  bucket_grid = tuple(int(x) for x in ol_cfg.intrinsics_coverage_grid)

  # Step 1 — enumerate per-dataset candidate frames (cheap, sequential).
  # Specs with an explicit frame_indices list skip the LeRobotDataset
  # open entirely — that's the fast-resume path. Each queue item is
  # (frame_idx, bbox_or_None).
  BboxOrNone = "tuple[int, int, int, int] | None"
  t0 = time.perf_counter()
  per_dataset_queues: dict[str, list[tuple[int, Any]]] = {}
  per_dataset_counts: dict[str, PerDatasetCounts] = {d: PerDatasetCounts() for d in dataset_ids}
  for spec in dataset_specs:
    if spec.frame_indices is not None:
      bboxes = spec.frame_bboxes or [None] * len(spec.frame_indices)
      per_dataset_queues[spec.dataset_id] = list(zip(spec.frame_indices, bboxes))
      n_bbox = sum(1 for b in bboxes if b is not None)
      logger.info("[pool]   %s: using %d explicit frame indices "
                  "(%d with bbox).",
                  spec.dataset_id, len(spec.frame_indices), n_bbox)
      continue

    logger.info("[pool] enumerating frames for %s (episode %d) via metadata ...",
                spec.dataset_id, ol_cfg.episode_checkerboard)
    try:
      ep_fr, ep_to = episode_bounds_from_meta(
        spec.dataset_id, ol_cfg.episode_checkerboard)
    except Exception as e:
      raise IntrinsicsAbort(
        f"could not read episode bounds for {spec.dataset_id}: {e}")
    if ep_to <= ep_fr:
      logger.warning("[pool] %s episode %d is empty; skipping.",
                     spec.dataset_id, ol_cfg.episode_checkerboard)
      per_dataset_queues[spec.dataset_id] = []
      continue
    target = ol_cfg.intrinsics_n_frames_target
    length = ep_to - ep_fr
    if length <= target:
      initial = list(range(ep_fr, ep_to))
      fallback = []
    else:
      initial = sorted({int(x) for x in np.linspace(ep_fr, ep_to - 1, num=target)})
      used = set(initial)
      fallback = [i for i in range(ep_fr, ep_to) if i not in used]
    indices = list(initial) + list(fallback)
    per_dataset_queues[spec.dataset_id] = [(i, None) for i in indices]
    logger.info("[pool]   %s: episode spans [%d, %d) length=%d  "
                "queue=%d (initial=%d + fallback=%d)",
                spec.dataset_id, ep_fr, ep_to, length,
                len(indices), len(initial), len(fallback))

  # Step 2 — interleave the per-dataset queues into a single flat queue.
  # Each item: (dataset_id, frame_idx, bbox_or_None).
  flat_queue: list[tuple[str, int, Any]] = _interleave_queues(per_dataset_queues)
  total_queue = len(flat_queue)
  if total_queue == 0:
    raise IntrinsicsAbort("no frames available across any input dataset.")

  logger.info("[pool] total work items: %d; target: %d distinct buckets; "
              "workers=%d", total_queue, ol_cfg.intrinsics_n_frames_target, workers)

  # Step 3 — fan out via ProcessPool. We use a small bounded-inflight
  # pattern so we don't submit the entire queue at once (each pending
  # future pins a frame's worth of memory once it runs).
  detections: list[DetectedFrame] = []
  occupied_buckets: set[tuple[int, int]] = set()
  image_size: tuple[int, int] | None = None
  target = int(ol_cfg.intrinsics_n_frames_target)
  inflight_cap = max(workers * 3, 8)

  # Open one append-only JSONL per dataset so the user can grep `frame_idx`
  # back after a crash and feed them in via the `id#f1,f2,...` CLI syntax.
  # Truncate on start so the file always describes only this run.
  hit_logs: dict[str, Any] = {}
  for did in dataset_ids:
    p = dataset_cache_dir(cache_dir, did) / "intrinsics_hits.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    hit_logs[did] = open(p, "w", buffering=1)  # line-buffered

  def _record_hit(det: DetectedFrame) -> None:
    f = hit_logs.get(det.dataset_id)
    if f is None:
      return
    f.write(json.dumps({
      "dataset_id": det.dataset_id,
      "frame_idx":  int(det.frame_idx),
      "bucket":     list(det.bucket),
      "elapsed_s":  round(time.perf_counter() - t0, 3),
    }) + "\n")

  log_state = {"last": time.perf_counter()}

  def _log_hit(det: DetectedFrame) -> None:
    """Always-on per-HIT log so the user sees every accepted frame."""
    totals = _totals(per_dataset_counts)
    logger.info(
      "[pool] %s f%d: HIT@%s  hits=%d misses=%d dupes=%d  buckets=%d/%d  "
      "queue_left=%d  elapsed=%.1fs",
      det.dataset_id, det.frame_idx, det.bucket,
      totals["hits"], totals["misses"], totals["dupes"],
      len(occupied_buckets), target,
      total_queue - totals["attempts"], time.perf_counter() - t0,
    )
    log_state["last"] = time.perf_counter()

  def _maybe_log(verdict: str, item: tuple[str, int, Any]) -> None:
    now = time.perf_counter()
    if (now - log_state["last"]) < 5.0:
      return
    log_state["last"] = now
    totals = _totals(per_dataset_counts)
    logger.info(
      "[pool] %s %s/%d: %s  hits=%d misses=%d dupes=%d  buckets=%d/%d  "
      "queue_left=%d  elapsed=%.1fs",
      item[0], _short_frame(item[1]), total_queue, verdict,
      totals["hits"], totals["misses"], totals["dupes"],
      len(occupied_buckets), target,
      total_queue - totals["attempts"], now - t0,
    )

  with ProcessPoolExecutor(max_workers=workers) as pool:
    queue_iter = iter(flat_queue)
    pending: dict[Any, tuple[str, int, Any]] = {}

    def _submit_more() -> None:
      while len(pending) < inflight_cap and (no_gates or len(occupied_buckets) < target):
        try:
          item = next(queue_iter)
        except StopIteration:
          return
        fut = pool.submit(_worker.detect_corners_worker,
                          item[0], item[1], ol_cfg.camera_key, ck_size,
                          item[2])
        pending[fut] = item

    _submit_more()

    while pending:
      done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
      for fut in done:
        item = pending.pop(fut)
        dataset_id, frame_idx, _bbox = item
        per_dataset_counts[dataset_id].attempts += 1
        try:
          ok, payload = fut.result()
        except Exception as e:
          per_dataset_counts[dataset_id].misses += 1
          logger.warning("[pool] %s frame %d: worker raised %s",
                         dataset_id, frame_idx, e)
          _maybe_log("ERR", item)
          continue

        if not ok:
          per_dataset_counts[dataset_id].misses += 1
          _maybe_log("MISS", item)
          continue

        corners, frame_image_size = payload
        if image_size is None:
          image_size = frame_image_size
          logger.info("[pool] first detection on %s frame %d; image_size=%dx%d",
                      dataset_id, frame_idx, image_size[0], image_size[1])
        bucket = _centroid_bucket(corners, image_size, bucket_grid)
        if not no_gates and bucket in occupied_buckets:
          per_dataset_counts[dataset_id].dupes += 1
          _maybe_log(f"DUP@{bucket}", item)
          continue

        occupied_buckets.add(bucket)
        per_dataset_counts[dataset_id].hits += 1
        det = DetectedFrame(
          dataset_id=dataset_id, frame_idx=frame_idx,
          corners=corners, bucket=bucket,
        )
        detections.append(det)
        _record_hit(det)
        _log_hit(det)

      _submit_more()

  for f in hit_logs.values():
    try:
      f.close()
    except Exception:
      pass

  totals = _totals(per_dataset_counts)
  logger.info(
    "[pool] detection complete in %.1fs: hits=%d misses=%d dupes=%d  "
    "distinct buckets=%d/%d.",
    time.perf_counter() - t0, totals["hits"], totals["misses"], totals["dupes"],
    len(occupied_buckets), target,
  )
  for did, c in per_dataset_counts.items():
    logger.info("[pool]   %s: attempts=%d hits=%d misses=%d dupes=%d",
                did, c.attempts, c.hits, c.misses, c.dupes)

  if image_size is None:
    raise IntrinsicsAbort("no checkerboard detected in any frame from any dataset.")

  if not no_gates and len(detections) < ol_cfg.intrinsics_n_frames_min:
    _write_failure_coverage(detections, ol_cfg, cache_dir, dataset_ids, image_size)
    raise IntrinsicsAbort(
      f"only {len(detections)} usable detections across all "
      f"{len(dataset_ids)} datasets (need >= {ol_cfg.intrinsics_n_frames_min}). "
      f"Re-record episode {ol_cfg.episode_checkerboard} with the board "
      f"swept through the full field of view, or re-run with --no-gates "
      f"to accept the small sample.")

  filled, total = _coverage_fill(detections, image_size, bucket_grid)
  logger.info("[pool] spatial coverage: %d / %d cells (%.0f%%).",
              filled, total, 100.0 * filled / max(1, total))
  if not no_gates and filled / max(1, total) < 0.8:
    _write_failure_coverage(detections, ol_cfg, cache_dir, dataset_ids, image_size)
    raise IntrinsicsAbort(
      f"checkerboard covers only {filled}/{total} grid cells (<80%). "
      f"Re-record episode {ol_cfg.episode_checkerboard} with the board "
      f"swept through the full field of view, or re-run with --no-gates "
      f"to accept the partial coverage.")
  if no_gates:
    logger.info("[pool] --no-gates: skipping bucket dedupe + coverage/min-frame gates; "
                "using every detection.")

  import cv2
  obj_points = [ck_obj.copy() for _ in detections]
  img_points = [d.corners for d in detections]

  calib_flags = _distortion_flags(cv2, ol_cfg.intrinsics_distortion_model)
  if ol_cfg.intrinsics_fix_aspect_ratio:
    calib_flags |= cv2.CALIB_FIX_ASPECT_RATIO
  if ol_cfg.intrinsics_fix_principal_point:
    calib_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
  logger.info("[pool] running cv2.calibrateCamera on %d views "
              "(distortion_model=%d coefficients, fix_aspect_ratio=%s, "
              "fix_principal_point=%s)...",
              len(detections), ol_cfg.intrinsics_distortion_model,
              ol_cfg.intrinsics_fix_aspect_ratio,
              ol_cfg.intrinsics_fix_principal_point)
  t_cal = time.perf_counter()
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
  K_init: np.ndarray | None = None
  dist_init: np.ndarray | None = None
  if (ol_cfg.intrinsics_fix_aspect_ratio
      or ol_cfg.intrinsics_fix_principal_point):
    # CALIB_FIX_* requires K to be initialized with the desired values
    # (OpenCV holds whatever's in K at the corresponding positions).
    # Seed with a sane guess: fx=fy=image_width (~50° FOV) and the
    # principal point at the image center.
    W, H = image_size
    K_init = np.array([
      [float(W), 0.0,      W / 2.0],
      [0.0,      float(W), H / 2.0],
      [0.0,      0.0,      1.0],
    ], dtype=np.float64)
    dist_init = np.zeros(5, dtype=np.float64)
    calib_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
  rms_px, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, image_size, K_init, dist_init,
    flags=calib_flags, criteria=criteria,
  )
  logger.info("[pool]   calibrateCamera done in %.1fs (rms=%.3fpx).",
              time.perf_counter() - t_cal, float(rms_px))
  K    = np.asarray(K, dtype=np.float64)
  dist = np.asarray(dist, dtype=np.float64).reshape(-1)
  if dist.size < 5:
    dist = np.concatenate([dist, np.zeros(5 - dist.size, dtype=np.float64)])
  else:
    dist = dist[:5]

  per_view_rms = _per_view_rms(obj_points, img_points, rvecs, tvecs, K, dist)
  intrinsics = Intrinsics(
    image_size   = image_size,
    K            = K,
    dist         = dist,
    rms_px       = float(rms_px),
    n_frames_used = len(detections),
  )

  # Step 4 — write the same intrinsics into each dataset's cache dir.
  pool_meta = {
    "source_datasets": list(dataset_ids),
    "per_dataset_view_counts": {
      did: sum(1 for d in detections if d.dataset_id == did)
      for did in dataset_ids
    },
    "per_view_rms_px":  [float(x) for x in per_view_rms],
    "rms_threshold_px": float(ol_cfg.intrinsics_rms_threshold_px),
    "view_provenance":  [
      {"dataset_id": d.dataset_id, "frame_idx": d.frame_idx,
       "bucket": list(d.bucket)} for d in detections
    ],
  }
  for did in dataset_ids:
    write_intrinsics(cache_dir, did, intrinsics, extra=pool_meta)
    logger.info("[pool]   wrote intrinsics.json for %s",
                dataset_cache_dir(cache_dir, did))

  # Step 5 — render verification artifacts. The coverage mosaic uses the
  # first dataset's empty-scene frame as backdrop; per-frame overlays
  # come from whichever dataset that view originated in.
  for did in dataset_ids:
    verify_dir = dataset_cache_dir(cache_dir, did) / "intrinsics_verify"
    _render_pooled_verification(
      detections, ol_cfg, verify_dir, image_size,
      K=K, dist=dist, rvecs=rvecs, tvecs=tvecs,
      per_view_rms=per_view_rms, ck_obj=ck_obj, backdrop_dataset=did,
    )

  return PooledIntrinsicsResult(
    intrinsics            = intrinsics,
    detections            = detections,
    per_view_rms          = list(per_view_rms),
    coverage_filled_cells = filled,
    coverage_total_cells  = total,
    per_dataset           = per_dataset_counts,
  )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interleave_queues(per_dataset: dict[str, list[tuple[int, Any]]]
                       ) -> list[tuple[str, int, Any]]:
  """Round-robin interleave per-dataset (frame, bbox) lists into a flat queue.

  Datasets with longer queues end up tailing the output, but the first
  ``min(N) * D`` items hit every dataset before any one is exhausted —
  important because the early hits are what populate the bucket set and
  trigger the early-stop.
  """
  out: list[tuple[str, int, Any]] = []
  iters = {k: iter(v) for k, v in per_dataset.items()}
  while iters:
    drained: list[str] = []
    for k, it in iters.items():
      try:
        frame_idx, bbox = next(it)
        out.append((k, frame_idx, bbox))
      except StopIteration:
        drained.append(k)
    for k in drained:
      del iters[k]
  return out


def _totals(per_dataset: dict[str, PerDatasetCounts]) -> dict[str, int]:
  return {
    "attempts": sum(c.attempts for c in per_dataset.values()),
    "hits":     sum(c.hits     for c in per_dataset.values()),
    "misses":   sum(c.misses   for c in per_dataset.values()),
    "dupes":    sum(c.dupes    for c in per_dataset.values()),
  }


def _short_frame(idx: int) -> str:
  return f"f{idx}"


def _distortion_flags(cv2_mod: Any, n_coeffs: int) -> int:
  """OpenCV calibrateCamera flag mask for the requested distortion model.

  - 0: pinhole (everything zero).
  - 2: k1, k2 only; tangential + k3 zeroed.
  - 5: default Brown-Conrady (k1, k2, p1, p2, k3).
  """
  if n_coeffs == 5:
    return 0
  if n_coeffs == 2:
    return cv2_mod.CALIB_ZERO_TANGENT_DIST | cv2_mod.CALIB_FIX_K3
  if n_coeffs == 0:
    return (cv2_mod.CALIB_ZERO_TANGENT_DIST
            | cv2_mod.CALIB_FIX_K1 | cv2_mod.CALIB_FIX_K2
            | cv2_mod.CALIB_FIX_K3)
  raise ValueError(f"unsupported distortion coefficient count: {n_coeffs}")


def _checkerboard_object_points(size: tuple[int, int], square_mm: float) -> np.ndarray:
  cols, rows = size
  objp = np.zeros((cols * rows, 3), dtype=np.float32)
  grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
  objp[:, :2] = grid * float(square_mm)
  return objp


def _centroid_bucket(corners: np.ndarray, image_size: tuple[int, int],
                     grid: tuple[int, int]) -> tuple[int, int]:
  """Quantize the checkerboard centroid into a ``grid``-cell bucket."""
  pts = corners.reshape(-1, 2)
  cu = float(pts[:, 0].mean())
  cv = float(pts[:, 1].mean())
  W, H = image_size
  gx, gy = max(1, grid[0]), max(1, grid[1])
  bx = min(gx - 1, max(0, int(cu / (W / gx))))
  by = min(gy - 1, max(0, int(cv / (H / gy))))
  return (bx, by)


def _coverage_fill(detections: list[DetectedFrame], image_size: tuple[int, int],
                   grid: tuple[int, int]) -> tuple[int, int]:
  W, H = image_size
  gx, gy = max(1, grid[0]), max(1, grid[1])
  cell_w, cell_h = W / gx, H / gy
  filled = np.zeros((gy, gx), dtype=bool)
  for d in detections:
    pts = d.corners.reshape(-1, 2)
    cx = np.clip((pts[:, 0] / cell_w).astype(int), 0, gx - 1)
    cy = np.clip((pts[:, 1] / cell_h).astype(int), 0, gy - 1)
    filled[cy, cx] = True
  return int(filled.sum()), int(gx * gy)


def _per_view_rms(obj_points: list[np.ndarray], img_points: list[np.ndarray],
                  rvecs: list[np.ndarray], tvecs: list[np.ndarray],
                  K: np.ndarray, dist: np.ndarray) -> list[float]:
  import cv2
  out: list[float] = []
  for op, ip, rv, tv in zip(obj_points, img_points, rvecs, tvecs):
    proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
    diff = (proj.reshape(-1, 2) - ip.reshape(-1, 2))
    rms = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    out.append(rms)
  return out


def _backdrop_frame(dataset_id: str, ol_cfg: ObjectLocalizationConfig) -> np.ndarray | None:
  """Pull the first frame of the empty episode from ``dataset_id``."""
  try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    ds = LeRobotDataset(dataset_id)
    fr, _ = episode_bounds(ds, ol_cfg.episode_empty)
    return get_frame(ds, fr, ol_cfg.camera_key)
  except Exception as e:
    logger.warning("backdrop frame from %s unavailable (%s)", dataset_id, e)
    return None


def _render_pooled_verification(
  detections: list[DetectedFrame], ol_cfg: ObjectLocalizationConfig,
  out_dir: Path, image_size: tuple[int, int],
  K: np.ndarray, dist: np.ndarray, rvecs: list[np.ndarray], tvecs: list[np.ndarray],
  per_view_rms: list[float], ck_obj: np.ndarray, backdrop_dataset: str,
) -> None:
  import cv2
  out_dir.mkdir(parents=True, exist_ok=True)
  _render_coverage_mosaic(detections, ol_cfg, out_dir, image_size, backdrop_dataset)

  per_frame_dir = out_dir / "per_frame"
  per_frame_dir.mkdir(parents=True, exist_ok=True)

  if not detections:
    return

  # Render up to 10 sample frames spaced across the deduped pool. We only
  # render frames from the same dataset we wrote the mosaic backdrop for,
  # so each cache dir stays self-contained even though the calibration
  # was pooled.
  same_ds = [(i, d) for i, d in enumerate(detections) if d.dataset_id == backdrop_dataset]
  if not same_ds:
    return
  picks = np.linspace(0, len(same_ds) - 1, num=min(10, len(same_ds)), dtype=int)

  try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    ds = LeRobotDataset(backdrop_dataset)
  except Exception as e:
    logger.warning("[%s] per-frame render unavailable (%s)", backdrop_dataset, e)
    return

  for k in picks:
    idx, det = same_ds[int(k)]
    try:
      frame = get_frame(ds, det.frame_idx, ol_cfg.camera_key)
    except Exception as e:
      logger.warning("[%s] frame %d render skipped (%s)",
                     backdrop_dataset, det.frame_idx, e)
      continue
    canvas = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
    proj, _ = cv2.projectPoints(ck_obj, rvecs[int(idx)], tvecs[int(idx)], K, dist)
    obs = det.corners.reshape(-1, 2)
    rep = proj.reshape(-1, 2)
    for o in obs:
      cv2.circle(canvas, (int(o[0]), int(o[1])), 4, (0, 200, 0), -1)
    for r in rep:
      cv2.drawMarker(canvas, (int(r[0]), int(r[1])), (0, 0, 255),
                     cv2.MARKER_CROSS, 8, 1)
    text = f"frame={det.frame_idx}  rms={per_view_rms[int(idx)]:.3f}px"
    cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(per_frame_dir / f"{det.frame_idx:05d}.png"), canvas)


def _render_coverage_mosaic(detections: list[DetectedFrame],
                             ol_cfg: ObjectLocalizationConfig,
                             out_dir: Path, image_size: tuple[int, int],
                             backdrop_dataset: str) -> None:
  import cv2
  out_dir.mkdir(parents=True, exist_ok=True)
  W, H = image_size
  base = _backdrop_frame(backdrop_dataset, ol_cfg)
  if base is None:
    base = np.full((H, W, 3), 32, dtype=np.uint8)
  canvas = cv2.cvtColor(base.copy(), cv2.COLOR_RGB2BGR)

  gx, gy = ol_cfg.intrinsics_coverage_grid
  cell_w, cell_h = W / gx, H / gy
  filled = np.zeros((gy, gx), dtype=bool)
  for d in detections:
    pts = d.corners.reshape(-1, 2)
    cx = np.clip((pts[:, 0] / cell_w).astype(int), 0, gx - 1)
    cy = np.clip((pts[:, 1] / cell_h).astype(int), 0, gy - 1)
    filled[cy, cx] = True

  overlay = canvas.copy()
  for j in range(gy):
    for i in range(gx):
      x0, y0 = int(i * cell_w), int(j * cell_h)
      x1, y1 = int((i + 1) * cell_w), int((j + 1) * cell_h)
      if not filled[j, i]:
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 200), -1)
  canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)

  for j in range(gy + 1):
    y = int(j * cell_h)
    cv2.line(canvas, (0, y), (W, y), (140, 140, 140), 1)
  for i in range(gx + 1):
    x = int(i * cell_w)
    cv2.line(canvas, (x, 0), (x, H), (140, 140, 140), 1)

  for k, d in enumerate(detections):
    hue = int(180 * (k / max(1, len(detections))))
    color_bgr = tuple(int(v) for v in cv2.cvtColor(
      np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
    for u, v in d.corners.reshape(-1, 2):
      cv2.circle(canvas, (int(u), int(v)), 2, color_bgr, -1)

  cv2.imwrite(str(out_dir / "coverage_mosaic.png"), canvas)


def _write_failure_coverage(detections: list[DetectedFrame],
                             ol_cfg: ObjectLocalizationConfig,
                             cache_dir: Path, dataset_ids: list[str],
                             image_size: tuple[int, int]) -> None:
  """On abort, dump the coverage mosaic into every input cache dir so
  the user can see *why* it failed.
  """
  for did in dataset_ids:
    out_dir = dataset_cache_dir(cache_dir, did) / "intrinsics_verify"
    _render_coverage_mosaic(detections, ol_cfg, out_dir, image_size, did)
