"""Build and cache the scene background model from the empty-mat episode.

The mat is static across episodes within one recording session. Episode 0
is the empty mat (no object), so the per-pixel median + std across a
sample of those frames is a tight reference for "what the scene looks
like without the object". Subtracting this from a frame from episodes
2/3/4 isolates the object far more reliably than any color heuristic.

The image-space model itself (:class:`SceneBackground`, the ROI helpers)
lives in ``chuck_dreamer.perception.scene_bg``; this module owns the
LeRobot sampling and the ``<cache>/<slug>/scene_bg.npz`` cache (paid once
per dataset, ~10 s to rebuild).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from chuck_dreamer.perception.scene_bg import SceneBackground, build_mat_mask
from chuck_dreamer.store import dataset_cache_dir

from .dataset import episode_bounds_from_meta, get_frame

logger = logging.getLogger(__name__)


def build_scene_bg(
  dataset_id: str, camera_key: str, episode_idx: int,
  n_samples: int = 16,
) -> SceneBackground:
  """Sample ``n_samples`` frames evenly across ``episode_idx`` and return
  the per-pixel median + std. Builds its own ``LeRobotDataset`` (slow on
  first call due to PyAV index).
  """
  from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

  t0 = time.perf_counter()
  fr, to = episode_bounds_from_meta(dataset_id, episode_idx)
  if to <= fr:
    raise ValueError(f"episode {episode_idx} of {dataset_id} is empty.")
  length = to - fr
  n = min(n_samples, length)
  indices = sorted({int(x) for x in np.linspace(fr, to - 1, num=n)})
  logger.info("[bg] %s: sampling %d frames from episode %d [%d, %d) ...",
              dataset_id, len(indices), episode_idx, fr, to)

  ds = LeRobotDataset(dataset_id)
  stack: list[np.ndarray] = []
  for i, idx in enumerate(indices):
    t = time.perf_counter()
    frame = get_frame(ds, idx, camera_key)
    stack.append(frame)
    logger.info("[bg]   frame %d/%d (idx=%d) in %.2fs",
                i + 1, len(indices), idx, time.perf_counter() - t)

  arr = np.stack(stack, axis=0)
  median = np.median(arr, axis=0).astype(np.uint8)
  std    = arr.astype(np.float32).std(axis=0)
  logger.info("[bg] %s: scene bg built in %.1fs (%d frames, shape=%s)",
              dataset_id, time.perf_counter() - t0, len(stack), median.shape)
  return SceneBackground(median=median, std=std, n_frames=len(stack))


def scene_bg_path(cache_dir: Path | str, dataset_id: str) -> Path:
  return dataset_cache_dir(cache_dir, dataset_id) / "scene_bg.npz"


def save_scene_bg(cache_dir: Path | str, dataset_id: str,
                  bg: SceneBackground) -> Path:
  p = scene_bg_path(cache_dir, dataset_id)
  p.parent.mkdir(parents=True, exist_ok=True)
  payload: dict[str, np.ndarray] = {
    "median":   bg.median,
    "std":      bg.std,
    "n_frames": np.asarray(bg.n_frames),
  }
  if bg.mat_mask is not None:
    payload["mat_mask"] = bg.mat_mask.astype(bool)
  # numpy stub models savez_compressed's **kwds poorly (collides with allow_pickle: bool).
  np.savez_compressed(p, **payload)  # type: ignore[arg-type]
  return p


def load_scene_bg(cache_dir: Path | str, dataset_id: str
                  ) -> SceneBackground | None:
  p = scene_bg_path(cache_dir, dataset_id)
  if not p.exists():
    return None
  with np.load(p) as f:
    mat_mask = np.asarray(f["mat_mask"], dtype=bool) if "mat_mask" in f else None
    return SceneBackground(
      median   = np.asarray(f["median"], dtype=np.uint8),
      std      = np.asarray(f["std"], dtype=np.float32),
      n_frames = int(f["n_frames"]),
      mat_mask = mat_mask,
    )


def ensure_scene_bg(
  cache_dir: Path | str, dataset_id: str, camera_key: str,
  episode_idx: int, n_samples: int = 16, rebuild: bool = False,
  calibration: "Any | None" = None,
  ol_cfg: "Any | None" = None,
  rebuild_mat_mask: bool = False,
) -> SceneBackground:
  """Load cached scene_bg, or build + save it. Idempotent.

  When ``calibration`` and ``ol_cfg`` are provided, attach a mat-region
  mask projected from the configured mat_extent_* bounds. Pass
  ``rebuild_mat_mask=True`` to force recomputation of an existing
  cached mat_mask (use this once after changing the extent values).
  """
  bg: SceneBackground | None = None
  if not rebuild:
    bg = load_scene_bg(cache_dir, dataset_id)
  if bg is None:
    bg = build_scene_bg(dataset_id, camera_key, episode_idx, n_samples)
  needs_mat = (bg.mat_mask is None) or rebuild_mat_mask
  if needs_mat and calibration is not None and ol_cfg is not None:
    try:
      mat_mask = build_mat_mask(
        image_size = bg.median.shape[1::-1],   # (W, H) from (H, W, 3)
        calibration = calibration,
        extent_xmin_mm = ol_cfg.mat_extent_xmin_mm,
        extent_xmax_mm = ol_cfg.mat_extent_xmax_mm,
        extent_ymin_mm = ol_cfg.mat_extent_ymin_mm,
        extent_ymax_mm = ol_cfg.mat_extent_ymax_mm,
      )
      bg.mat_mask = mat_mask
      logger.info("[bg] %s: attached mat_mask from world bbox "
                  "x=[%.0f, %.0f] y=[%.0f, %.0f] mm "
                  "(%d/%d pixels in ROI = %.1f%%)",
                  dataset_id,
                  ol_cfg.mat_extent_xmin_mm, ol_cfg.mat_extent_xmax_mm,
                  ol_cfg.mat_extent_ymin_mm, ol_cfg.mat_extent_ymax_mm,
                  int(mat_mask.sum()), mat_mask.size,
                  100.0 * mat_mask.sum() / mat_mask.size)
    except Exception as e:
      logger.warning("[bg] %s: could not build mat_mask (%s); proceeding "
                     "without ROI.", dataset_id, e)
  save_scene_bg(cache_dir, dataset_id, bg)
  return bg
