"""Per-pixel background model built from the empty-mat episode.

The mat is static across episodes within one recording session. Episode 0
is the empty mat (no object), so the per-pixel median + std across a
sample of those frames is a tight reference for "what the scene looks
like without the object". Subtracting this from a frame from episodes
2/3/4 isolates the object far more reliably than any color heuristic.

Cached at ``<cache>/<slug>/scene_bg.npz`` so the ~10s sampling cost is
paid once per dataset.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import episode_bounds_from_meta, get_frame
from .types import dataset_cache_dir

logger = logging.getLogger(__name__)


@dataclass
class SceneBackground:
  """Per-pixel reference scene built from the empty episode."""
  median: np.ndarray   # (H, W, 3) uint8 — per-pixel median across samples
  std:    np.ndarray   # (H, W, 3) float32 — per-pixel std across samples
  n_frames: int

  def foreground_mask(self, rgb: np.ndarray,
                       k: float = 3.0, floor: float = 15.0,
                       open_radius: int = 1) -> np.ndarray:
    """Binary mask where ``rgb`` differs from the reference.

    ``threshold = max(k*std, floor)`` per pixel + channel. A pixel is
    foreground iff the max-channel difference exceeds the threshold.
    A small morphological open removes 1-2 px speckle from sensor noise.
    """
    import cv2
    diff = np.abs(rgb.astype(np.float32) - self.median.astype(np.float32))
    thr  = np.maximum(self.std * k, floor)
    fg_per_channel = diff > thr
    fg = fg_per_channel.max(axis=-1)
    if open_radius > 0:
      m = fg.astype(np.uint8)
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (2 * open_radius + 1, 2 * open_radius + 1))
      m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
      m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
      fg = m.astype(bool)
    return fg


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
  np.savez_compressed(p, median=bg.median, std=bg.std, n_frames=bg.n_frames)
  return p


def load_scene_bg(cache_dir: Path | str, dataset_id: str
                  ) -> SceneBackground | None:
  p = scene_bg_path(cache_dir, dataset_id)
  if not p.exists():
    return None
  with np.load(p) as f:
    return SceneBackground(
      median   = np.asarray(f["median"], dtype=np.uint8),
      std      = np.asarray(f["std"], dtype=np.float32),
      n_frames = int(f["n_frames"]),
    )


def ensure_scene_bg(
  cache_dir: Path | str, dataset_id: str, camera_key: str,
  episode_idx: int, n_samples: int = 16, rebuild: bool = False,
) -> SceneBackground:
  """Load cached scene_bg, or build + save it. Idempotent."""
  if not rebuild:
    cached = load_scene_bg(cache_dir, dataset_id)
    if cached is not None:
      return cached
  bg = build_scene_bg(dataset_id, camera_key, episode_idx, n_samples)
  save_scene_bg(cache_dir, dataset_id, bg)
  return bg
