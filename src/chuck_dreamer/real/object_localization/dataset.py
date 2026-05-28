"""LeRobotDataset access helpers.

Frames are stored as ``CHW float32`` tensors in LeRobot's pipeline; the
rest of the calibration code wants ``HWC uint8`` numpy arrays. ``_frame``
handles the conversion in one place so callers never have to.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def episode_bounds(ds: Any, episode_idx: int) -> tuple[int, int]:
  """Return ``(from_idx, to_idx_exclusive)`` for ``episode_idx`` in ``ds``.

  Works against both the new and old LeRobotDataset APIs by sniffing
  for ``episode_data_index``; falls back to a per-frame scan otherwise.
  """
  # New API: ds.meta.episodes is a HF Dataset with dataset_from/to_index.
  # That table is already cached on disk; reading it touches no video.
  try:
    row = ds.meta.episodes[episode_idx]
    return int(row["dataset_from_index"]), int(row["dataset_to_index"])
  except (AttributeError, KeyError, IndexError, TypeError):
    pass
  try:
    idx_table = ds.episode_data_index
    fr = int(idx_table["from"][episode_idx])
    to = int(idx_table["to"][episode_idx])
    return fr, to
  except (AttributeError, KeyError, IndexError, TypeError):
    pass
  starts: list[int] = []
  current = -1
  for i in range(len(ds)):
    ep = int(ds[i]["episode_index"])
    if ep != current:
      if current != -1 and ep < current:
        break
      starts.append(i)
      current = ep
  starts.append(len(ds))
  return starts[episode_idx], starts[episode_idx + 1]


def episode_bounds_from_meta(dataset_id: str, episode_idx: int) -> tuple[int, int]:
  """``(from_idx, to_idx_exclusive)`` for one episode, metadata only.

  Derived from :meth:`chuck_dreamer.common.episode_spec.EpisodeSpec.read_episodes`,
  which reads only the parquet sidecars under ``meta/`` — no video touch.
  """
  for ep, bounds in all_episode_bounds_from_meta(dataset_id):
    if ep == episode_idx:
      return bounds
  raise KeyError(f"{dataset_id}: episode {episode_idx} not in metadata")


def all_episode_bounds_from_meta(dataset_id: str
                                 ) -> list[tuple[int, tuple[int, int]]]:
  """Return ``[(episode_idx, (from, to)), ...]`` for every episode in the
  dataset, sourced from cached parquet metadata only.
  """
  from chuck_dreamer.common.episode_spec import EpisodeSpec

  slices, _ = EpisodeSpec(dataset_id=dataset_id).read_episodes()
  return [(s.episode_index, s.bounds) for s in slices]


def get_frame(ds: Any, frame_idx: int, camera_key: str) -> np.ndarray:
  """Pull frame ``frame_idx`` and return it as ``HxWx3 uint8`` RGB."""
  sample = ds[frame_idx]
  img = sample[camera_key]
  return _to_uint8_hwc(img)


def sample_episode_frames(
  ds: Any, episode_idx: int, camera_key: str, n: int,
  start: int | None = None, stop: int | None = None,
) -> list[np.ndarray]:
  """Evenly sample ``n`` frames from one episode.

  Returns whatever frames exist (up to ``n``); callers handle the
  empty-episode case. If ``start``/``stop`` are given they override
  the lookup against ``episode_data_index``.
  """
  if start is None or stop is None:
    fr, to = episode_bounds(ds, episode_idx)
    start = start if start is not None else fr
    stop  = stop  if stop  is not None else to
  if stop <= start:
    return []
  length = stop - start
  if length <= n:
    indices = list(range(start, stop))
  else:
    indices = np.linspace(start, stop - 1, num=n, dtype=int).tolist()
  return [get_frame(ds, i, camera_key) for i in indices]


def iter_episode_frame_indices(
  ds: Any, episode_idx: int, target: int,
) -> tuple[list[int], list[int]]:
  """Return ``(initial_indices, fallback_indices)`` for an episode.

  ``initial_indices`` are the evenly-spaced sample, ``fallback_indices``
  is the complementary set ordered by distance from the initial picks,
  used by the intrinsics calibrator to backfill failed detections.
  """
  fr, to = episode_bounds(ds, episode_idx)
  length = to - fr
  if length <= 0:
    return [], []
  if length <= target:
    return list(range(fr, to)), []
  initial = sorted(int(x) for x in np.linspace(fr, to - 1, num=target))
  used = set(initial)
  fallback = [i for i in range(fr, to) if i not in used]
  return initial, fallback


def _to_uint8_hwc(img: Any) -> np.ndarray:
  """Coerce a LeRobotDataset image into ``HxWx3 uint8`` (RGB)."""
  try:
    import torch
  except Exception:  # pragma: no cover - torch is a hard dep but be defensive
    torch = None
  if torch is not None and isinstance(img, torch.Tensor):
    arr = img.detach().cpu().numpy()
  else:
    arr = np.asarray(img)
  if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
    arr = np.transpose(arr, (1, 2, 0))
  if arr.dtype != np.uint8:
    arr = np.clip(arr.astype(np.float32), 0.0, 1.0) * 255.0
    arr = arr.astype(np.uint8)
  if arr.ndim == 2:
    arr = np.stack([arr, arr, arr], axis=-1)
  if arr.shape[-1] == 1:
    arr = np.repeat(arr, 3, axis=-1)
  return arr
