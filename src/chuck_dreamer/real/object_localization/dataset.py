"""LeRobotDataset loaders + per-episode frame samplers."""

from __future__ import annotations

from typing import Iterator

import numpy as np


def _frame_to_uint8_rgb(t) -> np.ndarray:
  """Convert ``ds[i][camera_key]`` to an HxWx3 uint8 RGB numpy array.

  Accepts a torch tensor in CHW float [0, 1] (the LeRobot default) or
  any array-like already in HWC uint8.
  """
  if hasattr(t, "permute") and hasattr(t, "numpy"):
    if t.ndim == 3 and t.shape[0] in (1, 3):
      t = t.permute(1, 2, 0)
    arr = t.detach().cpu().numpy()
  else:
    arr = np.asarray(t)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
      arr = np.transpose(arr, (1, 2, 0))

  if arr.dtype != np.uint8:
    a = np.asarray(arr, dtype=np.float32)
    if a.max() <= 1.5:
      a = a * 255.0
    arr = np.clip(a, 0.0, 255.0).astype(np.uint8)

  if arr.ndim == 3 and arr.shape[2] == 1:
    arr = np.repeat(arr, 3, axis=2)
  return np.asarray(arr)


def episode_frame_indices(ds, episode_idx: int) -> range:
  """Return the global frame indices for ``episode_idx`` in ``ds``.

  Reads ``meta.episodes`` (LeRobot v3) for the slice. Falls back to
  ``ds.episode_data_index`` if the v2-style helper is present.
  """
  try:
    ep = ds.meta.episodes[int(episode_idx)]
    if isinstance(ep, dict):
      lo = int(ep["dataset_from_index"])
      hi = int(ep["dataset_to_index"])
    else:  # HF Dataset row
      lo = int(ep["dataset_from_index"])
      hi = int(ep["dataset_to_index"])
    return range(lo, hi)
  except Exception:
    pass

  if hasattr(ds, "episode_data_index"):
    idx = ds.episode_data_index
    lo  = int(idx["from"][episode_idx])
    hi  = int(idx["to"][episode_idx])
    return range(lo, hi)
  raise RuntimeError(f"cannot resolve frame range for episode {episode_idx}")


def iter_episode_frames(ds, episode_idx: int, camera_key: str) -> Iterator[np.ndarray]:
  """Yield uint8 RGB frames for the requested episode."""
  for i in episode_frame_indices(ds, episode_idx):
    yield _frame_to_uint8_rgb(ds[i][camera_key])


def sample_episode_frames(
  ds, episode_idx: int, camera_key: str, n: int,
) -> list[np.ndarray]:
  """Return up to ``n`` evenly spaced frames from the episode."""
  rng = episode_frame_indices(ds, episode_idx)
  if len(rng) == 0:
    return []
  if n >= len(rng):
    picks = list(rng)
  else:
    picks = np.linspace(rng.start, rng.stop - 1, n, dtype=int).tolist()
  return [_frame_to_uint8_rgb(ds[i][camera_key]) for i in picks]
