"""Reconstruction / reward / baseline metrics for eval rollouts.

All metrics live here so the online evaluator, the deep-eval runner, and
the notebooks compute identical numbers. Every per-step metric returns a
shape-``(T,)`` numpy array indexed by the post-burn-in window unless
documented otherwise.

The recon metric is L2 norm of the per-step difference, in the same
``[-0.5, 0.5]`` float space the encoder consumes. For dict observations
(``image_proprio``) we sum the image L2 and the proprio L2 — they're on
different scales but the sum is what training optimizes against.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_float32(x: Any) -> np.ndarray:
  return np.asarray(x, dtype=np.float32)


def recon_l2(pred: Any, target: Any) -> np.ndarray:
  """Per-step L2 norm of ``pred - target``, summed across feature axes.

  Both ``pred`` (decoder output) and ``target`` (processor output) must
  share leading time axis ``T`` and live in the same float space.
  Image obs are expected to be already normalized to ``[-0.5, 0.5]`` —
  use :class:`~chuck_dreamer.training.observation.Observation.normalize_image`
  if you're feeding raw uint8 frames.

  For dict observations the per-step L2 of every entry is summed.
  """
  if isinstance(pred, dict):
    if not isinstance(target, dict):
      raise TypeError(f"target must be a dict when pred is a dict; got {type(target).__name__}")
    total = None
    for key in pred:
      part = recon_l2(pred[key], target[key])
      total = part if total is None else total + part
    assert total is not None
    return total

  p = _as_float32(pred)
  t = _as_float32(target)
  diff = (p - t).reshape(p.shape[0], -1)
  return np.asarray(np.linalg.norm(diff, axis=-1).astype(np.float32))


def reward_abs_error(pred: Any, target: Any) -> np.ndarray:
  """``|pred - target|`` per step. Returns shape ``(T,)``."""
  p = _as_float32(pred).reshape(-1)
  t = _as_float32(target).reshape(-1)
  return np.asarray(np.abs(p - t).astype(np.float32))


def zero_dynamics_prediction(target: Any, anchor_idx: int) -> Any:
  """Tile ``target[anchor_idx]`` across the leading time axis.

  Produces the "predict obs stays constant" baseline. Returns the same
  shape and type (ndarray or dict-of-ndarrays) as ``target``.
  """
  if isinstance(target, dict):
    return {k: zero_dynamics_prediction(v, anchor_idx) for k, v in target.items()}
  arr = np.asarray(target)
  if arr.shape[0] == 0:
    return arr
  idx = min(max(anchor_idx, 0), arr.shape[0] - 1)
  last = arr[idx]
  T = arr.shape[0]
  return np.tile(last[None], (T, *([1] * last.ndim)))


def stack_curves(curves: list[np.ndarray]) -> np.ndarray:
  """Stack per-episode error curves into ``(N, T_max)`` with NaN padding.

  Empty input → empty ``(0, 0)`` array. Lets callers ``np.nanmean`` over
  variable-length episodes without a manual pad loop.
  """
  if not curves:
    return np.empty((0, 0), dtype=np.float32)
  T_max = max(c.shape[0] for c in curves)
  out = np.full((len(curves), T_max), np.nan, dtype=np.float32)
  for i, c in enumerate(curves):
    out[i, : c.shape[0]] = c
  return out
