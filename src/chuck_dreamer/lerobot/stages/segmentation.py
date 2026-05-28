"""SAM2 segmentation stage family, parameterised by target.

One :class:`SegmentationStage` instance segments one target (``"object"``
today; ``"arm"`` etc. in future). Masks are cached on the
:class:`StageContext` so downstream stages (e.g. object pose) can consume
them; the per-target seg/uv arrays are written onto the episode.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .base import Requirement, StageContext, as_uint8_hwc

# Key names each target writes onto the episode. The object target keeps
# its historical names so the writers + training code are untouched;
# future targets (e.g. "arm") get target-suffixed keys.
_SEG_KEYS: dict[str, tuple[str, str]] = {
  "object": ("segmentation_target", "object_uv"),
}


def seg_keys(target_name: str) -> tuple[str, str]:
  return _SEG_KEYS.get(
    target_name, (f"segmentation_{target_name}", f"{target_name}_uv"))


def _store_masks_and_uv(episode: dict[str, Any],
                        imgs: list[np.ndarray],
                        masks: list[np.ndarray | None],
                        *,
                        seg_key: str = "segmentation_target",
                        uv_key: str = "object_uv") -> None:
  """Fill a ``(T, H, W)`` uint8 seg array and ``(T, 2)`` float32 centroids
  on the episode dict from per-frame SAM2 masks, under ``seg_key`` /
  ``uv_key``.

  Pixels carry label 1 inside the target mask, 0 elsewhere. For frames
  where SAM2 returned no mask we emit a zero mask and ``NaN`` centroids
  so downstream tooling can tell drop-outs apart from "mask centred at
  origin".
  """
  T = len(imgs)
  if T == 0:
    return
  ref = next((np.asarray(m) for m in masks if m is not None), None)
  if ref is None:
    return
  H, W = ref.shape[:2]
  seg_arr = np.zeros((T, H, W), dtype=np.uint8)
  uv_arr  = np.full((T, 2), np.nan, dtype=np.float32)
  for t, m in enumerate(masks):
    if m is None:
      continue
    m_arr = np.asarray(m, dtype=bool)
    if m_arr.shape[:2] != (H, W):
      continue
    seg_arr[t][m_arr] = 1
    ys, xs = np.where(m_arr)
    if xs.size:
      uv_arr[t, 0] = float(xs.mean())
      uv_arr[t, 1] = float(ys.mean())
  episode[seg_key] = seg_arr
  episode[uv_key]  = uv_arr


def _dump_debug_masks(out_root: Path, source_repo: str, episode_index: int,
                      imgs: list[np.ndarray],
                      masks: list[np.ndarray | None] | None) -> None:
  """Dump per-frame mask overlays for visual inspection.

  Triggered by setting ``IMPORT_LEROBOT_DEBUG_MASKS=<dir>``. Writes one
  PNG per frame to ``<dir>/<dataset_slug>/ep<NNN>/frame_<NNNNN>.png``,
  with the SAM2 mask shown as a red 50% overlay on the source image.
  Frames where segmentation failed are still dumped (no overlay) so
  drop-outs are visible by scrubbing the folder.
  """
  if masks is None:
    print("[debug-masks] no masks available on estimator; skipping dump")
    return
  try:
    from PIL import Image
  except ImportError:
    print("[debug-masks] PIL not available; skipping dump")
    return

  from chuck_dreamer.real.object_localization.types import dataset_slug

  ep_dir = out_root / dataset_slug(source_repo) / f"ep{episode_index:03d}"
  ep_dir.mkdir(parents=True, exist_ok=True)
  n_with_mask = 0
  for t, (img, mask) in enumerate(zip(imgs, masks)):
    if img.dtype != np.uint8:
      img_u8 = (np.clip(img.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
      img_u8 = img
    if img_u8.ndim == 3 and img_u8.shape[-1] == 1:
      img_u8 = np.repeat(img_u8, 3, axis=-1)
    if img_u8.ndim == 2:
      img_u8 = np.stack([img_u8] * 3, axis=-1)
    out = img_u8.copy()
    if mask is not None:
      n_with_mask += 1
      m = np.asarray(mask, dtype=bool)
      if m.shape[:2] == out.shape[:2]:
        red = np.array([255, 0, 0], dtype=np.uint8)
        out[m] = (out[m].astype(np.uint16) * 1 // 2
                  + red.astype(np.uint16) * 1 // 2).astype(np.uint8)
    Image.fromarray(out).save(ep_dir / f"frame_{t:05d}.png")
  print(f"[debug-masks] wrote {len(imgs)} frames "
        f"({n_with_mask} with mask) to {ep_dir}")


class SegmentationStage:
  """Run SAM2 for one target, caching masks on ``ctx`` for downstream
  stages (e.g. pose fitting) and filling the per-target seg/uv arrays."""
  requires: tuple[str, ...] = ()

  def __init__(self, target_name: str = "object") -> None:
    self.target_name = target_name
    self.name = f"segment:{target_name}"
    self.produces: tuple[str, ...] = seg_keys(target_name)

  def requirements(self, ctx: StageContext) -> list[Requirement]:
    from chuck_dreamer.real.object_localization.types import dataset_cache_dir
    ol_cfg = ctx.ol_cfg()
    ds_dir = dataset_cache_dir(Path(ol_cfg.calibration_cache), ctx.source_repo)
    return [
      Requirement(
        "camera intrinsics", ds_dir / "intrinsics.json",
        f"uv run python main.py calibrate-intrinsics {ctx.source_repo}"),
      Requirement(
        "camera extrinsics", ds_dir / "extrinsics.json",
        f"uv run python main.py annotate-mat {ctx.source_repo}"),
      Requirement(
        "frame-0 object prompts", ds_dir / "object_prompts.json",
        f"uv run python main.py prompt-episodes --dataset {ctx.source_repo}"),
      Requirement(
        "object mesh", Path(ol_cfg.mesh_path),
        "set object_localization.mesh_path in configs/default.yaml "
        "to a valid .obj/.ply"),
    ]

  def apply(self, episode, metadata, ctx) -> None:
    images = episode.get("image")
    if images is None or len(images) == 0:
      return
    cfg = metadata.get("config")
    if not isinstance(cfg, dict):
      raise RuntimeError("segment: metadata['config'] missing or invalid")
    source_repo = cfg.get("source_repo")
    if not isinstance(source_repo, str) or not source_repo:
      raise RuntimeError("segment: metadata['config']['source_repo'] missing")

    imgs = as_uint8_hwc(images)
    estimator = ctx.estimator()
    ol_cfg = ctx.ol_cfg()
    cache_dir = Path(ol_cfg.calibration_cache)
    episode_index = int(cfg.get("episode_index", 0))

    from chuck_dreamer.real.object_localization.prompts import (
      load_keyframe_prompts,
    )
    keyframes = load_keyframe_prompts(cache_dir, source_repo, episode_index)
    if 0 not in keyframes:
      raise FileNotFoundError(
        f"no cached frame-0 prompt for {source_repo} episode {episode_index}. "
        f"Run: `python main.py prompt-episodes --dataset {source_repo}`")
    T = len(imgs)
    keyframes = {fi: pr for fi, pr in keyframes.items() if 0 <= fi < T}
    print(f"[segment:{self.target_name}] {source_repo} ep{episode_index}: "
          f"using {len(keyframes)} keyframe prompt(s) at frames "
          f"{sorted(keyframes.keys())}")

    masks = estimator.segment_episode(
      list(imgs),
      first_frame_prompt=keyframes[0],
      keyframe_prompts=keyframes,
      target_name=self.target_name,
    )
    ctx.masks[self.target_name] = masks

    if masks is not None:
      seg_key, uv_key = seg_keys(self.target_name)
      _store_masks_and_uv(episode, list(imgs), masks,
                          seg_key=seg_key, uv_key=uv_key)

    debug_dir = os.environ.get("IMPORT_LEROBOT_DEBUG_MASKS")
    if debug_dir:
      _dump_debug_masks(
        Path(debug_dir), source_repo, episode_index, list(imgs), masks)
