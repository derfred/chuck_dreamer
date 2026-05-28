"""Object-pose stage: per-frame pose fit + RTS smoothing over object masks."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Requirement, StageContext, as_uint8_hwc


class ObjectPoseStage:
  name = "object_pose"
  produces: tuple[str, ...] = ("object_xy", "object_gap_too_long")
  requires: tuple[str, ...] = ("segment:object",)

  def requirements(self, ctx: StageContext) -> list[Requirement]:
    ol_cfg = ctx.ol_cfg()
    return [Requirement(
      "object mesh", Path(ol_cfg.mesh_path),
      "set object_localization.mesh_path in configs/default.yaml "
      "to a valid .obj/.ply")]

  def apply(self, episode, metadata, ctx) -> None:
    images = episode.get("image")
    if images is None or len(images) == 0:
      return
    cfg = metadata.get("config")
    if not isinstance(cfg, dict):
      raise RuntimeError("object_pose: metadata['config'] missing or invalid")
    source_repo = cfg.get("source_repo")
    if not isinstance(source_repo, str) or not source_repo:
      raise RuntimeError("object_pose: metadata['config']['source_repo'] missing")
    episode_index = int(cfg.get("episode_index", 0))

    masks = ctx.masks.get("object")
    if masks is None:
      raise RuntimeError(
        "object_pose: no object masks on context — segment:object must "
        "run first")

    imgs = as_uint8_hwc(images)
    estimator = ctx.estimator()

    # fit_poses_from_masks needs the frame-0 prompt for keyframe
    # bookkeeping; reuse the same cached prompts the segmentation stage used.
    from chuck_dreamer.real.object_localization.prompts import (
      load_keyframe_prompts,
    )
    cache_dir = Path(ctx.ol_cfg().calibration_cache)
    keyframes = load_keyframe_prompts(cache_dir, source_repo, episode_index)
    T = len(imgs)
    keyframes = {fi: pr for fi, pr in keyframes.items() if 0 <= fi < T}
    poses = estimator.fit_poses_from_masks(
      list(imgs), masks,
      first_frame_prompt=keyframes[0],
      keyframe_prompts=keyframes,
    )
    n_ok = sum(1 for p in poses if p is not None)
    print(f"[object_pose] {source_repo} ep{episode_index}: "
          f"{n_ok}/{len(poses)} non-None poses")

    smoother = ctx.smoother()
    if all(p is None for p in poses) or poses[0] is None:
      out = np.zeros((T, 2), dtype=np.float32)
      for i, p in enumerate(poses):
        if p is not None:
          out[i] = p.xy_mm.astype(np.float32)
      episode["object_xy"] = out
      episode["object_gap_too_long"] = np.ones((T,), dtype=np.bool_)
      return

    frames = smoother.smooth_episode(poses)
    episode["object_xy"] = np.stack(
      [f.xy_mm for f in frames], axis=0).astype(np.float32)
    episode["object_gap_too_long"] = np.array(
      [f.gap_too_long for f in frames], dtype=np.bool_)
