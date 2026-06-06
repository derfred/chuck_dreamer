"""Object localizer perception module — wraps the SAM2 + pose-fit estimator.

This is M3's "first real perception module": it wraps the existing
``ObjectPoseEstimator`` (``lerobot/object_localization/estimator.py``) behind
the runtime :class:`~chuck_dreamer.runtime.perception.PerceptionModule`
protocol, emitting world-frame object position (``object_xy``) and its image
projection (``object_uv``).

It is **wired but off by default** (the ``runtime.perception`` config list is
empty). Two real dependencies keep it out of the default sim loop:

* **Calibration is an M4 deliverable.** ``CameraCalibration.load`` is strict —
  it raises if intrinsics/extrinsics aren't cached — so this module's
  ``from_config`` fails fast with a clear message when calibration is absent.
* **It is not real-time** (the estimator's CPU rasterization + SAM2 are slow).
  That is acceptable per spec §3.2/§4.1 — inline perception reduces the policy
  rate while the control loop keeps slewing — but it is why M3 ships the cheap
  ``EePoseModule`` for the runnable sim path and leaves this for later.

Because there is no UI until M7, the first-frame SAM2 prompt is a required
``first_frame_prompt: [u, v]`` from config; subsequent frames warm-start from
the previous pose (held as module state, cleared each episode by ``reset``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .modalities import IMAGE, OBJECT_UV, OBJECT_XY

__all__ = ["ObjectLocalizerModule"]


class ObjectLocalizerModule:
  """Per-frame object pose via SAM2 + silhouette/edge fit, as a perception module."""

  name = "object_localizer"
  consumes: tuple[str, ...] = (IMAGE,)
  emits: tuple[str, ...] = (OBJECT_XY, OBJECT_UV)

  def __init__(self, estimator, *, first_frame_prompt: tuple[int, int]) -> None:
    self._estimator = estimator
    self._first_frame_prompt = (int(first_frame_prompt[0]), int(first_frame_prompt[1]))
    self._prev_pose = None

  @classmethod
  def from_config(cls, cfg, **params) -> "ObjectLocalizerModule":
    from chuck_dreamer.lerobot.object_localization import (
      CameraCalibration,
      ObjectPoseEstimator,
      init_from_config,
    )

    prompt = params.get("first_frame_prompt")
    if prompt is None:
      raise ValueError(
        "ObjectLocalizerModule requires 'first_frame_prompt: [u, v]' in its "
        "params (no UI to click the object at M3)")
    dataset_id = params.get("dataset_id")
    if dataset_id is None:
      raise ValueError(
        "ObjectLocalizerModule requires 'dataset_id' in its params to locate the "
        "cached camera calibration")

    ol = init_from_config(cfg)
    # Strict load — raises CalibrationMissingError if intrinsics/extrinsics are
    # not cached (calibration is the M4 deliverable; surface that here).
    calibration = CameraCalibration.load(Path(ol.cache_dir), str(dataset_id))
    estimator = ObjectPoseEstimator(
      calibration=calibration,
      mesh_path=Path(ol.mesh_path),
      config={
        "sam2_checkpoint": ol.sam2_checkpoint,
        "use_sam2": ol.use_sam2,
      },
      device=ol.device,
      use_sam2=ol.use_sam2,
    )
    return cls(estimator, first_frame_prompt=tuple(prompt))

  def reset(self) -> None:
    self._prev_pose = None

  def process(self, data: dict[str, np.ndarray], *, t: float) -> dict[str, np.ndarray]:
    from chuck_dreamer.lerobot.object_localization.estimator import _pose_to_prompt

    image = data[IMAGE]
    pose = self._estimator.estimate(
      image,
      prev_pose=self._prev_pose,
      prompt=None if self._prev_pose is not None else self._first_frame_prompt,
    )
    if pose is None:
      # Gap (segmentation failed): keep the warm-start anchor, emit nothing.
      return {}
    self._prev_pose = pose
    u, v = _pose_to_prompt(pose, self._estimator.camera)
    return {
      OBJECT_XY: np.asarray(pose.xy_mm, dtype=np.float32),
      OBJECT_UV: np.asarray([u, v], dtype=np.float32),
    }
