"""High-level ObjectPoseEstimator class — drives segmentation + refinement."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..calibration import CameraCalibration
from ..geometry import line_endpoints_world
from .initial import (
  RestingFaceHypothesis,
  build_resting_hypotheses,
  initial_translation_from_mask,
)
from .mesh import ObjectMesh, load_mesh
from .refinement import RefinedPose, refine_edges, refine_silhouette
from .segmentation import (
  SegmentationPrompt,
  Sam2ImageSegmenter,
  Sam2VideoSegmenter,
  heuristic_mask,
  make_segmenter,
)

logger = logging.getLogger(__name__)


@dataclass
class ObjectPose:
  xy_mm:       np.ndarray   # (2,)
  xyz_mm:      np.ndarray   # (3,)
  R_world_obj: np.ndarray   # (3, 3) object -> world
  confidence:  float
  reprojection_error_px: float


def _mat_roi_mask(image_size: tuple[int, int], cal: CameraCalibration,
                  pad_mm: float = 100.0) -> np.ndarray:
  """A loose ROI mask covering the mat region in image space."""
  import cv2

  W, H = image_size
  eps = line_endpoints_world()
  L = float(abs(eps["line1_far"][0]))
  D = float(eps["line1_near"][1] - eps["line2_near"][1])
  corners = np.array([
    [+pad_mm,         +D / 2 + pad_mm, 0.0],
    [-L - pad_mm,     +D / 2 + pad_mm, 0.0],
    [-L - pad_mm,     -D / 2 - pad_mm, 0.0],
    [+pad_mm,         -D / 2 - pad_mm, 0.0],
  ], dtype=np.float64)
  rvec, _ = cv2.Rodrigues(cal.R)
  proj, _ = cv2.projectPoints(corners.reshape(-1, 1, 3),
                              rvec, cal.t.reshape(3, 1),
                              cal.K, cal.dist)
  pts = proj.reshape(-1, 2).astype(np.int32)
  mask = np.zeros((H, W), dtype=np.uint8)
  cv2.fillConvexPoly(mask, pts, 255)
  return mask.astype(bool)


class ObjectPoseEstimator:
  """Stateful per-episode estimator for the rhombicuboctahedron."""

  def __init__(
    self,
    calibration: CameraCalibration,
    mesh_path:   str | Path,
    device:      str = "cuda",
    use_sam2:    bool = True,
  ):
    self.cal       = calibration
    self.mesh      = load_mesh(mesh_path)
    self.device    = device
    self.image_size = (int(calibration.image_size[0]), int(calibration.image_size[1]))
    self.hypotheses: list[RestingFaceHypothesis] = build_resting_hypotheses(self.mesh)
    self._mat_roi  = _mat_roi_mask(self.image_size, calibration)

    self._image_seg: Optional[Sam2ImageSegmenter] = (
      make_segmenter(device=device) if use_sam2 else None
    )
    self._video_seg_factory = (
      (lambda: Sam2VideoSegmenter(device=device)) if use_sam2 else None
    )

    # Stateful (per-episode) cache:
    self._chosen_hypothesis: RestingFaceHypothesis | None = None
    self._prev_pose: ObjectPose | None = None

  # ------------------------------------------------------------------ #
  # Public API
  # ------------------------------------------------------------------ #

  def reset(self) -> None:
    """Clear per-episode state (chosen face, previous pose)."""
    self._chosen_hypothesis = None
    self._prev_pose = None

  def estimate(
    self,
    image: np.ndarray,
    prev_pose: Optional[ObjectPose] = None,
    prompt:    Optional[SegmentationPrompt] = None,
  ) -> ObjectPose | None:
    """Estimate object pose from a single frame."""
    mask = self._segment_image(image, prompt)
    if mask is None or mask.sum() < 100:
      return None

    cal = self.cal
    if prev_pose is None and self._prev_pose is not None:
      prev_pose = self._prev_pose

    if self._chosen_hypothesis is None:
      best = self._search_resting_face(image, mask)
      if best is None:
        return None
      self._chosen_hypothesis = best
    hypothesis = self._chosen_hypothesis

    if prev_pose is not None:
      x, y = float(prev_pose.xy_mm[0]), float(prev_pose.xy_mm[1])
      yaw0 = float(np.arctan2(prev_pose.R_world_obj[1, 0],
                              prev_pose.R_world_obj[0, 0]))
    else:
      t0 = initial_translation_from_mask(
        mask, cal.K, cal.dist, cal.R, cal.t, hypothesis.z_center)
      x, y = float(t0[0]), float(t0[1])
      yaw0 = 0.0

    refined = refine_silhouette(
      self.mesh, hypothesis,
      init_xy=(x, y), init_yaw=yaw0,
      observed_mask=mask,
      cal_K=cal.K, cal_dist=cal.dist, cal_R_cw=cal.R, cal_t_cw=cal.t,
      image_size=self.image_size,
    )
    refined = refine_edges(
      self.mesh, hypothesis, refined,
      image=image,
      cal_K=cal.K, cal_dist=cal.dist, cal_R_cw=cal.R, cal_t_cw=cal.t,
      image_size=self.image_size,
    )
    out = _to_object_pose(refined)
    self._prev_pose = out
    return out

  def estimate_episode(
    self,
    frames: Iterable[np.ndarray],
    first_frame_prompt: Optional[SegmentationPrompt] = None,
  ) -> list[ObjectPose | None]:
    """Run on a full episode. Uses SAM2 video propagation if available."""
    frame_list = [f for f in frames]
    if not frame_list:
      return []

    masks = self._propagate_masks(frame_list, first_frame_prompt)

    out: list[ObjectPose | None] = []
    self.reset()
    prev: ObjectPose | None = None
    for i, frame in enumerate(frame_list):
      mask = masks[i] if i < len(masks) else None
      if mask is None or mask.sum() < 100:
        out.append(None)
        continue
      pose = self._fit_with_mask(frame, mask, prev_pose=prev)
      out.append(pose)
      if pose is not None:
        prev = pose
    return out

  # ------------------------------------------------------------------ #
  # Internals
  # ------------------------------------------------------------------ #

  def _segment_image(
    self,
    image:  np.ndarray,
    prompt: Optional[SegmentationPrompt],
  ) -> np.ndarray | None:
    if self._image_seg is not None:
      if prompt is None:
        prompt = self._auto_prompt(image)
        if prompt is None:
          return None
      try:
        return self._image_seg.predict(image, prompt)
      except Exception as e:
        logger.warning("SAM2 predict failed (%s) — falling back to heuristic", e)
    return heuristic_mask(image, mat_roi_mask=self._mat_roi)

  def _propagate_masks(
    self,
    frames: list[np.ndarray],
    first_prompt: Optional[SegmentationPrompt],
  ) -> list[np.ndarray | None]:
    if self._video_seg_factory is not None:
      try:
        seg = self._video_seg_factory()
        prompt = first_prompt
        if prompt is None:
          prompt = self._auto_prompt(frames[0])
        if prompt is None:
          raise RuntimeError("no first-frame prompt available")
        return seg.propagate(frames, prompt)
      except Exception as e:
        logger.warning("SAM2 video propagation failed (%s) — falling back per-frame", e)
    # Fallback: per-frame heuristic masks
    return [heuristic_mask(f, mat_roi_mask=self._mat_roi) for f in frames]

  def _auto_prompt(self, image: np.ndarray) -> SegmentationPrompt | None:
    m = heuristic_mask(image, mat_roi_mask=self._mat_roi)
    if m is None:
      return None
    ys, xs = np.where(m)
    return SegmentationPrompt(
      bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
      points=[(int(xs.mean()), int(ys.mean()))],
    )

  def _fit_with_mask(
    self,
    image:     np.ndarray,
    mask:      np.ndarray,
    prev_pose: ObjectPose | None,
  ) -> ObjectPose | None:
    if self._chosen_hypothesis is None:
      best = self._search_resting_face(image, mask)
      if best is None:
        return None
      self._chosen_hypothesis = best
    hypothesis = self._chosen_hypothesis

    cal = self.cal
    if prev_pose is not None:
      x, y = float(prev_pose.xy_mm[0]), float(prev_pose.xy_mm[1])
      yaw0 = float(np.arctan2(prev_pose.R_world_obj[1, 0],
                              prev_pose.R_world_obj[0, 0]))
    else:
      t0 = initial_translation_from_mask(
        mask, cal.K, cal.dist, cal.R, cal.t, hypothesis.z_center)
      x, y = float(t0[0]), float(t0[1])
      yaw0 = 0.0

    refined = refine_silhouette(
      self.mesh, hypothesis,
      init_xy=(x, y), init_yaw=yaw0,
      observed_mask=mask,
      cal_K=cal.K, cal_dist=cal.dist, cal_R_cw=cal.R, cal_t_cw=cal.t,
      image_size=self.image_size,
    )
    refined = refine_edges(
      self.mesh, hypothesis, refined,
      image=image,
      cal_K=cal.K, cal_dist=cal.dist, cal_R_cw=cal.R, cal_t_cw=cal.t,
      image_size=self.image_size,
    )
    return _to_object_pose(refined)

  def _search_resting_face(
    self,
    image: np.ndarray,
    mask:  np.ndarray,
  ) -> RestingFaceHypothesis | None:
    """Try square + triangle hypotheses with a handful of yaw starts; keep the best."""
    cal = self.cal

    best_residual = np.inf
    best_hyp: RestingFaceHypothesis | None = None
    best_refined: RefinedPose | None = None

    for hypothesis in self.hypotheses:
      t0 = initial_translation_from_mask(
        mask, cal.K, cal.dist, cal.R, cal.t, hypothesis.z_center)
      x, y = float(t0[0]), float(t0[1])

      if hypothesis.face_type == "square":
        yaw_inits = [0.0, np.pi / 4]
      else:
        yaw_inits = [0.0, np.pi / 3]

      for yaw0 in yaw_inits:
        try:
          refined = refine_silhouette(
            self.mesh, hypothesis,
            init_xy=(x, y), init_yaw=float(yaw0),
            observed_mask=mask,
            cal_K=cal.K, cal_dist=cal.dist, cal_R_cw=cal.R, cal_t_cw=cal.t,
            image_size=self.image_size,
          )
        except Exception as e:
          logger.debug("hypothesis %s yaw=%.2f failed: %s",
                       hypothesis.face_type, yaw0, e)
          continue
        if refined.residual < best_residual:
          best_residual = refined.residual
          best_hyp      = hypothesis
          best_refined  = refined

    if best_refined is None:
      return None
    logger.info("chosen resting face: %s (residual %.2f px)",
                best_hyp.face_type if best_hyp else "?", best_residual)
    return best_hyp


def _to_object_pose(r: RefinedPose) -> ObjectPose:
  return ObjectPose(
    xy_mm=np.array([float(r.t_wo[0]), float(r.t_wo[1])]),
    xyz_mm=np.asarray(r.t_wo, dtype=np.float64),
    R_world_obj=np.asarray(r.R_wo, dtype=np.float64),
    confidence=float(np.exp(-max(r.residual, 0.0) / 20.0)),
    reprojection_error_px=float(r.reprojection_error_px),
  )
