"""Structured wrapper for an RGB observation plus its segmentation masks.

The env returns this under ``obs["image"]`` so consumers can reach both
the raw RGB (``.image``) and the per-geom segmentation masks without
re-running the segmentation renderer.

Masks are boolean ``(H, W)`` arrays aligned with ``image``. Clutter is
addressable both per-piece (``clutter_masks[i]``) and as a single union
(``clutter_mask``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ObservationImage:
  """RGB image + segmentation masks for a single timestep.

  Fields:
    image:         (H, W, 3) uint8 RGB.
    target_mask:   (H, W) bool — the pushable target object.
    goal_mask:     (H, W) bool — the goal marker.
    clutter_masks: list of (H, W) bool, one per clutter piece, in scene order.
    arm_mask:      (H, W) bool — union of all robot-arm geoms.
    background_mask: (H, W) bool — pixels not covered by any tracked geom.
  """

  image: np.ndarray
  target_mask: np.ndarray
  goal_mask: np.ndarray
  clutter_masks: list[np.ndarray] = field(default_factory=list)
  arm_mask: np.ndarray | None = None
  background_mask: np.ndarray | None = None

  @property
  def clutter_mask(self) -> np.ndarray:
    """Union of all clutter masks. Zeros (H, W) if there is no clutter."""
    if not self.clutter_masks:
      return np.zeros(self.image.shape[:2], dtype=bool)
    return np.logical_or.reduce(self.clutter_masks)

  @property
  def masks(self) -> dict[str, np.ndarray]:
    """All named masks in one dict — handy for iteration / debugging."""
    out: dict[str, np.ndarray] = {
      "target": self.target_mask,
      "goal":   self.goal_mask,
      "clutter": self.clutter_mask,
    }
    for i, m in enumerate(self.clutter_masks):
      out[f"clutter_{i}"] = m
    if self.arm_mask is not None:
      out["arm"] = self.arm_mask
    if self.background_mask is not None:
      out["background"] = self.background_mask
    return out

  @classmethod
  def from_seg(
      cls,
      image: np.ndarray,
      mask_by_name: Mapping[str, np.ndarray],
  ) -> "ObservationImage":
    """Build from the env's per-geom mask dict.

    Expects keys ``target_object_geom``, ``goal_marker``, ``background``
    (optional), ``arm`` (optional), and any number of ``clutter_{i}_geom``
    entries. Clutter keys are sorted by ``i`` so ``clutter_masks[i]``
    matches the scene's clutter order.
    """
    target = mask_by_name["target_object_geom"]
    goal   = mask_by_name["goal_marker"]
    background = mask_by_name.get("background")
    arm        = mask_by_name.get("arm")

    clutter_items = [
      (int(k.split("_")[1]), m)
      for k, m in mask_by_name.items()
      if k.startswith("clutter_") and k.endswith("_geom")
    ]
    clutter_items.sort(key=lambda kv: kv[0])
    clutter_masks = [m for _, m in clutter_items]

    return cls(
      image=image,
      target_mask=target,
      goal_mask=goal,
      clutter_masks=clutter_masks,
      arm_mask=arm,
      background_mask=background,
    )
