"""Composable analysis stages for the LeRobot importer.

Each freshly-imported episode dict is passed through an ordered list of
:class:`Stage` objects. A stage declares ``produces`` (episode/metadata
keys it fills), ``requires`` (names of stages that must run before it),
its external :class:`Requirement` list, and an :meth:`Stage.apply`
transform.

:func:`resolve_stages` turns a set of enabled stage names into a
dependency-ordered list, auto-pulling each stage's ``requires``. The
importer runs the list per episode; ``import-lerobot --doctor`` iterates
the same list and prints each stage's :meth:`Stage.requirements`, so
neither the loop nor the doctor hard-codes any single stage's details.

Adding a new analysis step = add a :class:`Stage` in its own module and
register it in :func:`build_registry`; the loop and doctor pick it up
unchanged.
"""
from __future__ import annotations

from .base import Requirement, Stage, StageContext
from .ee_pos import EePosStage
from .object_pose import ObjectPoseStage
from .registry import build_registry, enabled_from_flags, resolve_stages
from .segmentation import SegmentationStage

__all__ = [
  "Requirement",
  "Stage",
  "StageContext",
  "EePosStage",
  "SegmentationStage",
  "ObjectPoseStage",
  "build_registry",
  "resolve_stages",
  "enabled_from_flags",
]
