"""Composable analysis stages for the LeRobot importer.

Each freshly-imported episode dict is passed through an ordered list of
:class:`Stage` objects. A stage is bound to a :class:`RunContext` at
construction and declares ``produces`` (episode/metadata keys it fills),
``requires`` (names of stages that must run before it), its external
:class:`Requirement` list, and an :meth:`Stage.apply` transform.

A :class:`Pipeline` is built per episode from the importer's flags: it
resolves the enabled stage names into a dependency-ordered list
(auto-pulling each stage's ``requires``) and is iterable to yield those
stages. The importer iterates it calling ``stage.apply``;
``import-lerobot --doctor`` iterates a pipeline *per episode* and prints
each stage's :meth:`Stage.requirements`, so neither the loop nor the
doctor hard-codes any single stage's details and per-episode requirements
are surfaced.

Adding a new analysis step = add a :class:`Stage` in its own module and
register it in :func:`build_registry`; the pipeline and doctor pick it up
unchanged.
"""
from __future__ import annotations

from .base import Requirement, RunContext, Stage
from .ee_pos import EePosStage
from .object_pose import ObjectPoseStage
from .pipeline import Pipeline, Run
from .registry import build_registry, resolve_stages
from .segmentation import SegmentationStage

__all__ = [
  "Requirement",
  "Stage",
  "RunContext",
  "EePosStage",
  "SegmentationStage",
  "ObjectPoseStage",
  "Pipeline",
  "Run",
  "build_registry",
  "resolve_stages",
]
