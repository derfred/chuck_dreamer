"""Stage registry and dependency resolution.

Stages are bound to a :class:`RunContext` at construction, so building the
registry needs the context. The flag → stage-set mapping lives on
:class:`Run`.
"""
from __future__ import annotations

from .base import RunContext, Stage
from .ee_pos import EePosStage
from .object_pose import ObjectPoseStage
from .segmentation import SegmentationStage


def build_registry(ctx: RunContext) -> dict[str, Stage]:
  stages: list[Stage] = [
    EePosStage(ctx),
    SegmentationStage(ctx),
    ObjectPoseStage(ctx),
  ]
  return {s.name: s for s in stages}


def resolve_stages(enabled: set[str], ctx: RunContext,
                   registry: dict[str, Stage] | None = None) -> list[Stage]:
  """Return ``enabled`` stages plus their transitive ``requires``,
  dependency-ordered. Raises on unknown names or dependency cycles.

  ``enabled`` is a set of stage names; the flag→stage-set mapping lives on
  :class:`Run`. ``registry`` defaults to a fresh :func:`build_registry`."""
  registry = registry if registry is not None else build_registry(ctx)
  unknown = enabled - registry.keys()
  if unknown:
    raise ValueError(
      f"unknown stage(s): {sorted(unknown)}. "
      f"Known: {sorted(registry)}")

  ordered: list[Stage] = []
  visiting: set[str] = set()
  done: set[str] = set()

  def visit(name: str) -> None:
    if name in done:
      return
    if name in visiting:
      raise ValueError(f"stage dependency cycle through {name!r}")
    visiting.add(name)
    stage = registry[name]
    for dep in stage.requires:
      if dep not in registry:
        raise ValueError(
          f"stage {name!r} requires unknown stage {dep!r}")
      visit(dep)
    visiting.discard(name)
    done.add(name)
    ordered.append(stage)

  for name in enabled:
    visit(name)
  return ordered
