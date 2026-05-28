"""Stage registry, dependency resolution, and flag → stage-set mapping."""
from __future__ import annotations

from .base import Stage
from .ee_pos import EePosStage
from .object_pose import ObjectPoseStage
from .segmentation import SegmentationStage


def build_registry() -> dict[str, Stage]:
  stages: list[Stage] = [
    EePosStage(),
    SegmentationStage("object"),
    ObjectPoseStage(),
  ]
  return {s.name: s for s in stages}


def resolve_stages(enabled: set[str],
                   registry: dict[str, Stage] | None = None) -> list[Stage]:
  """Return ``enabled`` stages plus their transitive ``requires``,
  dependency-ordered. Raises on unknown names or dependency cycles."""
  registry = registry if registry is not None else build_registry()
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


def enabled_from_flags(*, with_ee_pos: bool, with_object_pose: bool) -> set[str]:
  """Map the importer's boolean flags onto the enabled stage-name set.

  ``object_pose`` pulls ``segment:object`` transitively via
  :func:`resolve_stages`, so it need not be listed explicitly here.
  """
  enabled: set[str] = set()
  if with_ee_pos:
    enabled.add("ee_pos")
  if with_object_pose:
    enabled.add("object_pose")
  return enabled
