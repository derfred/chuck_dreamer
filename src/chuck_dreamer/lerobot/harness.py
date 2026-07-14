"""Pipeline harness core: node declarations and dependency resolution.

Implements the node half of ``docs/trainer/harness_design.md``: a
:class:`Node` declares typed inputs and outputs; the producer relation and
execution order are *derived* from those declarations (there are no
hand-maintained stage-name dependencies). The TrackSet/NodeView execution
layer builds on this; the existing ``stages/`` run through a compatibility
shim until each is rewritten as a native node.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from chuck_dreamer.common.tracks import ChannelUnits, Frame, TrackSpec, Unit

if TYPE_CHECKING:
  from chuck_dreamer.common.tracks import Track


class SpecError(ValueError):
  """Static specification error (spec §10.1): missing/duplicate producer,
  or a dependency cycle."""


@dataclass(frozen=True)
class InputDecl:
  """One declared node input: a track or artifact type name, plus the
  expected frame/unit for per-frame tracks (``None`` = don't care)."""
  name: str
  frame: Frame | None = None
  unit: Unit | ChannelUnits | None = None


@runtime_checkable
class Node(Protocol):
  """A pipeline node (spec §7): pure at episode granularity, no knowledge of
  the graph. ``inputs`` may name tracks, artifacts, assets, or constants;
  ``outputs`` are the TrackSpecs it produces."""
  name: str
  scope_level: str                    # "camera" | "dataset" | "episode"
  inputs: tuple[InputDecl, ...]
  outputs: tuple[TrackSpec, ...]

  def run(self, view: "NodeView") -> "list[Track]": ...


class NodeView(Protocol):
  """What a node sees at run time: exactly its declared inputs, resolved,
  validated, and unit-converted by the harness."""
  def track(self, name: str) -> "Track": ...
  def artifact(self, name: str) -> object: ...


def producer_map(nodes: list[Node]) -> dict[str, Node]:
  """Output name → producing node. Two producers of one name is a static
  specification error (spec §10.1)."""
  out: dict[str, Node] = {}
  for node in nodes:
    for spec in node.outputs:
      if spec.name in out:
        raise SpecError(
          f"{spec.name!r} is produced by both {out[spec.name].name!r} "
          f"and {node.name!r}")
      out[spec.name] = node
  return out


def resolve_order(nodes: list[Node], targets: set[str],
                  leaves: frozenset[str] | set[str] = frozenset()) -> list[Node]:
  """Dependency-ordered nodes needed to produce ``targets``.

  Edges are derived from the declarations: node A precedes node B iff B
  declares an input that A produces. ``leaves`` are the user-supplied names
  (dataset components, assets, annotations) that need no producer; an input
  that is neither a leaf nor produced by any node is a :class:`SpecError`.
  """
  producers = producer_map(nodes)
  ordered: list[Node] = []
  done: set[str] = set()
  visiting: set[str] = set()

  def visit(node: Node) -> None:
    if node.name in done:
      return
    if node.name in visiting:
      raise SpecError(f"dependency cycle through node {node.name!r}")
    visiting.add(node.name)
    for decl in node.inputs:
      dep = producers.get(decl.name)
      if dep is not None:
        visit(dep)
      elif decl.name not in leaves:
        raise SpecError(
          f"node {node.name!r} input {decl.name!r} has no producer and is "
          "not a declared leaf")
    visiting.discard(node.name)
    done.add(node.name)
    ordered.append(node)

  for target in sorted(targets):
    node = producers.get(target)
    if node is None:
      raise SpecError(f"no node produces requested target {target!r}")
    visit(node)
  return ordered
