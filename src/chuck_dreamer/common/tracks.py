"""Track types for the pipeline harness (docs/trainer/harness_design.md).

A :class:`Track` is an immutable, per-frame array with a declared
:class:`TrackSpec` — name, dtype, per-frame shape, reference frame, unit,
optional validity mask, and persistence policy. Frames and units are
validated at node boundaries by the harness: frame changes must be explicit
nodes; mechanical scalar unit conversions (deg↔rad, mm↔m) are applied
automatically through the single registry below, never inline in node code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import numpy as np


class Frame(str, Enum):
  """Reference frames for spatial tracks.

  ``TABLE`` is the world frame defined by the mat fiducials (origin at the
  circle center, Z = 0 on the mat) — older code and docs call it the "world"
  frame; the two names are the same frame. ``ARM`` is the arm-base frame FK
  outputs live in; the ``table_to_arm`` transform (spec §7.2) bridges the
  two, and a track only crosses frames through an explicit node.
  """
  IMAGE  = "image"
  CAMERA = "camera"
  ARM    = "arm"
  TABLE  = "table"
  NONE   = "none"


class Unit(str, Enum):
  PX       = "px"
  M        = "m"
  MM       = "mm"
  RAD      = "rad"
  DEG      = "deg"
  S        = "s"
  UNITLESS = "unitless"


@dataclass(frozen=True)
class ChannelUnits:
  """`unit=mixed` in the spec: contiguous channel ranges with distinct units.

  ``groups`` maps ``(start, stop)`` channel ranges (on the last axis) to a
  :class:`Unit`, e.g. joint values = ``ChannelUnits({(0, 5): Unit.DEG,
  (5, 6): Unit.UNITLESS})``. Ranges must not overlap."""
  groups: tuple[tuple[tuple[int, int], Unit], ...]

  def __init__(self, groups: dict[tuple[int, int], Unit]) -> None:
    items = tuple(sorted(groups.items()))
    for (a, b), _ in items:
      if a >= b:
        raise ValueError(f"empty channel range ({a}, {b})")
    for ((_, b1), _u1), ((a2, _), _u2) in zip(items, items[1:]):
      if b1 > a2:
        raise ValueError("overlapping channel ranges")
    object.__setattr__(self, "groups", items)


# The single unit-conversion registry (spec §10.2): lossless scalar factors
# only. Anything not listed here is not mechanically convertible.
_CONVERSIONS: dict[tuple[Unit, Unit], float] = {
  (Unit.DEG, Unit.RAD): math.pi / 180.0,
  (Unit.RAD, Unit.DEG): 180.0 / math.pi,
  (Unit.MM,  Unit.M):   1e-3,
  (Unit.M,   Unit.MM):  1e3,
}


class FrameMismatch(ValueError):
  """A track's reference frame differs from a node's declared expectation.
  Never auto-converted — frame changes are geometry and must be a node."""


class UnitMismatch(ValueError):
  """A track's unit differs from the expectation and no mechanical scalar
  conversion exists."""


class Persist(Enum):
  EPHEMERAL  = "ephemeral"    # in-memory only, dropped after last consumer
  CACHED     = "cached"       # written to the artifact store
  TRAJECTORY = "trajectory"   # handed to the trajectory writer
  DIAGNOSTIC = "diagnostic"   # trajectory writer, marked non-training


@dataclass(frozen=True)
class TrackSpec:
  name: str
  dtype: Any                              # numpy dtype-like
  shape: tuple[int, ...]                  # per-frame shape, () for scalars
  frame: Frame = Frame.NONE
  unit: Unit | ChannelUnits = Unit.UNITLESS
  validity: bool = False
  persist: Persist = Persist.EPHEMERAL


@dataclass(frozen=True)
class Track:
  """Immutable per-frame data + optional validity, conforming to its spec.

  A node that "modifies" a track declares a new output track instead — the
  in-place mutation pattern is exactly how the historical deg/rad bug arose.
  """
  spec: TrackSpec
  values: np.ndarray
  valid: np.ndarray | None = None

  def __post_init__(self) -> None:
    v = np.asarray(self.values)
    if v.shape[1:] != self.spec.shape:
      raise ValueError(
        f"track {self.spec.name!r}: per-frame shape {v.shape[1:]} != "
        f"declared {self.spec.shape}")
    if v.dtype != np.dtype(self.spec.dtype):
      raise ValueError(
        f"track {self.spec.name!r}: dtype {v.dtype} != declared "
        f"{np.dtype(self.spec.dtype)}")
    object.__setattr__(self, "values", v)
    if self.spec.validity:
      if self.valid is None:
        raise ValueError(
          f"track {self.spec.name!r} declares validity but none was given")
      m = np.asarray(self.valid, dtype=bool)
      if m.shape != (len(v),):
        raise ValueError(
          f"track {self.spec.name!r}: validity shape {m.shape} != ({len(v)},)")
      object.__setattr__(self, "valid", m)
    elif self.valid is not None:
      raise ValueError(
        f"track {self.spec.name!r} does not declare validity but a mask "
        "was given")
    v.setflags(write=False)
    if self.valid is not None:
      self.valid.setflags(write=False)

  def __len__(self) -> int:
    return int(self.values.shape[0])

  def with_values(self, values: np.ndarray,
                  valid: np.ndarray | None = None) -> "Track":
    return Track(self.spec, values, valid if valid is not None else self.valid)


def _convert_array(values: np.ndarray, src: Unit, dst: Unit,
                   name: str) -> np.ndarray:
  if src == dst:
    return values
  factor = _CONVERSIONS.get((src, dst))
  if factor is None:
    raise UnitMismatch(
      f"track {name!r}: no mechanical conversion {src.value} -> {dst.value}")
  return (values.astype(np.float64) * factor).astype(values.dtype
          if np.issubdtype(values.dtype, np.floating) else np.float32)


def convert_unit(track: Track, target: Unit | ChannelUnits) -> Track:
  """Convert a track to ``target`` units through the registry.

  Returns the track unchanged if units already match; raises
  :class:`UnitMismatch` when no lossless scalar conversion exists. For
  :class:`ChannelUnits`, source and target must declare identical channel
  ranges — the conversion is per range."""
  src, name = track.spec.unit, track.spec.name
  if src == target:
    return track

  if isinstance(src, Unit) and isinstance(target, Unit):
    out = _convert_array(track.values, src, target, name)
  elif isinstance(src, ChannelUnits) and isinstance(target, ChannelUnits):
    if tuple(r for r, _ in src.groups) != tuple(r for r, _ in target.groups):
      raise UnitMismatch(
        f"track {name!r}: channel ranges differ between source and target")
    values = track.values
    if not np.issubdtype(values.dtype, np.floating):
      values = values.astype(np.float32)
    out = np.array(values, copy=True)
    for ((a, b), s_unit), (_, t_unit) in zip(src.groups, target.groups):
      out[..., a:b] = _convert_array(values[..., a:b], s_unit, t_unit, name)
  else:
    raise UnitMismatch(
      f"track {name!r}: cannot convert between scalar and per-channel units")

  return Track(replace(track.spec, unit=target, dtype=out.dtype),
               out, track.valid)


def check_frame(track: Track, expected: Frame, node: str) -> None:
  """Frame soundness (spec §10.2): mismatches are always errors."""
  if track.spec.frame != expected:
    raise FrameMismatch(
      f"node {node!r}: input {track.spec.name!r} is in frame "
      f"{track.spec.frame.value!r}, expected {expected.value!r} — frame "
      "changes must be an explicit pipeline node")
