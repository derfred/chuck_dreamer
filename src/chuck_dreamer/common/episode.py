"""The in-memory episode container that flows through the importer / sim
collect pipelines into the writers.

An :class:`Episode` is a dict-like of per-step arrays *plus* a small bundle of
per-field metadata: how each field should be serialized (:class:`FieldKind`) and
whether it should be persisted at all (``persist``). This replaces the bare
``dict[str, np.ndarray]`` that used to flow around, and folds two things that
used to live elsewhere back into the episode where they belong:

  * **Non-persisted scratch** — e.g. the raw per-frame SAM2 masks handed from the
    segmentation stage to the object-pose stage. Previously stashed on the
    ``RunContext`` (and hand-plumbed across the parallel-import process boundary
    as a special ``_Task.masks`` field). Now it is just an ``Episode`` field with
    ``persist=False``: it rides along with the episode automatically (including
    across the process boundary, since the episode is the queue payload) and the
    writers skip it.
  * **Per-field persistence policy** — previously hardcoded in each writer
    (``_REQUIRED_KEYS``, the ``segmentation_*`` scan, the Rerun-only mesh-overlay
    special-case). Now each field declares its own :class:`FieldKind`, and the
    writers dispatch on it. Adding a persisted field no longer needs a writer
    edit.

``Episode`` implements the :class:`~collections.abc.Mapping` protocol returning
*values*, so legacy call sites (``episode["x"]``, ``episode.get("x")``, ``"x" in
episode``, ``episode.items()``) keep working unchanged. ``episode["x"] = arr``
infers the field's kind/persist from its name via :data:`FIELD_REGISTRY`; use
:meth:`Episode.set` to override.

The container holds only plain arrays + a tiny metadata mapping, so it pickles
cleanly for the ``spawn``-based parallel importer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator


class FieldKind(str, Enum):
  """How a field is serialized by the writers. The string values are stable —
  they're how a field's disposition is named in logs / debugging, not on disk.

  The *kind* determines the writer dispatch, reproducing exactly the layout the
  hardcoded writers produced before this refactor:

    * ``SCALAR``   — a per-step ``(T, ...)`` numeric array. HDF5: one dataset
      named by the field (or its ``_HDF5_DATASET`` alias). Rerun: per-step
      ``rr.Scalars`` on an entity named by the field.
    * ``IMAGE``    — the ``(T, H, W, 3)`` RGB stack. HDF5: ``images`` dataset.
      Rerun: a single encoded ``camera/video`` blob + per-step frame refs.
    * ``ACTION``   — ``joint_action`` / ``ee_action``; like ``SCALAR`` but the
      writers require *at least one* action field present.
    * ``MASK``     — a ``(T, H, W)`` (or ``(T, K, H, W)``) boolean segmentation
      stack. HDF5: under the ``segmentation/`` group (name with the
      ``segmentation_`` prefix stripped). Rerun: per-step tinted PNG under
      ``camera/seg/<name>``.
    * ``OVERLAY``  — the fitted-mesh silhouette. **Rerun-only** (HDF5 drops it,
      matching prior behaviour): per-step tinted PNG under
      ``camera/mesh_overlay``.
    * ``SCRATCH``  — never persisted; an inter-stage handoff (e.g. raw masks).
  """
  SCALAR = "scalar"
  IMAGE = "image"
  ACTION = "action"
  MASK = "mask"
  OVERLAY = "overlay"
  SCRATCH = "scratch"


@dataclass
class Field:
  """One episode field: its array, how to serialize it, and whether to persist.

  ``persist`` is redundant with ``kind == SCRATCH`` for the scratch case but is
  kept explicit so a normally-persisted kind can be suppressed for one field
  without inventing a new kind."""
  value: Any
  kind: FieldKind
  persist: bool = True


# Default (kind, persist) for every field name the pipelines produce. A bare
# ``episode["name"] = arr`` looks the name up here; unknown names default to a
# persisted SCALAR (the common case for a new per-step numeric signal).
#
# This table is the single source of truth for field disposition — it encodes
# exactly what the old hardcoded writers did, so on-disk output is unchanged.
_SEG_PREFIX = "segmentation_"

FIELD_REGISTRY: dict[str, tuple[FieldKind, bool]] = {
  # Core persisted per-step signals.
  "image":        (FieldKind.IMAGE,  True),
  "reward":       (FieldKind.SCALAR, True),
  "timestamp":    (FieldKind.SCALAR, True),
  "joint_qpos":   (FieldKind.SCALAR, True),
  "ee_pos":       (FieldKind.SCALAR, True),
  "ee_quat":      (FieldKind.SCALAR, True),
  "object_xy":    (FieldKind.SCALAR, True),
  "object_uv":    (FieldKind.SCALAR, True),
  "object_gap_too_long": (FieldKind.SCALAR, True),
  # Actions — at least one required by the writers.
  "joint_action": (FieldKind.ACTION, True),
  "ee_action":    (FieldKind.ACTION, True),
  # Segmentation masks (also matched by the ``segmentation_`` prefix below).
  "segmentation_target":     (FieldKind.MASK, True),
  "segmentation_goal":       (FieldKind.MASK, True),
  "segmentation_arm":        (FieldKind.MASK, True),
  "segmentation_background": (FieldKind.MASK, True),
  "segmentation_clutter":    (FieldKind.MASK, True),
  # Fitted-mesh silhouette: Rerun-only overlay.
  "object_mesh_overlay":     (FieldKind.OVERLAY, True),
  # Non-persisted scratch carried on the episode in memory but never written:
  #   * object_masks — segment:object → object_pose handoff (importer)
  #   * step_info    — per-step StepInfo columns the replay buffer reads at
  #                    sample time; not part of a recording's on-disk format
  "object_masks":            (FieldKind.SCRATCH, False),
  "step_info":               (FieldKind.SCRATCH, False),
}


def default_disposition(name: str) -> tuple[FieldKind, bool]:
  """``(kind, persist)`` for a field ``name`` from :data:`FIELD_REGISTRY`.

  Unknown ``segmentation_*`` names are treated as persisted masks (so a new
  named mask just works); any other unknown name is a persisted scalar."""
  if name in FIELD_REGISTRY:
    return FIELD_REGISTRY[name]
  if name.startswith(_SEG_PREFIX):
    return (FieldKind.MASK, True)
  return (FieldKind.SCALAR, True)


class Episode(Mapping):
  """A field-attributed, dict-like episode container (see module docstring).

  Mapping access returns field *values*, so it is a drop-in for the old
  ``dict[str, np.ndarray]`` everywhere a value is read. Writes via ``[]`` infer
  disposition from the field name; :meth:`set` overrides it explicitly.
  """

  __slots__ = ("_fields",)

  def __init__(self, fields: dict[str, Field] | None = None) -> None:
    self._fields: dict[str, Field] = dict(fields) if fields else {}

  # ---- construction --------------------------------------------------------
  @classmethod
  def from_arrays(cls, arrays: Mapping[str, Any]) -> "Episode":
    """Build an episode from a plain ``{name: array}`` mapping, inferring each
    field's disposition from its name. The bridge for producers that already
    have a dict (and for tests)."""
    ep = cls()
    for name, value in arrays.items():
      ep[name] = value
    return ep

  # ---- field-level API -----------------------------------------------------
  def set(self, name: str, value: Any, *,
          kind: FieldKind | None = None, persist: bool | None = None) -> None:
    """Set a field, optionally overriding its inferred ``kind`` / ``persist``.

    With both left ``None`` this matches ``episode[name] = value``. Pass
    ``persist=False`` (or ``kind=FieldKind.SCRATCH``) for an inter-stage handoff
    that must not reach disk."""
    d_kind, d_persist = default_disposition(name)
    if kind is None:
      kind = d_kind
    if persist is None:
      persist = False if kind is FieldKind.SCRATCH else d_persist
    self._fields[name] = Field(value, kind, persist)

  def field(self, name: str) -> Field:
    """The :class:`Field` (value + metadata) for ``name``."""
    return self._fields[name]

  def fields(self) -> Iterator[tuple[str, Field]]:
    """Iterate ``(name, Field)`` — for the writers, which dispatch on kind."""
    return iter(self._fields.items())

  def persisted(self) -> Iterator[tuple[str, Field]]:
    """Iterate ``(name, Field)`` for the fields that should be written."""
    return ((n, f) for n, f in self._fields.items() if f.persist)

  # ---- Mapping protocol (returns values) -----------------------------------
  def __getitem__(self, name: str) -> Any:
    return self._fields[name].value

  def __setitem__(self, name: str, value: Any) -> None:
    self.set(name, value)

  def __delitem__(self, name: str) -> None:
    del self._fields[name]

  def __contains__(self, name: object) -> bool:
    return name in self._fields

  def __iter__(self) -> Iterator[str]:
    return iter(self._fields)

  def __len__(self) -> int:
    return len(self._fields)

  def get(self, name: str, default: Any = None) -> Any:
    f = self._fields.get(name)
    return f.value if f is not None else default

  # ---- pickle (spawn-safe; __slots__ has no __dict__) ----------------------
  def __getstate__(self) -> dict[str, Field]:
    return self._fields

  def __setstate__(self, state: dict[str, Field]) -> None:
    self._fields = state

  def __repr__(self) -> str:
    parts = ", ".join(
      f"{n}:{f.kind.value}{'' if f.persist else '*'}" for n, f in self._fields.items())
    return f"Episode({parts})"
