"""TrackSet + NodeView: the execution layer of the pipeline harness.

Implements the layered view from ``docs/trainer/harness_design.md``: a
:class:`TrackSet` resolves track names against (1) tracks staged by upstream
nodes in this run, then (2) the episode's base fields (the legacy
:class:`~chuck_dreamer.common.episode.Episode` container, which doubles as
the flush target the writers consume). Native nodes see only a
:class:`NodeView` restricted to their declaration — an undeclared read is an
error, and unit conversion happens at the boundary through the single
registry in :mod:`chuck_dreamer.common.tracks`.

Coarser-scoped inputs (assets, annotation artifacts, calibration) resolve
through :meth:`NodeView.artifact` and the provider registry below, keyed by
the *artifact names the spec declares* — so a node's ``InputDecl`` list is
the complete statement of what it consumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from chuck_dreamer.common.tracks import (
  ChannelUnits,
  Frame,
  Persist,
  Track,
  TrackSpec,
  Unit,
  check_frame,
  convert_unit,
)
from chuck_dreamer.store import Scope, for_episode

from .context import RunContext
from .harness import InputDecl, Node

if TYPE_CHECKING:
  from chuck_dreamer.common.episode import Episode


class MissingTrack(KeyError):
  """A node asked for a track that is neither staged nor a base field."""


class UndeclaredInput(RuntimeError):
  """A native node read a track/artifact it did not declare — declarations
  are the provenance contract, so this is a hard error, not a warning."""


@dataclass(frozen=True)
class EpisodeScope:
  dataset_id: str
  episode_index: int

  def store_scope(self) -> Scope:
    return for_episode(self.dataset_id, self.episode_index)


# Declared specs for the base tracks the dataset adapter provides. Base
# fields not listed here can still be read (their spec is inferred from the
# array with frame/unit unknown); listing a track makes frame/unit conversion
# at node boundaries possible. Positioning joints arrive in degrees, the
# gripper channel is unitless — the historical deg/rad bug lives exactly
# here.
def _joint_units(n_joints: int) -> ChannelUnits:
  n_pos = min(5, n_joints)
  groups: dict[tuple[int, int], Unit] = {(0, n_pos): Unit.DEG}
  if n_joints > n_pos:
    groups[(n_pos, n_joints)] = Unit.UNITLESS
  return ChannelUnits(groups)


# Base fields with a fixed (frame, unit) declaration. Coordinate tracks
# carry an explicit frame suffix (`_arm` / `_table`; `_uv` implies the image
# plane) and are persisted in millimetres — a track only crosses frames
# through an explicit node (the ``table_frame`` module, consuming
# ``table_to_arm``). These declarations matter in the parallel importer:
# producer-lane outputs cross the process boundary through the episode
# container and are re-typed here for the worker lane.
_POSE_MM = ChannelUnits({(0, 3): Unit.MM, (3, 7): Unit.UNITLESS})
_BASE_SPECS: dict[str, tuple[Frame, Unit | ChannelUnits]] = {
  "timestamp": (Frame.NONE,  Unit.S),
  "image":     (Frame.IMAGE, Unit.UNITLESS),
  "ee_pos_arm":      (Frame.ARM,   Unit.MM),
  "ee_quat_arm":     (Frame.ARM,   Unit.UNITLESS),
  "ee_action_arm":   (Frame.ARM,   _POSE_MM),
  "ee_pos_table":    (Frame.TABLE, Unit.MM),
  "ee_quat_table":   (Frame.TABLE, Unit.UNITLESS),
  "ee_action_table": (Frame.TABLE, _POSE_MM),
  "obj_pos_table":   (Frame.TABLE, Unit.MM),
  "obj_pos_arm":     (Frame.ARM,   Unit.MM),
  "obj_uv":          (Frame.IMAGE, Unit.PX),
  # scratch handoff from segmentation to obj_uv / object_pose
  "object_masks": (Frame.IMAGE, Unit.UNITLESS),
}


def base_track_spec(name: str, values: np.ndarray,
                    validity: bool = False) -> TrackSpec:
  """Spec for a base episode field, with declared frame/unit where known."""
  per_frame = tuple(values.shape[1:])
  if name == "joint_qpos":
    return TrackSpec(name, values.dtype, per_frame,
                     Frame.NONE, _joint_units(per_frame[0] if per_frame else 0),
                     validity=validity)
  declared = _BASE_SPECS.get(name)
  if declared is not None:
    return TrackSpec(name, values.dtype, per_frame, declared[0], declared[1],
                     validity=validity)
  return TrackSpec(name, values.dtype, per_frame, Frame.NONE, Unit.UNITLESS,
                   validity=validity)


# ---------------------------------------------------------------------------
# Artifact providers
# ---------------------------------------------------------------------------
def _sha_of(obj: Any) -> str:
  import hashlib
  import json
  blob = json.dumps(obj, sort_keys=True).encode()
  return "sha256:" + hashlib.sha256(blob).hexdigest()


def _file_sha(path: Any) -> str | None:
  from pathlib import Path
  import hashlib
  p = Path(path)
  if not p.exists():
    return None
  return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ResolvedArtifact:
  """A provider result: the artifact value plus the content identity the
  harness records in ``input_versions``. ``version=None`` means the input
  can't be identified — the runner then disables store caching for any node
  consuming it (never a silently-wrong cache key)."""
  value: Any
  version: str | None


# Coarser-than-EPISODE inputs, keyed by the artifact names the spec's node
# declarations use. Providers resolve through the RunContext (which owns the
# config and the artifact store), so both the serial importer and every
# parallel producer/worker resolve identically.
_ARTIFACT_PROVIDERS: dict[str, Callable[[RunContext, EpisodeScope],
                                        ResolvedArtifact]] = {
  # STATIC assets
  "fk_model":    lambda ctx, scope: _resolve_fk(ctx),
  "object_mesh": lambda ctx, scope: _load_mesh(ctx),
  # annotation artifacts
  "object_prompts": lambda ctx, scope: ResolvedArtifact(
    *ctx.resolve_keyframe_prompts(scope.episode_index)),
  # calibration: store-only, resolved through dataset_config
  "intrinsics": lambda ctx, scope: ResolvedArtifact(*ctx.resolve_intrinsics()),
  "extrinsics": lambda ctx, scope: ResolvedArtifact(*ctx.resolve_extrinsics()),
  # cached DATASET-scoped transform produced by extract-table-to-arm
  "table_to_arm": lambda ctx, scope: ResolvedArtifact(
    *ctx.resolve_table_to_arm()),
  # dataset component: this episode's video slice (MP4 path + decode window)
  "dataset.episodes": lambda ctx, scope: _resolve_slice(ctx, scope),
  # STATIC asset identity, pinned by assets/manifest.json
  "sam2_weights": lambda ctx, scope: _resolve_sam2(),
}


def _resolve_fk(ctx: RunContext) -> ResolvedArtifact:
  from chuck_dreamer.common import FK_MODEL_PATH
  return ResolvedArtifact(ctx.fk(), _file_sha(FK_MODEL_PATH))


def _load_mesh(ctx: RunContext) -> ResolvedArtifact:
  from pathlib import Path

  from chuck_dreamer.perception.mesh import load_mesh
  mesh_path = Path(ctx.ol_cfg.mesh_path)
  return ResolvedArtifact(load_mesh(mesh_path), _file_sha(mesh_path))


def _resolve_slice(ctx: RunContext, scope: EpisodeScope) -> ResolvedArtifact:
  """The episode's video slice; its version is a content proxy (file name +
  size + decode window) that avoids hashing multi-GB MP4s."""
  from pathlib import Path
  sl = ctx.slice_for(scope.episode_index)
  if sl.video_path is None:
    return ResolvedArtifact(sl, None)
  video_p = Path(sl.video_path)
  if not video_p.exists():
    return ResolvedArtifact(sl, None)
  return ResolvedArtifact(sl, _sha_of({
    "name": video_p.name, "size": video_p.stat().st_size,
    "from_ts": float(sl.video_from_ts), "to_ts": float(sl.video_to_ts)}))


def _resolve_sam2() -> ResolvedArtifact:
  """The SAM2 asset's pinned identity (``{"hf_repo", "revision"}``). The
  revision pin is provenance-level: it keys the mask cache, but the loader
  resolves from the local HF cache (SAM2 exposes no revision kwarg)."""
  from chuck_dreamer.common import asset_manifest_entry
  entry = asset_manifest_entry("sam2_weights")
  return ResolvedArtifact(
    entry, _sha_of({**entry, "postproc": "largest_cc/v1"}))


class TrackSet:
  """One episode's tracks: staged native outputs layered over the base
  :class:`Episode` container.

  The base container is also the flush target — committing a track both
  stages the typed :class:`Track` (with its validity mask, for downstream
  native nodes) and materializes its values as an episode field (for the
  writers and legacy stages). Until every stage is native, the two layers
  deliberately share storage rather than diverge.
  """

  def __init__(self, scope: EpisodeScope, episode: "Episode",
               metadata: dict[str, Any]) -> None:
    self.scope    = scope
    self.episode  = episode
    self.metadata = metadata
    self._staged: dict[str, Track] = {}
    self._versions: dict[str, str] = {}

  def get_track(self, name: str) -> Track:
    staged = self._staged.get(name)
    if staged is not None:
      return staged
    values = self.episode.get(name)
    if values is None:
      raise MissingTrack(
        f"track {name!r} is neither staged nor a base field of episode "
        f"{self.scope.dataset_id}#{self.scope.episode_index}")
    values = np.asarray(values)
    # A `<name>.valid` sibling field marks a validity-carrying track that
    # crossed a process boundary through the episode container (see commit).
    valid = self.episode.get(f"{name}.valid")
    if valid is not None:
      valid = np.asarray(valid, dtype=bool)
    return Track(base_track_spec(name, values, validity=valid is not None),
                 values, valid)

  def has(self, name: str) -> bool:
    return name in self._staged or self.episode.get(name) is not None

  def track_version(self, name: str) -> str:
    """Content hash of a track (values + validity), cached per name. Staged
    tracks are immutable and base fields don't change after assembly, so one
    hash per name per episode is sound."""
    cached = self._versions.get(name)
    if cached is not None:
      return cached
    import hashlib
    track = self.get_track(name)
    h = hashlib.sha256()
    v = np.ascontiguousarray(track.values)
    h.update(str(v.dtype).encode())
    h.update(str(v.shape).encode())
    h.update(v.tobytes())
    if track.valid is not None:
      h.update(np.ascontiguousarray(track.valid).tobytes())
    version = "sha256:" + h.hexdigest()
    self._versions[name] = version
    return version

  def commit(self, track: Track) -> None:
    self._staged[track.spec.name] = track
    self._versions.pop(track.spec.name, None)
    # Materialize into the episode container: values for the writers/pickling,
    # and the validity mask as a `<name>.valid` sibling so it survives the
    # parallel importer's process handoff (the staged dict does not cross
    # pickling; the episode does). The sibling persists for trajectory-bound
    # tracks — that is how validity reaches the on-disk formats — and stays
    # scratch for EPHEMERAL/CACHED tracks, which never reach the writers.
    # np arrays committed as tracks are write-protected; the writer gets a
    # view, not a copy.
    self.episode[track.spec.name] = track.values
    if track.valid is not None:
      persist = track.spec.persist in (Persist.TRAJECTORY, Persist.DIAGNOSTIC)
      self.episode.set(f"{track.spec.name}.valid", track.valid,
                       persist=persist)

  def release(self, name: str) -> None:
    """Drop a staged EPHEMERAL track after its last declared consumer ran.
    Tracks with any other persistence policy are kept — they are either in
    the store already (CACHED) or the writers' input (TRAJECTORY /
    DIAGNOSTIC)."""
    staged = self._staged.get(name)
    if staged is None or staged.spec.persist is not Persist.EPHEMERAL:
      return
    del self._staged[name]
    self._versions.pop(name, None)
    if name in self.episode:
      del self.episode[name]
    if f"{name}.valid" in self.episode:
      del self.episode[f"{name}.valid"]


class NodeView:
  """What a native node sees: exactly its declared inputs, frame-checked and
  unit-converted at the boundary.

  Every read records the input's content identity in :attr:`input_versions`
  — the provenance-record map is assembled as a side effect of resolution,
  never reconstructed afterwards (harness_design.md). A ``None`` version
  means the input can't be identified; the runner then skips store caching
  for this node's outputs."""

  def __init__(self, ts: TrackSet, node: Node) -> None:
    self._ts = ts
    self._node = node
    self._decls: dict[str, InputDecl] = {d.name: d for d in node.inputs}
    self.input_versions: dict[str, str | None] = {}

  @property
  def scope(self) -> EpisodeScope:
    return self._ts.scope

  def track(self, name: str) -> Track:
    decl = self._decls.get(name)
    if decl is None:
      raise UndeclaredInput(
        f"node {self._node.name!r} read undeclared input {name!r}; "
        f"declared: {sorted(self._decls)}")
    track = self._ts.get_track(name)
    self.input_versions[name] = self._ts.track_version(name)
    if decl.frame is not None:
      check_frame(track, decl.frame, node=self._node.name)
    if decl.unit is not None:
      track = convert_unit(track, decl.unit)
    return track

  def artifact(self, name: str) -> Any:
    """Coarser-scoped declared inputs: assets, annotation artifacts,
    calibration, and the episode's dataset slice, resolved through the
    provider registry (which reads the node's bound RunContext)."""
    if name not in self._decls:
      raise UndeclaredInput(
        f"node {self._node.name!r} read undeclared artifact {name!r}")
    provider = _ARTIFACT_PROVIDERS.get(name)
    if provider is None:
      raise MissingTrack(
        f"no artifact provider for {name!r} "
        f"(known: {sorted(_ARTIFACT_PROVIDERS)})")
    ctx = getattr(self._node, "ctx", None)
    if ctx is None:
      raise MissingTrack(
        f"node {self._node.name!r} has no bound RunContext to resolve "
        f"artifact {name!r}")
    resolved = provider(ctx, self._ts.scope)
    self.input_versions[name] = resolved.version
    return resolved.value

  def resolve_declared(self) -> None:
    """Resolve every declared input once, populating :attr:`input_versions`
    without running the node — the runner's cache-key pass. Track reads
    skip the decl's unit conversion (only identity is needed here)."""
    for name in self._decls:
      if name in _ARTIFACT_PROVIDERS:
        self.artifact(name)
      else:
        self._ts.get_track(name)
        self.input_versions[name] = self._ts.track_version(name)


def _runtime_spec(decl: TrackSpec, values: np.ndarray) -> TrackSpec:
  """A declared output spec finalized against a runtime array (declared
  per-frame shapes are templates when they depend on the dataset)."""
  from dataclasses import replace
  return replace(decl, dtype=values.dtype, shape=tuple(values.shape[1:]))


def _cached_outputs(node: Node, ts: TrackSet,
                    versions: dict[str, str | None],
                    store: Any) -> list[Track] | None:
  """All the node's CACHED outputs loaded from the store, or ``None`` when
  any is missing/stale (or any input version is unidentifiable)."""
  if any(v is None for v in versions.values()):
    return None
  scope = ts.scope.store_scope()
  tracks: list[Track] = []
  for spec in node.outputs:
    if not store.has(spec.name, scope):
      return None
    payload, record = store.get(spec.name, scope)
    if record.input_versions != versions:
      print(f"[{node.name}] ep{ts.scope.episode_index}: cached "
            f"{spec.name} stale (inputs changed); recomputing.")
      return None
    values = payload["values"]
    valid = payload["valid"].astype(bool) if spec.validity else None
    tracks.append(Track(_runtime_spec(spec, values), values, valid))
  return tracks


def _store_cached_output(node: Node, ts: TrackSet, track: Track,
                         versions: dict[str, str | None], store: Any) -> None:
  if any(v is None for v in versions.values()):
    return
  from chuck_dreamer.store import ProvenanceRecord, git_sha
  payload: dict[str, np.ndarray] = {"values": np.asarray(track.values)}
  if track.valid is not None:
    payload["valid"] = np.asarray(track.valid)
  store.put(track.spec.name, ts.scope.store_scope(), payload,
            ProvenanceRecord(
              artifact=track.spec.name, scope=ts.scope.store_scope(),
              node=node.name,
              input_versions={k: v for k, v in versions.items()
                              if v is not None},
              code_version=git_sha()))


def run_native_node(node: Node, ts: TrackSet) -> None:
  """Resolve, run, validate, and commit one native node.

  ``Persist.CACHED`` handling is generic runner behavior: when *all* of a
  node's outputs are CACHED and the store holds them keyed by the same
  ``input_versions``, the node body is skipped entirely; on a miss the node
  runs and its CACHED outputs are put back with a provenance record. Every
  node's resolved ``input_versions`` land in ``ts.metadata["provenance"]``
  so the trajectory file records how each persisted track was produced."""
  view = NodeView(ts, node)
  ctx = getattr(node, "ctx", None)
  store = getattr(ctx, "store", None)

  all_cached = bool(node.outputs) and all(
    s.persist is Persist.CACHED for s in node.outputs)
  outputs: list[Track] | None = None
  from_store = False
  if all_cached and store is not None:
    view.resolve_declared()
    outputs = _cached_outputs(node, ts, view.input_versions, store)
    from_store = outputs is not None
    if from_store:
      print(f"[{node.name}] ep{ts.scope.episode_index}: outputs loaded "
            "from store (node skipped).")

  if outputs is None:
    outputs = node.run(view)

  declared = {spec.name: spec for spec in node.outputs}
  for track in outputs:
    spec = declared.get(track.spec.name)
    if spec is None:
      raise UndeclaredInput(
        f"node {node.name!r} produced undeclared output {track.spec.name!r}")
    if (track.spec.frame, track.spec.validity) != (spec.frame, spec.validity):
      raise ValueError(
        f"node {node.name!r} output {track.spec.name!r} does not match its "
        f"declared spec (frame/validity)")
    ts.commit(track)
    if (not from_store and store is not None
        and spec.persist is Persist.CACHED):
      _store_cached_output(node, ts, track, view.input_versions, store)
  missing = declared.keys() - {t.spec.name for t in outputs}
  if missing:
    raise ValueError(
      f"node {node.name!r} did not produce declared output(s) "
      f"{sorted(missing)}")

  provenance = ts.metadata.setdefault("provenance", {})
  record = {
    "node": node.name,
    "input_versions": {k: v for k, v in view.input_versions.items()
                       if v is not None},
  }
  for track in outputs:
    provenance[track.spec.name] = record


def run_episode(nodes: list[Any], ts: TrackSet) -> None:
  """Run one episode's node list, in order, against ``ts``.

  Every node goes through :func:`run_native_node`: declaration enforcement,
  frame/unit checks at the boundary, output validation, staged commit, and
  generic CACHED-output handling. Order is the caller's responsibility
  (:meth:`Run.pipeline` builds it).

  EPHEMERAL tracks are released after their last declared consumer *in this
  node list* — the consumer count is per lane, so anything that must cross
  the parallel importer's process boundary uses a non-EPHEMERAL policy.
  """
  consumers: dict[str, int] = {}
  for node in nodes:
    for decl in node.inputs:
      if decl.name not in _ARTIFACT_PROVIDERS:
        consumers[decl.name] = consumers.get(decl.name, 0) + 1

  for node in nodes:
    run_native_node(node, ts)
    for decl in node.inputs:
      if decl.name in _ARTIFACT_PROVIDERS:
        continue
      consumers[decl.name] -= 1
      if consumers[decl.name] == 0:
        ts.release(decl.name)
