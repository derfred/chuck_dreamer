"""The artifact store: addressed payloads + mandatory provenance records.

Implements ``docs/trainer/artifact_store.md``: every artifact is addressed by
``(type, scope)``, stored as one payload file plus a ``*.prov.json`` sidecar,
written atomically. A payload without a record is not in the store — ``put``
takes both, ``get`` returns both.

Staleness / graph-walking helpers land with the harness (they need the static
node graph); this module is the persistence layer only.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class MissingArtifact(FileNotFoundError):
  """Requested artifact is not in the store at the resolved scope."""


class CorruptArtifact(RuntimeError):
  """Payload bytes no longer match the record's ``payload_hash``, or the
  sidecar is unreadable."""


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scope:
  """One of the spec's scope keys (§2). Exactly the fields for its level are
  set: CAMERA → ``camera``; DATASET → ``dataset``; EPISODE → ``dataset`` +
  ``episode``. STATIC assets are referenced, not stored, so they have no
  Scope."""
  camera: str | None = None
  dataset: str | None = None
  episode: int | None = None

  @property
  def level(self) -> str:
    if self.camera is not None:
      return "camera"
    if self.episode is not None:
      return "episode"
    if self.dataset is not None:
      return "dataset"
    raise ValueError("empty scope")

  def to_json(self) -> dict[str, Any]:
    return {k: v for k, v in (("camera", self.camera),
                              ("dataset", self.dataset),
                              ("episode", self.episode)) if v is not None}

  @classmethod
  def from_json(cls, blob: dict[str, Any]) -> "Scope":
    return cls(camera=blob.get("camera"), dataset=blob.get("dataset"),
               episode=blob.get("episode"))


def for_camera(camera_id: str) -> Scope:
  return Scope(camera=camera_id)


def for_dataset(dataset_id: str) -> Scope:
  return Scope(dataset=dataset_id)


def for_episode(dataset_id: str, episode_id: int) -> Scope:
  return Scope(dataset=dataset_id, episode=int(episode_id))


# ---------------------------------------------------------------------------
# Artifact type registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArtifactTypeSpec:
  name: str
  scope_level: str          # "camera" | "dataset" | "episode"
  codec: str                # "json" | "npz"


# The spec's §5/§6 tables. ``mat_annotation`` is not a separate artifact in
# the target model (the clicks live inside ``extrinsics``), but the migrator
# reads the legacy file, so the merged payload is still one JSON.
REGISTRY: dict[str, ArtifactTypeSpec] = {s.name: s for s in (
  ArtifactTypeSpec("intrinsics",     "camera",  "json"),
  ArtifactTypeSpec("dataset_config", "dataset", "json"),
  ArtifactTypeSpec("extrinsics",     "dataset", "json"),
  ArtifactTypeSpec("table_to_arm",   "dataset", "json"),
  ArtifactTypeSpec("object_prompts", "episode", "json"),
  ArtifactTypeSpec("object_masks",   "episode", "npz"),
)}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def git_sha(repo_root: Path | None = None) -> str:
  """Best-effort current commit sha; ``"unknown"`` outside a repo."""
  try:
    out = subprocess.run(
      ["git", "rev-parse", "--short", "HEAD"],
      cwd=repo_root, capture_output=True, text=True, timeout=5)
    return out.stdout.strip() or "unknown"
  except Exception:  # noqa: BLE001 - provenance must never block a write
    return "unknown"


@dataclass
class ProvenanceRecord:
  artifact: str
  scope: Scope
  node: str                                   # producing node, or "annotation:<tool>"
  input_versions: dict[str, str] = field(default_factory=dict)
  advisory_versions: dict[str, str] = field(default_factory=dict)
  code_version: str = "unknown"
  schema_version: int = 1
  payload_hash: str = ""                      # stamped by ArtifactStore.put
  created_at: str = ""                        # stamped by ArtifactStore.put

  def to_json(self) -> dict[str, Any]:
    return {
      "artifact":          self.artifact,
      "scope":             self.scope.to_json(),
      "node":              self.node,
      "input_versions":    dict(self.input_versions),
      "advisory_versions": dict(self.advisory_versions),
      "code_version":      self.code_version,
      "schema_version":    self.schema_version,
      "payload_hash":      self.payload_hash,
      "created_at":        self.created_at,
    }

  @classmethod
  def from_json(cls, blob: dict[str, Any]) -> "ProvenanceRecord":
    return cls(
      artifact          = blob["artifact"],
      scope             = Scope.from_json(blob["scope"]),
      node              = blob["node"],
      input_versions    = dict(blob.get("input_versions", {})),
      advisory_versions = dict(blob.get("advisory_versions", {})),
      code_version      = blob.get("code_version", "unknown"),
      schema_version    = int(blob.get("schema_version", 1)),
      payload_hash      = blob.get("payload_hash", ""),
      created_at        = blob.get("created_at", ""),
    )


def annotation_record(artifact: str, scope: Scope, tool: str,
                      advisory: dict[str, str] | None = None) -> ProvenanceRecord:
  """Record for a human-supplied artifact: provenance terminates here
  (spec §9.3) — no input versions, optional advisory versions."""
  return ProvenanceRecord(
    artifact=artifact, scope=scope, node=f"annotation:{tool}",
    advisory_versions=dict(advisory or {}), code_version=git_sha())


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
def _sha256(data: bytes) -> str:
  return "sha256:" + hashlib.sha256(data).hexdigest()


def _slug(dataset_id: str) -> str:
  from .calibration_cache import dataset_slug
  return dataset_slug(dataset_id)


@dataclass(frozen=True)
class ArtifactInfo:
  artifact: str
  scope: Scope
  path: Path
  created_at: str
  node: str


class ArtifactStore:
  """Filesystem artifact store (see module docstring)."""

  def __init__(self, root: Path | str,
               registry: dict[str, ArtifactTypeSpec] | None = None) -> None:
    self.root = Path(root)
    self.registry = registry if registry is not None else REGISTRY

  # ---- paths ---------------------------------------------------------------
  def _dir_for(self, scope: Scope) -> Path:
    if scope.level == "camera":
      return self.root / "camera" / scope.camera          # type: ignore[operator]
    assert scope.dataset is not None
    d = self.root / "dataset" / _slug(scope.dataset)
    if scope.level == "episode":
      d = d / "episode" / f"{scope.episode:05d}"
    return d

  def _paths(self, artifact: str, scope: Scope) -> tuple[Path, Path]:
    spec = self.registry.get(artifact)
    if spec is None:
      raise KeyError(f"unknown artifact type {artifact!r}; "
                     f"known: {sorted(self.registry)}")
    if scope.level != spec.scope_level:
      raise ValueError(
        f"{artifact} is {spec.scope_level.upper()}-scoped; got a "
        f"{scope.level.upper()} scope key")
    ext = "json" if spec.codec == "json" else "npz"
    d = self._dir_for(scope)
    return d / f"{artifact}.{ext}", d / f"{artifact}.prov.json"

  # ---- codecs ----------------------------------------------------------------
  def _encode(self, artifact: str, payload: Any) -> bytes:
    if self.registry[artifact].codec == "json":
      return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    return buf.getvalue()

  def _decode(self, artifact: str, data: bytes) -> Any:
    if self.registry[artifact].codec == "json":
      return json.loads(data.decode())
    import io
    with np.load(io.BytesIO(data)) as npz:
      return {k: npz[k] for k in npz.files}

  # ---- primitives ------------------------------------------------------------
  def has(self, artifact: str, scope: Scope) -> bool:
    payload_p, prov_p = self._paths(artifact, scope)
    return payload_p.exists() and prov_p.exists()

  def get(self, artifact: str, scope: Scope) -> tuple[Any, ProvenanceRecord]:
    payload_p, prov_p = self._paths(artifact, scope)
    if not payload_p.exists() or not prov_p.exists():
      raise MissingArtifact(
        f"{artifact} not in store for scope {scope.to_json()} "
        f"(expected {payload_p})")
    data = payload_p.read_bytes()
    try:
      record = ProvenanceRecord.from_json(json.loads(prov_p.read_text()))
    except Exception as e:  # noqa: BLE001 - surface as corruption, not a crash
      raise CorruptArtifact(f"unreadable provenance sidecar {prov_p}: {e}")
    if record.payload_hash and record.payload_hash != _sha256(data):
      raise CorruptArtifact(
        f"{payload_p} was modified outside the store "
        f"(payload hash no longer matches its provenance record)")
    return self._decode(artifact, data), record

  def put(self, artifact: str, scope: Scope, payload: Any,
          provenance: ProvenanceRecord) -> Path:
    payload_p, prov_p = self._paths(artifact, scope)
    data = self._encode(artifact, payload)
    provenance.artifact     = artifact
    provenance.scope        = scope
    provenance.payload_hash = _sha256(data)
    provenance.created_at   = (provenance.created_at
                               or datetime.now(timezone.utc).isoformat())
    payload_p.parent.mkdir(parents=True, exist_ok=True)
    # Atomic per file: write tmp + rename. Payload first, then the sidecar,
    # so a crash between the two leaves a payload without a record — which
    # `has()` treats as absent, never as corrupt.
    for target, blob in ((payload_p, data),
                         (prov_p, (json.dumps(provenance.to_json(), indent=2)
                                   + "\n").encode())):
      tmp = target.with_suffix(target.suffix + ".tmp")
      tmp.write_bytes(blob)
      tmp.replace(target)
    return payload_p

  def provenance(self, artifact: str, scope: Scope) -> ProvenanceRecord:
    _, prov_p = self._paths(artifact, scope)
    if not prov_p.exists():
      raise MissingArtifact(f"no provenance record at {prov_p}")
    return ProvenanceRecord.from_json(json.loads(prov_p.read_text()))

  def payload_hash(self, artifact: str, scope: Scope) -> str:
    """Current payload hash (for building downstream ``input_versions``)."""
    return self.provenance(artifact, scope).payload_hash

  # ---- inspection ------------------------------------------------------------
  def ls(self) -> list[ArtifactInfo]:
    out: list[ArtifactInfo] = []
    if not self.root.exists():
      return out
    for prov_p in sorted(self.root.rglob("*.prov.json")):
      try:
        rec = ProvenanceRecord.from_json(json.loads(prov_p.read_text()))
      except Exception:  # noqa: BLE001 - a broken sidecar shouldn't hide the rest
        continue
      try:
        payload_p = self._paths(rec.artifact, rec.scope)[0]
      except (KeyError, ValueError):
        continue  # sidecar for a type this registry doesn't know
      out.append(ArtifactInfo(
        artifact=rec.artifact, scope=rec.scope, path=payload_p,
        created_at=rec.created_at, node=rec.node))
    return out
