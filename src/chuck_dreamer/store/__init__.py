"""Artifact persistence: calibration cache today, the artifact store next.

See ``docs/trainer/artifact_store.md``. Shared ground between the import
pipeline (which writes artifacts) and the runtime (which reads calibration);
neither may import the other.
"""
from .artifacts import (
  ArtifactInfo,
  ArtifactStore,
  ArtifactTypeSpec,
  CorruptArtifact,
  MissingArtifact,
  ProvenanceRecord,
  Scope,
  annotation_record,
  for_camera,
  for_dataset,
  for_episode,
  git_sha,
)
from .calibration_cache import (
  dataset_cache_dir,
  dataset_slug,
  load_calibration,
  read_extrinsics,
  read_intrinsics,
  read_mat_annotation,
  write_extrinsics,
  write_intrinsics,
  write_mat_annotation,
)

__all__ = [
  "ArtifactInfo",
  "ArtifactStore",
  "ArtifactTypeSpec",
  "CorruptArtifact",
  "MissingArtifact",
  "ProvenanceRecord",
  "Scope",
  "annotation_record",
  "for_camera",
  "for_dataset",
  "for_episode",
  "git_sha",
  "dataset_cache_dir",
  "dataset_slug",
  "load_calibration",
  "read_extrinsics",
  "read_intrinsics",
  "read_mat_annotation",
  "write_extrinsics",
  "write_intrinsics",
  "write_mat_annotation",
]
