"""Artifact persistence: the artifact store.

See ``docs/trainer/artifact_store.md``. Shared ground between the import
pipeline (which writes artifacts) and the runtime (which reads calibration);
neither may import the other.
"""
from typing import Any

from .artifacts import (
  ArtifactInfo,
  ArtifactStore,
  ArtifactTypeSpec,
  CorruptArtifact,
  MissingArtifact,
  ProvenanceRecord,
  Scope,
  annotation_record,
  dataset_slug,
  for_camera,
  for_dataset,
  for_episode,
  git_sha,
)


def load_calibration(store: ArtifactStore, dataset_id: str
                     ) -> tuple[Any, dict[str, str]]:
  """Strict loader for a dataset's full camera calibration from the store.

  Resolves the CAMERA-scoped ``intrinsics`` via ``dataset_config.camera_id``
  plus the DATASET-scoped ``extrinsics``. Returns the
  :class:`~chuck_dreamer.perception.types.CameraCalibration` and the
  ``{artifact: payload_hash}`` map of the versions read (for advisory
  records). Missing artifacts raise :class:`MissingArtifact` naming the
  producing command.
  """
  from chuck_dreamer.perception.types import (
    CameraCalibration,
    Extrinsics,
    Intrinsics,
  )

  ds_scope = for_dataset(dataset_id)
  if not store.has("dataset_config", ds_scope):
    raise MissingArtifact(
      f"dataset_config not in store for {dataset_id}; run: "
      f"uv run python main.py import-lerobot annotate-mat {dataset_id}")
  ds_cfg, _ = store.get("dataset_config", ds_scope)
  camera_id = ds_cfg.get("camera_id")
  if not camera_id:
    raise MissingArtifact(
      f"dataset_config.camera_id is not set for {dataset_id}; re-run "
      f"annotate-mat for it")

  cam_scope = for_camera(str(camera_id))
  if not store.has("intrinsics", cam_scope):
    raise MissingArtifact(
      f"intrinsics for camera {camera_id!r} not in store; run: "
      f"uv run python main.py import-lerobot calibrate-intrinsics "
      f"--camera-id {camera_id} ...")
  intr_payload, intr_record = store.get("intrinsics", cam_scope)

  if not store.has("extrinsics", ds_scope):
    raise MissingArtifact(
      f"extrinsics not in store for {dataset_id}; run: "
      f"uv run python main.py import-lerobot annotate-mat {dataset_id}")
  extr_payload, extr_record = store.get("extrinsics", ds_scope)

  calibration = CameraCalibration(
    dataset_id=dataset_id,
    intrinsics=Intrinsics.from_json(intr_payload),
    extrinsics=Extrinsics.from_json(extr_payload),
  )
  return calibration, {"intrinsics": intr_record.payload_hash,
                       "extrinsics": extr_record.payload_hash}


__all__ = [
  "ArtifactInfo",
  "ArtifactStore",
  "ArtifactTypeSpec",
  "CorruptArtifact",
  "MissingArtifact",
  "ProvenanceRecord",
  "Scope",
  "annotation_record",
  "dataset_slug",
  "for_camera",
  "for_dataset",
  "for_episode",
  "git_sha",
  "load_calibration",
]
