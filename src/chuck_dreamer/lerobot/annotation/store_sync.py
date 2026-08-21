"""Artifact-store write path for the annotation tools.

The store is the *only* persistence for annotation artifacts (the legacy
``calibration_cache/`` is gone; ``scripts/migrate_calibration_cache.py``
imported its contents once). Every accepted annotation lands via
``store.put`` with an ``annotation:<tool>`` provenance record; tools that
leaned on pre-computed artifacts (e.g. the extrinsics fit consulting
``intrinsics``) record those as **advisory versions** — never staleness
inputs, only re-annotation hints (spec §9.3).

Annotations are latest-only: a re-annotation overwrites the store entry
(the record's ``created_at`` marks the supersession).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def store_from_config(config: Any) -> Any | None:
  """The configured :class:`~chuck_dreamer.store.ArtifactStore`, or ``None``
  when the config declares no ``store.root``."""
  if config is None:
    return None
  from omegaconf import OmegaConf
  root = OmegaConf.select(config, "store.root")
  if not root:
    return None
  from chuck_dreamer.store import ArtifactStore
  return ArtifactStore(Path(root))


def require_store(config: Any, tool: str) -> Any:
  """The configured store, or a hard error naming the tool that needs it."""
  store = store_from_config(config)
  if store is None:
    raise RuntimeError(
      f"{tool} persists to the artifact store; set store.root in the config")
  return store


def camera_id_from_key(camera_key: str) -> str:
  """Camera identity derived from the configured LeRobot camera key
  (``observation.images.front`` → ``front``)."""
  return str(camera_key).rsplit(".", 1)[-1]


def put_intrinsics(store: Any, camera_id: str, payload: dict[str, Any]) -> None:
  """Persist a calibrated intrinsics payload at CAMERA scope."""
  from chuck_dreamer.store import annotation_record, for_camera
  scope = for_camera(camera_id)
  store.put("intrinsics", scope,
            payload, annotation_record("intrinsics", scope,
                                       "calibrate-intrinsics"))
  print(f"store: intrinsics @ camera/{camera_id} updated")


def get_intrinsics(store: Any, camera_id: str) -> tuple[dict[str, Any], Any]:
  """``(payload, record)`` of a camera's intrinsics; raises
  :class:`~chuck_dreamer.store.MissingArtifact` naming the producer."""
  from chuck_dreamer.store import MissingArtifact, for_camera
  scope = for_camera(camera_id)
  if not store.has("intrinsics", scope):
    raise MissingArtifact(
      f"intrinsics for camera {camera_id!r} not in store; run: "
      f"uv run python main.py import-lerobot calibrate-intrinsics "
      f"--camera-id {camera_id} ...")
  payload, record = store.get("intrinsics", scope)
  return dict(payload), record


def put_extrinsics(store: Any, dataset_id: str, payload: dict[str, Any],
                   *, camera_id: str | None = None,
                   advisory: dict[str, str] | None = None) -> None:
  """Persist an accepted extrinsics fit (clicks embedded in the payload) at
  DATASET scope, and seed the dataset's ``dataset_config`` when absent
  (annotate-mat is the first per-dataset annotation step, so the camera
  identity lands here)."""
  from chuck_dreamer.store import annotation_record, for_dataset

  scope = for_dataset(dataset_id)
  store.put("extrinsics", scope, payload,
            annotation_record("extrinsics", scope, "annotate-mat",
                              advisory=advisory))
  print(f"store: extrinsics @ {dataset_id} updated")

  if camera_id and not store.has("dataset_config", scope):
    store.put("dataset_config", scope,
              {"camera_id": camera_id, "touchpoint_variant": None},
              annotation_record("dataset_config", scope, "annotate-mat"))
    print(f"store: dataset_config @ {dataset_id} seeded (camera_id={camera_id})")
