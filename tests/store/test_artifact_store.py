"""Artifact store: addressing, provenance sidecars, corruption detection."""
from __future__ import annotations

import numpy as np
import pytest

from chuck_dreamer.store import (
  ArtifactStore,
  CorruptArtifact,
  MissingArtifact,
  ProvenanceRecord,
  annotation_record,
  for_camera,
  for_dataset,
  for_episode,
)


@pytest.fixture
def store(tmp_path):
  return ArtifactStore(tmp_path / "store")


# ---------------------------------------------------------------------------
# put / get round-trips
# ---------------------------------------------------------------------------
def test_json_roundtrip_with_record(store):
  scope = for_camera("cam0")
  rec = ProvenanceRecord(artifact="", scope=scope, node="calibrate_intrinsics",
                         input_versions={"checkerboard_capture": "sha256:abc"})
  store.put("intrinsics", scope, {"K": [[1, 0, 0]]}, rec)

  payload, out = store.get("intrinsics", scope)
  assert payload == {"K": [[1, 0, 0]]}
  assert out.node == "calibrate_intrinsics"
  assert out.artifact == "intrinsics"
  assert out.payload_hash.startswith("sha256:")
  assert out.created_at
  assert out.input_versions == {"checkerboard_capture": "sha256:abc"}


def test_npz_roundtrip(store):
  scope = for_episode("user/ds", 3)
  masks = {"seg": np.zeros((4, 8, 8), dtype=np.uint8),
           "valid": np.array([1, 0, 1, 1], dtype=bool)}
  store.put("object_masks", scope, masks,
            annotation_record("object_masks", scope, "test"))
  payload, _ = store.get("object_masks", scope)
  np.testing.assert_array_equal(payload["seg"], masks["seg"])
  np.testing.assert_array_equal(payload["valid"], masks["valid"])


def test_layout_matches_design(store):
  store.put("intrinsics", for_camera("cam0"), {},
            annotation_record("intrinsics", for_camera("cam0"), "t"))
  store.put("extrinsics", for_dataset("user/ds"), {},
            annotation_record("extrinsics", for_dataset("user/ds"), "t"))
  store.put("object_prompts", for_episode("user/ds", 7), {"0": [1, 2]},
            annotation_record("object_prompts", for_episode("user/ds", 7), "t"))
  root = store.root
  assert (root / "camera" / "cam0" / "intrinsics.json").exists()
  assert (root / "camera" / "cam0" / "intrinsics.prov.json").exists()
  assert (root / "dataset" / "user__ds" / "extrinsics.json").exists()
  assert (root / "dataset" / "user__ds" / "episode" / "00007"
          / "object_prompts.json").exists()


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------
def test_missing_artifact_names_scope(store):
  with pytest.raises(MissingArtifact, match="intrinsics"):
    store.get("intrinsics", for_camera("nope"))


def test_scope_level_mismatch_rejected(store):
  with pytest.raises(ValueError, match="CAMERA-scoped"):
    store.put("intrinsics", for_dataset("user/ds"), {},
              annotation_record("intrinsics", for_dataset("user/ds"), "t"))


def test_unknown_artifact_type_rejected(store):
  with pytest.raises(KeyError, match="unknown artifact type"):
    store.has("bogus", for_camera("cam0"))


def test_out_of_band_edit_detected(store):
  scope = for_camera("cam0")
  store.put("intrinsics", scope, {"K": 1},
            annotation_record("intrinsics", scope, "t"))
  payload_p = store.root / "camera" / "cam0" / "intrinsics.json"
  payload_p.write_text('{"K": 2}\n')
  with pytest.raises(CorruptArtifact, match="modified outside the store"):
    store.get("intrinsics", scope)


def test_payload_without_record_is_absent(store):
  p = store.root / "camera" / "cam0"
  p.mkdir(parents=True)
  (p / "intrinsics.json").write_text("{}")
  assert not store.has("intrinsics", for_camera("cam0"))


def test_ls(store):
  store.put("intrinsics", for_camera("cam0"), {},
            annotation_record("intrinsics", for_camera("cam0"), "t"))
  store.put("object_prompts", for_episode("user/ds", 1), {},
            annotation_record("object_prompts", for_episode("user/ds", 1), "t"))
  infos = store.ls()
  assert {(i.artifact, i.scope.level) for i in infos} == {
    ("intrinsics", "camera"), ("object_prompts", "episode")}
