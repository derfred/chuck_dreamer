"""Annotation tools' store write path: records, seeding, advisory versions.

The annotation artifacts terminate provenance (empty ``input_versions``,
spec §9.3); tools that leaned on pre-computed artifacts record those as
``advisory_versions`` — re-annotation hints, never staleness inputs.
"""
from __future__ import annotations

import pytest

from chuck_dreamer.lerobot.annotation.prompts import (
  load_keyframe_prompts,
  save_keyframe_prompt,
)
from chuck_dreamer.lerobot.annotation.store_sync import (
  camera_id_from_key,
  put_extrinsics,
  put_intrinsics,
)
from chuck_dreamer.store import ArtifactStore, for_camera, for_dataset, for_episode

DS = "user/ds"


@pytest.fixture
def store(tmp_path):
  return ArtifactStore(tmp_path / "store")


def test_camera_id_from_key():
  assert camera_id_from_key("observation.images.front") == "front"


def test_put_intrinsics_annotation_record(store):
  put_intrinsics(store, "front", {"K": [[1]]})
  _, rec = store.get("intrinsics", for_camera("front"))
  assert rec.node == "annotation:calibrate-intrinsics"
  assert rec.input_versions == {}


def test_put_extrinsics_records_advisory_intrinsics(store):
  put_intrinsics(store, "front", {"K": [[1]]})
  intr_hash = store.payload_hash("intrinsics", for_camera("front"))

  put_extrinsics(store, DS, {"R": [[1]], "t": [0]},
                 camera_id="front", advisory={"intrinsics": intr_hash})
  _, rec = store.get("extrinsics", for_dataset(DS))
  assert rec.node == "annotation:annotate-mat"
  # Advisory, not staleness: input_versions stays empty (spec §9.3).
  assert rec.input_versions == {}
  assert rec.advisory_versions == {"intrinsics": intr_hash}


def test_put_extrinsics_seeds_dataset_config_once(store):
  put_extrinsics(store, DS, {"R": [[1]], "t": [0]}, camera_id="front")
  cfg, _ = store.get("dataset_config", for_dataset(DS))
  assert cfg == {"camera_id": "front", "touchpoint_variant": None}

  # A later re-annotation must not clobber an enriched dataset_config.
  from chuck_dreamer.store import annotation_record
  store.put("dataset_config", for_dataset(DS),
            {"camera_id": "front", "touchpoint_variant": 7},
            annotation_record("dataset_config", for_dataset(DS), "t"))
  put_extrinsics(store, DS, {"R": [[2]], "t": [1]}, camera_id="front")
  cfg, _ = store.get("dataset_config", for_dataset(DS))
  assert cfg["touchpoint_variant"] == 7


def test_prompts_merge_and_advisory(store):
  save_keyframe_prompt(store, DS, 3, 0, (10, 20))
  save_keyframe_prompt(store, DS, 3, 99, (30, 40),
                       advisory={"intrinsics": "sha256:x"})

  assert load_keyframe_prompts(store, DS, 3) == {0: (10, 20), 99: (30, 40)}
  _, rec = store.get("object_prompts", for_episode(DS, 3))
  assert rec.node == "annotation:prompt-episodes"
  assert rec.advisory_versions == {"intrinsics": "sha256:x"}


def test_prompts_empty_when_unannotated(store):
  assert load_keyframe_prompts(store, DS, 5) == {}
