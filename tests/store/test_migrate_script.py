"""Tests for the one-time ``scripts/migrate_calibration_cache.py``.

The migration is throwaway (the legacy cache dissolves into the store);
this test module loads the script by path and is deleted together with it.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from chuck_dreamer.store import (
  ArtifactStore,
  annotation_record,
  for_camera,
  for_dataset,
  for_episode,
)

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "migrate_calibration_cache.py")
_spec = importlib.util.spec_from_file_location("migrate_calibration_cache", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _mod   # dataclass processing needs the module registered
_spec.loader.exec_module(_mod)
migrate_calibration_cache = _mod.migrate_calibration_cache


@pytest.fixture
def store(tmp_path):
  return ArtifactStore(tmp_path / "store")


@pytest.fixture
def legacy_cache(tmp_path):
  slug = tmp_path / "cache" / "user__ds"
  slug.mkdir(parents=True)
  (slug / "intrinsics.json").write_text(json.dumps({"K": [[1]], "rms_px": 0.5}))
  (slug / "extrinsics.json").write_text(json.dumps({"R": [[1]], "t": [0]}))
  (slug / "mat_annotation.json").write_text(json.dumps({"circle_angle": 3.0}))
  (slug / "object_prompts.json").write_text(json.dumps(
    {"0": {"0": [10, 20]}, "2": {"0": [30, 40], "99": [50, 60]}}))
  (slug / "scene_bg.npz").write_bytes(b"not-a-real-npz")
  (slug / "notes.txt").write_text("hello")
  return tmp_path / "cache"


def test_migration_moves_everything(legacy_cache, store):
  actions = migrate_calibration_cache(
    legacy_cache, store, camera_id_for=lambda ds: "cam0")
  kinds = [a.kind for a in actions]
  assert kinds.count("migrated") == 5  # intrinsics, config, extrinsics, 2 prompt episodes
  assert "skipped-tool-internal" in kinds
  assert any(a.kind == "unknown-file" and a.source.name == "notes.txt"
             for a in actions)

  intr, rec = store.get("intrinsics", for_camera("cam0"))
  assert intr["rms_px"] == 0.5
  assert rec.node == "migration"
  assert rec.input_versions == {}

  extr, rec = store.get("extrinsics", for_dataset("user/ds"))
  assert extr["mat_annotation"] == {"circle_angle": 3.0}  # clicks merged in
  assert rec.node == "annotation:migration"

  cfg, _ = store.get("dataset_config", for_dataset("user/ds"))
  assert cfg == {"camera_id": "cam0", "touchpoint_variant": None}

  prompts, _ = store.get("object_prompts", for_episode("user/ds", 2))
  assert prompts == {"0": [30, 40], "99": [50, 60]}


def test_migration_is_idempotent(legacy_cache, store):
  migrate_calibration_cache(legacy_cache, store, camera_id_for=lambda ds: "cam0")
  again = migrate_calibration_cache(legacy_cache, store, camera_id_for=lambda ds: "cam0")
  assert not [a for a in again if a.kind == "migrated"]
  assert not [a for a in again if a.kind == "conflict"]


def test_migration_never_overwrites_conflicts(legacy_cache, store):
  scope = for_dataset("user/ds")
  store.put("extrinsics", scope, {"R": [[9]]},
            annotation_record("extrinsics", scope, "annotate-mat"))
  actions = migrate_calibration_cache(
    legacy_cache, store, camera_id_for=lambda ds: "cam0")
  assert any(a.kind == "conflict" and a.artifact == "extrinsics" for a in actions)
  payload, _ = store.get("extrinsics", scope)
  assert payload == {"R": [[9]]}  # untouched


def test_migration_dry_run_writes_nothing(legacy_cache, store):
  actions = migrate_calibration_cache(
    legacy_cache, store, camera_id_for=lambda ds: "cam0", dry_run=True)
  assert [a for a in actions if a.kind == "migrated"]
  assert store.ls() == []


def test_migration_shared_camera_intrinsics_are_equivalent(legacy_cache, store, tmp_path):
  # Two datasets carrying the same camera calibration with different
  # bookkeeping fields (source_dataset vs per_dataset_view_counts) must not
  # conflict on the shared CAMERA scope — the K/dist/image_size decide.
  slug2 = legacy_cache / "user__ds2"
  slug2.mkdir()
  (slug2 / "intrinsics.json").write_text(json.dumps(
    {"K": [[1]], "rms_px": 0.5, "source_dataset": "user/cal"}))

  actions = migrate_calibration_cache(
    legacy_cache, store, camera_id_for=lambda ds: "cam0")
  intr_actions = [a for a in actions if a.artifact == "intrinsics"]
  assert sorted(a.kind for a in intr_actions) == ["migrated", "skipped-exists"]
  assert not [a for a in actions if a.kind == "conflict"]
