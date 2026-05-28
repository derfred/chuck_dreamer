"""Tests for the composable importer stage system.

Covers the pure-logic surface — dependency resolution / ordering, the
flag→stage-set mapping, and each stage's declared ``produces`` / ``requires``
/ ``requirements()`` contract. The stages' ``apply()`` bodies need SAM2 /
MuJoCo and are not exercised here; ``requirements()`` is driven with a
``StageContext`` whose ``_ol_cfg`` is pre-seeded so no real config loads.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chuck_dreamer.lerobot.stages import (
  EePosStage,
  ObjectPoseStage,
  Requirement,
  SegmentationStage,
  StageContext,
  build_registry,
  enabled_from_flags,
  resolve_stages,
)


# ---------------------------------------------------------------------------
# enabled_from_flags
# ---------------------------------------------------------------------------


def test_enabled_from_flags_both():
  assert enabled_from_flags(with_ee_pos=True, with_object_pose=True) == {
    "ee_pos", "object_pose"}


def test_enabled_from_flags_ee_only():
  assert enabled_from_flags(with_ee_pos=True, with_object_pose=False) == {"ee_pos"}


def test_enabled_from_flags_object_only_omits_segment():
  # object_pose pulls segment:object transitively in resolve_stages, so the
  # flag set itself does NOT list it.
  assert enabled_from_flags(with_ee_pos=False, with_object_pose=True) == {
    "object_pose"}


def test_enabled_from_flags_none():
  assert enabled_from_flags(with_ee_pos=False, with_object_pose=False) == set()


# ---------------------------------------------------------------------------
# build_registry
# ---------------------------------------------------------------------------


def test_registry_keys():
  assert set(build_registry()) == {"ee_pos", "segment:object", "object_pose"}


# ---------------------------------------------------------------------------
# resolve_stages — dependency pull-in + ordering
# ---------------------------------------------------------------------------


def test_resolve_object_pose_pulls_in_segmentation():
  names = [s.name for s in resolve_stages({"object_pose"})]
  assert "segment:object" in names
  assert names.index("segment:object") < names.index("object_pose")


def test_resolve_full_default_set_orders_segment_before_pose():
  names = [s.name for s in resolve_stages({"ee_pos", "object_pose"})]
  assert set(names) == {"ee_pos", "segment:object", "object_pose"}
  assert names.index("segment:object") < names.index("object_pose")


def test_resolve_ee_only():
  assert [s.name for s in resolve_stages({"ee_pos"})] == ["ee_pos"]


def test_resolve_empty():
  assert resolve_stages(set()) == []


def test_resolve_unknown_stage_raises():
  with pytest.raises(ValueError, match="unknown stage"):
    resolve_stages({"nope"})


def test_resolve_detects_cycle():
  # Build a registry with a 2-node cycle and confirm it's rejected rather
  # than recursing forever.
  a = SimpleNamespace(name="a", requires=("b",), produces=())
  b = SimpleNamespace(name="b", requires=("a",), produces=())
  registry = {"a": a, "b": b}
  with pytest.raises(ValueError, match="cycle"):
    resolve_stages({"a"}, registry)


def test_resolve_dep_appears_once_even_if_reachable_twice():
  names = [s.name for s in resolve_stages({"object_pose", "segment:object"})]
  assert names.count("segment:object") == 1


# ---------------------------------------------------------------------------
# Stage declarations: produces / requires
# ---------------------------------------------------------------------------


def test_ee_pos_declares():
  s = EePosStage()
  assert s.name == "ee_pos"
  assert s.requires == ()
  assert set(s.produces) == {"joint_qpos", "ee_pos", "ee_quat", "ee_action"}


def test_object_pose_declares():
  s = ObjectPoseStage()
  assert s.name == "object_pose"
  assert s.requires == ("segment:object",)
  assert set(s.produces) == {"object_xy", "object_gap_too_long"}


def test_segmentation_object_target_uses_legacy_keys():
  s = SegmentationStage("object")
  assert s.name == "segment:object"
  assert s.requires == ()
  # The object target keeps the historical key names for writer/training
  # compatibility.
  assert s.produces == ("segmentation_target", "object_uv")


def test_segmentation_other_target_uses_suffixed_keys():
  s = SegmentationStage("arm")
  assert s.name == "segment:arm"
  assert s.produces == ("segmentation_arm", "arm_uv")


# ---------------------------------------------------------------------------
# requirements()
# ---------------------------------------------------------------------------


def _ctx_with_ol(tmp_path: Path) -> StageContext:
  """A StageContext whose ol_cfg() is pre-seeded so requirements() that
  read calibration_cache / mesh_path don't load the real config."""
  ctx = StageContext(source_repo="user/ds")
  ctx._ol_cfg = SimpleNamespace(
    calibration_cache=str(tmp_path / "cache"),
    mesh_path=str(tmp_path / "mesh.obj"),
  )
  return ctx


def test_ee_pos_requirements_lists_fk_model():
  reqs = EePosStage().requirements(StageContext(source_repo="user/ds"))
  assert [r.label for r in reqs] == ["FK MuJoCo model"]
  assert reqs[0].path.name == "so101_arm.xml"


def test_segmentation_requirements_cover_calibration_and_mesh(tmp_path):
  reqs = SegmentationStage("object").requirements(_ctx_with_ol(tmp_path))
  labels = [r.label for r in reqs]
  assert labels == [
    "camera intrinsics", "camera extrinsics",
    "frame-0 object prompts", "object mesh"]
  # Each requirement names the dataset in its remediation command.
  for r in reqs[:3]:
    assert "user/ds" in r.remediation


def test_object_pose_requirements_list_mesh(tmp_path):
  reqs = ObjectPoseStage().requirements(_ctx_with_ol(tmp_path))
  assert [r.label for r in reqs] == ["object mesh"]


# ---------------------------------------------------------------------------
# Requirement
# ---------------------------------------------------------------------------


def test_requirement_satisfied_true_when_path_exists(tmp_path):
  p = tmp_path / "f.json"
  p.write_text("{}")
  assert Requirement("x", p, "make x").satisfied() is True


def test_requirement_satisfied_false_when_missing(tmp_path):
  assert Requirement("x", tmp_path / "nope", "make x").satisfied() is False
