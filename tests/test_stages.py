"""Tests for the composable importer stage system.

Covers the pure-logic surface — dependency resolution / ordering, the
:class:`Run` flag→stage mapping, and each stage's declared
``produces`` / ``requires`` / ``requirements()`` contract. The stages'
``apply()`` bodies need SAM2 / MuJoCo and are not exercised here;
``requirements()`` is driven with a :class:`RunContext` whose ``_ol_cfg``
is pre-seeded so no real config loads.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chuck_dreamer.lerobot.pipeline import Run
from chuck_dreamer.lerobot.stages import (
  EePosStage,
  ObjectPoseStage,
  Requirement,
  RunContext,
  SegmentationStage,
  build_registry,
  resolve_stages,
)


def _ctx(**kw) -> RunContext:
  return RunContext(source_repo=kw.pop("source_repo", "user/ds"), **kw)


class _FakeSpec:
  """Minimal EpisodeSpec stand-in: Run only reads .dataset_id and calls
  .read_episodes(video_key=, root=)."""

  def __init__(self, dataset_id="user/ds", slices=(("ep", 0), ("ep", 1))):
    self.dataset_id = dataset_id
    self._slices = [SimpleNamespace(episode_index=i) for _, i in slices]
    self.read_calls: list[dict] = []

  def read_episodes(self, *, video_key=None, root=None):
    self.read_calls.append({"video_key": video_key, "root": root})
    return self._slices, "observation.images.wrist"


# ---------------------------------------------------------------------------
# Run — shared importer/doctor wiring
# ---------------------------------------------------------------------------


def test_run_builds_context_from_spec():
  run = Run(_FakeSpec("user/ds"))
  assert run.ctx.source_repo == "user/ds"


def test_run_local_root_none_for_repo_id():
  assert Run(_FakeSpec("user/ds")).local_root is None


def test_run_local_root_set_for_on_disk_dataset(tmp_path):
  assert Run(_FakeSpec(str(tmp_path))).local_root == tmp_path


def test_run_read_slices_passes_root_and_video_key(tmp_path):
  spec = _FakeSpec(str(tmp_path))
  run = Run(spec)
  run.read_slices(video_key="observation.images.wrist")
  assert spec.read_calls == [
    {"video_key": "observation.images.wrist", "root": tmp_path}]


def test_run_pipeline_threads_flags():
  run = Run(_FakeSpec(), with_ee_pos=True, with_object_pose=False)
  assert [s.name for s in run.pipeline(0)] == ["ee_pos"]


def test_run_pipeline_reuses_context_and_registry_across_episodes():
  run = Run(_FakeSpec(), with_ee_pos=True, with_object_pose=False)
  p0 = run.pipeline(0)
  p1 = run.pipeline(1)
  # Same stage instances (shared registry) are reused, not rebuilt per episode.
  assert p0[0] is p1[0]
  assert p0[0].ctx is run.ctx


def test_run_pipeline_sets_episode_index():
  run = Run(_FakeSpec(), with_ee_pos=True, with_object_pose=False)
  run.pipeline(4)
  assert run.ctx.episode_index == 4


# ---------------------------------------------------------------------------
# Run.pipeline — flag→stage mapping + per-episode behaviour
# ---------------------------------------------------------------------------


def test_pipeline_both_flags():
  names = [s.name for s in Run(_FakeSpec(), with_ee_pos=True, with_object_pose=True).pipeline(0)]
  assert set(names) == {"ee_pos", "segment:object", "object_pose"}


def test_pipeline_ee_only():
  names = [s.name for s in Run(_FakeSpec(), with_ee_pos=True, with_object_pose=False).pipeline(0)]
  assert names == ["ee_pos"]


def test_pipeline_object_pulls_in_segmentation():
  names = [s.name for s in Run(_FakeSpec(), with_ee_pos=False, with_object_pose=True).pipeline(0)]
  assert "segment:object" in names
  assert names.index("segment:object") < names.index("object_pose")


def test_pipeline_no_flags_is_empty():
  assert Run(_FakeSpec(), with_ee_pos=False, with_object_pose=False).pipeline(0) == []


def test_pipeline_sets_episode_index_on_context():
  run = Run(_FakeSpec(), with_ee_pos=True, with_object_pose=False)
  run.pipeline(7)
  assert run.ctx.episode_index == 7


def test_pipeline_clears_masks_on_construction():
  run = Run(_FakeSpec(), with_ee_pos=True, with_object_pose=False)
  run.ctx.masks["object"] = ["stale"]
  run.pipeline(1)
  assert run.ctx.masks == {}


# ---------------------------------------------------------------------------
# build_registry
# ---------------------------------------------------------------------------


def test_registry_keys():
  assert set(build_registry(_ctx())) == {"ee_pos", "segment:object", "object_pose"}


def test_registry_binds_context_to_stages():
  ctx = _ctx()
  registry = build_registry(ctx)
  assert all(s.ctx is ctx for s in registry.values())


# ---------------------------------------------------------------------------
# resolve_stages — dependency pull-in + ordering
# ---------------------------------------------------------------------------


def test_resolve_object_pose_pulls_in_segmentation():
  names = [s.name for s in resolve_stages({"object_pose"}, _ctx())]
  assert "segment:object" in names
  assert names.index("segment:object") < names.index("object_pose")


def test_resolve_full_default_set_orders_segment_before_pose():
  names = [s.name for s in resolve_stages({"ee_pos", "object_pose"}, _ctx())]
  assert set(names) == {"ee_pos", "segment:object", "object_pose"}
  assert names.index("segment:object") < names.index("object_pose")


def test_resolve_ee_only():
  assert [s.name for s in resolve_stages({"ee_pos"}, _ctx())] == ["ee_pos"]


def test_resolve_empty():
  assert resolve_stages(set(), _ctx()) == []


def test_resolve_unknown_stage_raises():
  with pytest.raises(ValueError, match="unknown stage"):
    resolve_stages({"nope"}, _ctx())


def test_resolve_detects_cycle():
  # Build a registry with a 2-node cycle and confirm it's rejected rather
  # than recursing forever.
  a = SimpleNamespace(name="a", requires=("b",), produces=())
  b = SimpleNamespace(name="b", requires=("a",), produces=())
  registry = {"a": a, "b": b}
  with pytest.raises(ValueError, match="cycle"):
    resolve_stages({"a"}, _ctx(), registry)


def test_resolve_dep_appears_once_even_if_reachable_twice():
  names = [s.name for s in resolve_stages({"object_pose", "segment:object"}, _ctx())]
  assert names.count("segment:object") == 1


# ---------------------------------------------------------------------------
# Stage declarations: produces / requires
# ---------------------------------------------------------------------------


def test_ee_pos_declares():
  s = EePosStage(_ctx())
  assert s.name == "ee_pos"
  assert s.requires == ()
  assert set(s.produces) == {"joint_qpos", "ee_pos", "ee_quat", "ee_action"}


def test_object_pose_declares():
  s = ObjectPoseStage(_ctx())
  assert s.name == "object_pose"
  assert s.requires == ("segment:object",)
  assert set(s.produces) == {"object_xy", "object_gap_too_long"}


def test_segmentation_object_target_uses_legacy_keys():
  s = SegmentationStage(_ctx(), "object")
  assert s.name == "segment:object"
  assert s.requires == ()
  # The object target keeps the historical key names for writer/training
  # compatibility.
  assert s.produces == ("segmentation_target", "object_uv")


def test_segmentation_other_target_uses_suffixed_keys():
  s = SegmentationStage(_ctx(), "arm")
  assert s.name == "segment:arm"
  assert s.produces == ("segmentation_arm", "arm_uv")


# ---------------------------------------------------------------------------
# requirements()
# ---------------------------------------------------------------------------


def _ctx_with_ol(tmp_path: Path, **kw) -> RunContext:
  """A RunContext whose ol_cfg() is pre-seeded so requirements() that
  read calibration_cache / mesh_path don't load the real config."""
  ctx = _ctx(**kw)
  ctx._ol_cfg = SimpleNamespace(
    calibration_cache=str(tmp_path / "cache"),
    mesh_path=str(tmp_path / "mesh.obj"),
  )
  return ctx


# ---------------------------------------------------------------------------
# RunContext object-localization accessors
# ---------------------------------------------------------------------------


def test_context_cache_dir_reads_ol_cfg(tmp_path):
  ctx = _ctx_with_ol(tmp_path)
  assert ctx.cache_dir() == tmp_path / "cache"


def test_context_dataset_cache_dir_is_per_repo(tmp_path):
  ctx = _ctx_with_ol(tmp_path, source_repo="user/ds")
  ds_dir = ctx.dataset_cache_dir()
  # Lives under the cache root and is repo-specific.
  assert (tmp_path / "cache") in ds_dir.parents
  assert ctx.dataset_slug() in ds_dir.parts


def test_context_keyframe_prompts_delegates(tmp_path, monkeypatch):
  import chuck_dreamer.lerobot.object_localization.prompts as prompts
  seen = {}

  def fake(cache_dir, dataset_id, ep):
    seen.update(cache_dir=cache_dir, dataset_id=dataset_id, ep=ep)
    return {0: "p"}

  monkeypatch.setattr(prompts, "load_keyframe_prompts", fake)
  ctx = _ctx_with_ol(tmp_path, source_repo="user/ds")
  assert ctx.keyframe_prompts(5) == {0: "p"}
  assert seen == {"cache_dir": tmp_path / "cache", "dataset_id": "user/ds", "ep": 5}


def test_ee_pos_requirements_lists_fk_model():
  reqs = EePosStage(_ctx()).requirements()
  assert [r.label for r in reqs] == ["FK MuJoCo model"]
  assert reqs[0].path.name == "so101_arm.xml"


def test_segmentation_requirements_cover_calibration_and_mesh(tmp_path):
  ctx = _ctx_with_ol(tmp_path)
  ctx.episode_index = 3
  reqs = SegmentationStage(ctx, "object").requirements()
  labels = [r.label for r in reqs]
  assert labels == [
    "camera intrinsics", "camera extrinsics",
    "frame-0 object prompt (episode 3)", "object mesh"]
  # Each requirement names the dataset in its remediation command.
  for r in reqs[:3]:
    assert "user/ds" in r.remediation


def test_segmentation_frame0_requirement_is_per_episode(tmp_path, monkeypatch):
  # The frame-0 prompt requirement checks the *episode's* entry inside the
  # shared sidecar, not just the sidecar file's existence.
  import chuck_dreamer.lerobot.object_localization.prompts as prompts

  present = {2: {0: object()}}  # episode 2 has a frame-0 prompt; others don't
  monkeypatch.setattr(
    prompts, "load_keyframe_prompts",
    lambda cache_dir, dataset_id, ep: present.get(ep, {}))

  def frame0_req(ep: int) -> Requirement:
    ctx = _ctx_with_ol(tmp_path)
    ctx.episode_index = ep
    reqs = SegmentationStage(ctx, "object").requirements()
    return next(r for r in reqs if r.label.startswith("frame-0"))

  assert frame0_req(2).satisfied() is True
  assert frame0_req(5).satisfied() is False


def test_object_pose_requirements_list_mesh(tmp_path):
  reqs = ObjectPoseStage(_ctx_with_ol(tmp_path)).requirements()
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


def test_requirement_custom_check_overrides_path_existence(tmp_path):
  missing = tmp_path / "nope"
  assert Requirement("x", missing, "make x", check=lambda: True).satisfied() is True
  present = tmp_path / "f.json"
  present.write_text("{}")
  assert Requirement("x", present, "make x", check=lambda: False).satisfied() is False
