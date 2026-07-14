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

import numpy as np
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
  run = Run(None, _FakeSpec("user/ds"), {})
  assert run.ctx.source_repo == "user/ds"


def test_run_local_root_none_for_repo_id():
  assert Run(None, _FakeSpec("user/ds"), {}).local_root is None


def test_run_local_root_set_for_on_disk_dataset(tmp_path):
  assert Run(None, _FakeSpec(str(tmp_path)), {}).local_root == tmp_path


def test_run_read_slices_passes_root_and_video_key(tmp_path):
  spec = _FakeSpec(str(tmp_path))
  run = Run(None, spec, {})
  run.read_slices(video_key="observation.images.wrist")
  assert spec.read_calls == [
    {"video_key": "observation.images.wrist", "root": tmp_path}]


def test_run_pipeline_threads_flags():
  run = Run(None, _FakeSpec(), {"with_ee_pos": True, "with_object_pose": False})
  assert [s.name for s in run.pipeline(0)] == ["ee_pos"]


def test_run_pipeline_reuses_context_and_registry_across_episodes():
  run = Run(None, _FakeSpec(), {"with_ee_pos": True, "with_object_pose": False})
  p0 = run.pipeline(0)
  p1 = run.pipeline(1)
  # Same stage instances (shared registry) are reused, not rebuilt per episode.
  assert p0[0] is p1[0]
  assert p0[0].ctx is run.ctx


def test_run_pipeline_sets_episode_index():
  run = Run(None, _FakeSpec(), {"with_ee_pos": True, "with_object_pose": False})
  run.pipeline(4)
  assert run.ctx.episode_index == 4


# ---------------------------------------------------------------------------
# Run.pipeline — flag→stage mapping + per-episode behaviour
# ---------------------------------------------------------------------------


def test_pipeline_both_flags():
  names = [s.name for s in Run(None, _FakeSpec(), {"with_ee_pos": True, "with_object_pose": True}).pipeline(0)]
  assert set(names) == {"ee_pos", "segment:object", "object_pose"}


def test_pipeline_ee_only():
  names = [s.name for s in Run(None, _FakeSpec(), {"with_ee_pos": True, "with_object_pose": False}).pipeline(0)]
  assert names == ["ee_pos"]


def test_pipeline_object_pulls_in_segmentation():
  names = [s.name for s in Run(None, _FakeSpec(), {"with_ee_pos": False, "with_object_pose": True}).pipeline(0)]
  assert "segment:object" in names
  assert names.index("segment:object") < names.index("object_pose")


def test_pipeline_no_flags_is_empty():
  assert Run(None, _FakeSpec(), {"with_ee_pos": False, "with_object_pose": False}).pipeline(0) == []


def test_pipeline_sets_episode_index_on_context():
  run = Run(None, _FakeSpec(), {"with_ee_pos": True, "with_object_pose": False})
  run.pipeline(7)
  assert run.ctx.episode_index == 7


def test_context_carries_no_episode_scratch():
  # Inter-stage episode state (e.g. SAM2 masks) lives on the Episode now, not
  # the RunContext — the context is purely run-scoped. Guard that the old
  # per-episode mask cache is gone so concurrent producers can't collide on it.
  run = Run(None, _FakeSpec(), {"with_ee_pos": True, "with_object_pose": False})
  assert not hasattr(run.ctx, "masks")


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
  assert set(s.produces) == {
    "object_xy", "object_gap_too_long", "object_mesh_overlay"}


def test_segmentation_declares():
  s = SegmentationStage(_ctx())
  assert s.name == "segment:object"
  assert s.requires == ()
  assert s.produces == ("segmentation_target", "object_uv")


# ---------------------------------------------------------------------------
# requirements()
# ---------------------------------------------------------------------------


def _ctx_with_ol(tmp_path: Path, **kw) -> RunContext:
  """A RunContext whose ol_cfg is pre-seeded so requirements() that
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
  import chuck_dreamer.lerobot.annotation.prompts as prompts
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


def test_segmentation_requires_only_the_frame0_prompt(tmp_path):
  # SAM2 needs only the video + a frame-0 prompt; calibration and the mesh
  # are the object-pose stage's requirements, not segmentation's.
  ctx = _ctx_with_ol(tmp_path)
  ctx.episode_index = 3
  reqs = SegmentationStage(ctx).requirements()
  assert [r.label for r in reqs] == ["frame-0 object prompt (episode 3)"]
  # The requirement names the dataset in its remediation command.
  assert "user/ds" in reqs[0].remediation


def test_segmentation_frame0_requirement_is_per_episode(tmp_path, monkeypatch):
  # The frame-0 prompt requirement checks the *episode's* entry inside the
  # shared sidecar, not just the sidecar file's existence.
  import chuck_dreamer.lerobot.annotation.prompts as prompts

  present = {2: {0: object()}}  # episode 2 has a frame-0 prompt; others don't
  monkeypatch.setattr(
    prompts, "load_keyframe_prompts",
    lambda cache_dir, dataset_id, ep: present.get(ep, {}))

  def frame0_req(ep: int) -> Requirement:
    ctx = _ctx_with_ol(tmp_path)
    ctx.episode_index = ep
    reqs = SegmentationStage(ctx).requirements()
    return next(r for r in reqs if r.label.startswith("frame-0"))

  assert frame0_req(2).satisfied() is True
  assert frame0_req(5).satisfied() is False


def test_object_pose_requirements_list_calibration_and_mesh(tmp_path):
  reqs = ObjectPoseStage(_ctx_with_ol(tmp_path)).requirements()
  assert [r.label for r in reqs] == [
    "camera intrinsics", "camera extrinsics", "object mesh"]


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


# ---------------------------------------------------------------------------
# _masks_to_seg_and_uv — densify + vectorized centroids
# ---------------------------------------------------------------------------


def _mask(H, W, pts):
  m = np.zeros((H, W), dtype=bool)
  for v, u in pts:
    m[v, u] = True
  return m


def test_masks_to_seg_and_uv_seg_array_and_centroids():
  from chuck_dreamer.lerobot.stages.segmentation import _masks_to_seg_and_uv

  H, W = 4, 5
  # frame 0: two pixels; frame 1: dropout (None); frame 2: one pixel.
  masks = [_mask(H, W, [(1, 1), (1, 3)]), None, _mask(H, W, [(2, 4)])]
  seg, uv = _masks_to_seg_and_uv(3, masks)

  assert seg.shape == (3, H, W) and seg.dtype == np.uint8
  assert seg[0].sum() == 2 and seg[1].sum() == 0 and seg[2].sum() == 1
  # centroid is the mean (u, v) pixel coordinate.
  np.testing.assert_allclose(uv[0], [2.0, 1.0])   # u=(1+3)/2, v=1
  np.testing.assert_allclose(uv[2], [4.0, 2.0])


def test_masks_to_seg_and_uv_dropouts_are_nan():
  from chuck_dreamer.lerobot.stages.segmentation import _masks_to_seg_and_uv

  H, W = 3, 3
  # A present-but-empty mask and a None both count as drop-outs.
  _, uv = _masks_to_seg_and_uv(2, [np.zeros((H, W), bool), None])
  assert np.isnan(uv).all()


def test_masks_to_seg_and_uv_pads_short_list_to_T():
  from chuck_dreamer.lerobot.stages.segmentation import _masks_to_seg_and_uv

  # The segmenter may return fewer entries than T; trailing frames are zero.
  seg, uv = _masks_to_seg_and_uv(4, [_mask(3, 3, [(0, 0)])])
  assert seg.shape == (4, 3, 3)
  assert seg[1:].sum() == 0
  assert np.isnan(uv[1:]).all()


def test_masks_to_seg_and_uv_skips_shape_mismatched_mask():
  from chuck_dreamer.lerobot.stages.segmentation import _masks_to_seg_and_uv

  ref = _mask(4, 5, [(0, 0)])
  seg, uv = _masks_to_seg_and_uv(2, [ref, np.ones((6, 5), bool)])
  # Reference shape wins; the mismatched second mask is dropped.
  assert seg.shape == (2, 4, 5)
  assert seg[1].sum() == 0 and np.isnan(uv[1]).all()


def test_masks_to_seg_and_uv_returns_none_when_no_masks():
  from chuck_dreamer.lerobot.stages.segmentation import _masks_to_seg_and_uv

  assert _masks_to_seg_and_uv(3, [None, None, None]) is None
  assert _masks_to_seg_and_uv(0, []) is None
