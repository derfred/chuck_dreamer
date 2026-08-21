"""Tests for the Run → node pipeline wiring and node declarations.

Covers the pure-logic surface — the :class:`Run` flag→node mapping, each
node's declared inputs/outputs, and the mask math. The nodes' ``run()``
bodies that need SAM2 / MuJoCo are not exercised here.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from chuck_dreamer.lerobot.context import RunContext
from chuck_dreamer.lerobot.nodes import (
  ObjectPoseNode,
  ObjUvExtractionNode,
  SegmentationNode,
  build_ee_nodes,
  build_ee_table_nodes,
  build_object_nodes,
  build_object_table_nodes,
)
from chuck_dreamer.lerobot.pipeline import Run

EE_CHAIN = ["normalize_joints", "ee_pos_arm", "ee_rotation", "ee_action_arm"]
EE_TABLE_CHAIN = ["ee_pos_table", "ee_quat_table", "ee_action_table"]
OBJECT_CHAIN = ["segmentation", "obj_uv_extraction", "object_pose"]
OBJECT_TABLE_CHAIN = ["obj_pos_arm"]


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
  assert [n.name for n in run.pipeline(0)] == EE_CHAIN


def test_run_pipeline_reuses_node_instances_across_episodes():
  run = Run(None, _FakeSpec(), {"with_ee_pos": False, "with_object_pose": True})
  p0 = run.pipeline(0)
  p1 = run.pipeline(1)
  # Nodes are episode-stateless; the same bound instances are reused.
  assert p0[0] is p1[0]
  assert p0[0].ctx is run.ctx


def test_run_pipeline_native_nodes_bind_run_context():
  run = Run(None, _FakeSpec(), {"with_ee_pos": True, "with_object_pose": True})
  assert all(n.ctx is run.ctx for n in run.pipeline(0))


def test_run_pipeline_sets_episode_index():
  run = Run(None, _FakeSpec(), {"with_ee_pos": True, "with_object_pose": False})
  run.pipeline(4)
  assert run.ctx.episode_index == 4


# ---------------------------------------------------------------------------
# Run.pipeline — flag→node mapping + per-episode behaviour
# ---------------------------------------------------------------------------


def test_pipeline_both_flags():
  names = [n.name for n in Run(None, _FakeSpec(), {"with_ee_pos": True, "with_object_pose": True}).pipeline(0)]
  assert names == EE_CHAIN + OBJECT_CHAIN


def test_pipeline_ee_only():
  names = [n.name for n in Run(None, _FakeSpec(), {"with_ee_pos": True, "with_object_pose": False}).pipeline(0)]
  assert names == EE_CHAIN


def test_pipeline_object_orders_segmentation_first():
  names = [n.name for n in Run(None, _FakeSpec(), {"with_ee_pos": False, "with_object_pose": True}).pipeline(0)]
  assert names == OBJECT_CHAIN


def test_pipeline_no_flags_is_empty():
  assert Run(None, _FakeSpec(), {"with_ee_pos": False, "with_object_pose": False}).pipeline(0) == []


def test_pipeline_table_frame_appends_frame_crossing_nodes():
  names = [n.name for n in Run(None, _FakeSpec(), {
    "with_ee_pos": True, "with_object_pose": True,
    "with_table_frame": True}).pipeline(0)]
  assert names == (EE_CHAIN + EE_TABLE_CHAIN
                   + OBJECT_CHAIN + OBJECT_TABLE_CHAIN)


def test_pipeline_table_frame_without_base_chains_is_empty():
  run = Run(None, _FakeSpec(), {"with_ee_pos": False,
                                "with_object_pose": False,
                                "with_table_frame": True})
  assert run.pipeline(0) == []


def test_pipeline_lanes_split_at_the_pose_fit():
  run = Run(None, _FakeSpec(), {"with_ee_pos": True, "with_object_pose": True})
  producer = [n.name for n in run.lane_pipeline(0, "producer")]
  worker = [n.name for n in run.lane_pipeline(0, "worker")]
  assert producer == EE_CHAIN + ["segmentation", "obj_uv_extraction"]
  assert worker == ["object_pose"]


def test_pipeline_lanes_table_nodes():
  # The EE table chain is cheap and rides the producer; obj_pos_arm consumes
  # the pose fit so it must run on the worker lane, after object_pose.
  run = Run(None, _FakeSpec(), {"with_ee_pos": True, "with_object_pose": True,
                                "with_table_frame": True})
  producer = [n.name for n in run.lane_pipeline(0, "producer")]
  worker = [n.name for n in run.lane_pipeline(0, "worker")]
  assert producer == (EE_CHAIN + EE_TABLE_CHAIN
                      + ["segmentation", "obj_uv_extraction"])
  assert worker == ["object_pose", "obj_pos_arm"]


def test_context_carries_no_episode_scratch():
  # Inter-node episode state (e.g. SAM2 masks) lives on the TrackSet/Episode,
  # not the RunContext — the context is purely run-scoped. Guard that the old
  # per-episode mask cache is gone so concurrent producers can't collide on it.
  run = Run(None, _FakeSpec(), {"with_ee_pos": True, "with_object_pose": False})
  assert not hasattr(run.ctx, "masks")


# ---------------------------------------------------------------------------
# Node declarations: inputs / outputs
# ---------------------------------------------------------------------------


def test_ee_chain_declares():
  nodes = build_ee_nodes(_ctx())
  assert [n.name for n in nodes] == EE_CHAIN
  produced = {spec.name for n in nodes for spec in n.outputs}
  assert produced == {"joint_qpos_rad", "ee_pos_arm", "ee_quat_arm",
                      "ee_action_arm"}
  assert all(n.lane == "producer" for n in nodes)


def test_ee_table_chain_declares():
  nodes = build_ee_table_nodes(_ctx())
  assert [n.name for n in nodes] == EE_TABLE_CHAIN
  produced = {spec.name for n in nodes for spec in n.outputs}
  assert produced == {"ee_pos_table", "ee_quat_table", "ee_action_table"}
  # the frame-crossing nodes consume the extracted transform; the action
  # node derives purely from the table-frame pose
  transform_consumers = {n.name for n in nodes
                         if any(d.name == "table_to_arm" for d in n.inputs)}
  assert transform_consumers == {"ee_pos_table", "ee_quat_table"}


def test_object_chain_declares():
  nodes = build_object_nodes(_ctx())
  assert [n.name for n in nodes] == OBJECT_CHAIN
  produced = {spec.name for n in nodes for spec in n.outputs}
  assert produced == {"object_masks", "segmentation_target", "obj_uv",
                      "obj_pos_table", "object_mesh_overlay"}


def test_object_table_chain_declares():
  nodes = build_object_table_nodes(_ctx())
  assert [n.name for n in nodes] == OBJECT_TABLE_CHAIN
  produced = {spec.name for n in nodes for spec in n.outputs}
  assert produced == {"obj_pos_arm"}
  assert all(any(d.name == "table_to_arm" for d in n.inputs) for n in nodes)


def test_object_pose_declares_camera_not_arm_inputs():
  # The pose fit renders through the camera (intrinsics + extrinsics); no
  # arm-frame input belongs in its declaration (spec §7.10).
  inputs = {d.name for d in ObjectPoseNode(_ctx()).inputs}
  assert inputs == {"image", "object_masks", "intrinsics", "extrinsics",
                    "object_mesh"}


def test_segmentation_declares_spec_inputs():
  inputs = {d.name for d in SegmentationNode(_ctx()).inputs}
  assert inputs == {"timestamp", "object_prompts", "dataset.episodes",
                    "sam2_weights"}


def test_obj_uv_consumes_masks():
  inputs = {d.name for d in ObjUvExtractionNode(_ctx()).inputs}
  assert "object_masks" in inputs


# ---------------------------------------------------------------------------
# _masks_to_seg_and_uv — densify + vectorized centroids
# ---------------------------------------------------------------------------


def _mask(H, W, pts):
  m = np.zeros((H, W), dtype=bool)
  for v, u in pts:
    m[v, u] = True
  return m


def test_masks_to_seg_and_uv_seg_array_and_centroids():
  from chuck_dreamer.lerobot.nodes.segmentation import _masks_to_seg_and_uv

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
  from chuck_dreamer.lerobot.nodes.segmentation import _masks_to_seg_and_uv

  H, W = 3, 3
  # A present-but-empty mask and a None both count as drop-outs.
  _, uv = _masks_to_seg_and_uv(2, [np.zeros((H, W), bool), None])
  assert np.isnan(uv).all()


def test_masks_to_seg_and_uv_pads_short_list_to_T():
  from chuck_dreamer.lerobot.nodes.segmentation import _masks_to_seg_and_uv

  # The segmenter may return fewer entries than T; trailing frames are zero.
  seg, uv = _masks_to_seg_and_uv(4, [_mask(3, 3, [(0, 0)])])
  assert seg.shape == (4, 3, 3)
  assert seg[1:].sum() == 0
  assert np.isnan(uv[1:]).all()


def test_masks_to_seg_and_uv_skips_shape_mismatched_mask():
  from chuck_dreamer.lerobot.nodes.segmentation import _masks_to_seg_and_uv

  ref = _mask(4, 5, [(0, 0)])
  seg, uv = _masks_to_seg_and_uv(2, [ref, np.ones((6, 5), bool)])
  # Reference shape wins; the mismatched second mask is dropped.
  assert seg.shape == (2, 4, 5)
  assert seg[1].sum() == 0 and np.isnan(uv[1]).all()


def test_masks_to_seg_and_uv_returns_none_when_no_masks():
  from chuck_dreamer.lerobot.nodes.segmentation import _masks_to_seg_and_uv

  assert _masks_to_seg_and_uv(3, [None, None, None]) is None
  assert _masks_to_seg_and_uv(0, []) is None


# ---------------------------------------------------------------------------
# masks_to_dense / dense_to_masks — the track ↔ ragged-list bridge
# ---------------------------------------------------------------------------


def test_masks_dense_roundtrip():
  from chuck_dreamer.lerobot.nodes.segmentation import (
    dense_to_masks,
    masks_to_dense,
  )

  masks = [_mask(3, 4, [(0, 1)]), None, _mask(3, 4, [(2, 2)])]
  dense, valid = masks_to_dense(3, masks)
  assert dense.shape == (3, 3, 4) and dense.dtype == bool
  np.testing.assert_array_equal(valid, [True, False, True])

  back = dense_to_masks(dense, valid)
  assert back[1] is None
  np.testing.assert_array_equal(back[0], masks[0])
  np.testing.assert_array_equal(back[2], masks[2])


def test_masks_dense_pads_and_placeholders():
  from chuck_dreamer.lerobot.nodes.segmentation import masks_to_dense

  dense, valid = masks_to_dense(3, [_mask(2, 2, [(0, 0)])])
  assert dense.shape == (3, 2, 2)
  np.testing.assert_array_equal(valid, [True, False, False])

  dense, valid = masks_to_dense(2, [None, None])
  assert dense.shape == (2, 1, 1)   # placeholder dims when nothing detected
  assert not valid.any()
