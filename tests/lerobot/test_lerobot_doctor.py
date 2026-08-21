"""Tests for the declaration- and store-driven doctor (``lerobot.doctor``).

The doctor derives every leaf input from the node declarations of the run's
pipeline and checks it against the artifact store / assets at the resolved
scope key — no per-node ``requirements()`` lists. These tests drive it with
a fake ``Run`` whose pipeline yields fake nodes with controlled
declarations, over a real :class:`ArtifactStore` in a tmp dir.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from chuck_dreamer.lerobot.context import RunContext
from chuck_dreamer.lerobot.doctor import doctor_run
from chuck_dreamer.store import (
  ArtifactStore,
  annotation_record,
  for_camera,
  for_dataset,
  for_episode,
)

DS = "user/ds"


def _node(name, inputs, outputs=()):
  return SimpleNamespace(
    name=name,
    inputs=tuple(SimpleNamespace(name=i) for i in inputs),
    outputs=tuple(SimpleNamespace(name=o) for o in outputs),
  )


class _FakeRun:
  def __init__(self, nodes, ctx, *, episodes=(0,), params=None):
    self.spec = SimpleNamespace(dataset_id=DS)
    self.params = params or {}
    self.ctx = ctx
    self._nodes = nodes
    self._episodes = episodes

  def read_slices(self):
    return [SimpleNamespace(episode_index=i) for i in self._episodes], "k"

  def pipeline(self, episode_index):
    self.ctx.episode_index = episode_index
    return list(self._nodes)


@pytest.fixture
def store(tmp_path):
  return ArtifactStore(tmp_path / "store")


@pytest.fixture
def ctx(tmp_path, store):
  cfg = OmegaConf.create({"store": {"root": str(store.root)}})
  c = RunContext(source_repo=DS, config=cfg)
  mesh = tmp_path / "mesh.obj"
  mesh.write_text("o mesh\n")
  c._ol_cfg = SimpleNamespace(mesh_path=str(mesh))
  return c


INTRINSICS = {
  "image_size": [4, 3],
  "K": [[1.0, 0.0, 2.0], [0.0, 1.0, 1.5], [0.0, 0.0, 1.0]],
  "dist": [0.0] * 5,
  "rms_px": 0.1,
  "n_frames_used": 1,
}
EXTRINSICS = {
  "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
  "t": [0.0, 0.0, 500.0],
  "rms_px": 0.2,
}


def _seed_calibration(store):
  store.put("intrinsics", for_camera("front"), INTRINSICS,
            annotation_record("intrinsics", for_camera("front"), "test"))
  store.put("dataset_config", for_dataset(DS),
            {"camera_id": "front", "touchpoint_variant": None},
            annotation_record("dataset_config", for_dataset(DS), "test"))
  store.put("extrinsics", for_dataset(DS), EXTRINSICS,
            annotation_record("extrinsics", for_dataset(DS), "test"))


def _seed_prompts(store, episodes=(0,)):
  for ep in episodes:
    store.put("object_prompts", for_episode(DS, ep), {"0": [1, 2]},
              annotation_record("object_prompts", for_episode(DS, ep), "test"))


# ---------------------------------------------------------------------------
# pass / fail per leaf kind
# ---------------------------------------------------------------------------


def test_doctor_passes_when_store_holds_everything(store, ctx, capsys):
  _seed_calibration(store)
  _seed_prompts(store)
  nodes = [_node("segmentation", ["object_prompts", "dataset.episodes"]),
           _node("object_pose", ["intrinsics", "extrinsics", "object_mesh"])]
  assert doctor_run(_FakeRun(nodes, ctx)) is True
  out = capsys.readouterr().out
  assert "MISS" not in out


def test_doctor_reports_missing_extrinsics_with_remediation(store, ctx, capsys):
  _seed_calibration(store)
  # wipe extrinsics only
  (store.root / "dataset").exists()
  import shutil
  shutil.rmtree(store.root)
  store.put("dataset_config", for_dataset(DS),
            {"camera_id": "front", "touchpoint_variant": None},
            annotation_record("dataset_config", for_dataset(DS), "test"))
  nodes = [_node("object_pose", ["extrinsics"])]
  assert doctor_run(_FakeRun(nodes, ctx)) is False
  out = capsys.readouterr().out
  assert "extrinsics" in out and "MISS" in out


def test_doctor_reports_table_to_arm_remediation(store, ctx, capsys):
  nodes = [_node("ee_pos_table", ["table_to_arm"])]
  assert doctor_run(_FakeRun(nodes, ctx)) is False
  out = capsys.readouterr().out
  assert "extract-table-to-arm" in out


def test_doctor_names_missing_prompt_tool(store, ctx, capsys):
  nodes = [_node("segmentation", ["object_prompts"])]
  assert doctor_run(_FakeRun(nodes, ctx)) is False
  out = capsys.readouterr().out
  assert "prompt-episodes" in out


# ---------------------------------------------------------------------------
# declaration-derived leaves
# ---------------------------------------------------------------------------


def test_doctor_skips_inputs_produced_upstream(store, ctx, capsys):
  # object_masks is produced by segmentation, so obj_uv_extraction's input
  # must not be treated as a leaf.
  _seed_prompts(store)
  nodes = [
    _node("segmentation", ["object_prompts"], outputs=["object_masks"]),
    _node("obj_uv_extraction", ["object_masks", "image"]),
  ]
  assert doctor_run(_FakeRun(nodes, ctx)) is True
  assert "object_masks" not in capsys.readouterr().out


def test_doctor_skips_dataset_supplied_inputs(store, ctx, capsys):
  nodes = [_node("ee", ["timestamp", "image", "joint_qpos", "dataset.episodes"])]
  assert doctor_run(_FakeRun(nodes, ctx)) is True
  out = capsys.readouterr().out
  assert "timestamp" not in out


def test_doctor_is_generic_over_declarations(store, ctx, capsys):
  # A brand-new node consuming a known leaf type is checked with no doctor
  # edit — the declarations drive everything.
  nodes = [_node("brand_new_node", ["intrinsics"])]
  assert doctor_run(_FakeRun(nodes, ctx)) is False
  assert "intrinsics" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# scope keys: per-episode checks + dedupe
# ---------------------------------------------------------------------------


def test_doctor_checks_prompts_per_episode(store, ctx, capsys):
  _seed_prompts(store, episodes=(0,))   # ep 1 unannotated
  nodes = [_node("segmentation", ["object_prompts"])]
  assert doctor_run(_FakeRun(nodes, ctx, episodes=(0, 1))) is False
  out = capsys.readouterr().out
  assert "episode 0" in out and "episode 1" in out


def test_doctor_dedups_shared_leaves(store, ctx, capsys):
  # Two nodes and three episodes naming the same DATASET-scoped artifact
  # print it once.
  nodes = [_node("a", ["extrinsics"]), _node("b", ["extrinsics"])]
  doctor_run(_FakeRun(nodes, ctx, episodes=(0, 1, 2)))
  out = capsys.readouterr().out
  assert out.count("extrinsics @") == 1


def test_doctor_fk_model_checked_against_repo_asset(store, ctx, capsys):
  # The FK model ships in the repo, so this leaf is present.
  nodes = [_node("ee_pos_arm", ["fk_model", "joint_qpos"])]
  assert doctor_run(_FakeRun(nodes, ctx)) is True
  assert "fk_model" in capsys.readouterr().out
