"""TrackSet / NodeView execution layer: layered resolution, declaration
enforcement, boundary unit conversion, validity siblings, and the artifact
resolver."""
from __future__ import annotations

import numpy as np
import pytest

from chuck_dreamer.common.episode import Episode
from chuck_dreamer.common.tracks import (
  Frame,
  FrameMismatch,
  Persist,
  Track,
  TrackSpec,
  Unit,
)
from chuck_dreamer.lerobot.harness import InputDecl
from chuck_dreamer.lerobot.trackset import (
  EpisodeScope,
  MissingTrack,
  NodeView,
  TrackSet,
  UndeclaredInput,
  run_episode,
)

SCOPE = EpisodeScope("user/ds", 3)


def _ts(fields: dict | None = None, metadata: dict | None = None) -> TrackSet:
  episode = Episode.from_arrays(fields or {})
  return TrackSet(SCOPE, episode, metadata if metadata is not None else {})


class _Node:
  """Minimal native node for tests."""
  scope_level = "episode"

  def __init__(self, name, inputs, outputs, fn):
    self.name = name
    self.inputs = tuple(inputs)
    self.outputs = tuple(outputs)
    self._fn = fn

  def run(self, view):
    return self._fn(view)


# ---------------------------------------------------------------------------
# Resolution layers
# ---------------------------------------------------------------------------
def test_staged_shadows_base():
  ts = _ts({"x": np.zeros((4, 2), np.float32)})
  spec = TrackSpec("x", np.float32, (2,))
  ts.commit(Track(spec, np.ones((4, 2), np.float32)))
  assert ts.get_track("x").values[0, 0] == 1.0        # staged wins
  assert np.asarray(ts.episode.get("x"))[0, 0] == 1.0  # materialized for writers


def test_missing_track_names_scope():
  with pytest.raises(MissingTrack, match="user/ds#3"):
    _ts().get_track("nope")


def test_base_joint_qpos_gets_channel_units():
  ts = _ts({"joint_qpos": np.zeros((4, 6), np.float32)})
  spec = ts.get_track("joint_qpos").spec
  assert spec.unit.groups[0] == ((0, 5), Unit.DEG)
  assert spec.unit.groups[1] == ((5, 6), Unit.UNITLESS)


def test_base_xyz_tracks_declare_frames():
  """The coordinate base fields carry their reference frame: every frame
  suffix is explicit and positions are millimetres."""
  ts = _ts({"ee_pos_arm":    np.zeros((4, 3), np.float32),
            "obj_pos_table": np.zeros((4, 3), np.float32)})
  ee = ts.get_track("ee_pos_arm").spec
  assert (ee.frame, ee.unit) == (Frame.ARM, Unit.MM)
  obj = ts.get_track("obj_pos_table").spec
  assert (obj.frame, obj.unit) == (Frame.TABLE, Unit.MM)


def test_base_frame_mismatch_is_enforced():
  """A node declaring table-frame ee_pos_arm must not silently read the
  arm-frame base field — the exact mixup frames exist to prevent."""
  ts = _ts({"ee_pos_arm": np.zeros((4, 3), np.float32)})
  node = _Node("n", [InputDecl("ee_pos_arm", frame=Frame.TABLE)], [], lambda v: [])
  with pytest.raises(FrameMismatch):
    NodeView(ts, node).track("ee_pos_arm")


# ---------------------------------------------------------------------------
# NodeView enforcement
# ---------------------------------------------------------------------------
def test_undeclared_read_raises():
  ts = _ts({"x": np.zeros((4,), np.float32)})
  node = _Node("n", [InputDecl("y")], [], lambda v: [])
  with pytest.raises(UndeclaredInput, match="undeclared input 'x'"):
    NodeView(ts, node).track("x")


def test_boundary_unit_conversion():
  ts = _ts()
  ts.commit(Track(TrackSpec("d", np.float32, (), unit=Unit.DEG),
                  np.full((2,), 180.0, np.float32)))
  node = _Node("n", [InputDecl("d", unit=Unit.RAD)], [], lambda v: [])
  out = NodeView(ts, node).track("d")
  np.testing.assert_allclose(out.values, np.pi, rtol=1e-6)


def test_boundary_frame_check():
  ts = _ts()
  ts.commit(Track(TrackSpec("p", np.float32, (3,), frame=Frame.ARM),
                  np.zeros((2, 3), np.float32)))
  node = _Node("n", [InputDecl("p", frame=Frame.TABLE)], [], lambda v: [])
  with pytest.raises(FrameMismatch):
    NodeView(ts, node).track("p")


# ---------------------------------------------------------------------------
# run_episode: native validation + legacy shim
# ---------------------------------------------------------------------------
def test_native_node_outputs_validated_and_committed():
  out_spec = TrackSpec("y", np.float32, (), frame=Frame.ARM,
                       persist=Persist.EPHEMERAL)
  node = _Node("n", [], [out_spec],
               lambda v: [Track(out_spec, np.ones((3,), np.float32))])
  ts = _ts()
  run_episode([node], ts)
  assert ts.get_track("y").spec.frame == Frame.ARM


def test_native_node_must_produce_declared_outputs():
  out_spec = TrackSpec("y", np.float32, ())
  node = _Node("n", [], [out_spec], lambda v: [])
  with pytest.raises(ValueError, match="did not produce"):
    run_episode([node], _ts())


def test_native_node_undeclared_output_rejected():
  rogue = TrackSpec("z", np.float32, ())
  node = _Node("n", [], [], lambda v: [Track(rogue, np.zeros((1,), np.float32))])
  with pytest.raises(UndeclaredInput, match="undeclared output"):
    run_episode([node], _ts())


def test_validity_survives_the_episode_container():
  """Commit materializes `<name>.valid` as a non-persisted sibling field;
  a fresh TrackSet over the same episode (the parallel worker's view after
  pickling) reconstructs the validity-carrying track."""
  ts = _ts()
  spec = TrackSpec("m", np.bool_, (2, 2), frame=Frame.IMAGE, validity=True)
  valid = np.array([True, False, True], bool)
  ts.commit(Track(spec, np.ones((3, 2, 2), bool), valid))

  fresh = TrackSet(SCOPE, ts.episode, {})
  track = fresh.get_track("m")
  assert track.spec.validity
  np.testing.assert_array_equal(track.valid, valid)
  # The sibling is a non-persisted scratch field.
  assert dict(ts.episode.fields())["m.valid"].persist is False


def test_artifact_resolution_goes_through_node_ctx():
  from types import SimpleNamespace

  class _CtxNode:
    name = "n"
    scope_level = "episode"
    inputs = (InputDecl("fk_model"),)
    outputs = ()

    def __init__(self):
      self.ctx = SimpleNamespace(fk=lambda: "the-fk-evaluator")

    def run(self, view):
      return []

  node = _CtxNode()
  view = NodeView(_ts(), node)
  assert view.artifact("fk_model") == "the-fk-evaluator"
  with pytest.raises(UndeclaredInput, match="undeclared artifact"):
    view.artifact("object_mesh")


# ---------------------------------------------------------------------------
# Store-first resolution, runner-level caching side contracts
# ---------------------------------------------------------------------------
def test_intrinsics_resolve_store_first(tmp_path):
  from types import SimpleNamespace

  from omegaconf import OmegaConf

  from chuck_dreamer.lerobot.context import RunContext
  from chuck_dreamer.store import (
    ArtifactStore,
    annotation_record,
    for_camera,
    for_dataset,
  )

  store = ArtifactStore(tmp_path / "store")
  cam = for_camera("front")
  store.put("dataset_config", for_dataset("user/ds"),
            {"camera_id": "front", "touchpoint_variant": None},
            annotation_record("dataset_config", for_dataset("user/ds"), "t"))
  payload = {"image_size": [4, 3], "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
             "dist": [0, 0, 0, 0, 0], "rms_px": 0.1, "n_frames_used": 9}
  store.put("intrinsics", cam, payload,
            annotation_record("intrinsics", cam, "t"))

  ctx = RunContext(source_repo="user/ds",
                   config=OmegaConf.create({"store": {"root": str(tmp_path / "store")}}))
  ctx._ol_cfg = SimpleNamespace(calibration_cache=str(tmp_path / "cache"))
  intr, version = ctx.resolve_intrinsics()
  assert intr.image_size == (4, 3)
  assert version == store.payload_hash("intrinsics", cam)


def test_prompts_resolve_store_first_with_legacy_fallback(tmp_path):
  from types import SimpleNamespace

  from omegaconf import OmegaConf

  from chuck_dreamer.lerobot.context import RunContext
  from chuck_dreamer.store import ArtifactStore, annotation_record, for_episode

  store = ArtifactStore(tmp_path / "store")
  store.put("object_prompts", for_episode("user/ds", 1), {"0": [7, 8]},
            annotation_record("object_prompts", for_episode("user/ds", 1), "t"))

  ctx = RunContext(source_repo="user/ds",
                   config=OmegaConf.create({"store": {"root": str(tmp_path / "store")}}))
  ctx._ol_cfg = SimpleNamespace(calibration_cache=str(tmp_path / "cache"))

  prompts, version = ctx.resolve_keyframe_prompts(1)
  assert prompts == {0: [7, 8]}          # store hit, int-keyed
  assert version                          # the store record's payload hash

  prompts, version = ctx.resolve_keyframe_prompts(2)
  assert prompts == {} and version is None   # neither store nor sidecar


def test_commit_persists_validity_sibling_for_trajectory_tracks():
  ts = _ts()
  spec = TrackSpec("object_uv", np.float32, (2,), frame=Frame.IMAGE,
                   unit=Unit.PX, validity=True, persist=Persist.TRAJECTORY)
  valid = np.array([True, False], bool)
  ts.commit(Track(spec, np.zeros((2, 2), np.float32), valid))
  fld = dict(ts.episode.fields())["object_uv.valid"]
  assert fld.persist is True             # reaches the writers
  np.testing.assert_array_equal(fld.value, valid)


def test_run_episode_records_provenance_in_metadata():
  out_spec = TrackSpec("y", np.float32, (), persist=Persist.TRAJECTORY)
  node = _Node("n", [InputDecl("x")], [out_spec],
               lambda v: [Track(out_spec, v.track("x").values + 1)])
  ts = _ts({"x": np.zeros((3,), np.float32)})
  run_episode([node], ts)
  prov = ts.metadata["provenance"]["y"]
  assert prov["node"] == "n"
  assert set(prov["input_versions"]) == {"x"}
  assert prov["input_versions"]["x"].startswith("sha256:")


def test_run_episode_releases_ephemeral_after_last_consumer():
  eph = TrackSpec("scratch", np.float32, (), persist=Persist.EPHEMERAL)
  out = TrackSpec("y", np.float32, (), persist=Persist.TRAJECTORY)
  producer = _Node("p", [], [eph],
                   lambda v: [Track(eph, np.ones((2,), np.float32))])
  consumer = _Node("c", [InputDecl("scratch")], [out],
                   lambda v: [Track(out, v.track("scratch").values * 2)])
  ts = _ts()
  run_episode([producer, consumer], ts)
  assert "scratch" not in ts.episode      # dropped after its last consumer
  with pytest.raises(MissingTrack):
    ts.get_track("scratch")
  np.testing.assert_array_equal(ts.get_track("y").values, [2.0, 2.0])
