"""Table-frame coordinate nodes: transform math + harness integration.

Runs the frame-crossing nodes through ``run_episode`` with a real store
holding a ``table_to_arm`` transform, and checks the geometry against
hand-computed values (transform payload is in metres; tracks are mm).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from omegaconf import OmegaConf

from chuck_dreamer.common.episode import Episode
from chuck_dreamer.lerobot.context import RunContext
from chuck_dreamer.lerobot.nodes.table_frame import (
  ObjPosArmNode,
  _quat_from_R,
  build_ee_table_nodes,
)
from chuck_dreamer.lerobot.trackset import EpisodeScope, TrackSet, run_episode
from chuck_dreamer.store import (
  ArtifactStore,
  ProvenanceRecord,
  for_dataset,
)

DS = "user/ds"
T = 4

# 90° about z plus a translation — easy to verify by hand.
_THETA = math.pi / 2
_R = [[math.cos(_THETA), -math.sin(_THETA), 0.0],
      [math.sin(_THETA),  math.cos(_THETA), 0.0],
      [0.0, 0.0, 1.0]]
_T_M = [0.1, -0.2, 0.05]      # metres in the payload


@pytest.fixture
def ctx(tmp_path):
  store = ArtifactStore(tmp_path / "store")
  scope = for_dataset(DS)
  store.put("table_to_arm", scope,
            {"R": _R, "t": _T_M, "convention": "p_table = R @ p_arm + t",
             "units": "m"},
            ProvenanceRecord(artifact="table_to_arm", scope=scope,
                             node="extract_table_to_arm"))
  cfg = OmegaConf.create({"store": {"root": str(store.root)}})
  return RunContext(source_repo=DS, config=cfg)


def _run(ctx, fields, nodes):
  ts = TrackSet(EpisodeScope(DS, 0), Episode.from_arrays(fields), {})
  run_episode(nodes, ts)
  return ts


def test_ee_table_chain_transforms_pose(ctx):
  rng = np.random.default_rng(0)
  p_arm_mm = rng.standard_normal((T, 3)).astype(np.float32) * 100
  q_arm = np.tile(np.array([1.0, 0, 0, 0], np.float32), (T, 1))

  ts = _run(ctx, {"ee_pos_arm": p_arm_mm, "ee_quat_arm": q_arm},
            build_ee_table_nodes(ctx))

  R = np.asarray(_R)
  t_mm = np.asarray(_T_M) * 1000.0
  expected = (p_arm_mm @ R.T + t_mm).astype(np.float32)
  np.testing.assert_allclose(
    ts.get_track("ee_pos_table").values, expected, atol=1e-3)

  # Identity arm quat → table quat is the transform's own rotation.
  q_table = ts.get_track("ee_quat_table").values
  np.testing.assert_allclose(q_table[0], _quat_from_R(R), atol=1e-6)

  # Action: shift-by-one of the table-frame pose, last frame repeated.
  pose = np.concatenate(
    [ts.get_track("ee_pos_table").values, q_table], axis=-1)
  action = ts.get_track("ee_action_table").values
  np.testing.assert_array_equal(action[:-1], pose[1:])
  np.testing.assert_array_equal(action[-1], pose[-1])


def test_obj_pos_arm_inverts_the_transform(ctx):
  rng = np.random.default_rng(1)
  p_arm_mm = rng.standard_normal((T, 3)).astype(np.float32) * 100
  R = np.asarray(_R)
  t_mm = np.asarray(_T_M) * 1000.0
  p_table_mm = (p_arm_mm @ R.T + t_mm).astype(np.float32)
  valid = np.array([True, False, True, True])

  ts = _run(ctx, {"obj_pos_table": p_table_mm, "obj_pos_table.valid": valid},
            [ObjPosArmNode(ctx)])

  track = ts.get_track("obj_pos_arm")
  np.testing.assert_allclose(track.values, p_arm_mm, atol=1e-3)
  # Validity propagates untouched (spec §10.2).
  np.testing.assert_array_equal(track.valid, valid)


def test_quat_from_R_roundtrip_axes():
  # All three principal 180° rotations hit the non-trace branches.
  for axis in range(3):
    R = -np.eye(3)
    R[axis, axis] = 1.0
    q = _quat_from_R(R)
    assert np.isclose(np.linalg.norm(q), 1.0)
    # Rebuild the rotation from the quaternion and compare.
    w, x, y, z = q
    R2 = np.array([
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    np.testing.assert_allclose(R2, R, atol=1e-9)
