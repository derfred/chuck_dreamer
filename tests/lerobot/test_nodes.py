"""Native EE-chain nodes: numeric equivalence with the legacy EePosStage.

Runs the chain through the harness with a fake FK and checks each output
against the legacy stage's exact formulas (deg→rad on positioning joints
only, per-frame FK, constant down-quat, shift-by-one action)."""
from __future__ import annotations

import numpy as np
import pytest

from chuck_dreamer.common.episode import Episode
from chuck_dreamer.lerobot.nodes.ee import (
  EE_QUAT_DOWN,
  N_POSITIONING_JOINTS,
  build_ee_nodes,
)
from chuck_dreamer.lerobot.context import RunContext
from chuck_dreamer.lerobot.trackset import EpisodeScope, TrackSet, run_episode

T, J = 5, 6


@pytest.fixture
def ctx():
  c = RunContext(source_repo="user/ds")
  # Fake FK: deterministic function of the positioning joints.
  c._fk = lambda q: np.array([q.sum(), q[0], 1.0], dtype=np.float64)
  return c


def _run_chain(ctx, qpos_deg):
  episode = Episode.from_arrays({
    "joint_qpos": qpos_deg,
    "timestamp":  np.arange(T, dtype=np.float32),
  })
  ts = TrackSet(EpisodeScope("user/ds", 0), episode, {})
  run_episode(build_ee_nodes(ctx), ts)
  return episode


def test_chain_matches_legacy_formulas(ctx):
  rng = np.random.default_rng(0)
  qpos_deg = rng.uniform(-90, 90, size=(T, J)).astype(np.float32)
  episode = _run_chain(ctx, qpos_deg)

  # normalize_joints: deg→rad on the first 5 joints, gripper untouched,
  # emitted under its own name.
  rescaled = np.asarray(episode.get("joint_qpos_rad"))
  np.testing.assert_array_equal(
    rescaled[:, :N_POSITIONING_JOINTS],
    np.deg2rad(qpos_deg[:, :N_POSITIONING_JOINTS]))
  np.testing.assert_array_equal(rescaled[:, 5], qpos_deg[:, 5])
  assert rescaled.dtype == np.float32

  # …and the raw degrees reading survives untouched alongside it.
  np.testing.assert_array_equal(np.asarray(episode.get("joint_qpos")), qpos_deg)

  # ee_pos_arm: per-frame FK on the rescaled positioning joints, in mm.
  ee_pos = np.asarray(episode.get("ee_pos_arm"))
  expected = np.stack([
    (ctx._fk(rescaled[t, :N_POSITIONING_JOINTS]) * 1000.0).astype(np.float32)
    for t in range(T)])
  np.testing.assert_array_equal(ee_pos, expected)

  # ee_rotation: constant down-quat.
  ee_quat = np.asarray(episode.get("ee_quat_arm"))
  np.testing.assert_array_equal(
    ee_quat, np.tile(np.asarray(EE_QUAT_DOWN, np.float32), (T, 1)))

  # ee_action_arm: pose shifted by one, last frame repeated.
  ee_action = np.asarray(episode.get("ee_action_arm"))
  pose = np.concatenate([ee_pos, ee_quat], axis=-1)
  np.testing.assert_array_equal(ee_action[:-1], pose[1:])
  np.testing.assert_array_equal(ee_action[-1], pose[-1])
  assert ee_action.shape == (T, 7) and ee_action.dtype == np.float32


def test_chain_tracks_carry_declared_specs(ctx):
  from chuck_dreamer.common.tracks import Frame, Unit

  qpos_deg = np.zeros((T, J), dtype=np.float32)
  episode = Episode.from_arrays({
    "joint_qpos": qpos_deg,
    "timestamp":  np.arange(T, dtype=np.float32),
  })
  ts = TrackSet(EpisodeScope("user/ds", 0), episode, {})
  run_episode(build_ee_nodes(ctx), ts)

  assert ts.get_track("ee_pos_arm").spec.frame == Frame.ARM
  assert ts.get_track("ee_pos_arm").spec.unit == Unit.MM
  # joint_qpos_rad declares radians on the positioning joints; the raw
  # joint_qpos leaf keeps the dataset's degrees.
  assert ts.get_track("joint_qpos_rad").spec.unit.groups[0] == ((0, 5), Unit.RAD)
  assert ts.get_track("joint_qpos").spec.unit.groups[0] == ((0, 5), Unit.DEG)


def test_ee_chain_resolves_without_a_cycle(ctx):
  """The split names let the harness derive the order — the in-place form
  was a self-edge and `resolve_order` rejected it as a dependency cycle."""
  from chuck_dreamer.lerobot.harness import resolve_order

  order = resolve_order(
    build_ee_nodes(ctx), {"ee_action_arm"},
    leaves=frozenset({"joint_qpos", "timestamp", "fk_model"}))
  names = [n.name for n in order]
  assert names.index("normalize_joints") < names.index("ee_pos_arm")
