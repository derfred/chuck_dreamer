"""Harness node declarations: producer map and declaration-derived ordering."""
from __future__ import annotations

import numpy as np
import pytest

from chuck_dreamer.common.tracks import Persist, TrackSpec
from chuck_dreamer.lerobot.harness import (
  InputDecl,
  SpecError,
  producer_map,
  resolve_order,
)


class _FakeNode:
  scope_level = "episode"

  def __init__(self, name, inputs, outputs):
    self.name = name
    self.inputs = tuple(InputDecl(i) for i in inputs)
    self.outputs = tuple(
      TrackSpec(o, np.float32, (), persist=Persist.EPHEMERAL) for o in outputs)

  def run(self, view):
    return []


def test_resolve_order_follows_declared_tracks():
  nodes = [
    _FakeNode("ee_pos_table", ["table_to_arm", "ee_pos_arm"], ["ee_pos_table"]),
    _FakeNode("normalize_joints", ["joint_values"], ["joint_values_rad"]),
    _FakeNode("ee_pos_arm", ["fk_model", "joint_values_rad"], ["ee_pos_arm"]),
  ]
  leaves = {"joint_values", "fk_model", "table_to_arm"}
  order = [n.name for n in resolve_order(nodes, {"ee_pos_table"}, leaves)]
  assert order == ["normalize_joints", "ee_pos_arm", "ee_pos_table"]


def test_missing_producer_is_spec_error():
  nodes = [_FakeNode("a", ["mystery"], ["out"])]
  with pytest.raises(SpecError, match="no producer"):
    resolve_order(nodes, {"out"})


def test_duplicate_producer_is_spec_error():
  nodes = [_FakeNode("a", [], ["x"]), _FakeNode("b", [], ["x"])]
  with pytest.raises(SpecError, match="produced by both"):
    producer_map(nodes)


def test_cycle_is_spec_error():
  nodes = [_FakeNode("a", ["y"], ["x"]), _FakeNode("b", ["x"], ["y"])]
  with pytest.raises(SpecError, match="cycle"):
    resolve_order(nodes, {"x"})


def test_unknown_target_is_spec_error():
  with pytest.raises(SpecError, match="no node produces"):
    resolve_order([], {"obj_uv"})
