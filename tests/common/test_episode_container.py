"""Tests for the field-attributed :class:`Episode` container."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from chuck_dreamer.common.episode import (
  Episode,
  Field,
  FieldKind,
  default_disposition,
)


# ---------------------------------------------------------------------------
# Disposition inference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,kind,persist", [
  ("image",        FieldKind.IMAGE,  True),
  ("object_xy",    FieldKind.SCALAR, True),
  ("joint_action", FieldKind.ACTION, True),
  ("segmentation_target", FieldKind.MASK, True),
  ("object_mesh_overlay", FieldKind.OVERLAY, True),
  ("object_masks", FieldKind.SCRATCH, False),
])
def test_known_field_disposition(name, kind, persist):
  assert default_disposition(name) == (kind, persist)


def test_unknown_segmentation_is_a_persisted_mask():
  assert default_disposition("segmentation_widget") == (FieldKind.MASK, True)


def test_unknown_field_defaults_to_persisted_scalar():
  assert default_disposition("some_new_signal") == (FieldKind.SCALAR, True)


# ---------------------------------------------------------------------------
# Mapping protocol — drop-in for the old dict
# ---------------------------------------------------------------------------


def test_setitem_getitem_returns_value():
  ep = Episode()
  arr = np.zeros((3, 2), dtype=np.float32)
  ep["object_xy"] = arr
  assert ep["object_xy"] is arr


def test_contains_and_get_and_len():
  ep = Episode()
  ep["reward"] = np.zeros(3)
  assert "reward" in ep
  assert "missing" not in ep
  assert ep.get("missing") is None
  assert ep.get("missing", 5) == 5
  assert len(ep) == 1


def test_items_and_iter_yield_names_and_values():
  ep = Episode.from_arrays({"a": np.arange(3), "b": np.arange(2)})
  assert set(ep) == {"a", "b"}
  assert {n: v.tolist() for n, v in ep.items()} == {"a": [0, 1, 2], "b": [0, 1]}


def test_delitem():
  ep = Episode()
  ep["x"] = np.zeros(1)
  del ep["x"]
  assert "x" not in ep


# ---------------------------------------------------------------------------
# Field-level API + scratch
# ---------------------------------------------------------------------------


def test_setitem_infers_field_metadata():
  ep = Episode()
  ep["segmentation_target"] = np.ones((2, 3, 3), bool)
  f = ep.field("segmentation_target")
  assert isinstance(f, Field)
  assert f.kind is FieldKind.MASK and f.persist is True


def test_set_overrides_kind_and_persist():
  ep = Episode()
  ep.set("object_masks", [None, None], kind=FieldKind.SCRATCH)
  f = ep.field("object_masks")
  assert f.kind is FieldKind.SCRATCH and f.persist is False


def test_scratch_field_excluded_from_persisted():
  ep = Episode()
  ep["object_xy"] = np.zeros((2, 2))
  ep.set("object_masks", ["m"], persist=False)
  names = {n for n, _ in ep.persisted()}
  assert names == {"object_xy"}
  # but it's still readable as ordinary content (and crosses pickle below).
  assert ep["object_masks"] == ["m"]


def test_explicit_persist_false_on_normal_kind():
  ep = Episode()
  ep.set("object_xy", np.zeros((2, 2)), persist=False)
  assert ep.field("object_xy").kind is FieldKind.SCALAR
  assert ep.field("object_xy").persist is False
  assert list(ep.persisted()) == []


# ---------------------------------------------------------------------------
# Picklability — required for the spawn-based parallel importer
# ---------------------------------------------------------------------------


def test_episode_pickle_roundtrip_preserves_fields_and_scratch():
  ep = Episode()
  ep["image"] = np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3)
  ep["object_xy"] = np.ones((2, 2), dtype=np.float32)
  ep.set("object_masks", [np.ones((2, 2), bool), None], persist=False)

  ep2 = pickle.loads(pickle.dumps(ep))
  assert set(ep2) == {"image", "object_xy", "object_masks"}
  np.testing.assert_array_equal(ep2["image"], ep["image"])
  # scratch field survives the pickle (it's the parallel mask handoff).
  assert ep2.field("object_masks").persist is False
  np.testing.assert_array_equal(ep2["object_masks"][0], np.ones((2, 2), bool))
  assert ep2["object_masks"][1] is None
