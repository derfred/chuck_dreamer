"""Track/TrackSpec validation, unit conversion, and frame checks."""
from __future__ import annotations

import numpy as np
import pytest

from chuck_dreamer.common.tracks import (
  ChannelUnits,
  Frame,
  FrameMismatch,
  Track,
  TrackSpec,
  Unit,
  UnitMismatch,
  check_frame,
  convert_unit,
)


def _track(name="t", shape=(3,), frame=Frame.ARM, unit=Unit.M,
           validity=False, T=4, valid=None, dtype=np.float32):
  spec = TrackSpec(name, dtype, shape, frame, unit, validity)
  values = np.ones((T, *shape), dtype=dtype)
  return Track(spec, values, valid)


# ---------------------------------------------------------------------------
# Track validation
# ---------------------------------------------------------------------------
def test_shape_and_dtype_enforced():
  with pytest.raises(ValueError, match="per-frame shape"):
    Track(TrackSpec("t", np.float32, (3,)), np.zeros((4, 2), np.float32))
  with pytest.raises(ValueError, match="dtype"):
    Track(TrackSpec("t", np.float32, (3,)), np.zeros((4, 3), np.float64))


def test_validity_contract():
  with pytest.raises(ValueError, match="declares validity"):
    _track(validity=True)                      # mask required
  with pytest.raises(ValueError, match="does not declare validity"):
    _track(valid=np.ones(4, bool))             # mask forbidden
  t = _track(validity=True, valid=np.array([1, 0, 1, 1], bool))
  assert t.valid is not None and not t.valid[1]


def test_tracks_are_immutable():
  t = _track()
  with pytest.raises(ValueError):
    t.values[0] = 9.0


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
def test_scalar_conversion_deg_to_rad():
  t = _track(unit=Unit.DEG)
  out = convert_unit(t, Unit.RAD)
  np.testing.assert_allclose(out.values, np.deg2rad(1.0), rtol=1e-6)
  assert out.spec.unit == Unit.RAD
  assert convert_unit(t, Unit.DEG) is t      # no-op returns same object


def test_mm_to_m():
  t = _track(unit=Unit.MM)
  np.testing.assert_allclose(convert_unit(t, Unit.M).values, 1e-3)


def test_non_mechanical_conversion_raises():
  with pytest.raises(UnitMismatch, match="no mechanical conversion"):
    convert_unit(_track(unit=Unit.PX), Unit.M)


def test_channel_group_conversion_leaves_gripper_alone():
  # The joint_values case: positioning joints deg -> rad, gripper untouched.
  src = ChannelUnits({(0, 5): Unit.DEG, (5, 6): Unit.UNITLESS})
  dst = ChannelUnits({(0, 5): Unit.RAD, (5, 6): Unit.UNITLESS})
  spec = TrackSpec("joint_values", np.float32, (6,), Frame.NONE, src)
  values = np.full((2, 6), 90.0, dtype=np.float32)
  out = convert_unit(Track(spec, values), dst)
  np.testing.assert_allclose(out.values[:, :5], np.pi / 2, rtol=1e-6)
  np.testing.assert_allclose(out.values[:, 5], 90.0)


def test_channel_ranges_must_match():
  src = ChannelUnits({(0, 5): Unit.DEG})
  dst = ChannelUnits({(0, 4): Unit.RAD})
  spec = TrackSpec("t", np.float32, (5,), Frame.NONE, src)
  with pytest.raises(UnitMismatch, match="channel ranges differ"):
    convert_unit(Track(spec, np.zeros((1, 5), np.float32)), dst)


def test_overlapping_channel_ranges_rejected():
  with pytest.raises(ValueError, match="overlapping"):
    ChannelUnits({(0, 5): Unit.DEG, (4, 6): Unit.UNITLESS})


# ---------------------------------------------------------------------------
# Frame soundness
# ---------------------------------------------------------------------------
def test_frame_mismatch_is_an_error():
  with pytest.raises(FrameMismatch, match="explicit pipeline node"):
    check_frame(_track(frame=Frame.ARM), Frame.TABLE, node="ee_action")
  check_frame(_track(frame=Frame.ARM), Frame.ARM, node="ok")
