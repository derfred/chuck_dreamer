"""Tests for the ``prompt-episodes`` keyframe-spec parsing.

Cover the pure-logic surface of ``--keyframes`` parsing and offset
resolution: plain manual tokens, the ``A:B:step`` augment-range syntax,
endpoint deduplication, error cases, and episode-relative resolution.
Nothing here touches LeRobot, OpenCV, or the pose estimator.
"""
from __future__ import annotations

import pytest

from chuck_dreamer.lerobot.object_localization.cli.prompt_episodes import (
  _AugmentRange, _kf_for_len, _parse_keyframes, _resolve_keyframe_offset,
)


# ---------------------------------------------------------------------------
# _parse_keyframes: plain (manual-only) specs
# ---------------------------------------------------------------------------


def test_plain_start_end_no_ranges():
  manual, ranges = _parse_keyframes("start,end")
  assert manual == ["start", "end"]
  assert ranges == []


def test_plain_signed_ints():
  manual, ranges = _parse_keyframes("start,400,-1")
  assert manual == ["start", 400, -1]
  assert ranges == []


def test_tokens_are_lowercased_and_stripped():
  manual, ranges = _parse_keyframes("  START , End ")
  assert manual == ["start", "end"]
  assert ranges == []


def test_manual_offsets_deduped_preserving_order():
  manual, _ = _parse_keyframes("start,start,end,start")
  assert manual == ["start", "end"]


# ---------------------------------------------------------------------------
# _parse_keyframes: A:B:step augment ranges
# ---------------------------------------------------------------------------


def test_single_range_adds_endpoints_to_manual():
  manual, ranges = _parse_keyframes("start:end:30")
  assert manual == ["start", "end"]
  assert ranges == [_AugmentRange(a="start", b="end", step=30)]


def test_multi_range_per_range_steps():
  manual, ranges = _parse_keyframes("start:400:20,400:end:40")
  # 400 appears as the right endpoint of the first range and the left
  # endpoint of the second, but is deduped in the manual list.
  assert manual == ["start", 400, "end"]
  assert ranges == [
    _AugmentRange(a="start", b=400, step=20),
    _AugmentRange(a=400, b="end", step=40),
  ]


def test_range_endpoints_dedupe_against_plain_token():
  manual, ranges = _parse_keyframes("start:end:30,end")
  assert manual == ["start", "end"]
  assert ranges == [_AugmentRange(a="start", b="end", step=30)]


def test_range_with_signed_int_endpoints():
  manual, ranges = _parse_keyframes("0:-1:25")
  assert manual == [0, -1]
  assert ranges == [_AugmentRange(a=0, b=-1, step=25)]


def test_mixed_plain_and_range_tokens():
  manual, ranges = _parse_keyframes("100,start:end:30")
  assert manual == [100, "start", "end"]
  assert ranges == [_AugmentRange(a="start", b="end", step=30)]


# ---------------------------------------------------------------------------
# _parse_keyframes: error cases (raise ValueError)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ["", "   ", ",,"])
def test_empty_spec_raises(spec):
  with pytest.raises(ValueError, match="empty keyframes spec"):
    _parse_keyframes(spec)


@pytest.mark.parametrize("spec", ["start:end", "start:end:30:5", "1:2"])
def test_range_wrong_part_count_raises(spec):
  with pytest.raises(ValueError, match="must be 'A:B:step'"):
    _parse_keyframes(spec)


@pytest.mark.parametrize("spec", ["start:end:0", "start:end:-5"])
def test_range_nonpositive_step_raises(spec):
  with pytest.raises(ValueError, match="step must be positive"):
    _parse_keyframes(spec)


def test_range_noninteger_step_raises():
  with pytest.raises(ValueError, match="step .* is not an integer"):
    _parse_keyframes("start:end:abc")


@pytest.mark.parametrize("spec", ["foo", "a:b:10"])
def test_bad_offset_token_raises(spec):
  with pytest.raises(ValueError, match="is not an integer or 'start'/'end'"):
    _parse_keyframes(spec)


# ---------------------------------------------------------------------------
# _resolve_keyframe_offset
# ---------------------------------------------------------------------------


def test_resolve_start_end():
  assert _resolve_keyframe_offset("start", 100) == 0
  assert _resolve_keyframe_offset("end", 100) == 99


def test_resolve_positive_int_in_bounds():
  assert _resolve_keyframe_offset(50, 100) == 50


def test_resolve_negative_counts_from_end():
  assert _resolve_keyframe_offset(-1, 100) == 99
  assert _resolve_keyframe_offset(-10, 100) == 90


@pytest.mark.parametrize("offset", [100, 250, -101])
def test_resolve_out_of_bounds_is_none(offset):
  assert _resolve_keyframe_offset(offset, 100) is None


# ---------------------------------------------------------------------------
# _kf_for_len
# ---------------------------------------------------------------------------


def test_kf_for_len_resolves_and_dedupes():
  # 'start' -> 0, 0 -> 0 (dup, dropped), 'end' -> 9, -1 -> 9 (dup, dropped).
  assert _kf_for_len(["start", 0, "end", -1], 10) == [0, 9]


def test_kf_for_len_drops_out_of_bounds():
  assert _kf_for_len(["start", 500, "end"], 10) == [0, 9]
