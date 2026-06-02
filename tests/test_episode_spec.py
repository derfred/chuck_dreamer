"""Tests for the ``EpisodeSpec`` / ``EpisodeSlice`` episode-selection spec.

These cover the pure-logic surface — grammar parsing, the Click-error
wrapping, episode filtering and the ``EpisodeSlice`` accessors — without
touching LeRobot or the network. ``read_episodes`` itself (which reads
``LeRobotDatasetMetadata``) is not exercised here; only the filtering it
delegates to is, via the public surface.
"""
from __future__ import annotations

import click
import pytest

from chuck_dreamer.common.episode_spec import EpisodeSlice, EpisodeSpec


def _slices(indices: list[int]) -> list[EpisodeSlice]:
  """Minimal slices carrying just the episode_index we filter on."""
  return [
    EpisodeSlice(
      episode_index=i, data_from=i * 10, data_to=i * 10 + 5,
      video_from_ts=0.0, video_to_ts=0.5,
      data_chunk=0, data_file=0, video_chunk=0, video_file=0,
      task="", length=5,
    )
    for i in indices
  ]


# ---------------------------------------------------------------------------
# parse — dataset id + episode selection
# ---------------------------------------------------------------------------


def test_parse_bare_dataset_id_means_all_episodes():
  spec = EpisodeSpec.parse("user/ds")
  assert spec.dataset_id == "user/ds"
  assert spec.episodes is None
  assert spec.frames is None


def test_parse_single_episode():
  spec = EpisodeSpec.parse("user/ds#3")
  assert spec.dataset_id == "user/ds"
  assert spec.episodes == [3]


def test_parse_mixed_singletons_and_ranges():
  spec = EpisodeSpec.parse("user/ds#0,2-4")
  assert spec.episodes == [0, 2, 3, 4]


def test_parse_all_keyword_is_none():
  assert EpisodeSpec.parse("user/ds#all").episodes is None


def test_parse_dedupes_overlapping_ranges_preserving_order():
  spec = EpisodeSpec.parse("user/ds#2-4,3-5")
  assert spec.episodes == [2, 3, 4, 5]


def test_parse_episode_range_expands_inclusive():
  spec = EpisodeSpec.parse("user/ds#0-3")
  assert spec.episodes == [0, 1, 2, 3]


def test_parse_top_level_colon_is_episodes_frames_split_not_episode_stride():
  # The first ':' after '#' separates episodes from frames, so a stride
  # can't be expressed on the episodes portion: '0-10:5' means
  # "episodes 0..10, frame 5", NOT "episodes 0,5,10". Stride is only
  # reachable inside the frames slot (see the frames stride test).
  spec = EpisodeSpec.parse("user/ds#0-10:5")
  assert spec.episodes == list(range(0, 11))
  assert spec.frames == [5]


# ---------------------------------------------------------------------------
# parse — open-ended episode ranges (N- and -M)
# ---------------------------------------------------------------------------


def test_parse_open_upper_episode_range_defers_finite_list():
  # ``8-`` can't be expanded without the dataset, so it lands in
  # ``episode_open_from`` and leaves the finite list empty (but non-None).
  spec = EpisodeSpec.parse("user/ds#8-")
  assert spec.episodes == []
  assert spec.episode_open_from == 8


def test_parse_open_lower_episode_range_is_finite_from_zero():
  # ``-3`` is just ``0-3`` and expands immediately; no open bound.
  spec = EpisodeSpec.parse("user/ds#-3")
  assert spec.episodes == [0, 1, 2, 3]
  assert spec.episode_open_from is None


def test_parse_mixes_finite_picks_with_open_upper():
  spec = EpisodeSpec.parse("user/ds#0,2,9-")
  assert spec.episodes == [0, 2]
  assert spec.episode_open_from == 9


def test_parse_open_upper_takes_smallest_n_across_tokens():
  spec = EpisodeSpec.parse("user/ds#12-,5-")
  assert spec.episode_open_from == 5


def test_open_upper_episode_filter_is_none_unknown_until_resolved():
  # The selected set can't be known without reading the dataset, so the
  # filter reports None (callers use it only for a progress total).
  assert EpisodeSpec.parse("user/ds#8-").episode_filter is None


def test_select_open_upper_keeps_every_available_index_at_or_above_n():
  spec = EpisodeSpec.parse("user/ds#8-")
  kept = spec._select_episodes(_slices([5, 6, 7, 8, 9, 10]))
  assert [s.episode_index for s in kept] == [8, 9, 10]


def test_select_open_upper_unioned_with_finite_picks():
  spec = EpisodeSpec.parse("user/ds#0,2,9-")
  kept = spec._select_episodes(_slices([0, 1, 2, 3, 9, 10, 11]))
  assert [s.episode_index for s in kept] == [0, 2, 9, 10, 11]


def test_parse_open_upper_rejects_stride():
  # A stride needs a closed upper bound to count toward. (Reachable only
  # in the frames slot, since the first ':' splits episodes from frames.)
  with pytest.raises(click.ClickException):
    EpisodeSpec.parse("user/ds#1:5-:2")


# ---------------------------------------------------------------------------
# parse — frames portion
# ---------------------------------------------------------------------------


def test_parse_frames_when_allowed():
  spec = EpisodeSpec.parse("user/ds#1:0-200:50")
  assert spec.episodes == [1]
  assert spec.frames == [0, 50, 100, 150, 200]


def test_parse_frames_with_all_episodes():
  spec = EpisodeSpec.parse("user/ds#all:0-2")
  assert spec.episodes is None
  assert spec.frames == [0, 1, 2]


def test_parse_open_lower_frames_is_finite_from_zero():
  spec = EpisodeSpec.parse("user/ds#1:-2")
  assert spec.frames == [0, 1, 2]
  assert spec.frame_open_from is None


def test_parse_open_upper_frames_records_open_from():
  spec = EpisodeSpec.parse("user/ds#1:100-")
  assert spec.frames == []
  assert spec.frame_open_from == 100


def test_parse_rejects_frames_when_not_allowed():
  with pytest.raises(click.ClickException, match="FRAMES"):
    EpisodeSpec.parse("user/ds#1:0-200", allow_frames=False)


def test_parse_frames_rejection_names_the_command():
  with pytest.raises(click.ClickException, match="import-lerobot"):
    EpisodeSpec.parse(
      "user/ds#1:0-200", allow_frames=False, command="import-lerobot")


def test_parse_episode_only_is_fine_when_frames_disallowed():
  spec = EpisodeSpec.parse("user/ds#0,2", allow_frames=False)
  assert spec.episodes == [0, 2]
  assert spec.frames is None


# ---------------------------------------------------------------------------
# parse — error handling (ValueError -> ClickException)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [
  "",                    # empty
  "#0",                  # empty dataset id before '#'
  "user/ds#5-2",         # reversed range
  "user/ds#abc",         # non-integer token
  "user/ds#0:1:2:3",     # stride on a singleton (1:2 then :3 invalid)
  "user/ds#-",           # open range with neither bound
  "user/ds#-x",          # open-lower with a non-integer bound
])
def test_parse_malformed_raises_clickexception(raw):
  with pytest.raises(click.ClickException):
    EpisodeSpec.parse(raw)


def test_parse_does_not_leak_valueerror():
  # The grammar raises ValueError internally; parse must convert it so
  # callers only ever catch ClickException.
  with pytest.raises(click.ClickException):
    EpisodeSpec.parse("user/ds#9-3")


# ---------------------------------------------------------------------------
# parse_many
# ---------------------------------------------------------------------------


def test_parse_many_parses_each_token():
  specs = EpisodeSpec.parse_many(["user/a", "user/b#3"])
  assert [s.dataset_id for s in specs] == ["user/a", "user/b"]
  assert [s.episodes for s in specs] == [None, [3]]


def test_parse_many_propagates_allow_frames():
  with pytest.raises(click.ClickException, match="FRAMES"):
    EpisodeSpec.parse_many(["user/a#0:0-1"], allow_frames=False)


# ---------------------------------------------------------------------------
# episode_filter
# ---------------------------------------------------------------------------


def test_episode_filter_none_for_all():
  assert EpisodeSpec.parse("user/ds").episode_filter is None


def test_episode_filter_is_a_set():
  assert EpisodeSpec.parse("user/ds#0,2-4").episode_filter == {0, 2, 3, 4}


# ---------------------------------------------------------------------------
# _select_episodes (exercised through the public read path's contract)
# ---------------------------------------------------------------------------


def test_select_episodes_keeps_only_requested():
  spec = EpisodeSpec.parse("user/ds#0,2,4")
  kept = spec._select_episodes(_slices([0, 1, 2, 3, 4, 5]))
  assert [s.episode_index for s in kept] == [0, 2, 4]


def test_select_episodes_all_returns_everything():
  spec = EpisodeSpec.parse("user/ds")
  given = _slices([0, 1, 2])
  assert spec._select_episodes(given) == given


def test_select_episodes_preserves_input_order_not_spec_order():
  # Spec lists 4 before 1, but input slices are ascending; the result
  # follows the *input* order (matches the importer's sorted slices).
  spec = EpisodeSpec.parse("user/ds#4,1")
  kept = spec._select_episodes(_slices([0, 1, 2, 3, 4]))
  assert [s.episode_index for s in kept] == [1, 4]


def test_select_episodes_ignores_requested_but_absent():
  spec = EpisodeSpec.parse("user/ds#1,99")
  kept = spec._select_episodes(_slices([0, 1, 2]))
  assert [s.episode_index for s in kept] == [1]


# ---------------------------------------------------------------------------
# EpisodeSlice
# ---------------------------------------------------------------------------


def test_episode_slice_bounds():
  s = _slices([3])[0]
  assert s.bounds == (30, 35)
  assert s.bounds == (s.data_from, s.data_to)
