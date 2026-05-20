"""Tests for Tracker.maybe_log_collect_episode — periodic dumps of collected episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import chuck_dreamer.training.tracker as tracker_mod
from chuck_dreamer.training.tracker import Tracker


@dataclass
class _FakeScene:
  """Minimal dataclass shaped like SceneConfig so asdict() works."""
  goal_pos: list[float] = field(default_factory=lambda: [0.1, 0.2])
  max_steps: int = 100
  mat_centre: list[float] | None = None


class _FakeWriter:
  """Captures write_episode() calls so tests can inspect what was dumped."""
  def __init__(self, output_dir: str = "", format: str = "rerun") -> None:
    self.output_dir = output_dir
    self.format     = format
    self.calls: list[tuple[dict[str, Any], dict[str, Any], str]] = []

  def write_episode(self, episode: dict[str, Any], metadata: dict[str, Any], *, name_suffix: str) -> None:
    self.calls.append((episode, metadata, name_suffix))


@pytest.fixture
def patch_writer(monkeypatch):
  """Replace the EpisodeWriter factory with one that records all instances."""
  instances: list[_FakeWriter] = []

  def _factory(output_dir: str, format: str = "rerun"):
    w = _FakeWriter(output_dir, format)
    instances.append(w)
    return w

  monkeypatch.setattr(tracker_mod, "EpisodeWriter", _factory)
  return instances


def _make_config(*, every_n_collects, dump_path="/tmp/episodes", experiment_name=None,
                 dump_format="rerun", seed=42, logger="none", every_n_evals=None):
  """Build a SimpleNamespace config matching the new logging.episodes shape."""
  return SimpleNamespace(
    seed=seed,
    logging=SimpleNamespace(
      logger=logger,
      project_name="test",
      experiment_name=experiment_name,
      episodes=SimpleNamespace(
        dump_path=dump_path,
        dump_format=dump_format,
        every_n_collects=every_n_collects,
        every_n_evals=every_n_evals,
      ),
    ),
  )


# ---------------------------------------------------------------------------
# Disabled paths
# ---------------------------------------------------------------------------


def test_no_writer_when_every_n_collects_is_none(patch_writer):
  Tracker(_make_config(every_n_collects=None))
  assert patch_writer == []


def test_no_writer_when_dump_path_is_none(patch_writer):
  Tracker(_make_config(every_n_collects=10, dump_path=None))
  assert patch_writer == []


def test_no_writer_when_every_n_collects_is_zero(patch_writer):
  # 0 should disable just like None — the truthiness check guards both.
  Tracker(_make_config(every_n_collects=0))
  assert patch_writer == []


def test_maybe_log_is_noop_when_disabled(patch_writer):
  tracker = Tracker(_make_config(every_n_collects=None))
  for _ in range(5):
    tracker.maybe_log_collect_episode({"image": np.zeros(1)}, _FakeScene(), outcome="success")
  assert patch_writer == []


# ---------------------------------------------------------------------------
# Cadence
# ---------------------------------------------------------------------------


def test_writes_every_nth_call(patch_writer):
  tracker = Tracker(_make_config(every_n_collects=3))
  writer = patch_writer[0]
  for _ in range(9):
    tracker.maybe_log_collect_episode({}, _FakeScene(), outcome="success")
  # Calls #3, #6, #9 trigger writes.
  assert len(writer.calls) == 3


def test_first_write_happens_on_nth_call_not_first(patch_writer):
  tracker = Tracker(_make_config(every_n_collects=4))
  writer = patch_writer[0]
  for i in range(1, 5):
    tracker.maybe_log_collect_episode({}, _FakeScene(), outcome="success")
    expected = 1 if i == 4 else 0
    assert len(writer.calls) == expected


def test_cadence_n_equals_one_writes_every_call(patch_writer):
  tracker = Tracker(_make_config(every_n_collects=1))
  writer = patch_writer[0]
  for _ in range(5):
    tracker.maybe_log_collect_episode({}, _FakeScene(), outcome="success")
  assert len(writer.calls) == 5


# ---------------------------------------------------------------------------
# Writer construction
# ---------------------------------------------------------------------------


def test_writer_constructed_with_dump_path_and_format(patch_writer):
  Tracker(_make_config(every_n_collects=2, dump_path="/var/tmp/dumps", dump_format="hdf5"))
  assert len(patch_writer) == 1
  w = patch_writer[0]
  assert w.output_dir == "/var/tmp/dumps"
  assert w.format     == "hdf5"


# ---------------------------------------------------------------------------
# Metadata payload
# ---------------------------------------------------------------------------


def test_metadata_payload_shape(patch_writer):
  tracker = Tracker(_make_config(every_n_collects=1, seed=7))
  writer = patch_writer[0]
  scene = _FakeScene(goal_pos=[0.3, -0.4], max_steps=77)
  episode = {"image": np.zeros((1, 4, 4, 3), dtype=np.uint8)}

  tracker.maybe_log_collect_episode(episode, scene, outcome="timeout", data={"iteration": 12})

  assert len(writer.calls) == 1
  ep_arg, meta, _ = writer.calls[0]
  assert ep_arg is episode
  assert meta["config"]     == {"goal_pos": [0.3, -0.4], "max_steps": 77, "mat_centre": None}
  assert meta["seed"]       == 7
  assert meta["source"]     == "collect"
  assert meta["outcome"]    == "timeout"
  assert meta["goal_xy"]    == [0.3, -0.4]
  assert meta["mat_centre"] is None
  assert meta["iteration"]  == 12


def test_extra_data_does_not_clobber_required_metadata(patch_writer):
  # ``data`` is merged after the required keys, so a caller passing a
  # conflicting key would override — this test pins the current behavior so
  # any future change is intentional.
  tracker = Tracker(_make_config(every_n_collects=1))
  writer = patch_writer[0]
  tracker.maybe_log_collect_episode({}, _FakeScene(), outcome="success",
                                    data={"source": "override"})
  _, meta, _ = writer.calls[0]
  assert meta["source"] == "override"


# ---------------------------------------------------------------------------
# Suffix
# ---------------------------------------------------------------------------


def test_suffix_uses_zero_padded_dump_index(patch_writer):
  tracker = Tracker(_make_config(every_n_collects=2, experiment_name=None))
  writer = patch_writer[0]
  for _ in range(6):
    tracker.maybe_log_collect_episode({}, _FakeScene(), outcome="success")
  # Dumps numbered 1, 2, 3 — the index is (count // every_n).
  suffixes = [s for _, _, s in writer.calls]
  assert suffixes == ["001", "002", "003"]


def test_suffix_includes_experiment_name(patch_writer):
  tracker = Tracker(_make_config(every_n_collects=1, experiment_name="exp42"))
  writer = patch_writer[0]
  tracker.maybe_log_collect_episode({}, _FakeScene(), outcome="success")
  tracker.maybe_log_collect_episode({}, _FakeScene(), outcome="success")
  suffixes = [s for _, _, s in writer.calls]
  assert suffixes == ["exp42-001", "exp42-002"]


# ---------------------------------------------------------------------------
# Counter is per-root, shared across derived/scoped trackers
# ---------------------------------------------------------------------------


def test_derived_tracker_shares_counter_with_root(patch_writer):
  root = Tracker(_make_config(every_n_collects=2))
  writer = patch_writer[0]

  child = root.derive({"phase": "collect"})
  # Two calls on the child — cadence is keyed off the shared root counter,
  # so the second call should trigger the first dump.
  child.maybe_log_collect_episode({}, _FakeScene(), outcome="success")
  assert len(writer.calls) == 0
  child.maybe_log_collect_episode({}, _FakeScene(), outcome="success")
  assert len(writer.calls) == 1

  # Another call on the root then continues the same cadence.
  root.maybe_log_collect_episode({}, _FakeScene(), outcome="success")
  assert len(writer.calls) == 1
  root.maybe_log_collect_episode({}, _FakeScene(), outcome="success")
  assert len(writer.calls) == 2


def test_scope_shares_counter_with_root(patch_writer):
  root = Tracker(_make_config(every_n_collects=1))
  writer = patch_writer[0]
  with root.scope({"phase": "collect"}) as scoped:
    scoped.maybe_log_collect_episode({}, _FakeScene(), outcome="success")
    scoped.maybe_log_collect_episode({}, _FakeScene(), outcome="success")
  # Both calls flow through the root writer.
  assert len(writer.calls) == 2


def test_derive_does_not_create_extra_writer(patch_writer):
  root = Tracker(_make_config(every_n_collects=2))
  root.derive({"phase": "collect"})
  root.derive({"phase": "train"})
  # Exactly one writer should ever be constructed — derived trackers must
  # not re-instantiate it (which would silently dump to disk again).
  assert len(patch_writer) == 1
