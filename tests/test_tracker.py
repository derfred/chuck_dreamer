"""Tests for the training Tracker (derive + log forwarding)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import trackio

from chuck_dreamer.training.tracker import Tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def calls(monkeypatch):
  """Capture trackio.log payloads in-process, no real backend involved."""
  recorded: list[dict] = []
  monkeypatch.setattr(trackio, "log", lambda payload: recorded.append(payload))
  return recorded


def _make_root(logger: str = "trackio") -> Tracker:
  """Build a root Tracker pointed at the given logger backend.

  ``logging.episodes.dump_path`` is left None so the Tracker takes the
  disabled branch and no real EpisodeWriter is constructed.
  """
  config = SimpleNamespace(
    logging=SimpleNamespace(
      logger=logger,
      project_name="test",
      episodes=SimpleNamespace(every_n_collects=None, every_n_evals=None, dump_path=None, dump_format="rerun"),
    ),
  )
  return Tracker(config)


# ---------------------------------------------------------------------------
# Root behavior
# ---------------------------------------------------------------------------


def test_root_log_forwards_to_backend(calls):
  root = _make_root()
  root.log({"loss": 1.5})
  assert calls == [{"loss": 1.5}]


def test_root_log_with_unknown_logger_is_noop(calls):
  # logger != "wandb"/"trackio" → log() falls through silently.
  root = _make_root(logger="none")
  root.log({"loss": 0.1})
  assert calls == []


# ---------------------------------------------------------------------------
# Derive
# ---------------------------------------------------------------------------


def test_derive_returns_child_tracker_with_parent():
  root = _make_root()
  child = root.derive({"phase": "train"})

  assert isinstance(child, Tracker)
  assert child._parent is root
  assert child.data == {"phase": "train"}
  assert child.config is root.config


def test_derive_does_not_mutate_parent_data():
  root = _make_root()
  root.derive({"phase": "train"})

  assert root.data == {}


def test_derive_child_log_merges_context_into_payload(calls):
  root = _make_root()
  child = root.derive({"phase": "train", "epoch": 3})

  child.log({"loss": 0.42})

  assert calls == [{"phase": "train", "epoch": 3, "loss": 0.42}]


def test_payload_overrides_context_on_key_conflict(calls):
  # Child.log merges as {**self.data, **data}, so explicit log keys win.
  root = _make_root()
  child = root.derive({"phase": "train"})

  child.log({"phase": "eval", "loss": 0.1})

  assert calls == [{"phase": "eval", "loss": 0.1}]


# ---------------------------------------------------------------------------
# Scope (context manager)
# ---------------------------------------------------------------------------


def test_scope_yields_child_tracker_with_parent():
  root = _make_root()
  with root.scope({"phase": "train"}) as child:
    assert isinstance(child, Tracker)
    assert child._parent is root
    assert child.data == {"phase": "train"}


def test_scope_child_log_merges_context_into_payload(calls):
  root = _make_root()
  with root.scope({"phase": "train", "epoch": 3}) as child:
    child.log({"loss": 0.42})

  assert calls == [{"phase": "train", "epoch": 3, "loss": 0.42}]


def test_scope_supports_nested_derive(calls):
  root = _make_root()
  with root.scope({"phase": "train"}) as scoped:
    scoped.derive({"epoch": 1}).log({"loss": 0.5})

  assert calls == [{"phase": "train", "epoch": 1, "loss": 0.5}]


def test_scope_propagates_exceptions():
  root = _make_root()
  with pytest.raises(RuntimeError, match="boom"):
    with root.scope({"phase": "train"}):
      raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Nested derive
# ---------------------------------------------------------------------------


def test_nested_derive_accumulates_context(calls):
  root = _make_root()
  level1 = root.derive({"phase": "train"})
  level2 = level1.derive({"epoch": 7})

  level2.log({"loss": 0.5})

  assert calls == [{"phase": "train", "epoch": 7, "loss": 0.5}]


def test_nested_derive_inner_context_overrides_outer_on_conflict(calls):
  # The last (innermost) derive wins on key conflicts: at each hop log()
  # merges {**self.data, **data}, and inner derives sit closer to the
  # explicit payload as the call walks up to the root.
  root = _make_root()
  level1 = root.derive({"phase": "train", "epoch": 999})
  level2 = level1.derive({"epoch": 7})

  level2.log({"loss": 0.5})

  assert calls == [{"phase": "train", "epoch": 7, "loss": 0.5}]


def test_deep_chain_reaches_root_backend(calls):
  root = _make_root()
  t = root
  for i in range(5):
    t = t.derive({f"k{i}": i})

  t.log({"loss": 1.0})

  expected = {f"k{i}": i for i in range(5)}
  expected["loss"] = 1.0
  assert calls == [expected]


# ---------------------------------------------------------------------------
# Sibling isolation
# ---------------------------------------------------------------------------


def test_sibling_derives_have_independent_context(calls):
  root = _make_root()
  train = root.derive({"phase": "train"})
  eval_ = root.derive({"phase": "eval"})

  train.log({"loss": 1.0})
  eval_.log({"loss": 2.0})

  assert calls == [
    {"phase": "train", "loss": 1.0},
    {"phase": "eval", "loss": 2.0},
  ]


def test_child_log_does_not_mutate_child_data(calls):
  root = _make_root()
  child = root.derive({"phase": "train"})

  child.log({"loss": 0.1})
  child.log({"loss": 0.2})

  assert child.data == {"phase": "train"}
  assert calls == [
    {"phase": "train", "loss": 0.1},
    {"phase": "train", "loss": 0.2},
  ]


# ---------------------------------------------------------------------------
# init() backend selection
# ---------------------------------------------------------------------------


def test_init_with_unknown_logger_logs_are_silent(calls):
  config = SimpleNamespace(logging=SimpleNamespace(
    logger="none", project_name="test",
    episodes=SimpleNamespace(every_n_collects=None, every_n_evals=None, dump_path=None, dump_format="rerun"),
  ))
  tracker = Tracker(config)
  tracker.init()

  tracker.log({"loss": 1.0})
  assert calls == []


def test_init_resets_data():
  config = SimpleNamespace(logging=SimpleNamespace(
    logger="none", project_name="test",
    episodes=SimpleNamespace(every_n_collects=None, every_n_evals=None, dump_path=None, dump_format="rerun"),
  ))
  tracker = Tracker(config, data={"stale": True})
  tracker.init({"fresh": 1})

  assert tracker.data == {"fresh": 1}


# ---------------------------------------------------------------------------
# log_batch_tags — per-batch episode counts by tag
# ---------------------------------------------------------------------------


def test_log_batch_tags_buckets_untagged_separately(calls):
  root = _make_root()
  root.log_batch_tags([(), (), ()])
  assert calls == [{"batch_tag/untagged": 3}]


def test_log_batch_tags_counts_per_tag(calls):
  root = _make_root()
  root.log_batch_tags([("real",), ("real",), ("sim",), (), ("real",)])
  assert calls == [{
    "batch_tag/real":     3,
    "batch_tag/sim":      1,
    "batch_tag/untagged": 1,
  }]


def test_log_batch_tags_double_counts_multi_tag_sequences(calls):
  # A sequence tagged ("real", "demo") increments BOTH buckets — by design.
  root = _make_root()
  root.log_batch_tags([("real", "demo"), ("real",), ("demo",)])
  assert calls == [{"batch_tag/real": 2, "batch_tag/demo": 2}]


def test_log_batch_tags_empty_batch_does_not_log(calls):
  # Nothing to count → no payload sent. Prevents stray empty log entries
  # if the buffer ever returns an empty tags list.
  root = _make_root()
  root.log_batch_tags([])
  assert calls == []


def test_log_batch_tags_through_scope_inherits_scope_data(calls):
  # The trainer logs counts from within a scope({"phase": "train", ...})
  # context; make sure the scope's data rides along on the forwarded payload.
  root = _make_root()
  with root.scope({"phase": "train", "iteration": 3}) as scoped:
    scoped.log_batch_tags([("real",), ()])
  assert calls == [{
    "phase":              "train",
    "iteration":          3,
    "batch_tag/real":     1,
    "batch_tag/untagged": 1,
  }]
