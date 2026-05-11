"""Tests for Trainer._log_episode — Bernoulli-sampled disk dumps of collected episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from chuck_dreamer.trainer import Trainer


@dataclass
class _FakeScene:
  """Minimal dataclass shaped like SceneConfig for asdict() and goal_pos access."""
  goal_pos: list[float] = field(default_factory=lambda: [0.1, 0.2])
  max_steps: int = 100


class _FakeWriter:
  """Captures write_episode() calls so the test can inspect what was dumped."""
  def __init__(self) -> None:
    self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

  def write_episode(self, episode: dict[str, Any], metadata: dict[str, Any], *, name_suffix: str) -> None:
    self.calls.append((episode, metadata, name_suffix))


def _make_trainer(writer, prob: float, seed: int = 0) -> Trainer:
  """Build a Trainer without invoking __init__ — _log_episode only needs four attrs."""
  t = Trainer.__new__(Trainer)
  t._collect_writer = writer
  t._collect_dump_prob = prob
  t._dump_rng = np.random.default_rng(seed)

  class _Cfg:
    seed = 42
  t.config = _Cfg()
  return t


def test_log_episode_disabled_when_writer_is_none():
  t = _make_trainer(writer=None, prob=1.0)
  # Should be a silent no-op even with prob=1.0.
  t._log_episode({"image": np.zeros(1)}, _FakeScene(), outcome="success", iteration=0, ep_idx=0)
  # Nothing to assert beyond not raising.


def test_log_episode_never_writes_when_prob_is_zero():
  writer = _FakeWriter()
  t = _make_trainer(writer=writer, prob=0.0)
  for _ in range(50):
    t._log_episode({}, _FakeScene(), outcome="success", iteration=0, ep_idx=0)
  assert writer.calls == []


def test_log_episode_always_writes_when_prob_is_one():
  writer = _FakeWriter()
  t = _make_trainer(writer=writer, prob=1.0)
  for _ in range(20):
    t._log_episode({}, _FakeScene(), outcome="success", iteration=0, ep_idx=0)
  assert len(writer.calls) == 20


def test_log_episode_metadata_matches_generate_scenes_shape():
  writer = _FakeWriter()
  t = _make_trainer(writer=writer, prob=1.0)
  scene = _FakeScene(goal_pos=[0.3, -0.4], max_steps=77)
  episode = {"image": np.zeros((1, 4, 4, 3), dtype=np.uint8)}

  t._log_episode(episode, scene, outcome="timeout", iteration=12, ep_idx=3)

  assert len(writer.calls) == 1
  ep_arg, meta, suffix = writer.calls[0]
  assert ep_arg is episode
  # Same metadata keys the generate-scenes CLI emits.
  assert set(meta) == {"config", "seed", "source", "outcome", "goal_xy"}
  assert meta["config"] == {"goal_pos": [0.3, -0.4], "max_steps": 77}
  assert meta["seed"] == 42
  assert meta["source"] == "sim"
  assert meta["outcome"] == "timeout"
  assert meta["goal_xy"] == [0.3, -0.4]
  assert suffix == "step012-003"


def test_log_episode_bernoulli_rate_is_approximately_correct():
  writer = _FakeWriter()
  prob = 0.3
  n = 5000
  t = _make_trainer(writer=writer, prob=prob, seed=123)
  for _ in range(n):
    t._log_episode({}, _FakeScene(), outcome="success", iteration=0, ep_idx=0)
  observed = len(writer.calls) / n
  # 3-sigma band on a Bernoulli(0.3) mean across 5000 trials is ~±0.019; 0.03 is safe.
  assert observed == pytest.approx(prob, abs=0.03)


def test_log_episode_rng_is_seeded_for_reproducibility():
  scene = _FakeScene()

  w1 = _FakeWriter()
  t1 = _make_trainer(writer=w1, prob=0.5, seed=7)
  for _ in range(100):
    t1._log_episode({}, scene, outcome="success", iteration=0, ep_idx=0)

  w2 = _FakeWriter()
  t2 = _make_trainer(writer=w2, prob=0.5, seed=7)
  for _ in range(100):
    t2._log_episode({}, scene, outcome="success", iteration=0, ep_idx=0)

  assert len(w1.calls) == len(w2.calls)
