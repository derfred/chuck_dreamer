"""Unit tests for the runtime perception pipeline (pure, no SAM2/torch).

Covers: the pipeline folds emits through modules in order; ``reset`` clears
per-module state; a stub module threads warm-start state across ``process``
calls (the contract ``ObjectLocalizerModule`` relies on); and ``EePoseModule``
builds the 7-D ``ee`` modality from a fake backend.
"""

from __future__ import annotations

import numpy as np

from chuck_dreamer.runtime.modalities import EE, OBJECT_XY
from chuck_dreamer.runtime.perception import EePoseModule, PerceptionPipeline


class _Add:
  """Emits a key whose value is derived from a consumed key (records call order)."""

  def __init__(self, name, consumes, emit, calls):
    self.name = name
    self.consumes = tuple(consumes)
    self.emits = (emit,)
    self._emit = emit
    self._calls = calls

  def reset(self):
    pass

  def process(self, data, *, t):
    self._calls.append(self.name)
    base = sum(float(np.sum(data[c])) for c in self.consumes) if self.consumes else 0.0
    return {self._emit: np.array([base + 1.0], np.float32)}


class _WarmStart:
  """Counts how often a fresh (None-state) cold start happens vs warm steps."""

  def __init__(self):
    self.name = "warm"
    self.consumes = ()
    self.emits = (OBJECT_XY,)
    self._prev = None
    self.cold_starts = 0

  def reset(self):
    self._prev = None

  def process(self, data, *, t):
    if self._prev is None:
      self.cold_starts += 1
    val = np.array([0.0 if self._prev is None else self._prev + 1.0], np.float32)
    self._prev = float(val[0])
    return {OBJECT_XY: val}


def test_pipeline_runs_modules_in_order_and_merges():
  calls = []
  m1 = _Add("first", (), "a", calls)
  m2 = _Add("second", ("a",), "b", calls)   # consumes first's emit
  pipe = PerceptionPipeline([m1, m2])
  out = pipe.run({"t": 0.0}, t=0.0)
  assert calls == ["first", "second"]        # configured order preserved
  assert "a" in out and "b" in out
  assert float(out["b"][0]) == 2.0           # b = a(=1) + 1


def test_reset_clears_module_state():
  m = _WarmStart()
  pipe = PerceptionPipeline([m])
  pipe.reset()
  pipe.run({}, t=0.0)   # cold start (prev None -> 0)
  pipe.run({}, t=0.1)   # warm (0 -> 1)
  assert m.cold_starts == 1
  pipe.reset()          # new episode
  pipe.run({}, t=0.0)   # cold start again
  assert m.cold_starts == 2


def test_warm_start_state_threads_across_steps():
  m = _WarmStart()
  pipe = PerceptionPipeline([m])
  pipe.reset()
  vals = [float(pipe.run({}, t=i)[OBJECT_XY][0]) for i in range(3)]
  assert vals == [0.0, 1.0, 2.0]   # prev_pose-style warm-start chain


class _FakeBackend:
  def ee_pos(self):
    return np.array([0.1, 0.2, 0.3], np.float64)

  def ee_quat(self):
    return np.array([1.0, 0.0, 0.0, 0.0], np.float64)


def test_ee_pose_module_emits_7d_ee():
  m = EePoseModule(_FakeBackend())
  assert m.emits == (EE,)
  out = m.process({}, t=0.0)
  ee = out[EE]
  assert ee.shape == (7,)
  np.testing.assert_allclose(ee, [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_ee_pose_module_rejects_backend_without_ee():
  import pytest

  class _Bare:
    pass

  with pytest.raises(TypeError):
    EePoseModule(_Bare())
