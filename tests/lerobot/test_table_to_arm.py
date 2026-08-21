"""extract_table_to_arm: alignment math, the collinearity heuristic, the
residual gates, and the store-first extraction step."""
from __future__ import annotations

import math

import numpy as np
import pytest

from chuck_dreamer.lerobot.context import RunContext
from chuck_dreamer.lerobot.nodes.table_to_arm import (
  TOUCHPOINT_VARIANTS,
  ResidualGateError,
  extract_table_to_arm,
  fit_table_to_arm,
  resolve_variant,
  umeyama_rigid,
)
from chuck_dreamer.store import ArtifactStore, annotation_record, for_dataset

DS = "user/ds"


def _rz(deg: float) -> np.ndarray:
  c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(deg: float) -> np.ndarray:
  c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
  return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _arm_points(p_table: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
  """Invert the fit convention ``p_table = R @ p_arm + t``."""
  return (p_table - t) @ R


# ---------------------------------------------------------------------------
# Alignment math
# ---------------------------------------------------------------------------
def test_umeyama_recovers_full_3d_transform():
  rng = np.random.default_rng(0)
  P = rng.uniform(-0.3, 0.3, size=(7, 3))
  R_true = _rx(30.0) @ _rz(-70.0)
  t_true = np.array([0.2, -0.1, 0.05])
  R, t = umeyama_rigid(P, P @ R_true.T + t_true)
  np.testing.assert_allclose(R, R_true, atol=1e-9)
  np.testing.assert_allclose(t, t_true, atol=1e-9)


def test_fit_coplanar_targets_uses_full_umeyama():
  """All touch targets are on the mat plane (z = 0) — coplanar is the
  normal case and must still recover a transform with tilt."""
  p_table = np.array(
    [[x / 1000.0, y / 1000.0, 0.0] for x, y in TOUCHPOINT_VARIANTS[7].values()])
  R_true = _rx(2.0) @ _rz(35.0)          # slight base tilt + yaw
  t_true = np.array([0.25, -0.05, 0.01])
  fit = fit_table_to_arm(_arm_points(p_table, R_true, t_true), p_table)
  assert fit.method == "umeyama3d"
  assert fit.max_residual_mm < 1e-6
  np.testing.assert_allclose(fit.R, R_true, atol=1e-9)
  np.testing.assert_allclose(fit.t, t_true, atol=1e-9)


def test_fit_collinear_targets_falls_back_to_planar():
  """Collinear touches leave rotation about their line unobservable; the
  planar heuristic (arm z ∥ table z) must still recover a z-rotation +
  translation exactly."""
  p_table = np.array([[0.0, 0.0, 0.0], [-0.1, 0.0, 0.0], [-0.2, 0.0, 0.0]])
  R_true = _rz(40.0)
  t_true = np.array([0.3, 0.1, 0.02])
  fit = fit_table_to_arm(_arm_points(p_table, R_true, t_true), p_table)
  assert fit.method == "planar_z"
  assert fit.max_residual_mm < 1e-6
  np.testing.assert_allclose(fit.R, R_true, atol=1e-9)
  np.testing.assert_allclose(fit.t, t_true, atol=1e-9)


def test_fit_rejects_bad_shapes():
  with pytest.raises(ValueError, match="at least 2"):
    fit_table_to_arm(np.zeros((1, 3)), np.zeros((1, 3)))
  with pytest.raises(ValueError, match="shapes"):
    fit_table_to_arm(np.zeros((3, 3)), np.zeros((4, 3)))


# ---------------------------------------------------------------------------
# Extraction step (store-first, provenance-keyed)
# ---------------------------------------------------------------------------
class _FakeRun:
  """The slice of :class:`Run` the extraction consumes."""

  def __init__(self, ctx: RunContext, joints: dict[int, np.ndarray]) -> None:
    self.ctx = ctx
    self._joints = joints

  @property
  def dataset_id(self) -> str:
    return self.ctx.source_repo

  def episode_joint_values(self, episode_index: int) -> np.ndarray:
    return self._joints[episode_index]


def _touch_joints(p_arm_m: np.ndarray, frames: int = 9) -> np.ndarray:
  """A (T, 6) deg joint episode whose median maps to ``p_arm_m`` under the
  test FK ``fk(q_rad) = q_rad[:3]``. First/last frames are move-in/out
  garbage the median must shrug off."""
  q_deg = np.rad2deg(np.concatenate([p_arm_m, [0.0, 0.0]]))
  ep = np.tile(np.concatenate([q_deg, [1.0]]), (frames, 1)).astype(np.float32)
  ep[0, :5] += 25.0
  ep[-1, :5] -= 25.0
  return ep


@pytest.fixture
def rig(tmp_path):
  """(run, store) with a variant-3 dataset whose touchpoints match a known
  planar transform exactly."""
  store = ArtifactStore(tmp_path / "store")
  scope = for_dataset(DS)
  store.put("dataset_config", scope,
            {"camera_id": "front", "touchpoint_variant": 3},
            annotation_record("dataset_config", scope, "test"))

  ctx = RunContext(source_repo=DS)
  ctx._store = store
  ctx._store_resolved = True
  ctx._fk = lambda q_rad: np.asarray(q_rad[:3], dtype=np.float64)

  R_true, t_true = _rz(30.0), np.array([0.28, -0.04, 0.015])
  joints: dict[int, np.ndarray] = {}
  for ep, (x_mm, y_mm) in TOUCHPOINT_VARIANTS[3].items():
    p_table = np.array([x_mm / 1000.0, y_mm / 1000.0, 0.0])
    joints[ep] = _touch_joints(_arm_points(p_table, R_true, t_true))
  return _FakeRun(ctx, joints), store, R_true, t_true


def test_extract_fits_and_persists(rig):
  run, store, R_true, t_true = rig
  result = extract_table_to_arm(run)
  assert not result.cached
  assert store.has("table_to_arm", for_dataset(DS))

  p = result.payload
  assert p["convention"] == "p_table = R @ p_arm + t"
  assert p["variant"] == 3
  assert p["max_residual_mm"] < 0.5
  np.testing.assert_allclose(np.asarray(p["R"]), R_true, atol=1e-3)
  np.testing.assert_allclose(np.asarray(p["t"]), t_true, atol=1e-4)
  assert [tp["episode"] for tp in p["touchpoints"]] == [1, 2, 3]

  record = store.provenance("table_to_arm", for_dataset(DS))
  assert record.node == "extract_table_to_arm"
  assert set(record.input_versions) == {
    "fk_model", "touchpoint_targets", "dataset_config", "dataset.touchpoints"}


def test_extract_is_cached_until_inputs_change(rig):
  run, _store, _R, _t = rig
  first = extract_table_to_arm(run)
  again = extract_table_to_arm(run)
  assert not first.cached and again.cached
  assert again.payload == first.payload

  # A changed touchpoint episode invalidates the cache (nudge small enough
  # to stay inside the residual gates).
  run._joints[2] = run._joints[2] + 0.05
  recomputed = extract_table_to_arm(run)
  assert not recomputed.cached


def test_extract_residual_gate(rig):
  run, store, _R, _t = rig
  # Pull one touch ~28 mm off target: reject without --force.
  run._joints[3] = run._joints[3] + np.rad2deg(0.02)
  with pytest.raises(ResidualGateError, match="not persisting"):
    extract_table_to_arm(run)
  assert not store.has("table_to_arm", for_dataset(DS))

  forced = extract_table_to_arm(run, force=True)
  assert not forced.cached
  assert store.has("table_to_arm", for_dataset(DS))
  assert any("residual" in w for w in forced.warnings)


def test_extract_requires_variant(rig):
  run, store, _R, _t = rig
  scope = for_dataset(DS)
  store.put("dataset_config", scope,
            {"camera_id": "front", "touchpoint_variant": None},
            annotation_record("dataset_config", scope, "test"))
  with pytest.raises(ValueError, match="touchpoint_variant is not set"):
    resolve_variant(run)
