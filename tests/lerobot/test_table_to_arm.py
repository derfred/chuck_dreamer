"""extract_table_to_arm: the 2-parameter rig fit, the residual gates, and
the store-first extraction step."""
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
def _rig_arm_points(p_table: np.ndarray, theta_deg: float, distance: float,
                    touch_z: float) -> np.ndarray:
  """Arm-frame touches for a rig at yaw ``theta_deg``, whose base sits
  ``distance`` behind the middle target along the arm's +x axis, touching at
  height ``touch_z``."""
  R = _rz(theta_deg)
  t = -R @ np.array([distance, 0.0, 0.0])
  t[2] = -touch_z
  return _arm_points(p_table, R, t)


def test_fit_recovers_the_two_rig_parameters():
  """The exact-fit case: touches generated from a known (theta, distance)
  come back with those parameters and a zero residual."""
  p_table = np.array(
    [[x / 1000.0, y / 1000.0, 0.0] for x, y in TOUCHPOINT_VARIANTS[7].values()])
  theta_deg, distance, touch_z = 35.0, 0.28, 0.004
  fit = fit_table_to_arm(
    _rig_arm_points(p_table, theta_deg, distance, touch_z), p_table)
  assert fit.max_residual_mm < 1e-6
  assert math.degrees(fit.theta_rad) == pytest.approx(theta_deg)
  assert fit.distance_m == pytest.approx(distance)
  assert fit.z_offset_m == pytest.approx(-touch_z)
  np.testing.assert_allclose(fit.R, _rz(theta_deg), atol=1e-9)


def test_fit_handles_collinear_touches():
  """Collinear touch rows are the protocol's normal case and are fully
  determined under the 2-parameter model."""
  p_table = np.array([[-0.15, 0.0, 0.0], [0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])
  fit = fit_table_to_arm(_rig_arm_points(p_table, 40.0, 0.3, 0.0), p_table)
  assert fit.max_residual_mm < 1e-6
  assert math.degrees(fit.theta_rad) == pytest.approx(40.0)
  assert fit.distance_m == pytest.approx(0.3)


def test_z_offset_is_the_negative_mean_touch_height():
  """Touches at differing heights average into a single z offset rather
  than tilting the frame."""
  p_table = np.array([[-0.15, 0.0, 0.0], [0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])
  p_arm = _rig_arm_points(p_table, 0.0, 0.25, 0.0)
  p_arm[:, 2] += np.array([0.001, 0.003, 0.005])   # mean +3 mm
  fit = fit_table_to_arm(p_arm, p_table)
  assert fit.z_offset_m == pytest.approx(-0.003)
  assert fit.t[2] == pytest.approx(-0.003)


def test_translation_has_no_y_component_in_the_arm_frame():
  """The model asserts the middle target is on the arm's +x axis, so the
  base offset it recovers is purely axial."""
  p_table = np.array([[-0.15, 0.0, 0.0], [0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])
  fit = fit_table_to_arm(_rig_arm_points(p_table, 25.0, 0.3, 0.0), p_table)
  offset_arm = -fit.R.T @ fit.t          # base offset expressed in the arm frame
  assert offset_arm[1] == pytest.approx(0.0, abs=1e-12)


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

  R_true = _rz(30.0)
  t_true = -R_true @ np.array([0.28, 0.0, 0.0])
  t_true[2] = -0.015
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
  assert [tp["episode"] for tp in p["touchpoints"]] == sorted(TOUCHPOINT_VARIANTS[3])

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
  middle = sorted(TOUCHPOINT_VARIANTS[3])[1]
  run._joints[middle] = run._joints[middle] + 0.05
  recomputed = extract_table_to_arm(run)
  assert not recomputed.cached


def test_extract_residual_gate(rig):
  run, store, _R, _t = rig
  # Pull one touch ~28 mm off target: reject without --force.
  last = sorted(TOUCHPOINT_VARIANTS[3])[-1]
  run._joints[last] = run._joints[last] + np.rad2deg(0.02)
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
