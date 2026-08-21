"""``extract_table_to_arm`` (spec §7.2): the table↔arm transform from
touchpoint captures.

Each dataset's capture protocol includes a set of **touchpoint episodes**:
the gripper resting on a known point of the mat fiducial for the whole
episode. The variant number in ``dataset_config.touchpoint_variant`` names
the protocol — it fixes *which episode indices* are touchpoints and which
table-frame target each one touches (:data:`TOUCHPOINT_VARIANTS`). From the
per-episode joint medians, FK gives the touched points in the arm frame;
rigid alignment against the known table-frame targets gives the transform.

This is the pipeline's only DATASET-scoped computation, so it does not run
inside the per-episode harness lists: it runs once per dataset (the
``import-lerobot extract-table-to-arm`` command), persists ``table_to_arm``
in the artifact store keyed by its input versions, and downstream episode
nodes consume it through the ``table_to_arm`` artifact provider.

Degeneracy: all targets lie in the mat plane (z = 0), which full 3-D
Umeyama handles fine — but when the touched points are (near-)collinear the
rotation about the common line is unobservable. The captured datasets can't
be redone, so instead of rejecting we fall back to a **planar heuristic**:
the arm base is bolted flat to the table, so its z axis is assumed parallel
to the table z axis, reducing the fit to a rotation about z plus a 3-D
translation — well-posed for any two distinct touch points.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from chuck_dreamer.common import FK_MODEL_PATH

if TYPE_CHECKING:
  from ..pipeline import Run

NODE_NAME = "extract_table_to_arm"
ARTIFACT  = "table_to_arm"     # store artifact type (spec §6) this node produces

# Touch-target convention per variant: touchpoint episode index → (x, y) in
# millimetres in the table frame (every touch is on the mat surface, z = 0).
# The episode indices are part of the capture protocol itself — episode 0 is
# the empty-space capture, the touchpoints follow directly after it.
#
# PLACEHOLDER VALUES — the coordinates below are stand-ins pending the
# measured fiducial touch points; replace them with the real mat positions
# before trusting any extracted transform.
TOUCHPOINT_VARIANTS: dict[int, dict[int, tuple[float, float]]] = {
  3: {
    1: (0.0, 0.0),
    2: (-150.0, 0.0),
    3: (150.0, 0.0),
  },
  7: {
    1: (0.0, 0.0),
    2: (0.0, 80.0),
    3: (-150.0, 80.0),
    4: (-300.0, 80.0),
    5: (-14.0, 0.0),
    6: (0.0, -80.0),
    7: (-150.0, -80.0),
  },
}

# Residual quality gates (mm), from the validated calibrate-live procedure:
# warn above WARN, refuse to persist above REJECT — a transform that misses
# its own calibration points by ≥1 cm produces silently-wrong table-frame
# tracks downstream.
RESIDUAL_WARN_MM   = 5.0
RESIDUAL_REJECT_MM = 10.0

# Per-joint spread (max − min, degrees) above which a touch's median is
# suspect — the arm was not at rest for most of the episode.
SPREAD_WARN_DEG = 3.0

# Relative second-singular-value threshold below which the target set is
# treated as collinear (rotation about the common line unobservable).
_COLLINEAR_RTOL = 0.05


class ResidualGateError(ValueError):
  """The fitted transform misses its own touch targets by more than
  :data:`RESIDUAL_REJECT_MM` — not persisted."""


# ---------------------------------------------------------------------------
# Alignment math
# ---------------------------------------------------------------------------
def umeyama_rigid(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Closed-form rigid alignment ``q = R @ p + t`` over ``(N, 3)``
  correspondences (Umeyama/Kabsch, no scale).

  The reflection correction (the ``d`` term) is required, not optional:
  without it SVD can return an improper rotation (det −1) that looks
  geometrically plausible but is a mirror."""
  P = np.asarray(P, dtype=np.float64)
  Q = np.asarray(Q, dtype=np.float64)
  c_p = P.mean(axis=0)
  c_q = Q.mean(axis=0)
  H = (P - c_p).T @ (Q - c_q)
  U, _, Vt = np.linalg.svd(H)
  d = np.sign(np.linalg.det(Vt.T @ U.T))
  R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
  t = c_q - R @ c_p
  return R, t


def _planar_z_fit(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Rigid alignment constrained to a rotation about z (2-D Kabsch on the
  xy channels) plus a full 3-D translation.

  The degeneracy heuristic for collinear touch points: the arm base sits
  flat on the table, so arm z ∥ table z is assumed rather than estimated;
  the z component of ``t`` absorbs the (constant) touch height offset."""
  P = np.asarray(P, dtype=np.float64)
  Q = np.asarray(Q, dtype=np.float64)
  Pc = P[:, :2] - P[:, :2].mean(axis=0)
  Qc = Q[:, :2] - Q[:, :2].mean(axis=0)
  theta = math.atan2(float(np.sum(Pc[:, 0] * Qc[:, 1] - Pc[:, 1] * Qc[:, 0])),
                     float(np.sum(Pc[:, 0] * Qc[:, 0] + Pc[:, 1] * Qc[:, 1])))
  c, s = math.cos(theta), math.sin(theta)
  R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
  t = Q.mean(axis=0) - R @ P.mean(axis=0)
  return R, t


def _is_collinear(points: np.ndarray) -> bool:
  """Whether ``(N, 3)`` points are (near-)collinear: the second singular
  value of the centered set vanishes relative to the first."""
  pts = np.asarray(points, dtype=np.float64)
  if len(pts) < 3:
    return True
  s = np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)
  return bool(s[1] <= _COLLINEAR_RTOL * s[0])


@dataclass(frozen=True)
class TableToArmFit:
  """A fitted arm→table transform: ``p_table = R @ p_arm + t`` (metres)."""
  R: np.ndarray
  t: np.ndarray
  method: str                    # "umeyama3d" | "planar_z"
  residuals_mm: np.ndarray       # (N,) per-touch |predicted − target|

  @property
  def max_residual_mm(self) -> float:
    return float(self.residuals_mm.max())

  @property
  def rms_residual_mm(self) -> float:
    return float(np.sqrt(np.mean(self.residuals_mm ** 2)))


def fit_table_to_arm(p_arm_m: np.ndarray, p_table_m: np.ndarray) -> TableToArmFit:
  """Fit ``p_table = R @ p_arm + t`` from ``(N, 3)`` correspondences in
  metres, falling back to the planar-z heuristic when the touch points are
  collinear (see module docstring)."""
  p_arm_m   = np.asarray(p_arm_m, dtype=np.float64)
  p_table_m = np.asarray(p_table_m, dtype=np.float64)
  if p_arm_m.shape != p_table_m.shape or p_arm_m.ndim != 2 or p_arm_m.shape[1] != 3:
    raise ValueError(
      f"correspondence shapes differ or are not (N, 3): "
      f"{p_arm_m.shape} vs {p_table_m.shape}")
  if len(p_arm_m) < 2:
    raise ValueError("need at least 2 touch points to fit a transform")

  # Collinearity in either point set kills the same rotational DOF.
  if _is_collinear(p_table_m) or _is_collinear(p_arm_m):
    R, t = _planar_z_fit(p_arm_m, p_table_m)
    method = "planar_z"
  else:
    R, t = umeyama_rigid(p_arm_m, p_table_m)
    method = "umeyama3d"

  predicted = p_arm_m @ R.T + t
  residuals_mm = np.linalg.norm(predicted - p_table_m, axis=1) * 1000.0
  return TableToArmFit(R=R, t=t, method=method, residuals_mm=residuals_mm)


# ---------------------------------------------------------------------------
# Per-touch joint processing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Touch:
  """One processed touchpoint episode."""
  episode: int
  target_mm: tuple[float, float]
  q_median_deg: np.ndarray       # (5,) positioning joints
  q_spread_deg: np.ndarray       # (5,) max − min over the dwell window
  p_arm_m: np.ndarray            # (3,) FK of the median, metres


def _process_touch(episode: int, target_mm: tuple[float, float],
                   qpos_deg: np.ndarray, fk: Any) -> Touch:
  """Median the positioning joints over the episode, run FK.

  The median is robust as long as the arm dwells on the point for most of
  the episode; the spread is computed over the central half of the frames
  (move-in/move-out excluded) as the rest-quality diagnostic."""
  q = np.asarray(qpos_deg, dtype=np.float64)
  if q.ndim != 2 or q.shape[0] == 0:
    raise ValueError(f"episode {episode}: no joint values")
  n_pos = min(5, q.shape[1])
  q_pos = q[:, :n_pos]
  q_median = np.median(q_pos, axis=0)
  lo, hi = len(q) // 4, max(len(q) // 4 + 1, (3 * len(q)) // 4)
  window = q_pos[lo:hi]
  q_spread = window.max(axis=0) - window.min(axis=0)
  p_arm = np.asarray(fk(np.deg2rad(q_median)), dtype=np.float64).reshape(3)
  return Touch(episode=episode, target_mm=target_mm,
               q_median_deg=q_median, q_spread_deg=q_spread, p_arm_m=p_arm)


# ---------------------------------------------------------------------------
# Extraction (dataset-scoped step)
# ---------------------------------------------------------------------------
@dataclass
class ExtractionResult:
  payload: dict[str, Any]
  cached: bool
  warnings: list[str] = field(default_factory=list)


def _targets_version() -> str:
  from ..trackset import _sha_of
  return _sha_of({str(v): {str(e): list(xy) for e, xy in tbl.items()}
                  for v, tbl in TOUCHPOINT_VARIANTS.items()})


def _touch_inputs_version(joint_values: dict[int, np.ndarray]) -> str:
  import hashlib
  h = hashlib.sha256()
  for ep in sorted(joint_values):
    v = np.ascontiguousarray(joint_values[ep])
    h.update(str(ep).encode())
    h.update(str(v.dtype).encode())
    h.update(str(v.shape).encode())
    h.update(v.tobytes())
  return "sha256:" + h.hexdigest()


def resolve_variant(run: "Run") -> int:
  """The dataset's touchpoint variant from ``dataset_config``, validated
  against the known protocol tables."""
  cfg = run.ctx.dataset_config()
  variant = (cfg or {}).get("touchpoint_variant")
  if variant is None:
    raise ValueError(
      f"{run.dataset_id}: dataset_config.touchpoint_variant is not set; "
      f"run: uv run python main.py import-lerobot extract-table-to-arm "
      f"{run.dataset_id} --variant {{3|7}}")
  variant = int(variant)
  if variant not in TOUCHPOINT_VARIANTS:
    raise ValueError(
      f"{run.dataset_id}: unknown touchpoint_variant {variant}; "
      f"known: {sorted(TOUCHPOINT_VARIANTS)}")
  return variant


def extract_table_to_arm(run: "Run", *, force: bool = False) -> ExtractionResult:
  """Compute (or fetch) this dataset's ``table_to_arm`` transform.

  Store-first with input-version keying, mirroring the runner's
  ``Persist.CACHED`` behavior: when the store already holds a transform
  produced from the same FK model, target table, dataset_config and
  touchpoint joint values, it is returned without recomputation. ``force``
  recomputes unconditionally and bypasses the residual reject gate (the
  warn gate still reports).
  """
  from chuck_dreamer.store import ProvenanceRecord, for_dataset, git_sha

  from ..trackset import _file_sha

  ctx = run.ctx
  store = ctx.store
  if store is None:
    raise RuntimeError(
      "extract-table-to-arm needs the artifact store; set store.root in the "
      "config")

  variant = resolve_variant(run)
  targets = TOUCHPOINT_VARIANTS[variant]
  scope = for_dataset(run.dataset_id)

  joint_values = {ep: run.episode_joint_values(ep) for ep in sorted(targets)}
  input_versions: dict[str, str] = {
    "fk_model":            _file_sha(FK_MODEL_PATH) or "missing",
    "touchpoint_targets":  _targets_version(),
    "dataset_config":      store.payload_hash("dataset_config", scope),
    "dataset.touchpoints": _touch_inputs_version(joint_values),
  }

  if not force and store.has(ARTIFACT, scope):
    stored, record = store.get(ARTIFACT, scope)
    if record.input_versions == input_versions:
      return ExtractionResult(payload=stored, cached=True)

  fk = ctx.fk()
  touches = [_process_touch(ep, targets[ep], joint_values[ep], fk)
             for ep in sorted(targets)]

  warnings: list[str] = []
  for touch in touches:
    worst = float(touch.q_spread_deg.max()) if len(touch.q_spread_deg) else 0.0
    if worst > SPREAD_WARN_DEG:
      warnings.append(
        f"episode {touch.episode}: joint spread {worst:.1f}° over the dwell "
        f"window (> {SPREAD_WARN_DEG}°) — the median may not be a rest pose")

  p_arm   = np.stack([t.p_arm_m for t in touches])
  p_table = np.array([[t.target_mm[0] / 1000.0, t.target_mm[1] / 1000.0, 0.0]
                      for t in touches])
  fit = fit_table_to_arm(p_arm, p_table)

  if fit.method == "planar_z":
    warnings.append(
      "touch points are collinear — rotation about their common line is "
      "unobservable; fell back to the planar heuristic (arm z ∥ table z)")
  if fit.max_residual_mm >= RESIDUAL_WARN_MM:
    warnings.append(
      f"max residual {fit.max_residual_mm:.1f} mm ≥ {RESIDUAL_WARN_MM} mm — "
      "consider redoing the touch captures if precision matters")
  if fit.max_residual_mm >= RESIDUAL_REJECT_MM and not force:
    raise ResidualGateError(
      f"{run.dataset_id}: max touch residual {fit.max_residual_mm:.1f} mm ≥ "
      f"{RESIDUAL_REJECT_MM} mm; not persisting a transform that misses its "
      f"own calibration points (per-touch mm: "
      f"{np.round(fit.residuals_mm, 1).tolist()}). Rerun with --force to "
      "persist anyway.")

  payload: dict[str, Any] = {
    "R": fit.R.tolist(),
    "t": fit.t.tolist(),
    "convention": "p_table = R @ p_arm + t",
    "units": "m",
    "method": fit.method,
    "variant": variant,
    "touchpoints": [
      {
        "episode":      t.episode,
        "target_mm":    [t.target_mm[0], t.target_mm[1], 0.0],
        "p_arm_m":      t.p_arm_m.tolist(),
        "residual_mm":  float(r),
        "q_median_deg": t.q_median_deg.tolist(),
        "q_spread_deg": t.q_spread_deg.tolist(),
      }
      for t, r in zip(touches, fit.residuals_mm)
    ],
    "max_residual_mm": fit.max_residual_mm,
    "rms_residual_mm": fit.rms_residual_mm,
  }
  store.put(ARTIFACT, scope, payload, ProvenanceRecord(
    artifact=ARTIFACT, scope=scope, node=NODE_NAME,
    input_versions=input_versions, code_version=git_sha()))
  return ExtractionResult(payload=payload, cached=False, warnings=warnings)
