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

The fit is deliberately **2-parameter**, not a general 6-DoF rigid
alignment. The rig pins down everything else:

* the arm base is bolted flat to the mat, so arm z ∥ table z — no tilt to
  estimate, and the z translation is just the negative mean touch height
  (every touch is on the mat surface, z = 0 by definition);
* the arm is aimed at the middle touch target, so that target sits on the
  arm's +x axis — the base offset has no y component.

What is left is ``theta`` (the yaw of the touch line relative to the arm's
x axis) and ``distance`` (base → middle target, along that axis). Fitting
two parameters instead of six is well-conditioned on the collinear touch
rows the capture protocol produces, where a full Umeyama fit would leave
the rotation about the common line unobservable.
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
    5: (-150.0, 0.0),
    6: (0.0, 0.0),
    7: (150.0, 0.0),
  },
  7: {
    4: (-150.0, 0.0),
    5: (0.0, 0.0),
    6: (150.0, 0.0),
    7: (-150.0, 80.0),
    8: (150.0, 80.0),
    9: (-150.0, -80.0),
    10: (150.0, -80.0),
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


class ResidualGateError(ValueError):
  """The fitted transform misses its own touch targets by more than
  :data:`RESIDUAL_REJECT_MM` — not persisted."""


# ---------------------------------------------------------------------------
# Alignment math
# ---------------------------------------------------------------------------
def _middle_target_index(p_table_m: np.ndarray) -> int:
  """Index of the *middle* touch target: the one nearest the table-frame
  origin, which the capture protocol places under the arm's x axis."""
  return int(np.argmin(np.linalg.norm(np.asarray(p_table_m, dtype=np.float64)[:, :2], axis=1)))


def _fit_theta(p_arm_m: np.ndarray, p_table_m: np.ndarray) -> float:
  """Yaw of the table frame relative to the arm frame.

  With the translation constrained to lie along the arm's x axis, theta is
  still the ordinary 2-D Kabsch angle on the *centered* xy channels: the
  centering removes the translation whatever it turns out to be, so the
  rotation can be solved for first and independently."""
  Pc = p_arm_m[:, :2]   - p_arm_m[:, :2].mean(axis=0)
  Qc = p_table_m[:, :2] - p_table_m[:, :2].mean(axis=0)
  return math.atan2(float(np.sum(Pc[:, 0] * Qc[:, 1] - Pc[:, 1] * Qc[:, 0])),
                    float(np.sum(Pc[:, 0] * Qc[:, 0] + Pc[:, 1] * Qc[:, 1])))


@dataclass(frozen=True)
class TableToArmFit:
  """A fitted arm→table transform: ``p_table = R @ p_arm + t`` (metres).

  ``R`` and ``t`` are the *derived* dense form; the fit itself has only the
  two free parameters :attr:`theta_rad` and :attr:`distance_m` (plus the
  z offset, which is measured rather than fitted)."""
  R: np.ndarray
  t: np.ndarray
  theta_rad: float               # yaw of the touch line vs. the arm x axis
  distance_m: float              # arm base → middle touch target, along arm +x
  z_offset_m: float              # −mean touch height, folded into t[2]
  residuals_mm: np.ndarray       # (N,) per-touch |predicted − target|

  @property
  def max_residual_mm(self) -> float:
    return float(self.residuals_mm.max())

  @property
  def rms_residual_mm(self) -> float:
    return float(np.sqrt(np.mean(self.residuals_mm ** 2)))


def fit_table_to_arm(p_arm_m: np.ndarray, p_table_m: np.ndarray) -> TableToArmFit:
  """Fit ``p_table = R @ p_arm + t`` from ``(N, 3)`` correspondences in
  metres under the rig's 2-parameter model (see the module docstring).

  ``theta`` comes from the centered 2-D Kabsch angle; ``distance`` is the
  arm-frame x coordinate of the middle touch, which the protocol places on
  the arm's +x axis; the z translation is the negative mean touch height."""
  p_arm_m   = np.asarray(p_arm_m, dtype=np.float64)
  p_table_m = np.asarray(p_table_m, dtype=np.float64)
  if p_arm_m.shape != p_table_m.shape or p_arm_m.ndim != 2 or p_arm_m.shape[1] != 3:
    raise ValueError(f"correspondence shapes differ or are not (N, 3): {p_arm_m.shape} vs {p_table_m.shape}")

  if len(p_arm_m) < 2:
    raise ValueError("need at least 2 touch points to fit a transform")

  theta = _fit_theta(p_arm_m, p_table_m)
  c, s  = math.cos(theta), math.sin(theta)
  R     = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

  # The middle touch is on the arm's +x axis, so the base→middle distance is
  # that touch's arm-frame x. The whole translation is along that axis, which
  # after rotation into the table frame is −R @ [distance, 0, 0].
  middle     = _middle_target_index(p_table_m)
  distance   = float(p_arm_m[middle, 0])
  z_offset   = float(-p_arm_m[:, 2].mean())
  t          = -R @ np.array([distance, 0.0, 0.0])
  t[2]       = z_offset

  predicted    = p_arm_m @ R.T + t
  residuals_mm = np.linalg.norm(predicted - p_table_m, axis=1) * 1000.0
  return TableToArmFit(R=R, t=t, theta_rad=theta, distance_m=distance, z_offset_m=z_offset, residuals_mm=residuals_mm)


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


def _process_touch(episode: int, target_mm: tuple[float, float], qpos_deg: np.ndarray, fk: Any) -> Touch:
  """Median the positioning joints over the episode, run FK.

  The median is robust as long as the arm dwells on the point for most of
  the episode; the spread is computed over the central half of the frames
  (move-in/move-out excluded) as the rest-quality diagnostic."""

  q = np.asarray(qpos_deg, dtype=np.float64)
  if q.ndim != 2 or q.shape[0] == 0:
    raise ValueError(f"episode {episode}: no joint values")

  n_pos    = min(5, q.shape[1])
  q_pos    = q[:, :n_pos]
  q_median = np.median(q_pos, axis=0)
  lo, hi   = len(q) // 4, max(len(q) // 4 + 1, (3 * len(q)) // 4)
  window   = q_pos[lo:hi]
  q_spread = window.max(axis=0) - window.min(axis=0)
  p_arm    = np.asarray(fk(np.deg2rad(q_median)), dtype=np.float64).reshape(3)
  return Touch(episode=episode, target_mm=target_mm, q_median_deg=q_median, q_spread_deg=q_spread, p_arm_m=p_arm)


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

  ctx   = run.ctx
  store = ctx.store
  if store is None:
    raise RuntimeError("extract-table-to-arm needs the artifact store; set store.root in the config")

  variant = resolve_variant(run)
  targets = TOUCHPOINT_VARIANTS[variant]
  scope   = for_dataset(run.dataset_id)

  joint_values                   = {ep: run.episode_joint_values(ep) for ep in sorted(targets)}
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

  fk      = ctx.fk()
  touches = [_process_touch(ep, targets[ep], joint_values[ep], fk) for ep in sorted(targets)]

  warnings: list[str] = []
  for touch in touches:
    worst = float(touch.q_spread_deg.max()) if len(touch.q_spread_deg) else 0.0
    if worst > SPREAD_WARN_DEG:
      warnings.append(
        f"episode {touch.episode}: joint spread {worst:.1f}° over the dwell "
        f"window (> {SPREAD_WARN_DEG}°) — the median may not be a rest pose")

  p_arm   = np.stack([t.p_arm_m for t in touches])
  p_table = np.array([[t.target_mm[0] / 1000.0, t.target_mm[1] / 1000.0, 0.0] for t in touches])
  fit     = fit_table_to_arm(p_arm, p_table)

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
    "method": "rig2p",
    "theta_rad": fit.theta_rad,
    "distance_m": fit.distance_m,
    "z_offset_m": fit.z_offset_m,
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
