"""Frozen view over ``cfg.object_localization`` used across the pipeline.

``init_from_config`` parses + validates the YAML subtree once and stores
the resulting ``ObjectLocalizationConfig`` in module state. CLIs and the
runtime estimator call ``active()`` to retrieve it.

This indirection exists because the legacy processor wires the
estimator into the LeRobot import flow without having access to a CLI
context — it loads the project's default config and installs the OL
runtime once, then later code paths fetch the snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass
class ObjectLocalizationConfig:
  camera_key: str
  cache_dir: str
  calibration_cache: str        # legacy alias, mirrors cache_dir

  episode_empty: int
  episode_checkerboard: int
  episode_object_left: int
  episode_object_mid: int
  episode_object_right: int

  checkerboard_size: tuple[int, int]   # (cols, rows) inner corners
  checkerboard_square_mm: float

  mat_line_length_mm: float            # L
  mat_line_separation_mm: float        # D
  mat_circle_radius_mm: float          # r
  mat_circle_samples: int              # N

  intrinsics_n_frames_target: int
  intrinsics_n_frames_min:    int
  intrinsics_rms_threshold_px: float
  intrinsics_coverage_grid:   tuple[int, int]
  intrinsics_distortion_model: int   # 0, 2, or 5 (see configs/default.yaml)
  intrinsics_fix_aspect_ratio:    bool
  intrinsics_fix_principal_point: bool

  extrinsics_rms_threshold_px: float
  extrinsics_annotation_frame: int

  mesh_path: str
  sam2_checkpoint: str
  n_yaw_inits: int
  silhouette_chamfer_max_iter: int
  edges_canny_low: int
  edges_canny_high: int
  edges_search_range_px: int
  edges_max_iter: int

  use_sam2: bool
  device:   str

  raw: dict[str, Any] = field(default_factory=dict)


_active: ObjectLocalizationConfig | None = None


def init_from_config(cfg: DictConfig) -> ObjectLocalizationConfig:
  """Validate ``cfg.object_localization`` and remember the result.

  Raises ``ValueError`` with a precise message when a required key is
  missing or has the wrong type. The error message names the dotted
  config path so users know what to fix.
  """
  global _active
  if "object_localization" not in cfg:
    raise ValueError("config missing 'object_localization' section.")
  raw = OmegaConf.to_container(cfg.object_localization, resolve=True)
  assert isinstance(raw, dict)

  ol_cfg = _parse(raw)
  _active = ol_cfg
  return ol_cfg


def active() -> ObjectLocalizationConfig:
  if _active is None:
    raise RuntimeError(
      "object_localization runtime not initialized — call "
      "init_from_config(load_config()) before active().")
  return _active


def _req(d: dict[str, Any], key: str, parent: str) -> Any:
  if key not in d:
    raise ValueError(f"object_localization.{parent}{key}: missing required key.")
  return d[key]


def _parse(raw: dict[str, Any]) -> ObjectLocalizationConfig:
  episodes      = _req(raw, "episodes",     "")
  checkerboard  = _req(raw, "checkerboard", "")
  mat           = _req(raw, "mat",          "")
  intrinsics    = raw.get("intrinsics", {})
  extrinsics    = raw.get("extrinsics", {})
  object_block  = raw.get("object", {})
  silhouette    = (object_block.get("silhouette") or {})
  edges         = (object_block.get("edges") or {})

  ck_cols, ck_rows = _read_checkerboard_size(checkerboard)

  L = _float(_req(mat, "line_length_mm",     "mat."))
  D = _float(_req(mat, "line_separation_mm", "mat."))
  r = _float(_req(mat, "circle_radius_mm",   "mat."))

  cache_dir = str(raw.get("cache_dir") or raw.get("calibration_cache") or "./calibration_cache")
  legacy_cache = str(raw.get("calibration_cache") or cache_dir)

  mesh_path = str(object_block.get("mesh_path") or raw.get("mesh_path") or "")

  coverage = intrinsics.get("coverage_grid", [6, 4])
  coverage = (int(coverage[0]), int(coverage[1]))

  return ObjectLocalizationConfig(
    camera_key             = str(_req(raw, "camera_key", "")),
    cache_dir              = cache_dir,
    calibration_cache      = legacy_cache,

    episode_empty          = int(_req(episodes, "empty",        "episodes.")),
    episode_checkerboard   = int(_req(episodes, "checkerboard", "episodes.")),
    episode_object_left    = int(_req(episodes, "object_left",  "episodes.")),
    episode_object_mid     = int(_req(episodes, "object_mid",   "episodes.")),
    episode_object_right   = int(_req(episodes, "object_right", "episodes.")),

    checkerboard_size      = (ck_cols, ck_rows),
    checkerboard_square_mm = _float(_req(checkerboard, "square_mm", "checkerboard.")),

    mat_line_length_mm     = L,
    mat_line_separation_mm = D,
    mat_circle_radius_mm   = r,
    mat_circle_samples     = int(mat.get("circle_samples", 64)),

    intrinsics_n_frames_target   = int(intrinsics.get("n_frames_target", 64)),
    intrinsics_n_frames_min      = int(intrinsics.get("n_frames_min", 5)),
    intrinsics_rms_threshold_px  = float(intrinsics.get("rms_threshold_px", 0.3)),
    intrinsics_coverage_grid     = coverage,
    intrinsics_distortion_model  = _parse_distortion_model(intrinsics.get("distortion_model", 5)),
    intrinsics_fix_aspect_ratio    = bool(intrinsics.get("fix_aspect_ratio", False)),
    intrinsics_fix_principal_point = bool(intrinsics.get("fix_principal_point", False)),

    extrinsics_rms_threshold_px  = float(extrinsics.get("rms_threshold_px", 1.0)),
    extrinsics_annotation_frame  = int(extrinsics.get("annotation_frame", 0)),

    mesh_path                  = mesh_path,
    sam2_checkpoint            = str(object_block.get("sam2_checkpoint", "facebook/sam2-hiera-large")),
    n_yaw_inits                = int(object_block.get("n_yaw_inits", 4)),
    silhouette_chamfer_max_iter = int(silhouette.get("chamfer_max_iter", 100)),
    edges_canny_low            = int(edges.get("canny_low", 50)),
    edges_canny_high           = int(edges.get("canny_high", 150)),
    edges_search_range_px      = int(edges.get("search_range_px", 25)),
    edges_max_iter             = int(edges.get("max_iter", 50)),

    use_sam2 = bool(raw.get("use_sam2", True)),
    device   = str(raw.get("device", "cpu")),

    raw = raw,
  )


def _read_checkerboard_size(checkerboard: dict[str, Any]) -> tuple[int, int]:
  size = checkerboard.get("size") or checkerboard.get("inner_corners")
  if size is None:
    raise ValueError("object_localization.checkerboard.size (or inner_corners): missing required key.")
  return (int(size[0]), int(size[1]))


def _float(v: Any) -> float:
  return float(v)


def _parse_distortion_model(v: Any) -> int:
  """Validate the ``intrinsics.distortion_model`` value.

  Accepts the integer count of distortion coefficients to fit. Only
  ``0`` (pinhole), ``2`` (k1, k2), and ``5`` (full Brown-Conrady) are
  supported because OpenCV's calibration flags don't compose cleanly
  beyond those points.
  """
  n = int(v)
  if n not in (0, 2, 5):
    raise ValueError(
      f"object_localization.intrinsics.distortion_model: expected 0, 2, or 5, "
      f"got {v!r}.")
  return n
