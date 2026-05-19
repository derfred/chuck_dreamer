"""Runtime configuration for the object localization pipeline.

All values live under ``cfg.object_localization`` in the Chuck Dreamer
config (``configs/default.yaml``). Call :func:`init_from_config` once at
program start; the rest of the package reads from the function
accessors below (e.g. :func:`mat_line_length_mm`), which raise if the
config has not been installed yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class ObjectLocalizationConfig:
  camera_key:                str
  episode_empty:             int
  episode_checkerboard:      int
  episode_object_left:       int
  episode_object_mid:        int
  episode_object_right:      int
  checkerboard_inner_corners: tuple[int, int]
  checkerboard_square_mm:     float
  mat_line_length_mm:         float
  mat_line_separation_mm:     float
  mat_circle_radius_mm:       float
  calibration_cache:          str
  mesh_path:                  str
  use_sam2:                   bool
  device:                     str

  @property
  def object_episodes(self) -> tuple[int, int, int]:
    return (self.episode_object_left, self.episode_object_mid, self.episode_object_right)


_active: ObjectLocalizationConfig | None = None


def init_from_config(cfg: Any) -> ObjectLocalizationConfig:
  """Validate ``cfg.object_localization`` and install it as the active config."""
  global _active
  if isinstance(cfg, DictConfig):
    section = cfg.object_localization
  elif isinstance(cfg, dict):
    section = OmegaConf.create(cfg).object_localization
  else:
    raise TypeError(f"init_from_config: expected DictConfig/dict, got {type(cfg).__name__}")

  inner = tuple(int(v) for v in section.checkerboard.inner_corners)
  if len(inner) != 2:
    raise ValueError(
      f"object_localization.checkerboard.inner_corners must be (cols, rows), got {inner}")

  for key in ("line_length_mm", "line_separation_mm", "circle_radius_mm"):
    val = section.mat.get(key)
    if val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
      raise ValueError(
        f"object_localization.mat.{key} must be a number — set it to the measured "
        f"mat dimension in configs/default.yaml (got {val!r}).")

  _active = ObjectLocalizationConfig(
    camera_key                  = str(section.camera_key),
    episode_empty               = int(section.episodes.empty),
    episode_checkerboard        = int(section.episodes.checkerboard),
    episode_object_left         = int(section.episodes.object_left),
    episode_object_mid          = int(section.episodes.object_mid),
    episode_object_right        = int(section.episodes.object_right),
    checkerboard_inner_corners  = (inner[0], inner[1]),
    checkerboard_square_mm      = float(section.checkerboard.square_mm),
    mat_line_length_mm          = float(section.mat.line_length_mm),
    mat_line_separation_mm      = float(section.mat.line_separation_mm),
    mat_circle_radius_mm        = float(section.mat.circle_radius_mm),
    calibration_cache           = str(section.calibration_cache),
    mesh_path                   = str(section.mesh_path),
    use_sam2                    = bool(section.use_sam2),
    device                      = str(section.device),
  )
  return _active


def active() -> ObjectLocalizationConfig:
  if _active is None:
    raise RuntimeError(
      "object_localization runtime not initialised. Call "
      "chuck_dreamer.real.object_localization.init_from_config(cfg) once at "
      "startup (the chuck_dreamer `main.py` CLI does this automatically).")
  return _active


# Convenience accessors so call-sites don't need to spell out ``active().foo``.

def camera_key()                  -> str:               return active().camera_key
def episode_empty()               -> int:               return active().episode_empty
def episode_checkerboard()        -> int:               return active().episode_checkerboard
def episode_object_left()         -> int:               return active().episode_object_left
def episode_object_mid()          -> int:               return active().episode_object_mid
def episode_object_right()        -> int:               return active().episode_object_right
def object_episodes()             -> tuple[int, int, int]: return active().object_episodes
def checkerboard_inner_corners()  -> tuple[int, int]:   return active().checkerboard_inner_corners
def checkerboard_square_mm()      -> float:             return active().checkerboard_square_mm
def mat_line_length_mm()          -> float:             return active().mat_line_length_mm
def mat_line_separation_mm()      -> float:             return active().mat_line_separation_mm
def mat_circle_radius_mm()        -> float:             return active().mat_circle_radius_mm
def calibration_cache()           -> str:               return active().calibration_cache
def mesh_path()                   -> str:               return active().mesh_path
def use_sam2()                    -> bool:              return active().use_sam2
def device()                      -> str:               return active().device
