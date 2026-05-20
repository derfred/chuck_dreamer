"""Samples randomised SceneConfig instances from difficulty-controlled distributions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pathlib import Path

from .scene_config import (
    CameraConfig,
    LightingConfig,
    ObjectConfig,
    SceneConfig,
    joint_names_for_robot,
    object_half_z,
)

# z-coordinate of the table geom centre in the base MJCF (see base_scene.xml).
_TABLE_GEOM_CENTRE_Z = 0.02

# Pre-converted STL meshes available as the "mesh" shape. Keyed by a short
# tag stored in ObjectConfig.mesh_path so the builder knows which asset to
# register. The half-extent (native bounding-box half-side, in metres) lets
# the generator pick a per-instance scale that yields a sensible footprint.
_MESH_LIBRARY: dict[str, dict[str, Any]] = {
  "rhombicuboctahedron": {
    "stl": str(Path(__file__).parent.parent.parent.parent
               / "assets" / "mujoco" / "rhombicuboctahedron.stl"),
    "native_half_extent": 0.029,  # ~58mm bounding cube, centred at origin
  },
}

# ---------------------------------------------------------------------------
# Difficulty presets
# ---------------------------------------------------------------------------

# Distractor (obstacle/clutter) shapes per difficulty. ``target_shapes`` lists
# the shapes the pushable target object may take. ``color_mode`` controls how
# the target and obstacle/clutter colors are sampled — see ``_sample_color``.
_PRESETS: dict[str, dict[str, Any]] = {
  "easy": {
    "target_shapes": ["box", "cylinder"],
    "distractor_shapes": ["box", "cylinder"],
    "mass_range": (0.05, 0.2),
    "num_obstacles": (0, 0),
    "num_clutter": (0, 0),
    "camera_angle_range": 5.0,   # degrees
    "push_distance": (0.05, 0.10),
    "lighting_variation": 0.0,
    "robot_type": "so100",
    "color_mode": "target_red",
  },
  "medium": {
    "target_shapes": ["box", "cylinder", "mesh"],
    "distractor_shapes": ["box", "cylinder", "pyramid", "mesh"],
    "mass_range": (0.02, 0.5),
    "num_obstacles": (0, 2),
    "num_clutter": (0, 3),
    "camera_angle_range": 15.0,
    "push_distance": (0.05, 0.20),
    "lighting_variation": 0.2,
    "robot_type": "so100",
    "color_mode": "target_red_distractors_non_red",
  },
  "hard": {
    "target_shapes": ["box", "cylinder", "capsule", "sphere", "pyramid", "mesh"],
    "distractor_shapes": ["box", "cylinder", "capsule", "sphere", "pyramid", "mesh"],
    "mass_range": (0.01, 1.0),
    "num_obstacles": (0, 5),
    "num_clutter": (0, 8),
    "camera_angle_range": 30.0,
    "push_distance": (0.05, 0.30),
    "lighting_variation": 0.5,
    "robot_type": "so100",
    "color_mode": "shared",
  },
  # Mirrors the real-world calibration mat: black table-top with two white
  # 30 cm parallel lines (16 cm apart) in front of the arm and a 10 cm
  # diameter white circle sitting between the lines' far-side endpoints.
  # The pushable target is always the rhombicuboctahedron mesh — same object
  # used in the physical experiments — placed somewhere reachable on the mat.
  "real": {
    "target_shapes": ["mesh"],
    "distractor_shapes": [],
    "mass_range": (0.05, 0.2),
    "num_obstacles": (0, 0),
    "num_clutter": (0, 0),
    "camera_angle_range": 0.0,
    "push_distance": (0.05, 0.15),
    "lighting_variation": 0.0,
    "robot_type": "so100",
    "color_mode": "target_red",  # unused — target color overridden below
  },
}

# --- "real" preset layout (matches the physical calibration mat) ----------
# All coordinates are in metres. The pattern is rotated 90° relative to the
# arm's reach axis: the two lines run along the y-axis, separated by D along
# x, and the circle sits between their +y endpoints. The mat is centred on
# (x, y) = (_REAL_MAT_CENTRE_X, 0), placing it inside the arm's reach annulus.
_REAL_TABLE_COLOR     = [0.05, 0.05, 0.05, 1.0]
_REAL_MARKING_COLOR   = [0.95, 0.95, 0.95, 1.0]
_REAL_LINE_LENGTH     = 0.30
_REAL_LINE_SEPARATION = 0.16
_REAL_LINE_WIDTH      = 0.005   # 5 mm-wide painted lines
_REAL_CIRCLE_RADIUS   = 0.05    # 10 cm diameter
# Decal half-height — kept tiny so the markings read as paint on the mat.
_REAL_MARKING_HALF_HEIGHT = 0.0005
# Mat centred ~0.25 m in front of the arm base. Lines run from y = -L/2 to
# y = +L/2; the circle sits at (_REAL_MAT_CENTRE_X, +L/2).
_REAL_MAT_CENTRE_X = -0.35
_REAL_LINE_Y_NEAR  = -_REAL_LINE_LENGTH / 2.0
_REAL_LINE_Y_FAR   = _REAL_LINE_LENGTH / 2.0

# Circle is drawn as an unfilled outline (a ring of N tangent box segments).
# Width is the radial thickness of the painted ring on the real mat.
_REAL_CIRCLE_RING_SEGMENTS = 64
_REAL_CIRCLE_RING_WIDTH    = 0.003   # 3 mm ring thickness

# Per-episode randomisation of the mat's placement on the table. The pattern
# is rigidly translated within this box (centred on the nominal mat origin) to
# stop the policy memorising absolute positions. The "along" axis runs with
# the painted lines (y); the "across" axis is perpendicular to them (x).
_REAL_MAT_OFFSET_ALONG  = 0.050   # ±50 mm along the lines (y)
_REAL_MAT_OFFSET_ACROSS = 0.025   # ±25 mm across the lines (x)

# RGBA tuple for the "red" target color used in easy/medium difficulties.
_RED_RGBA = [0.85, 0.15, 0.15, 1.0]

# Threshold (in normalized RGB distance) below which a sampled color is
# considered "too red" and rejected when distractors must avoid red.
_RED_REJECTION_DIST = 0.45

# Maximum arm reach (3 links × 0.15 m each, but realistic reach ~0.35 m)
_ARM_MAX_REACH = 0.4
_ARM_MIN_REACH = 0.2

# Object radius used for overlap checks (conservative)
_OBJ_RADIUS = 0.06


def _object_footprint_radius(cfg: ObjectConfig) -> float:
  """Worst-case radius of the object's projection onto the xy plane."""
  s = cfg.size
  if cfg.shape == "box":
    return math.hypot(s[0], s[1])
  if cfg.shape == "cylinder":
    return s[0]
  if cfg.shape == "sphere":
    return s[0]
  if cfg.shape == "capsule":
    return s[0]
  if cfg.shape == "pyramid":
    # Square base with half-extent s[0]; worst-case diagonal radius.
    return math.hypot(s[0], s[0])
  if cfg.shape == "mesh":
    # size = [half_extent]; conservative xy radius is the bounding diagonal.
    return math.hypot(s[0], s[0])
  return 0.03


def _sample_non_red_color(rng: np.random.Generator) -> list[float]:
  """Sample an RGBA color whose RGB is sufficiently far from the target red.

  Rejection sampling — capped at a few attempts to avoid pathological loops.
  If we never find a valid color we fall back to a desaturated grey so the
  scene still renders something sensible.
  """
  red = np.array(_RED_RGBA[:3])
  for _ in range(32):
    rgb = np.array([rng.uniform(0.1, 1.0) for _ in range(3)])
    if np.linalg.norm(rgb - red) > _RED_REJECTION_DIST:
      return [float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0]
  return [0.4, 0.4, 0.5, 1.0]


def _sample_color(rng: np.random.Generator) -> list[float]:
  return [float(rng.uniform(0.1, 1.0)) for _ in range(3)] + [1.0]


def _sample_object(
    rng: np.random.Generator,
    shapes: list[str],
    mass_range: tuple[float, float],
    table_half_x: float,
    table_half_y: float,
    table_top_z: float,
    color: list[float],
    margin: float = 0.07,
    min_x: float | None = None,
    mesh_options: list[str] | None = None,
) -> ObjectConfig:
  shape       = str(rng.choice(shapes))
  mass        = float(rng.uniform(*mass_range))
  friction    = float(rng.uniform(0.3, 0.8))
  orientation = float(rng.uniform(0.0, 2 * math.pi))

  # Sample position anywhere on the accessible portion of the table
  x_low = -table_half_x + margin if min_x is None else max(-table_half_x + margin, min_x)
  x     = float(rng.uniform(x_low, table_half_x - margin))
  y     = float(rng.uniform(-table_half_y + margin, table_half_y - margin))

  mesh_tag: str | None = None
  if shape == "box":
    s    = float(rng.uniform(0.025, 0.05))
    size = [s, s, s]
  elif shape == "cylinder":
    r    = float(rng.uniform(0.02, 0.045))
    h    = float(rng.uniform(0.02, 0.06))
    size = [r, h]
  elif shape == "sphere":
    r = float(rng.uniform(0.02, 0.045))
    size = [r]
  elif shape == "capsule":
    r = float(rng.uniform(0.015, 0.035))
    h = float(rng.uniform(0.03, 0.07))
    size = [r, h]
  elif shape == "pyramid":
    half_base = float(rng.uniform(0.025, 0.05))
    height    = float(rng.uniform(0.04, 0.08))
    size      = [half_base, height]
  elif shape == "mesh":
    if not mesh_options:
      raise ValueError("'mesh' shape sampled but no mesh_options provided")
    mesh_tag = str(rng.choice(mesh_options))
    # Aim for a final bounding-cube half-extent in [0.020, 0.040] m so the
    # mesh stays comparable to the primitive objects.
    half = float(rng.uniform(0.020, 0.040))
    size = [half]
  else:
    size = [0.03, 0.03, 0.03]

  partial = ObjectConfig(
      shape=shape,
      size=size,
      mass=mass,
      friction=friction,
      pos=[x, y, 0.0],
      orientation=orientation,
      color=list(color),
      mesh_path=mesh_tag,
  )
  z = table_top_z + object_half_z(partial)
  partial.pos[2] = z
  return partial


class SceneGenerator:
  """Samples :class:`SceneConfig` instances for a given difficulty level."""

  def __init__(self, config) -> None:
    if config.env.difficulty not in _PRESETS:
      raise ValueError(f"Unknown difficulty '{config.env.difficulty}'. Choose from {list(_PRESETS)}")
    self.config     = config
    self._rng       = np.random.default_rng(config.seed)
    self.difficulty = config.env.difficulty
    self._preset    = _PRESETS[config.env.difficulty]
    self.table_size = [float(v) for v in config.env.table_size]

  @property
  def robot_type(self) -> Any:
    return self._preset["robot_type"]

  @property
  def joint_names(self) -> list[str]:
    return joint_names_for_robot(self.robot_type)

  @property
  def n_joints(self) -> int:
    return len(self.joint_names)

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  def sample(self) -> SceneConfig:
    """Return a valid :class:`SceneConfig`, rejecting invalid samples."""
    for _ in range(200):
      cfg = self._sample_unchecked(self._rng)
      if self._is_valid(cfg):
        return cfg
    # Return last attempt even if not perfectly valid (avoids infinite loops
    # in degenerate difficulty settings)
    return cfg  # type: ignore[return-value]

  # ------------------------------------------------------------------
  # Internal sampling
  # ------------------------------------------------------------------

  def _sample_unchecked(self, rng: np.random.Generator) -> SceneConfig:
      if self.difficulty == "real":
        return self._sample_real(rng)
      p = self._preset

      table_half_x, table_half_y, table_half_z = self.table_size
      table_size                               = list(self.table_size)
      table_friction                           = float(rng.uniform(0.4, 0.7))
      table_color                              = [0.6, 0.5, 0.4, 1.0]
      table_top_z                              = _TABLE_GEOM_CENTRE_Z + table_half_z

      # Robot base sits on top of the table at the left edge (−x), centered in y
      robot_base_pos = [-table_half_x, 0.0, table_half_z]

      # Colors — per difficulty:
      #   easy:   target red, no distractors anyway
      #   medium: target red, distractors any non-red color
      #   hard:   target and distractors share one shared random color
      color_mode = p["color_mode"]
      if color_mode == "target_red":
        target_color     = list(_RED_RGBA)
        distractor_color = _sample_non_red_color  # callable; sample per-instance
      elif color_mode == "target_red_distractors_non_red":
        target_color     = list(_RED_RGBA)
        distractor_color = _sample_non_red_color
      elif color_mode == "shared":
        shared           = _sample_color(rng)
        target_color     = shared
        distractor_color = (lambda _rng, c=shared: list(c))  # type: ignore[misc]
      else:
        raise ValueError(f"Unknown color_mode '{color_mode}' in preset")

      mesh_options = list(_MESH_LIBRARY.keys())

      # Target object — must end up on the reachable side
      target = _sample_object(
          rng, p["target_shapes"], p["mass_range"],
          table_half_x, table_half_y, table_top_z,
          color=target_color, margin=0.07,
          mesh_options=mesh_options,
      )

      # Goal: push direction at a random angle from target
      push_dist      = float(rng.uniform(*p["push_distance"]))
      push_angle     = float(rng.uniform(0.0, 2 * math.pi))
      gx             = target.pos[0] + push_dist * math.cos(push_angle)
      gy             = target.pos[1] + push_dist * math.sin(push_angle)
      goal_pos       = [gx, gy]
      goal_tolerance = 0.04

      # Obstacles (colliding distractors). Shapes follow the preset's
      # ``distractor_shapes`` list, which includes the pyramid at medium+.
      num_obs = int(rng.integers(p["num_obstacles"][0], p["num_obstacles"][1] + 1))
      obstacles: list[ObjectConfig] = []
      for _ in range(num_obs):
        obs = _sample_object(
            rng, p["distractor_shapes"], p["mass_range"],
            table_half_x, table_half_y, table_top_z,
            color=distractor_color(rng),
            min_x=-table_half_x / 2,
            mesh_options=mesh_options,
        )
        obstacles.append(obs)

      # Clutter (visual only — contype=0). Same shape/color rules as obstacles.
      num_clutter = int(rng.integers(p["num_clutter"][0], p["num_clutter"][1] + 1))
      clutter: list[ObjectConfig] = []
      for _ in range(num_clutter):
        cl = _sample_object(
            rng, p["distractor_shapes"], p["mass_range"],
            table_half_x, table_half_y, table_top_z,
            color=distractor_color(rng),
            min_x=-table_half_x / 2,
            mesh_options=mesh_options,
        )
        clutter.append(cl)

      # Camera — looking down at the table from above and to one side
      base_cam_pos = [0.0, -0.40, 0.55]
      angle_range = p["camera_angle_range"]
      cam_angle_offset = float(rng.uniform(-angle_range, angle_range))
      rad = math.radians(cam_angle_offset)
      cam_pos = [
          base_cam_pos[0] + 0.3 * math.sin(rad),
          base_cam_pos[1],
          base_cam_pos[2],
      ]
      camera = CameraConfig(pos=cam_pos, look_at=[0.0, 0.0, table_half_z], fov=60.0)

      # Lighting
      base_intensity = 0.8
      variation      = p["lighting_variation"]
      intensity      = float(np.clip(base_intensity + rng.uniform(-variation, variation), 0.1, 1.0))
      ambient        = float(np.clip(0.3 + rng.uniform(-variation * 0.5, variation * 0.5), 0.05, 1.0))
      lighting       = LightingConfig(direction=[0.0, -0.5, -1.0], intensity=intensity, ambient=ambient)

      return SceneConfig(
          table_size=table_size,
          table_friction=table_friction,
          table_color=table_color,
          robot_type=p["robot_type"],
          robot_base_pos=robot_base_pos,
          robot_initial_qpos=None,
          target=target,
          goal_pos=goal_pos,
          goal_tolerance=goal_tolerance,
          obstacles=obstacles,
          clutter=clutter,
          camera=camera,
          lighting=lighting,
          max_steps=150,
          control_dt=0.1,
      )

  # ------------------------------------------------------------------
  # "real" preset — fixed mat geometry, randomised target placement
  # ------------------------------------------------------------------

  def _sample_real(self, rng: np.random.Generator) -> SceneConfig:
      p = self._preset

      table_half_x, table_half_y, table_half_z = self.table_size
      table_size      = list(self.table_size)
      table_friction  = float(rng.uniform(0.4, 0.7))
      table_color     = list(_REAL_TABLE_COLOR)
      table_top_z     = _TABLE_GEOM_CENTRE_Z + table_half_z
      robot_base_pos  = [-table_half_x, 0.0, table_half_z]

      mesh_options = list(_MESH_LIBRARY.keys())

      # Per-episode mat offset: rigidly translate the whole pattern within a
      # box around the nominal centre. Across = x (perpendicular to lines),
      # along = y (parallel to lines).
      mat_offset_x = float(rng.uniform(-_REAL_MAT_OFFSET_ACROSS, _REAL_MAT_OFFSET_ACROSS))
      mat_offset_y = float(rng.uniform(-_REAL_MAT_OFFSET_ALONG,  _REAL_MAT_OFFSET_ALONG))
      mat_centre_x = _REAL_MAT_CENTRE_X + mat_offset_x
      line_y_near  = _REAL_LINE_Y_NEAR  + mat_offset_y
      line_y_far   = _REAL_LINE_Y_FAR   + mat_offset_y

      # Mat layout — pattern rotated 90°: lines run along the y-axis, separated
      # along x by D. The circle sits between the +y endpoints; the camera is
      # fixed on the same +y side so the operator's view always frames it.
      half_sep = _REAL_LINE_SEPARATION / 2.0
      line_x_left   = mat_centre_x - half_sep
      line_x_right  = mat_centre_x + half_sep
      circle_centre = (mat_centre_x, line_y_far)

      # Target: rhombicuboctahedron, placed somewhere between the lines and
      # within the arm's reachable annulus.
      target_color = [0.85, 0.15, 0.15, 1.0]
      margin = 0.02
      for _ in range(64):
        tx = float(rng.uniform(line_x_left + margin, line_x_right - margin))
        ty = float(rng.uniform(line_y_near + margin, line_y_far - margin))
        if _ARM_MIN_REACH <= math.hypot(tx - robot_base_pos[0], ty) <= _ARM_MAX_REACH:
          break
      mesh_tag = str(rng.choice(mesh_options))
      half = float(rng.uniform(0.020, 0.030))
      target = ObjectConfig(
          shape="mesh",
          size=[half],
          mass=float(rng.uniform(*p["mass_range"])),
          friction=float(rng.uniform(0.3, 0.8)),
          pos=[tx, ty, 0.0],
          orientation=float(rng.uniform(0.0, 2 * math.pi)),
          color=target_color,
          mesh_path=mesh_tag,
      )
      target.pos[2] = table_top_z + object_half_z(target)

      # Goal: the centre of the white circle — fixed by the mat geometry.
      goal_pos       = [circle_centre[0], circle_centre[1]]
      goal_tolerance = _REAL_CIRCLE_RADIUS

      # Visual mat markings: two white "lines" as thin elongated boxes (rotated
      # 90° → long axis along y) plus a white "circle" outline rendered as a
      # ring of N tangent box segments (unfilled — matches the painted ring on
      # the physical mat). All live in the clutter list so the builder renders
      # them with contype=0.
      line_cy = (line_y_near + line_y_far) / 2.0
      line_cz = table_top_z + _REAL_MARKING_HALF_HEIGHT
      line_half_length = _REAL_LINE_LENGTH / 2.0
      line_half_width  = _REAL_LINE_WIDTH / 2.0
      lines: list[ObjectConfig] = []
      for sign in (-1.0, 1.0):
        lines.append(ObjectConfig(
            shape="box",
            # size = (half_x, half_y, half_z); long axis is y, short axis is x
            size=[line_half_width, line_half_length, _REAL_MARKING_HALF_HEIGHT],
            mass=0.001,
            friction=0.5,
            pos=[mat_centre_x + sign * half_sep, line_cy, line_cz],
            orientation=0.0,
            color=list(_REAL_MARKING_COLOR),
        ))
      # Ring of tangent box segments. Each segment is a small chord:
      #   half-length along the tangent ≈ r·sin(π/N)
      #   half-width  along the radius  = ring_width/2
      n_seg          = _REAL_CIRCLE_RING_SEGMENTS
      seg_half_chord = _REAL_CIRCLE_RADIUS * math.sin(math.pi / n_seg)
      seg_half_width = _REAL_CIRCLE_RING_WIDTH / 2.0
      cx, cy_c = circle_centre
      ring: list[ObjectConfig] = []
      for i in range(n_seg):
        theta = 2.0 * math.pi * (i + 0.5) / n_seg
        ring.append(ObjectConfig(
            shape="box",
            # Local axes before yaw: x = tangent, y = radial. We rotate by
            # theta so x ends up tangent to the circle at angle theta.
            size=[seg_half_chord, seg_half_width, _REAL_MARKING_HALF_HEIGHT],
            mass=0.001,
            friction=0.5,
            pos=[
                cx + _REAL_CIRCLE_RADIUS * math.cos(theta),
                cy_c + _REAL_CIRCLE_RADIUS * math.sin(theta),
                line_cz,
            ],
            orientation=theta + math.pi / 2.0,
            color=list(_REAL_MARKING_COLOR),
        ))
      clutter = lines + ring

      # Camera — operator's-eye pose on a 110° front-facing arc centred on
      # the arm base, 50 cm above the table. Angle 0 = directly in front of
      # the arm (+x direction); we sample in ±55° so the camera always looks
      # back toward the arm and sees the full mat + arm.
      cam_radius = 0.65
      cam_height = 0.50
      cam_angle  = float(rng.uniform(-math.radians(55.0), math.radians(55.0)))
      bx, by, _bz = robot_base_pos
      cam_pos = [
          bx + cam_radius * math.cos(cam_angle),
          by + cam_radius * math.sin(cam_angle),
          cam_height,
      ]
      look_target = [mat_centre_x, mat_offset_y, table_top_z]
      camera = CameraConfig(pos=cam_pos, look_at=look_target, fov=70.0)

      lighting = LightingConfig(direction=[0.0, -0.5, -1.0], intensity=0.8, ambient=0.3)

      return SceneConfig(
          table_size=table_size,
          table_friction=table_friction,
          table_color=table_color,
          robot_type=p["robot_type"],
          robot_base_pos=robot_base_pos,
          robot_initial_qpos=None,
          target=target,
          goal_pos=goal_pos,
          goal_tolerance=goal_tolerance,
          obstacles=[],
          clutter=clutter,
          camera=camera,
          lighting=lighting,
          max_steps=150,
          control_dt=0.1,
          draw_goal_marker=False,
          mat_centre=[mat_centre_x, mat_offset_y],
      )

  # ------------------------------------------------------------------
  # Validity checks
  # ------------------------------------------------------------------

  def _is_valid(self, cfg: SceneConfig) -> bool:
    if self.difficulty == "real":
      # The mat-marking clutter intentionally sits past the arm's reach, and
      # the goal (circle centre) is allowed to as well — only the target's
      # reachability matters.
      return self._check_reachability(cfg) and self._check_goal_on_table(cfg)
    return (
        self._check_reachability(cfg)
        and self._check_goal_reachability(cfg)
        and self._check_goal_on_table(cfg)
        and self._check_no_overlaps(cfg)
        and self._check_push_path(cfg)
        and self._check_objects_in_frustum(cfg)
    )

  def _check_reachability(self, cfg: SceneConfig) -> bool:
    """Target must be within arm reach from robot base."""
    tx, ty = cfg.target.pos[:2]
    bx, by = cfg.robot_base_pos[:2]
    dist   = math.hypot(tx - bx, ty - by)
    return _ARM_MIN_REACH <= dist <= _ARM_MAX_REACH

  def _check_goal_reachability(self, cfg: SceneConfig) -> bool:
    """Goal must lie within the same reachability annulus as the target."""
    gx, gy = cfg.goal_pos
    bx, by = cfg.robot_base_pos[:2]
    dist   = math.hypot(gx - bx, gy - by)
    return _ARM_MIN_REACH <= dist <= _ARM_MAX_REACH

  def _check_goal_on_table(self, cfg: SceneConfig) -> bool:
    gx, gy = cfg.goal_pos
    hx, hy = cfg.table_size[:2]
    margin = 0.03
    return (
        (-hx + margin) <= gx <= (hx - margin)
        and (-hy + margin) <= gy <= (hy - margin)
    )

  def _check_no_overlaps(self, cfg: SceneConfig) -> bool:
    """No colliding object should overlap with another, the target, or the goal."""
    gx, gy = cfg.goal_pos
    colliders = [cfg.target] + cfg.obstacles
    for i, a in enumerate(colliders):
      ax, ay = a.pos[:2]
      for b in colliders[i + 1:]:
        bx, by = b.pos[:2]
        if math.hypot(ax - bx, ay - by) < 2 * _OBJ_RADIUS:
          return False
    # Goal must lie outside the target's current footprint, otherwise the
    # target is already "at the goal" at t=0.
    tx, ty = cfg.target.pos[:2]
    target_radius = _object_footprint_radius(cfg.target)
    if math.hypot(gx - tx, gy - ty) < target_radius + cfg.goal_tolerance:
      return False
    for obs in cfg.obstacles:
      ox, oy = obs.pos[:2]
      if math.hypot(gx - ox, gy - oy) < _OBJ_RADIUS:
        return False
    return True

  def _check_push_path(self, cfg: SceneConfig) -> bool:
    """Simple check: no obstacle fully blocks the straight line target→goal."""
    if not cfg.obstacles:
      return True
    tx, ty = cfg.target.pos[:2]
    gx, gy = cfg.goal_pos
    dx, dy = gx - tx, gy - ty
    length = math.hypot(dx, dy)
    if length < 1e-6:
      return True
    ux, uy = dx / length, dy / length
    for obs in cfg.obstacles:
      ox, oy = obs.pos[:2]
      # Project obstacle centre onto the push ray
      t  = max(0.0, min(length, (ox - tx) * ux + (oy - ty) * uy))
      cx = tx + t * ux
      cy = ty + t * uy
      if math.hypot(ox - cx, oy - cy) < _OBJ_RADIUS:
        return False
    return True

  def _check_objects_in_frustum(self, cfg: SceneConfig) -> bool:
    """All objects on the table should be visible by roughly checking camera FOV."""
    # Simple conservative check: all object x,y positions within table bounds
    hx, hy = cfg.table_size[:2]
    objects_to_check = [cfg.target] + cfg.obstacles + cfg.clutter
    for obj in objects_to_check:
      ox, oy = obj.pos[:2]
      if abs(ox) > hx or abs(oy) > hy:
        return False
    return True
