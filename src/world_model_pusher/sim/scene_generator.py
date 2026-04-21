"""Samples randomised SceneConfig instances from difficulty-controlled distributions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .scene_config import (
    CameraConfig,
    LightingConfig,
    ObjectConfig,
    SceneConfig,
    object_half_z,
)

# z-coordinate of the table geom centre in the base MJCF (see base_scene.xml).
_TABLE_GEOM_CENTRE_Z = 0.02

# ---------------------------------------------------------------------------
# Difficulty presets
# ---------------------------------------------------------------------------

_PRESETS: dict[str, dict[str, Any]] = {
    "easy": {
        "shapes": ["box", "cylinder"],
        "mass_range": (0.05, 0.2),
        "num_obstacles": (0, 0),
        "num_clutter": (0, 0),
        "camera_angle_range": 5.0,   # degrees
        "push_distance": (0.05, 0.10),
        "lighting_variation": 0.0,
        "robot_type": "stick",
    },
    "medium": {
        "shapes": ["box", "cylinder", "capsule"],
        "mass_range": (0.02, 0.5),
        "num_obstacles": (0, 2),
        "num_clutter": (0, 3),
        "camera_angle_range": 15.0,
        "push_distance": (0.05, 0.20),
        "lighting_variation": 0.2,
        "robot_type": "so100",
    },
    "hard": {
        "shapes": ["box", "cylinder", "capsule", "sphere"],
        "mass_range": (0.01, 1.0),
        "num_obstacles": (0, 5),
        "num_clutter": (0, 8),
        "camera_angle_range": 30.0,
        "push_distance": (0.05, 0.30),
        "lighting_variation": 0.5,
        "robot_type": "so100",
    },
}

# Maximum arm reach (3 links × 0.15 m each, but realistic reach ~0.35 m)
_ARM_MAX_REACH = 0.38
_ARM_MIN_REACH = 0.12

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
    return 0.03


def _sample_object(
    rng: np.random.Generator,
    shapes: list[str],
    mass_range: tuple[float, float],
    table_half_x: float,
    table_half_y: float,
    table_top_z: float,
    margin: float = 0.07,
    min_x: float | None = None,
) -> ObjectConfig:
    shape       = rng.choice(shapes)
    mass        = float(rng.uniform(*mass_range))
    friction    = float(rng.uniform(0.3, 0.8))
    orientation = float(rng.uniform(0.0, 2 * math.pi))
    color       = [float(rng.uniform(0.1, 1.0)) for _ in range(3)] + [1.0]

    # Sample position anywhere on the accessible portion of the table
    x_low = -table_half_x + margin if min_x is None else max(-table_half_x + margin, min_x)
    x     = float(rng.uniform(x_low, table_half_x - margin))
    y     = float(rng.uniform(-table_half_y + margin, table_half_y - margin))

    if shape == "box":
        s    = float(rng.uniform(0.025, 0.05))
        size = [s, s, s]
    elif shape == "cylinder":
        r = float(rng.uniform(0.02, 0.045))
        h = float(rng.uniform(0.02, 0.06))
        size = [r, h]
    elif shape == "sphere":
        r = float(rng.uniform(0.02, 0.045))
        size = [r]
    elif shape == "capsule":
        r = float(rng.uniform(0.015, 0.035))
        h = float(rng.uniform(0.03, 0.07))
        size = [r, h]
    else:
        size = [0.03, 0.03, 0.03]

    partial = ObjectConfig(
        shape=shape,
        size=size,
        mass=mass,
        friction=friction,
        pos=[x, y, 0.0],
        orientation=orientation,
        color=color,
    )
    z = table_top_z + object_half_z(partial)
    partial.pos[2] = z
    return partial


class SceneGenerator:
    """Samples :class:`SceneConfig` instances for a given difficulty level."""

    def __init__(self, table_size: list[float], difficulty: str = "easy") -> None:
        if difficulty not in _PRESETS:
            raise ValueError(f"Unknown difficulty '{difficulty}'. Choose from {list(_PRESETS)}")
        self.difficulty = difficulty
        self._preset    = _PRESETS[difficulty]
        self.table_size = [float(v) for v in table_size]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, rng: np.random.Generator) -> SceneConfig:
        """Return a valid :class:`SceneConfig`, rejecting invalid samples."""
        for _ in range(200):
            cfg = self._sample_unchecked(rng)
            if self._is_valid(cfg):
                return cfg
        # Return last attempt even if not perfectly valid (avoids infinite loops
        # in degenerate difficulty settings)
        return cfg  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal sampling
    # ------------------------------------------------------------------

    def _sample_unchecked(self, rng: np.random.Generator) -> SceneConfig:
        p = self._preset

        table_half_x, table_half_y, table_half_z = self.table_size
        table_size                               = list(self.table_size)
        table_friction                           = float(rng.uniform(0.4, 0.7))
        table_color                              = [0.6, 0.5, 0.4, 1.0]
        table_top_z                              = _TABLE_GEOM_CENTRE_Z + table_half_z

        # Robot base sits on top of the table at the left edge (−x), centered in y
        robot_base_pos = [-table_half_x, 0.0, table_half_z]

        # Target object — must end up on the reachable side
        target = _sample_object(rng, p["shapes"], p["mass_range"], table_half_x, table_half_y, table_top_z, margin=0.07)

        # Goal: push direction at a random angle from target
        push_dist      = float(rng.uniform(*p["push_distance"]))
        push_angle     = float(rng.uniform(0.0, 2 * math.pi))
        gx             = target.pos[0] + push_dist * math.cos(push_angle)
        gy             = target.pos[1] + push_dist * math.sin(push_angle)
        goal_pos       = [gx, gy]
        goal_tolerance = 0.04

        # Obstacles
        num_obs = int(rng.integers(p["num_obstacles"][0], p["num_obstacles"][1] + 1))
        obstacles: list[ObjectConfig] = []
        for _ in range(num_obs):
            obs = _sample_object(rng, ["box", "cylinder"], p["mass_range"], table_half_x, table_half_y, table_top_z, min_x=-table_half_x / 2)
            obstacles.append(obs)

        # Clutter (visual only — contype=0)
        num_clutter = int(rng.integers(p["num_clutter"][0], p["num_clutter"][1] + 1))
        clutter: list[ObjectConfig] = []
        for _ in range(num_clutter):
            cl = _sample_object(rng, p["shapes"], p["mass_range"], table_half_x, table_half_y, table_top_z, min_x=-table_half_x / 2)
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
    # Validity checks
    # ------------------------------------------------------------------

    def _is_valid(self, cfg: SceneConfig) -> bool:
        return (
            self._check_reachability(cfg)
            and self._check_goal_on_table(cfg)
            and self._check_no_overlaps(cfg)
            and self._check_push_path(cfg)
            and self._check_objects_in_frustum(cfg)
        )

    def _check_reachability(self, cfg: SceneConfig) -> bool:
        """Target must be within arm reach from robot base."""
        tx, ty = cfg.target.pos[:2]
        bx, by = cfg.robot_base_pos[:2]
        dist = math.hypot(tx - bx, ty - by)
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
            t = max(0.0, min(length, (ox - tx) * ux + (oy - ty) * uy))
            cx = tx + t * ux
            cy = ty + t * uy
            if math.hypot(ox - cx, oy - cy) < _OBJ_RADIUS:
                return False
        return True

    def _check_objects_in_frustum(self, cfg: SceneConfig) -> bool:
        """All objects on the table should be visible by roughly checking camera FOV."""
        # Simple conservative check: all object x,y positions within table
        # bounds
        hx, hy = cfg.table_size[:2]
        objects_to_check = [cfg.target] + cfg.obstacles + cfg.clutter
        for obj in objects_to_check:
            ox, oy = obj.pos[:2]
            if abs(ox) > hx or abs(oy) > hy:
                return False
        return True
