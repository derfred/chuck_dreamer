"""Dataclasses describing a single episode's scene configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ObjectConfig:
    shape: str                     # "box", "cylinder", "sphere", "capsule", "mesh"
    size: list[float]              # shape-dependent dimensions
    mass: float                    # kg
    friction: float                # sliding friction coefficient
    pos: list[float]               # [x, y, z] world position (z = object centre)
    orientation: float             # yaw angle in radians
    color: list[float]             # RGBA
    mesh_path: str | None = None   # for shape="mesh"


@dataclass
class CameraConfig:
    pos: list[float]               # [x, y, z]
    look_at: list[float]           # [x, y, z]
    fov: float                     # degrees


@dataclass
class LightingConfig:
    direction: list[float]         # [x, y, z]
    intensity: float               # 0.0–1.0
    ambient: float                 # ambient light level


@dataclass
class SceneConfig:
    # Table
    table_size: list[float]        # [half_x, half_y, half_z]
    table_friction: float
    table_color: list[float]       # RGBA

    # Robot
    robot_type: str                # "stick" | "so100"
    robot_base_pos: list[float]    # [x, y, z] relative to table
    # if None, use default home position
    robot_initial_qpos: list[float] | None

    # Target object (the one to push)
    target: ObjectConfig

    # Goal
    goal_pos: list[float]          # [x, y] on table surface
    goal_tolerance: float          # distance threshold for "done"

    # Obstacles (objects that block the path but aren't pushed)
    obstacles: list[ObjectConfig]

    # Clutter (visual-only objects for domain randomization)
    clutter: list[ObjectConfig]

    # Camera and lighting
    camera: CameraConfig
    lighting: LightingConfig

    # Episode parameters
    max_steps: int
    control_dt: float              # seconds per action step

    @property
    def joint_initial_qpos(self) -> list[float] | None:
        if self.robot_initial_qpos is not None:
            return self.robot_initial_qpos
        if self.robot_type == "stick":
            return [0.0]
        elif self.robot_type == "so100":
            return [0, -3.14, 3.14, 0, 0, 0]
        else:
            raise ValueError(f"Unknown robot type: {self.robot_type}")

    @property
    def joint_names(self) -> list[str]:
        if self.robot_type == "stick":
            return ["joint1"]
        elif self.robot_type == "so100":
            return ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        else:
            raise ValueError(f"Unknown robot type: {self.robot_type}")

    @property
    def actuator_names(self) -> list[str]:
        return self.joint_names


def object_half_z(cfg: ObjectConfig) -> float:
    """Half-extent along z for an object resting with its default orientation."""
    s = cfg.size
    if cfg.shape == "box":
        return s[2] if len(s) > 2 else s[0]
    if cfg.shape == "cylinder":
        return s[1] if len(s) > 1 else s[0]
    if cfg.shape == "sphere":
        return s[0]
    if cfg.shape == "capsule":
        return (s[0] + s[1]) if len(s) > 1 else s[0]
    return 0.03
