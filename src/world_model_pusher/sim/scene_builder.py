"""Builds a compiled mujoco.MjModel from a SceneConfig via XML manipulation."""

from __future__ import annotations

import math
import os
from pathlib import Path

import mujoco
import numpy as np
from lxml import etree

from .scene_config import ObjectConfig, SceneConfig

# Absolute path to the SO-100 assets directory (relative to this file)
_ASSETS_DIR = Path(__file__).parent.parent.parent.parent / "assets" / "mujoco"
_SO100_XML = _ASSETS_DIR / "trs_so_arm100" / "so_arm100.xml"

# ---------------------------------------------------------------------------
# Default base MJCF writer
# ---------------------------------------------------------------------------

_BASE_MJCF_TEMPLATE = """\
<mujoco model="pushing_base">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81" iterations="50" solver="Newton"/>
  <default>
    <joint damping="0.5" armature="0.01"/>
    <geom condim="3" friction="0.8 0.02 0.001" contype="1" conaffinity="1"/>
  </default>
  <worldbody>
    <!-- Ground plane -->
    <geom name="ground" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.3 1"
          pos="0 0 0" contype="1" conaffinity="1"/>
    <!-- Table -->
    <body name="table" pos="0 0 0">
      <geom name="table_top" type="box" size="0.30 0.25 0.02"
            pos="0 0 0.02" rgba="0.6 0.5 0.4 1"
            contype="1" conaffinity="1" friction="0.6 0.02 0.001"/>
    </body>
    <!-- Pusher: heavy dynamic body welded tightly to the mocap.
         The mocap body is pure kinematic (no collision).
         The ee_link body is dynamic and carries the collision geom.
         A very stiff weld makes ee_link track mocap_target exactly.
         Mass=2kg ensures contact forces push objects rather than deflecting the pusher. -->
    <body name="mocap_target" mocap="true" pos="0.0 0 0.075">
      <geom type="sphere" size="0.008" rgba="1 0 0 0.6"
            contype="0" conaffinity="0"/>
    </body>
    <body name="ee_link" pos="0.0 0 0.075">
      <joint name="joint1" type="free"/>
      <geom type="cylinder" size="0.025 0.015" rgba="0.2 0.2 0.9 0.8"
            contype="1" conaffinity="1" mass="2.0"/>
    </body>
  </worldbody>
  <equality>
    <weld body1="ee_link" body2="mocap_target"
          solref="0.002 1" solimp="0.99 0.999 0.0001"/>
  </equality>
</mujoco>
"""


def create_default_base_mjcf(path: str) -> None:
    """Write the default base MJCF (table + 3-DOF stick arm + mocap EE + weld) to *path*."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_BASE_MJCF_TEMPLATE)


def create_so100_base_mjcf(path: str) -> None:
    """Write a base MJCF that embeds the SO-100 arm on the table edge.

    The SO-100 ``Base`` body is placed at the negative-x edge of the table
    (x=-0.30) and rotated so the arm faces the table surface.  A mocap body
    ``mocap_target`` is welded to the ``ee_frame`` body inside the hand so that
    the existing mocap-based EE control still works unchanged.
    """
    meshdir = str(_ASSETS_DIR / "trs_so_arm100")
    template = f"""\
<mujoco model="pushing_so100">
  <compiler angle="radian" meshdir="{meshdir}" inertiafromgeom="false"/>
  <option timestep="0.005" gravity="0 0 -9.81" iterations="50" solver="Newton"
          cone="elliptic" impratio="10"/>
  <default>
    <geom condim="3" friction="0.8 0.02 0.001" contype="1" conaffinity="1"/>
    <default class="so_arm100">
      <joint frictionloss="0.1" armature="0.1"/>
      <position kp="50" dampratio="1" forcerange="-3.5 3.5"/>
      <default class="Rotation">
        <joint axis="0 1 0" range="-1.92 1.92"/>
      </default>
      <default class="Pitch">
        <joint axis="1 0 0" range="-3.32 0.174"/>
      </default>
      <default class="Elbow">
        <joint axis="1 0 0" range="-0.174 3.14"/>
      </default>
      <default class="Wrist_Pitch">
        <joint axis="1 0 0" range="-1.66 1.66"/>
      </default>
      <default class="Wrist_Roll">
        <joint axis="0 1 0" range="-2.79 2.79"/>
      </default>
      <default class="Jaw">
        <joint axis="0 0 1" range="-0.174 1.75"/>
      </default>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" density="0" group="2" material="white"/>
        <default class="motor_visual">
          <geom material="black"/>
        </default>
      </default>
      <default class="collision">
        <geom group="3" type="mesh" material="white"/>
        <default class="finger_collision">
          <geom type="box" solimp="2 1 0.01" solref="0.01 1" friction="1 0.005 0.0001"/>
        </default>
      </default>
    </default>
  </default>
  <asset>
    <material name="white" rgba="1 1 1 1"/>
    <material name="black" rgba="0.1 0.1 0.1 1"/>
    <mesh name="Base"                    file="Base.stl"/>
    <mesh name="Base_Motor"              file="Base_Motor.stl"/>
    <mesh name="Rotation_Pitch"          file="Rotation_Pitch.stl"/>
    <mesh name="Rotation_Pitch_Motor"    file="Rotation_Pitch_Motor.stl"/>
    <mesh name="Upper_Arm"               file="Upper_Arm.stl"/>
    <mesh name="Upper_Arm_Motor"         file="Upper_Arm_Motor.stl"/>
    <mesh name="Lower_Arm"               file="Lower_Arm.stl"/>
    <mesh name="Lower_Arm_Motor"         file="Lower_Arm_Motor.stl"/>
    <mesh name="Wrist_Pitch_Roll"        file="Wrist_Pitch_Roll.stl"/>
    <mesh name="Wrist_Pitch_Roll_Motor"  file="Wrist_Pitch_Roll_Motor.stl"/>
    <mesh name="Fixed_Jaw"               file="Fixed_Jaw.stl"/>
    <mesh name="Fixed_Jaw_Motor"         file="Fixed_Jaw_Motor.stl"/>
    <mesh name="Fixed_Jaw_Collision_1"   file="Fixed_Jaw_Collision_1.stl"/>
    <mesh name="Fixed_Jaw_Collision_2"   file="Fixed_Jaw_Collision_2.stl"/>
    <mesh name="Moving_Jaw"              file="Moving_Jaw.stl"/>
    <mesh name="Moving_Jaw_Collision_1"  file="Moving_Jaw_Collision_1.stl"/>
    <mesh name="Moving_Jaw_Collision_2"  file="Moving_Jaw_Collision_2.stl"/>
    <mesh name="Moving_Jaw_Collision_3"  file="Moving_Jaw_Collision_3.stl"/>
  </asset>
  <worldbody>
    <!-- Ground plane -->
    <geom name="ground" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.3 1"
          pos="0 0 0" contype="1" conaffinity="1"/>
    <!-- Table -->
    <body name="table" pos="0 0 0">
      <geom name="table_top" type="box" size="0.30 0.25 0.02"
            pos="0 0 0.02" rgba="0.6 0.5 0.4 1"
            contype="1" conaffinity="1" friction="0.6 0.02 0.001"/>
    </body>
    <!-- SO-100 arm: base at table edge, euler rotates it to face +x (table surface) -->
    <body name="Base" childclass="so_arm100" pos="-0.30 0 0.04" euler="0 0 0">
      <geom type="mesh" mesh="Base" class="visual"/>
      <geom type="mesh" mesh="Base_Motor" class="motor_visual"/>
      <geom type="mesh" mesh="Base" class="collision"/>
      <body name="Rotation_Pitch" pos="0 -0.0452 0.0165" quat="0.707105 0.707108 0 0">
        <inertial pos="-9.07886e-05 0.0590972 0.031089" quat="0.363978 0.441169 -0.623108 0.533504"
          mass="0.119226" diaginertia="5.94278e-05 5.89975e-05 3.13712e-05"/>
        <joint name="Rotation" class="Rotation"/>
        <geom type="mesh" mesh="Rotation_Pitch" class="visual"/>
        <geom type="mesh" mesh="Rotation_Pitch_Motor" class="motor_visual"/>
        <geom type="mesh" mesh="Rotation_Pitch" class="collision"/>
        <body name="Upper_Arm" pos="0 0.1025 0.0306" euler="1.57079 0 0">
          <inertial pos="-1.72052e-05 0.0701802 0.00310545"
            quat="0.50104 0.498994 -0.493562 0.50632" mass="0.162409"
            diaginertia="0.000213312 0.000167164 7.01522e-05"/>
          <joint name="Pitch" class="Pitch"/>
          <geom type="mesh" mesh="Upper_Arm" class="visual"/>
          <geom type="mesh" mesh="Upper_Arm_Motor" class="motor_visual"/>
          <geom type="mesh" mesh="Upper_Arm" class="collision"/>
          <body name="Lower_Arm" pos="0 0.11257 0.028" euler="-1.57079 0 0">
            <inertial pos="-0.00339604 0.00137796 0.0768007"
              quat="0.701995 0.0787996 0.0645626 0.704859"
              mass="0.147968" diaginertia="0.000138803 0.000107748 4.84242e-05"/>
            <joint name="Elbow" class="Elbow"/>
            <geom type="mesh" mesh="Lower_Arm" class="visual"/>
            <geom type="mesh" mesh="Lower_Arm_Motor" class="motor_visual"/>
            <geom type="mesh" mesh="Lower_Arm" class="collision"/>
            <body name="Wrist_Pitch_Roll" pos="0 0.0052 0.1349" euler="-1.57079 0 0">
              <inertial pos="-0.00852653 -0.0352279 -2.34622e-05"
                quat="-0.0522806 0.705235 0.0549524 0.704905"
                mass="0.0661321" diaginertia="3.45403e-05 2.39041e-05 1.94704e-05"/>
              <joint name="Wrist_Pitch" class="Wrist_Pitch"/>
              <geom type="mesh" mesh="Wrist_Pitch_Roll" class="visual"/>
              <geom type="mesh" mesh="Wrist_Pitch_Roll_Motor" class="motor_visual"/>
              <geom type="mesh" mesh="Wrist_Pitch_Roll" class="collision"/>
              <body name="Fixed_Jaw" pos="0 -0.0601 0" euler="0 1.57079 0">
                <inertial pos="0.00552377 -0.0280167 0.000483583"
                  quat="0.41836 0.620891 -0.350644 0.562599"
                  mass="0.0929859" diaginertia="5.03136e-05 4.64098e-05 2.72961e-05"/>
                <joint name="Wrist_Roll" class="Wrist_Roll"/>
                <geom type="mesh" mesh="Fixed_Jaw" class="visual"/>
                <geom type="mesh" mesh="Fixed_Jaw_Motor" class="motor_visual"/>
                <geom type="mesh" mesh="Fixed_Jaw_Collision_1" class="collision"/>
                <geom type="mesh" mesh="Fixed_Jaw_Collision_2" class="collision"/>
                <geom class="finger_collision" name="fixed_jaw_pad_1" size="0.001 0.005 0.004" pos="0.0089 -0.1014 0"/>
                <geom class="finger_collision" name="fixed_jaw_pad_2" size="0.001 0.005 0.006" pos="0.0109 -0.0914 0"/>
                <geom class="finger_collision" name="fixed_jaw_pad_3" size="0.001 0.01 0.007"  pos="0.0126 -0.0768 0"/>
                <geom class="finger_collision" name="fixed_jaw_pad_4" size="0.001 0.01 0.008"  pos="0.0143 -0.0572 0"/>
                <body name="vx300s_left/camera_focus" pos="0.0 -0.06 0">
                  <site pos="0 0 0" size="0.01" type="sphere" name="left_cam_focus" rgba="0 0 1 0"/>
                  <body name="ee_frame" pos="0 0 0" quat="0.5 0.5 -0.5 -0.5">
                    <site name="ee_site" pos="0 0 0" size="0.01" type="sphere" rgba="1 0 0 0.5"/>
                  </body>
                </body>
                <body name="Moving_Jaw" pos="-0.0202 -0.0244 0"
                  quat="1.34924e-11 -3.67321e-06 1 -3.67321e-06">
                  <inertial pos="-0.00161745 -0.0303473 0.000449646"
                    quat="0.696562 0.716737 -0.0239844 -0.0227026"
                    mass="0.0202444" diaginertia="1.11265e-05 8.99651e-06 2.99548e-06"/>
                  <joint name="Jaw" class="Jaw"/>
                  <geom type="mesh" mesh="Moving_Jaw" class="visual"/>
                  <geom type="mesh" mesh="Moving_Jaw_Collision_1" class="collision"/>
                  <geom type="mesh" mesh="Moving_Jaw_Collision_2" class="collision"/>
                  <geom type="mesh" mesh="Moving_Jaw_Collision_3" class="collision"/>
                  <geom class="finger_collision" name="moving_jaw_pad_1" size="0.001 0.005 0.004" pos="-0.0113 -0.077 0"/>
                  <geom class="finger_collision" name="moving_jaw_pad_2" size="0.001 0.005 0.006" pos="-0.0093 -0.067 0"/>
                  <geom class="finger_collision" name="moving_jaw_pad_3" size="0.001 0.01 0.006"  pos="-0.0073 -0.055 0"/>
                  <geom class="finger_collision" name="moving_jaw_pad_4" size="0.001 0.01 0.008"  pos="-0.0073 -0.035 0"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <!-- Mocap body welded to ee_frame for position-delta control -->
    <body name="mocap_target" mocap="true" pos="-0.10 0 0.08">
      <geom type="sphere" size="0.015" rgba="1 0 0 0.4"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <equality>
    <weld body1="ee_frame" body2="mocap_target" solref="0.01 1" solimp="0.9 0.95 0.001"/>
  </equality>
  <actuator>
    <position class="Rotation"    name="Rotation"    joint="Rotation"    inheritrange="1"/>
    <position class="Pitch"       name="Pitch"       joint="Pitch"       inheritrange="1"/>
    <position class="Elbow"       name="Elbow"       joint="Elbow"       inheritrange="1"/>
    <position class="Wrist_Pitch" name="Wrist_Pitch" joint="Wrist_Pitch" inheritrange="1"/>
    <position class="Wrist_Roll"  name="Wrist_Roll"  joint="Wrist_Roll"  inheritrange="1"/>
    <position class="Jaw"         name="Jaw"         joint="Jaw"         inheritrange="1"/>
  </actuator>
  <contact>
    <exclude body1="Base" body2="Rotation_Pitch"/>
  </contact>
</mujoco>
"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(template)


# ---------------------------------------------------------------------------
# Helper: build an XML body element for a primitive ObjectConfig
# ---------------------------------------------------------------------------

def _object_geom_element(cfg: ObjectConfig, name: str) -> etree._Element:
    """Return an lxml ``<geom>`` element for the given object config."""
    geom = etree.Element("geom")
    geom.set("name", f"{name}_geom")
    rgba = " ".join(f"{v:.3f}" for v in cfg.color)
    geom.set("rgba", rgba)
    geom.set("mass", str(cfg.mass))
    geom.set("friction", f"{cfg.friction:.3f} 0.02 0.001")

    shape = cfg.shape
    if shape == "box":
        geom.set("type", "box")
        s = cfg.size
        geom.set("size", f"{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}")
    elif shape == "cylinder":
        geom.set("type", "cylinder")
        geom.set("size", f"{cfg.size[0]:.4f} {cfg.size[1]:.4f}")
    elif shape == "sphere":
        geom.set("type", "sphere")
        geom.set("size", f"{cfg.size[0]:.4f}")
    elif shape == "capsule":
        geom.set("type", "capsule")
        geom.set("size", f"{cfg.size[0]:.4f} {cfg.size[1]:.4f}")
    else:
        # Fallback: small box
        geom.set("type", "box")
        geom.set("size", "0.03 0.03 0.03")

    return geom


def _make_object_body(
    cfg: ObjectConfig,
    name: str,
    table_top_z: float,
    colliding: bool,
) -> etree._Element:
    """Build a ``<body>`` element placed on the table surface."""
    body = etree.Element("body")
    body.set("name", name)

    # Position: x,y from config, z = table surface + half-height of object
    half_z = _half_z_for_object(cfg)
    pos_z = table_top_z + half_z
    yaw = cfg.orientation
    # MuJoCo euler is intrinsic XYZ
    body.set("pos", f"{cfg.pos[0]:.4f} {cfg.pos[1]:.4f} {pos_z:.4f}")
    body.set("euler", f"0 0 {yaw:.4f}")

    # Free joint allows the body to move under contact forces
    free_joint = etree.SubElement(body, "joint")
    free_joint.set("type", "free")

    geom = _object_geom_element(cfg, name)
    if not colliding:
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    body.append(geom)
    return body


def _half_z_for_object(cfg: ObjectConfig) -> float:
    """Estimate half-height of the object for placement."""
    shape = cfg.shape
    s = cfg.size
    if shape == "box":
        return s[2] if len(s) > 2 else s[0]
    elif shape == "cylinder":
        return s[1] if len(s) > 1 else s[0]
    elif shape == "sphere":
        return s[0]
    elif shape == "capsule":
        return (s[1] + s[0]) if len(s) > 1 else s[0]
    return 0.03


# ---------------------------------------------------------------------------
# Camera: convert look-at to MuJoCo quat
# ---------------------------------------------------------------------------

def _lookat_to_euler(
        pos: list[float], look_at: list[float]) -> tuple[str, str]:
    """Return (pos_str, euler_str) for a camera pointing from pos toward look_at."""
    px, py, pz = pos
    lx, ly, lz = look_at
    # Direction vector (forward = -z in MuJoCo camera frame)
    fwd = np.array([lx - px, ly - py, lz - pz], dtype=float)
    dist = np.linalg.norm(fwd)
    if dist < 1e-6:
        fwd = np.array([0.0, 0.0, -1.0])
    else:
        fwd /= dist

    # Pitch (rotation around x)
    pitch = math.atan2(-fwd[2], math.hypot(fwd[0], fwd[1]))
    # Yaw (rotation around z)
    yaw = math.atan2(fwd[1], fwd[0]) + math.pi / 2

    pos_str = f"{px:.4f} {py:.4f} {pz:.4f}"
    euler_str = f"{pitch:.4f} 0 {yaw:.4f}"
    return pos_str, euler_str


# ---------------------------------------------------------------------------
# SceneBuilder
# ---------------------------------------------------------------------------

class SceneBuilder:
    """Injects objects, camera, and lighting into a base MJCF and compiles it.

    Parameters
    ----------
    base_mjcf_path:
        Path to a pre-written base MJCF file.  When *None* the builder writes a
        fresh temporary file based on ``config.robot_type`` at build time.
    """

    def __init__(self, base_mjcf_path: str | None = None) -> None:
        self.base_mjcf_path = base_mjcf_path

    # ------------------------------------------------------------------
    # Internal: select / create the correct base MJCF for a config
    # ------------------------------------------------------------------

    def _get_base_path(self, config: SceneConfig) -> str:
        if self.base_mjcf_path is not None:
            return self.base_mjcf_path
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
        tmp.close()
        if config.robot_type == "so100":
            create_so100_base_mjcf(tmp.name)
        else:
            create_default_base_mjcf(tmp.name)
        return tmp.name

    def build(
        self,
        config: SceneConfig,
        render_size: tuple[int, int] | None = None,
    ) -> mujoco.MjModel:
        """Parse base MJCF, inject scene elements, compile and return MjModel.

        Parameters
        ----------
        render_size:
            (height, width) of the offscreen framebuffer.  When provided, a
            ``<visual><global offheight=... offwidth=.../></visual>`` element is
            injected so MuJoCo allocates a large enough framebuffer.
        """
        base_path = self._get_base_path(config)
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(base_path, parser)
        root = tree.getroot()

        # 0. Set offscreen framebuffer size if requested
        if render_size is not None:
            h, w = render_size
            visual = root.find("visual")
            if visual is None:
                visual = etree.SubElement(root, "visual")
            global_elem = visual.find("global")
            if global_elem is None:
                global_elem = etree.SubElement(visual, "global")
            global_elem.set("offheight", str(h))
            global_elem.set("offwidth", str(w))

        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("Base MJCF has no <worldbody> element")

        # Table top z = pos_z + half_z = 0.02 + 0.02 = 0.04
        table_top_z = 0.04

        # 1. Update table color and friction
        table_geom = worldbody.find(".//geom[@name='table_top']")
        if table_geom is not None:
            rgba = " ".join(f"{v:.3f}" for v in config.table_color)
            table_geom.set("rgba", rgba)
            hz = config.table_size[2]
            hx, hy = config.table_size[0], config.table_size[1]
            table_geom.set("size", f"{hx:.3f} {hy:.3f} {hz:.3f}")
            table_geom.set(
                "friction", f"{
                    config.table_friction:.3f} 0.02 0.001")

        # 2. Inject target body
        target_body = _make_object_body(
            config.target, "target_object", table_top_z, colliding=True
        )
        worldbody.append(target_body)

        # 3. Inject obstacle bodies (colliding)
        for i, obs_cfg in enumerate(config.obstacles):
            obs_body = _make_object_body(
                obs_cfg, f"obstacle_{i}", table_top_z, colliding=True)
            worldbody.append(obs_body)

        # 4. Inject clutter bodies (visual only)
        for i, cl_cfg in enumerate(config.clutter):
            cl_body = _make_object_body(
                cl_cfg, f"clutter_{i}", table_top_z, colliding=False)
            worldbody.append(cl_body)

        # 5. Set camera
        cam_elem = root.find(".//camera[@name='main_camera']")
        if cam_elem is None:
            cam_elem = etree.SubElement(worldbody, "camera")
            cam_elem.set("name", "main_camera")
        pos_str, euler_str = _lookat_to_euler(
            config.camera.pos, config.camera.look_at)
        cam_elem.set("pos", pos_str)
        cam_elem.set("euler", euler_str)
        cam_elem.set("fovy", f"{config.camera.fov:.1f}")

        # 6. Set lighting via <light> element
        # Remove existing non-headlight lights
        for lt in root.findall(".//light"):
            if lt.get("name", "") == "scene_light":
                lt.getparent().remove(lt)  # type: ignore[union-attr]

        light = etree.SubElement(worldbody, "light")
        light.set("name", "scene_light")
        d = config.lighting.direction
        light.set("dir", f"{d[0]:.3f} {d[1]:.3f} {d[2]:.3f}")
        iv = config.lighting.intensity
        light.set("diffuse", f"{iv:.3f} {iv:.3f} {iv:.3f}")
        av = config.lighting.ambient
        light.set("ambient", f"{av:.3f} {av:.3f} {av:.3f}")
        light.set("directional", "true")
        light.set("castshadow", "false")

        # 7. Compile
        xml_str = etree.tostring(root, pretty_print=True, encoding="unicode")
        model = mujoco.MjModel.from_xml_string(xml_str)
        return model
