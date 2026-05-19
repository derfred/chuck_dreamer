"""Builds a compiled mujoco.MjModel from a SceneConfig via XML manipulation."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco  # type: ignore[import-untyped]
import numpy as np
from lxml import etree  # type: ignore[import-untyped]

from .scene_config import ObjectConfig, SceneConfig

_ASSETS_DIR = Path(__file__).parent.parent.parent.parent / "assets" / "mujoco"
_BASE_SCENE_XML = _ASSETS_DIR / "base_scene.xml"
_SO101_ARM_XML = _ASSETS_DIR / "so101_arm.xml"
_SO101_MESHDIR = _ASSETS_DIR / "trs_so_arm100"

# Name of the inline square-based pyramid mesh registered in the <asset> section.
_PYRAMID_MESH_NAME = "unit_pyramid"

# Unit pyramid (square base of side 2 centred at origin in xy, apex at z=1).
# We scale it via the mesh-asset ``scale`` attribute for each instance.
_PYRAMID_VERTICES = "-1 -1 0  1 -1 0  1 1 0  -1 1 0  0 0 1"

# External STL meshes registered on demand. Each entry maps a short tag (as
# stored in ObjectConfig.mesh_path) to its absolute path and the native
# bounding-cube half-extent so per-instance scaling matches the requested
# size in metres.
_MESH_ASSETS: dict[str, dict[str, object]] = {
  "rhombicuboctahedron": {
    "path": _ASSETS_DIR / "rhombicuboctahedron.stl",
    "native_half_extent": 0.029,
  },
}

# z-coordinate of the table geom centre in the base MJCF (see base_scene.xml).
# Duplicated from scene_generator.py to keep this module standalone.
_TABLE_GEOM_CENTRE_Z = 0.02

# Half-height of the flat disc rendered to mark the goal. Tiny so it reads as a
# decal on the table from any reasonable camera angle.
_GOAL_DISC_HALF_HEIGHT = 0.0005


# ---------------------------------------------------------------------------
# Helper: build an XML body element for a primitive ObjectConfig
# ---------------------------------------------------------------------------

def _object_geom_element(
    cfg: ObjectConfig,
    name: str,
    pyramid_mesh_name: str | None = None,
    mesh_asset_name: str | None = None,
) -> etree._Element:
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
    elif shape == "pyramid":
        if pyramid_mesh_name is None:
            raise ValueError("pyramid shape requires a registered mesh name")
        geom.set("type", "mesh")
        geom.set("mesh", pyramid_mesh_name)
    elif shape == "mesh":
        if mesh_asset_name is None:
            raise ValueError("mesh shape requires a registered mesh-asset name")
        geom.set("type", "mesh")
        geom.set("mesh", mesh_asset_name)
    else:
        geom.set("type", "box")
        geom.set("size", "0.03 0.03 0.03")

    return geom


def _make_object_body(
    cfg: ObjectConfig,
    name: str,
    colliding: bool,
    pyramid_mesh_name: str | None = None,
    mesh_asset_name: str | None = None,
) -> etree._Element:
    """Build a ``<body>`` element at the config's world position."""
    body = etree.Element("body")
    body.set("name", name)

    yaw = cfg.orientation
    body.set("pos", f"{cfg.pos[0]:.4f} {cfg.pos[1]:.4f} {cfg.pos[2]:.4f}")
    body.set("euler", f"0 0 {yaw:.4f}")

    if colliding:
        free_joint = etree.SubElement(body, "joint")
        free_joint.set("type", "free")

        inertial = etree.SubElement(body, "inertial")
        inertial.set("pos", "0 0 0")
        inertial.set("mass", f"{cfg.mass:.6f}")
        r = cfg.size[0] if cfg.size else 0.03
        ival = max(2.0 / 5.0 * cfg.mass * r * r, 1e-6)
        inertial.set("diaginertia", f"{ival:.6e} {ival:.6e} {ival:.6e}")

    geom = _object_geom_element(
        cfg, name,
        pyramid_mesh_name=pyramid_mesh_name,
        mesh_asset_name=mesh_asset_name,
    )
    if not colliding:
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    body.append(geom)
    return body


def _register_stl_mesh(
    root: etree._Element,
    asset_name: str,
    stl_path: Path,
    scale: float,
) -> None:
    """Add an ``<asset><mesh file="..." scale="s s s"/>`` entry for an STL.

    A fresh asset is registered per instance so each object can carry its own
    uniform scale (MuJoCo bakes the scale into the asset, not the geom).
    """
    asset = root.find("asset")
    if asset is None:
        asset = etree.SubElement(root, "asset")
    mesh = etree.SubElement(asset, "mesh")
    mesh.set("name", asset_name)
    mesh.set("file", str(stl_path))
    mesh.set("scale", f"{scale:.6f} {scale:.6f} {scale:.6f}")


def _register_pyramid_mesh(
    root: etree._Element,
    name: str,
    half_base: float,
    height: float,
) -> None:
    """Add an inline square-based pyramid mesh to the <asset> section.

    The mesh is built from a unit pyramid (base half-extent 1, height 1) scaled
    per-instance so that we can vary pyramid dimensions without rebuilding the
    geometry definition.
    """
    asset = root.find("asset")
    if asset is None:
        asset = etree.SubElement(root, "asset")
    mesh = etree.SubElement(asset, "mesh")
    mesh.set("name", name)
    mesh.set("vertex", _PYRAMID_VERTICES)
    mesh.set("scale", f"{half_base:.4f} {half_base:.4f} {height:.4f}")


# ---------------------------------------------------------------------------
# Camera: convert look-at to MuJoCo euler
# ---------------------------------------------------------------------------

def _lookat_to_euler(
        pos: list[float], look_at: list[float]) -> tuple[str, str]:
    """Return (pos_str, euler_str) for a camera pointing from pos toward look_at."""
    px, py, pz = pos
    lx, ly, lz = look_at
    fwd = np.array([lx - px, ly - py, lz - pz], dtype=float)
    dist = np.linalg.norm(fwd)
    if dist < 1e-6:
        fwd = np.array([0.0, 0.0, -1.0])
    else:
        fwd /= dist

    pitch = math.atan2(-fwd[2], math.hypot(fwd[0], fwd[1]))
    yaw = math.atan2(fwd[1], fwd[0]) + math.pi / 2

    pos_str = f"{px:.4f} {py:.4f} {pz:.4f}"
    euler_str = f"{pitch:.4f} 0 {yaw:.4f}"
    return pos_str, euler_str


# ---------------------------------------------------------------------------
# Base MJCF assembly
# ---------------------------------------------------------------------------

def _inject_arm_fragment(root: etree._Element, arm_root: etree._Element) -> None:
    """Merge all sections from an arm fragment XML into the base scene root."""
    # compiler: copy extra attributes (e.g. cone, impratio, inertiafromgeom)
    arm_compiler = arm_root.find("compiler")
    if arm_compiler is not None:
        base_compiler = root.find("compiler")
        if base_compiler is None:
            base_compiler = etree.SubElement(root, "compiler")
        for k, v in arm_compiler.attrib.items():
            base_compiler.set(k, v)

    # default: append arm <default> children into base <default>
    arm_default = arm_root.find("default")
    if arm_default is not None:
        base_default = root.find("default")
        if base_default is None:
            base_default = etree.SubElement(root, "default")
        for child in arm_default:
            base_default.append(child)

    # asset: append all children into base <asset> (create if missing)
    arm_asset = arm_root.find("asset")
    if arm_asset is not None:
        base_asset = root.find("asset")
        if base_asset is None:
            base_asset = etree.SubElement(root, "asset")
        for child in arm_asset:
            base_asset.append(child)

    # worldbody: append arm worldbody children into base worldbody
    arm_worldbody = arm_root.find("worldbody")
    if arm_worldbody is not None:
        base_worldbody = root.find("worldbody")
        if base_worldbody is None:
            base_worldbody = etree.SubElement(root, "worldbody")
        for child in arm_worldbody:
            base_worldbody.append(child)

    # option: merge extra attributes (e.g. cone, impratio)
    arm_option = arm_root.find("option")
    if arm_option is not None:
        base_option = root.find("option")
        if base_option is None:
            base_option = etree.SubElement(root, "option")
        for k, v in arm_option.attrib.items():
            base_option.set(k, v)

    # top-level sections: actuator, equality, contact
    for tag in ("actuator", "equality", "contact"):
        arm_elem = arm_root.find(tag)
        if arm_elem is not None:
            root.append(arm_elem)


def _load_base_xml(config: SceneConfig) -> etree._Element:
    """Load base scene XML and inject the appropriate arm fragment."""
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.parse(str(_BASE_SCENE_XML), parser).getroot()

    if config.robot_type != "so100":
        raise ValueError(f"Unsupported robot_type '{config.robot_type}'; only 'so100' is supported")
    arm_root = etree.parse(str(_SO101_ARM_XML), parser).getroot()
    _inject_arm_fragment(root, arm_root)
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str(_SO101_MESHDIR))

    return root


# ---------------------------------------------------------------------------
# SceneBuilder
# ---------------------------------------------------------------------------

class SceneBuilder:
    """Injects objects, camera, and lighting into a base MJCF and compiles it."""

    def build(
        self,
        config: SceneConfig,
        render_size: tuple[int, int] | None = None,
    ) -> mujoco.MjModel:
        """Parse base MJCF, inject scene elements, compile and return MjModel."""
        root = _load_base_xml(config)

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

        table_geom = worldbody.find(".//geom[@name='table_top']")
        if table_geom is not None:
            rgba = " ".join(f"{v:.3f}" for v in config.table_color)
            table_geom.set("rgba", rgba)
            hz = config.table_size[2]
            hx, hy = config.table_size[0], config.table_size[1]
            table_geom.set("size", f"{hx:.3f} {hy:.3f} {hz:.3f}")
            table_geom.set("friction", f"{config.table_friction:.3f} 0.02 0.001")

        def _maybe_register_pyramid(cfg: ObjectConfig, name: str) -> str | None:
            if cfg.shape != "pyramid":
                return None
            mesh_name = f"{name}_pyramid_mesh"
            half_base = cfg.size[0] if len(cfg.size) > 0 else 0.03
            height    = cfg.size[1] if len(cfg.size) > 1 else 0.06
            _register_pyramid_mesh(root, mesh_name, half_base, height)
            return mesh_name

        def _maybe_register_mesh(cfg: ObjectConfig, name: str) -> str | None:
            if cfg.shape != "mesh":
                return None
            tag = cfg.mesh_path
            if tag is None or tag not in _MESH_ASSETS:
                raise ValueError(
                    f"Unknown mesh tag '{tag}'. Known: {list(_MESH_ASSETS)}"
                )
            spec = _MESH_ASSETS[tag]
            requested_half = cfg.size[0] if cfg.size else 0.03
            scale = float(requested_half) / float(spec["native_half_extent"])
            asset_name = f"{name}_{tag}_mesh"
            _register_stl_mesh(root, asset_name, spec["path"], scale)  # type: ignore[arg-type]
            return asset_name

        target_pyramid = _maybe_register_pyramid(config.target, "target_object")
        target_mesh    = _maybe_register_mesh(config.target, "target_object")
        target_body = _make_object_body(
            config.target, "target_object", colliding=True,
            pyramid_mesh_name=target_pyramid,
            mesh_asset_name=target_mesh,
        )
        worldbody.append(target_body)

        for i, obs_cfg in enumerate(config.obstacles):
            name = f"obstacle_{i}"
            pyramid_name = _maybe_register_pyramid(obs_cfg, name)
            mesh_name    = _maybe_register_mesh(obs_cfg, name)
            obs_body = _make_object_body(
                obs_cfg, name, colliding=True,
                pyramid_mesh_name=pyramid_name,
                mesh_asset_name=mesh_name,
            )
            worldbody.append(obs_body)

        for i, cl_cfg in enumerate(config.clutter):
            name = f"clutter_{i}"
            pyramid_name = _maybe_register_pyramid(cl_cfg, name)
            mesh_name    = _maybe_register_mesh(cl_cfg, name)
            cl_body = _make_object_body(
                cl_cfg, name, colliding=False,
                pyramid_mesh_name=pyramid_name,
                mesh_asset_name=mesh_name,
            )
            worldbody.append(cl_body)

        # Goal disc: a thin non-colliding cylinder lying flush on the table top.
        # Uses ``goal_tolerance`` as its radius so the rendered marker matches the
        # success threshold the reward function uses.
        table_top_z = _TABLE_GEOM_CENTRE_Z + config.table_size[2]
        goal_geom = etree.SubElement(worldbody, "geom")
        goal_geom.set("name", "goal_marker")
        goal_geom.set("type", "cylinder")
        goal_geom.set(
            "size", f"{config.goal_tolerance:.4f} {_GOAL_DISC_HALF_HEIGHT:.5f}"
        )
        goal_geom.set(
            "pos",
            f"{config.goal_pos[0]:.4f} {config.goal_pos[1]:.4f} "
            f"{table_top_z + _GOAL_DISC_HALF_HEIGHT:.5f}",
        )
        goal_geom.set("rgba", "0.95 0.95 0.95 1")
        goal_geom.set("contype", "0")
        goal_geom.set("conaffinity", "0")

        cam_elem = root.find(".//camera[@name='main_camera']")
        if cam_elem is None:
            cam_elem = etree.SubElement(worldbody, "camera")
            cam_elem.set("name", "main_camera")
        pos_str, euler_str = _lookat_to_euler(
            config.camera.pos, config.camera.look_at)
        cam_elem.set("pos", pos_str)
        cam_elem.set("euler", euler_str)
        cam_elem.set("fovy", f"{config.camera.fov:.1f}")

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

        xml_str = etree.tostring(root, pretty_print=True, encoding="unicode")
        model   = mujoco.MjModel.from_xml_string(xml_str)
        return model
