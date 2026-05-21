"""Ground-truth forward kinematics for the SO-101 arm via MuJoCo.

Single source of truth: load the SO-101 model, set qpos for the 5 positioning
joints, call mj_kinematics, read the EE site's world position. No world-frame
knowledge, no fiducials, no episodes — just q -> p_arm.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

# XML joint names in the canonical (real-robot) order.
_XML_JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]

# Canonical real-robot joint names. The public `q` argument is in this order.
CANONICAL_JOINT_NAMES = [
  "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
]

# Nominal home pose. Matches SceneConfig.default_qpos() in
# src/chuck_dreamer/sim/scene_config.py (SO-100 default), minus the Jaw entry.
Q_HOME = np.array([0.0, -3.14, 3.14, 0.0, 0.0], dtype=np.float64)


def _load_model(model_path: str | Path) -> mujoco.MjModel:
  """Load an SO-101 MJCF, injecting meshdir if the fragment lacks one."""
  model_path = Path(model_path)
  tree = ET.parse(model_path)
  root = tree.getroot()
  compiler = root.find("compiler")
  if compiler is None:
    compiler = ET.Element("compiler")
    root.insert(0, compiler)
  if not compiler.get("meshdir") or "PLACEHOLDER" in compiler.get("meshdir", ""):
    compiler.set("meshdir", str(model_path.parent / "trs_so_arm100"))
  xml_str = ET.tostring(root, encoding="unicode")
  return mujoco.MjModel.from_xml_string(xml_str)


class FK:
  """Stateful FK evaluator. Not thread-safe; that is acceptable."""

  def __init__(self, model_path: str | Path, ee_site_name: str = "ee_tip"):
    self.model = _load_model(model_path)
    self.data = mujoco.MjData(self.model)
    self.ee_site_name = ee_site_name
    self.ee_site_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
    if self.ee_site_id < 0:
      raise ValueError(f"site '{ee_site_name}' not found in model")
    self.qpos_idx = np.array([
      self.model.jnt_qposadr[
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
      for n in _XML_JOINT_NAMES
    ])
    if np.any(self.qpos_idx < 0):
      missing = [n for n, i in zip(_XML_JOINT_NAMES, self.qpos_idx) if i < 0]
      raise ValueError(f"joints not found in model: {missing}")

  def __call__(self, q, dq=None) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (5,):
      raise ValueError(f"q must be shape (5,), got {q.shape}")
    if dq is None:
      dq = np.zeros(5)
    else:
      dq = np.asarray(dq, dtype=np.float64)
      if dq.shape != (5,):
        raise ValueError(f"dq must be shape (5,), got {dq.shape}")
    self.data.qpos[self.qpos_idx] = q + dq
    mujoco.mj_kinematics(self.model, self.data)
    return self.data.site_xpos[self.ee_site_id].copy()


# ---------------------------------------------------------------------------
# Verify CLI
# ---------------------------------------------------------------------------

def _close_jaw(fk: "FK") -> None:
  """Set the Jaw qpos to its lower limit (closed)."""
  jaw_id = mujoco.mj_name2id(fk.model, mujoco.mjtObj.mjOBJ_JOINT, "Jaw")
  if jaw_id < 0:
    return
  adr = fk.model.jnt_qposadr[jaw_id]
  fk.data.qpos[adr] = fk.model.jnt_range[jaw_id, 0]


def _verify(model_path: str, out_png: str, delta: float) -> None:
  fk = FK(model_path)

  # Check 1: render closed-gripper pose with the red ee_tip site visible.
  fk(Q_HOME)
  _close_jaw(fk)
  mujoco.mj_kinematics(fk.model, fk.data)

  renderer = mujoco.Renderer(fk.model, height=480, width=640)
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(fk.model, cam)
  # Frame the gripper, not the whole arm — the EE-site dot is small.
  ee_pos = fk.data.site_xpos[fk.ee_site_id]
  cam.lookat[:] = ee_pos
  cam.distance = 0.15
  cam.azimuth = 90.0
  cam.elevation = -10.0
  renderer.update_scene(fk.data, camera=cam)
  pixels = renderer.render()
  out_path = Path(out_png)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  try:
    from PIL import Image
    Image.fromarray(pixels).save(out_path)
  except ImportError:
    # Fallback: write a raw PPM that any image viewer can open.
    h, w, _ = pixels.shape
    with open(out_path.with_suffix(".ppm"), "wb") as f:
      f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
      f.write(pixels.tobytes())
    out_path = out_path.with_suffix(".ppm")
  print(f"[check 1] EE site render saved to: {out_path.resolve()}")
  print("           inspect by eye: the red dot should sit at the closed-gripper tip.")
  print()

  # Check 2: joint sign sanity.
  p0 = fk(Q_HOME)
  print("[check 2] joint-sign sanity (delta = "
        f"{delta:+.3f} rad on each joint, others at Q_HOME):")
  print(f"           {'joint':>15s}  {'disp (m)':<30s}  dominant")
  for i, name in enumerate(CANONICAL_JOINT_NAMES):
    q1 = Q_HOME.copy()
    q1[i] += delta
    p1 = fk(q1)
    disp = p1 - p0
    axis_i = int(np.argmax(np.abs(disp)))
    axis = ["x", "y", "z"][axis_i]
    sign = "+" if disp[axis_i] >= 0 else "-"
    disp_str = f"[{disp[0]:+.4f} {disp[1]:+.4f} {disp[2]:+.4f}]"
    print(f"           {name:>15s}  {disp_str:<30s}  {sign}{axis}")
  print()

  # Check 3: home pose sanity.
  p_home = fk(Q_HOME)
  print("[check 3] home pose:")
  print(f"           q_home = {Q_HOME.tolist()}")
  print(f"           p_home = {p_home.tolist()}")


def main() -> None:
  p = argparse.ArgumentParser(description="SO-101 forward kinematics utility.")
  sub = p.add_subparsers(dest="cmd", required=True)
  v = sub.add_parser("verify", help="Run the three verification checks.")
  v.add_argument("--model", required=True, help="Path to SO-101 MJCF XML.")
  v.add_argument("--out", default="/tmp/fk_verify_ee_site.png",
                 help="Output PNG for the EE-site render.")
  v.add_argument("--delta", type=float, default=0.1,
                 help="Per-joint perturbation for the sign-sanity check (rad).")
  args = p.parse_args()
  if args.cmd == "verify":
    _verify(args.model, args.out, args.delta)


if __name__ == "__main__":
  main()
