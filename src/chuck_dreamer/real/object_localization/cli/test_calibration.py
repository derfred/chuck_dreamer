"""`test-calibration` CLI.

Pops a window on a configured frame. Every left-click is undistorted,
back-projected through the cached intrinsics + extrinsics, and
intersected with the mat plane Z = ``--z`` (default 0). The world
``(x, y, z) mm`` is printed and overlaid on the image so you can
sanity-check the calibration against known mat features:

  - The white circle center should report ~ (0, 0, 0).
  - A line's NEAR endpoint should report ~ (0, ±D/2, 0).
  - A line's FAR endpoint should report ~ (-L, ±D/2, 0).

The image starts at episode ``object_localization.extrinsics.annotation_frame``
of episode ``episodes.empty`` (the same frame ``annotate-mat`` used). Use
``--episode`` to look at a recording with the object in it, and ``--frame``
to scrub to a specific index.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np

from chuck_dreamer.config import load_config

from ..dataset import episode_bounds_from_meta, get_frame
from ..runtime import init_from_config
from ..types import CameraCalibration
from .common import override_option

logger = logging.getLogger(__name__)

_WIN_NAME = "test-calibration"
_BANNER_H = 64
_FOOTER_H = 24
_MAX_W    = 1280
_MAX_H    = 720


@click.command("test-calibration")
@click.argument("dataset_id", required=True)
@click.option("--episode", "episode_idx", default=None, type=int,
              help="Episode index (default: object_localization.episodes.empty).")
@click.option("--frame", "frame_idx", default=None, type=int,
              help="Frame index within the episode (default: 0).")
@click.option("--z", "world_z", default=0.0, type=float,
              help="Assumed world-frame Z (mm) of the clicked point. Default 0 "
                   "(the mat plane). Use e.g. --z 28 to back-project clicks on "
                   "top of an object 28 mm tall.")
@override_option
@click.pass_context
def test_calibration_cmd(ctx, dataset_id: str, episode_idx: int | None,
                         frame_idx: int | None, world_z: float,
                         overrides: tuple[str, ...]) -> None:
  """Click any image point; print its world-frame (x, y, z=Z) in mm.

  Verifies a per-dataset (intrinsics + extrinsics) pair end-to-end.
  Click known mat features and confirm the printed coords match the
  geometry you measured by hand.
  """
  import cv2

  cfg = load_config(ctx.obj["config_path"], overrides=overrides)
  ol_cfg = init_from_config(cfg)
  cache_root = Path(ol_cfg.cache_dir)

  try:
    cal = CameraCalibration.load(cache_root, dataset_id)
  except FileNotFoundError as e:
    raise click.ClickException(str(e))

  ep = ol_cfg.episode_empty if episode_idx is None else int(episode_idx)
  click.echo(f"{dataset_id}: loading metadata + frame ...")
  ep_fr, ep_to = episode_bounds_from_meta(dataset_id, ep)
  if ep_to <= ep_fr:
    raise click.ClickException(f"episode {ep} of {dataset_id} is empty.")
  local_fr = 0 if frame_idx is None else int(frame_idx)
  global_idx = ep_fr + local_fr
  if global_idx >= ep_to:
    raise click.ClickException(
      f"frame {local_fr} out of range for episode {ep} (length {ep_to - ep_fr}).")

  from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
  ds = LeRobotDataset(dataset_id)
  frame = get_frame(ds, global_idx, ol_cfg.camera_key)
  click.echo(f"  episode {ep} frame {local_fr} (global {global_idx}) loaded; "
             f"shape={frame.shape}.")

  K    = np.asarray(cal.intrinsics.K, dtype=np.float64)
  dist = np.asarray(cal.intrinsics.dist, dtype=np.float64).reshape(-1)
  R    = np.asarray(cal.extrinsics.R, dtype=np.float64)
  t    = np.asarray(cal.extrinsics.t, dtype=np.float64).reshape(3)
  click.echo(f"  intrinsics rms_px={cal.intrinsics.rms_px:.3f}  "
             f"extrinsics rms_px={cal.extrinsics.rms_px:.3f}")

  # Camera position in world: -R^T t.
  cam_pos_world = -R.T @ t
  click.echo(f"  camera pos (world, mm): "
             f"x={cam_pos_world[0]:+.1f} y={cam_pos_world[1]:+.1f} "
             f"z={cam_pos_world[2]:+.1f}")
  click.echo(f"  (z should be positive if +Z points away from the mat.)")

  H_img, W_img = frame.shape[:2]
  scale = min(_MAX_W / W_img, _MAX_H / H_img, 1.0)
  win_w = int(W_img * scale)
  canvas_h = int(H_img * scale)
  win_h = _BANNER_H + canvas_h + _FOOTER_H

  clicks: list[tuple[float, float, np.ndarray]] = []

  def _backproject(u: float, v: float) -> np.ndarray | None:
    """Image pixel -> world (x, y, world_z). Returns None if ray is parallel
    to the plane.
    """
    pt = np.array([[[u, v]]], dtype=np.float64)
    undist = cv2.undistortPoints(pt, K, dist).reshape(2)
    ray_cam = np.array([undist[0], undist[1], 1.0])
    ray_world = R.T @ ray_cam
    if abs(ray_world[2]) < 1e-9:
      return None
    s = (world_z - cam_pos_world[2]) / ray_world[2]
    if s < 0:
      # Plane is behind the camera; ray points away.
      return None
    return cam_pos_world + s * ray_world

  def on_mouse(event, x, y, _flags, _ud):
    if event != cv2.EVENT_LBUTTONDOWN:
      return
    if y < _BANNER_H or y >= _BANNER_H + canvas_h:
      return
    u = (x) / scale
    v = (y - _BANNER_H) / scale
    if not (0 <= u < W_img and 0 <= v < H_img):
      return
    world = _backproject(u, v)
    if world is None:
      logger.warning("ray-plane intersection invalid for this click "
                     "(ray parallel to or behind plane).")
      return
    clicks.append((u, v, world))
    print(f"click {len(clicks):2d}: u={u:7.1f} v={v:7.1f}  ->  "
          f"x={world[0]:+8.2f}mm  y={world[1]:+8.2f}mm  z={world[2]:+8.2f}mm")

  cv2.namedWindow(_WIN_NAME, cv2.WINDOW_AUTOSIZE)
  cv2.setMouseCallback(_WIN_NAME, on_mouse)

  bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  if scale != 1.0:
    bgr = cv2.resize(bgr, (win_w, canvas_h), interpolation=cv2.INTER_AREA)

  click.echo("\nClick any point on the mat. Press u to undo, c to clear, "
             "q/Esc to quit.\n")
  click.echo(f"Mat geometry to check against (L={ol_cfg.mat_line_length_mm}mm, "
             f"D={ol_cfg.mat_line_separation_mm}mm, r={ol_cfg.mat_circle_radius_mm}mm):")
  click.echo(f"  circle center                 -> ( +0.0, +0.0, +0.0 )")
  click.echo(f"  line 1 (bottom) NEAR endpoint -> ( +0.0, "
             f"{+ol_cfg.mat_line_separation_mm/2:+.1f}, +0.0 )")
  click.echo(f"  line 1 (bottom) FAR endpoint  -> ({-ol_cfg.mat_line_length_mm:+.1f}, "
             f"{+ol_cfg.mat_line_separation_mm/2:+.1f}, +0.0 )")
  click.echo(f"  line 2 (top) NEAR endpoint    -> ( +0.0, "
             f"{-ol_cfg.mat_line_separation_mm/2:+.1f}, +0.0 )")
  click.echo(f"  line 2 (top) FAR endpoint     -> ({-ol_cfg.mat_line_length_mm:+.1f}, "
             f"{-ol_cfg.mat_line_separation_mm/2:+.1f}, +0.0 )\n")

  try:
    while True:
      canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
      canvas[:_BANNER_H, :] = (32, 32, 32)
      canvas[_BANNER_H + canvas_h:, :] = (48, 48, 48)
      canvas[_BANNER_H:_BANNER_H + canvas_h, :, :] = bgr

      for i, (u, v, world) in enumerate(clicks):
        x = int(round(u * scale))
        y = int(round(v * scale + _BANNER_H))
        cv2.circle(canvas, (x, y), 6, (0, 220, 220), -1)
        cv2.circle(canvas, (x, y), 7, (0, 0, 0), 1)
        label = f"{i+1}: ({world[0]:+.0f},{world[1]:+.0f},{world[2]:+.0f})mm"
        cv2.putText(canvas, label, (x + 10, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (x + 10, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1, cv2.LINE_AA)

      title = (f"{dataset_id}  ep={ep} frame={local_fr} (global {global_idx})  "
               f"plane Z={world_z}mm  clicks={len(clicks)}")
      cv2.putText(canvas, title, (10, 24),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)
      cv2.putText(canvas, f"intr rms={cal.intrinsics.rms_px:.2f}px  "
                          f"extr rms={cal.extrinsics.rms_px:.2f}px",
                  (10, 50),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
      controls = "click: print world coords   u undo   c clear   q quit"
      cv2.putText(canvas, controls, (10, win_h - 8),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

      cv2.imshow(_WIN_NAME, canvas)
      key = cv2.waitKey(16) & 0xFFFF
      if key in (ord('q'), ord('Q'), 27):
        break
      if key in (ord('u'), ord('U')) and clicks:
        clicks.pop()
      elif key in (ord('c'), ord('C')):
        clicks.clear()
  finally:
    cv2.destroyWindow(_WIN_NAME)
    cv2.waitKey(1)
