"""`test-object-pose` CLI.

End-to-end smoke test for the ObjectPoseEstimator: opens one frame
from each of the three configured "object" episodes, asks for a single
click on the object (used as the segmentation prompt), runs the
estimator, and draws:

  - the SAM2-or-fallback mask outline in yellow,
  - the projected mesh wireframe in green,
  - the projected silhouette bbox in cyan,
  - the world (x, y) label.

If the wireframe sits on the actual object, the full
intrinsics + extrinsics + mesh + pose stack is healthy. If it's
visibly offset, the gap tells you which stage is wrong:

  - mask covers wrong thing -> segmenter or prompt
  - mask right but wireframe rotated 90deg -> wrong resting face class
  - wireframe slightly off in (x, y) -> extrinsics or back-projection
  - wireframe sits at wrong scale -> mesh mm vs unit mismatch
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import click
import numpy as np

from chuck_dreamer.config import load_config

from ..dataset import episode_bounds_from_meta, get_frame
from ..runtime import init_from_config
from ..scene_bg import ensure_scene_bg
from ..types import CameraCalibration
from .common import override_option

logger = logging.getLogger(__name__)

_WIN_NAME = "test-object-pose"
_BANNER_H = 64
_FOOTER_H = 24
_MAX_W    = 1280
_MAX_H    = 720


@click.command("test-object-pose")
@click.argument("dataset_id", required=True)
@click.option("--episodes", "episode_csv", default=None, type=str,
              help="Comma-separated episode indices to test. "
                   "Default: object_left,object_mid,object_right from config.")
@click.option("--frame", "frame_idx", default=0, type=int,
              help="Frame index within each episode to test (default 0).")
@click.option("--segmenter", type=click.Choice(["bg", "color", "sam2"]), default="bg",
              help="Segmentation backend. 'bg' builds a per-pixel reference "
                   "from the empty-mat episode and segments by background "
                   "subtraction (recommended). 'color' uses a brightness "
                   "floodfill from the click. 'sam2' uses the SAM2 image "
                   "predictor if installed.")
@click.option("--rebuild-bg", is_flag=True, default=False,
              help="Force recompute of the scene background (default: reuse "
                   "cached scene_bg.npz if present).")
@click.option("--bg-samples", default=16, type=int,
              help="Frames sampled from the empty episode to build the "
                   "background model (default 16).")
@click.option("--n-yaw-inits", default=None, type=int,
              help="Override object_localization.object.n_yaw_inits.")
@override_option
@click.pass_context
def test_object_pose_cmd(ctx, dataset_id: str, episode_csv: str | None,
                         frame_idx: int, segmenter: str,
                         rebuild_bg: bool, bg_samples: int,
                         n_yaw_inits: int | None,
                         overrides: tuple[str, ...]) -> None:
  """Click each object frame; print recovered world (x, y) and draw the
  mesh wireframe in the inferred pose.
  """
  import cv2
  from ..estimator import ObjectPoseEstimator
  from ..render import Camera, project_visible_edges, transform_object

  cfg = load_config(ctx.obj["config_path"], overrides=overrides)
  ol_cfg = init_from_config(cfg)
  cache_root = Path(ol_cfg.cache_dir)

  try:
    cal = CameraCalibration.load(cache_root, dataset_id)
  except FileNotFoundError as e:
    raise click.ClickException(str(e))
  click.echo(f"{dataset_id}: loaded calibration "
             f"(intr rms={cal.intrinsics.rms_px:.2f}px, "
             f"extr rms={cal.extrinsics.rms_px:.2f}px)")

  if episode_csv:
    episodes = [int(x) for x in episode_csv.split(",") if x.strip()]
  else:
    episodes = [ol_cfg.episode_object_left,
                ol_cfg.episode_object_mid,
                ol_cfg.episode_object_right]
  click.echo(f"  testing episodes: {episodes} (frame {frame_idx} of each)")

  mesh_path = Path(ol_cfg.mesh_path)
  if not mesh_path.exists():
    raise click.ClickException(f"mesh not found at {mesh_path}; "
                               f"set object_localization.object.mesh_path.")

  est_cfg = dict(ol_cfg.raw.get("object") or {})
  if n_yaw_inits is not None:
    est_cfg["n_yaw_inits"] = int(n_yaw_inits)
  est_cfg.setdefault("sam2_checkpoint", ol_cfg.sam2_checkpoint)

  scene_bg = None
  if segmenter == "bg":
    click.echo(f"  preparing scene background from episode "
               f"{ol_cfg.episode_empty} ({bg_samples} samples) ...")
    scene_bg = ensure_scene_bg(
      cache_root, dataset_id, ol_cfg.camera_key,
      ol_cfg.episode_empty, n_samples=bg_samples, rebuild=rebuild_bg,
    )
    click.echo(f"  scene background ready (n_frames={scene_bg.n_frames}, "
               f"shape={scene_bg.median.shape})")

  click.echo(f"  loading mesh from {mesh_path} ...")
  t = time.perf_counter()
  estimator = ObjectPoseEstimator(
    calibration = cal,
    mesh_path   = mesh_path,
    config      = est_cfg,
    device      = ol_cfg.device,
    use_sam2    = (segmenter == "sam2"),
    scene_bg    = scene_bg,
  )
  click.echo(f"  estimator ready in {time.perf_counter()-t:.1f}s  "
             f"(segmenter={segmenter}, n_yaw_inits={estimator.n_yaw_inits})")
  click.echo("\n  NOTE: first-frame estimate runs a 2 x n_yaw_inits grid search "
             "(8 silhouette fits by default). Expect 30-90s per first frame on CPU.")

  from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
  ds = LeRobotDataset(dataset_id)

  for ep in episodes:
    click.echo(f"\n=== episode {ep}, frame {frame_idx} ===")
    try:
      ep_fr, ep_to = episode_bounds_from_meta(dataset_id, ep)
    except Exception as e:
      click.echo(f"  episode {ep} bounds unavailable ({e}); skipping.")
      continue
    if ep_to <= ep_fr:
      click.echo(f"  episode {ep} is empty; skipping.")
      continue
    global_idx = ep_fr + max(0, min(ep_to - ep_fr - 1, int(frame_idx)))
    rgb = get_frame(ds, global_idx, ol_cfg.camera_key)

    from ..prompts import click_object
    prompt = click_object(rgb, banner=f"{dataset_id}  ep={ep} (global {global_idx})")
    if prompt is None:
      click.echo("  no click captured; skipping episode.")
      continue
    click.echo(f"  prompt: u={prompt[0]} v={prompt[1]}  -> running estimator ...")

    t = time.perf_counter()
    pose = estimator.estimate(rgb, prompt=prompt)
    dt = time.perf_counter() - t
    if pose is None:
      click.echo(f"  estimator returned None in {dt:.1f}s "
                 f"(segmentation or rest-class search failed).")
      _show_failure(rgb, prompt)
      continue

    click.echo(f"  pose recovered in {dt:.1f}s")
    click.echo(f"    xy_mm     = ({pose.xy_mm[0]:+8.2f}, {pose.xy_mm[1]:+8.2f})")
    click.echo(f"    xyz_mm    = ({pose.xyz_mm[0]:+8.2f}, {pose.xyz_mm[1]:+8.2f}, "
               f"{pose.xyz_mm[2]:+8.2f})")
    click.echo(f"    rest_face = {pose.resting_face_class}")
    click.echo(f"    yaw_deg   = {np.degrees(pose.yaw_rad):+.1f}  "
               f"tilt_deg = {np.degrees(pose.tilt_rad):.2f}")
    click.echo(f"    residual  = {pose.reprojection_error_px:.2f}px  "
               f"confidence={pose.confidence:.2f}")

    _show_pose(
      rgb, prompt, pose, estimator, ep, global_idx, dataset_id,
      segmenter=segmenter,
    )


def _show_pose(rgb: np.ndarray, prompt: tuple[int, int], pose, estimator,
               ep: int, global_idx: int, dataset_id: str,
               segmenter: str) -> None:
  """Overlay mesh wireframe + bbox + label; window blocks until key press."""
  import cv2
  from ..render import Camera, project_visible_edges, transform_object

  H, W = rgb.shape[:2]
  scale = min(_MAX_W / W, _MAX_H / H, 1.0)
  win_w = int(W * scale)
  canvas_h = int(H * scale)
  win_h = _BANNER_H + canvas_h + _FOOTER_H

  bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)

  # Wireframe via the same render path the estimator's edge refinement uses.
  verts_world = transform_object(estimator.mesh.vertices_mm,
                                 pose.R_world_obj, pose.xyz_mm)
  edges = project_visible_edges(verts_world, estimator.mesh.triangles, estimator.camera)
  for p0, p1 in edges:
    cv2.line(bgr,
             (int(round(p0[0])), int(round(p0[1]))),
             (int(round(p1[0])), int(round(p1[1]))),
             (0, 255, 0), 2, cv2.LINE_AA)

  # bbox from the projected vertices.
  uvs = np.array([p for edge in edges for p in edge])
  if len(uvs) > 0:
    x0, y0 = uvs.min(axis=0)
    x1, y1 = uvs.max(axis=0)
    cv2.rectangle(bgr,
                  (int(round(x0)), int(round(y0))),
                  (int(round(x1)), int(round(y1))),
                  (255, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(bgr,
                f"({pose.xy_mm[0]:+.1f}, {pose.xy_mm[1]:+.1f}) mm  "
                f"{pose.resting_face_class}",
                (int(round(x0)), max(int(round(y0)) - 6, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 200, 0), 2, cv2.LINE_AA)

  # The click prompt as a magenta crosshair.
  cv2.drawMarker(bgr, prompt, (255, 0, 255), cv2.MARKER_CROSS, 18, 2)

  if scale != 1.0:
    disp = cv2.resize(bgr, (win_w, canvas_h), interpolation=cv2.INTER_AREA)
  else:
    disp = bgr

  cv2.namedWindow(_WIN_NAME, cv2.WINDOW_AUTOSIZE)
  try:
    while True:
      canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
      canvas[:_BANNER_H, :] = (32, 32, 32)
      canvas[_BANNER_H + canvas_h:, :] = (48, 48, 48)
      canvas[_BANNER_H:_BANNER_H + canvas_h, :, :] = disp
      title = (f"{dataset_id}  ep={ep} (global {global_idx})  "
               f"segmenter={segmenter}")
      cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                  (230, 230, 230), 1, cv2.LINE_AA)
      stats = (f"xy=({pose.xy_mm[0]:+.1f},{pose.xy_mm[1]:+.1f})mm  "
               f"face={pose.resting_face_class}  yaw={np.degrees(pose.yaw_rad):+.1f}deg  "
               f"resid={pose.reprojection_error_px:.2f}px  conf={pose.confidence:.2f}")
      cv2.putText(canvas, stats, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                  (180, 220, 255), 1, cv2.LINE_AA)
      cv2.putText(canvas, "any key -> next episode  |  q quit",
                  (10, win_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                  (200, 200, 200), 1, cv2.LINE_AA)
      cv2.imshow(_WIN_NAME, canvas)
      key = cv2.waitKey(16) & 0xFFFF
      if key != 0xFFFF:
        break
  finally:
    cv2.destroyWindow(_WIN_NAME)
    cv2.waitKey(1)


def _show_failure(rgb: np.ndarray, prompt: tuple[int, int]) -> None:
  """Show the click on the frame so the user can see what they prompted."""
  import cv2
  H, W = rgb.shape[:2]
  scale = min(_MAX_W / W, _MAX_H / H, 1.0)
  bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
  cv2.drawMarker(bgr, prompt, (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
  if scale != 1.0:
    bgr = cv2.resize(bgr, (int(W * scale), int(H * scale)),
                     interpolation=cv2.INTER_AREA)
  cv2.imshow(_WIN_NAME, bgr)
  cv2.waitKey(0)
  cv2.destroyWindow(_WIN_NAME)
  cv2.waitKey(1)
