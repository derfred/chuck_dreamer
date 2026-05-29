#!/usr/bin/env python3
"""Standalone camera-intrinsics calibration from video, episodes, or a live camera.

Pools checkerboard views from any mix of frame sources and fits a single
``cv2.calibrateCamera`` model, writing one intrinsics JSON. Unlike the pooled
LeRobot command in ``chuck_dreamer.lerobot.object_localization.calibration``
(worktree ``issue-2-2``), this script is source-agnostic: it does not write
into per-dataset calibration_cache dirs, and it uses *every* detected view
(no dedup, no coverage/min-frame gating).

Sources:
  --video PATH       a video file (repeatable)
  --episodes SPEC    LeRobot EpisodeSpec, e.g. user/cal#1:0-200:5 (repeatable)
  --camera [INDEX]   live preview + capture (mutually exclusive, once only)

--video and --episodes can be combined and repeated; their frames are pooled.

Checkerboard geometry, distortion model, and the inspect coverage grid come
from ``object_localization`` in the project config (see configs/default.yaml).

Examples:
  python scripts/calibrate_intrinsics.py out.json --video clip.mp4 --inspect
  python scripts/calibrate_intrinsics.py out.json --episodes user/cal#1
  python scripts/calibrate_intrinsics.py out.json --camera --inspect
  python scripts/calibrate_intrinsics.py out.json --inspect   # re-inspect existing json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from chuck_dreamer.config import load_config
from chuck_dreamer.common.episode_spec import EpisodeSpec
from chuck_dreamer.lerobot.object_localization.dataset import episode_bounds, get_frame
from chuck_dreamer.lerobot.object_localization.runtime import (
  ObjectLocalizationConfig,
  init_from_config,
)
from chuck_dreamer.lerobot.object_localization.types import Intrinsics


# ---------------------------------------------------------------------------
# Frame container
# ---------------------------------------------------------------------------
@dataclass
class DetectedView:
  label: str                # source-tagged provenance, e.g. "clip.mp4:42"
  corners: np.ndarray       # (N, 1, 2) float32 image coords
  bucket: tuple[int, int]   # coverage-grid cell the centroid landed in


# ---------------------------------------------------------------------------
# Frame ingestion
# ---------------------------------------------------------------------------
def _frames_from_video(path: Path, stride: int = 1) -> Iterator[tuple[str, np.ndarray]]:
  """Decode a video file to (label, RGB-uint8) frames via PyAV."""
  import av  # type: ignore

  base = path.name
  with av.open(str(path)) as container:
    stream = container.streams.video[0]
    for i, frame in enumerate(container.decode(stream)):
      if stride > 1 and (i % stride) != 0:
        continue
      yield f"{base}:{i}", frame.to_ndarray(format="rgb24")


def _frames_from_episodes(
  spec_str: str, camera_key: str, n_target: int,
) -> Iterator[tuple[str, np.ndarray]]:
  """Resolve an EpisodeSpec and yield (label, RGB-uint8) frames.

  Frames listed in the spec are episode-relative and used verbatim; when the
  spec carries no ``:FRAMES`` portion we auto-sample ``n_target`` evenly-spaced
  frames per episode (same linspace strategy as the pooled calibrator).
  """
  from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

  spec = EpisodeSpec.parse(spec_str)
  slices, _ = spec.read_episodes()
  if not slices:
    return
  ds = LeRobotDataset(spec.dataset_id)
  for sl in slices:
    fr, to = sl.bounds
    length = to - fr
    if length <= 0:
      continue
    if spec.frames is not None:
      # Episode-relative indices, clipped to this episode's bounds.
      indices = sorted({fr + f for f in spec.frames if 0 <= f < length})
    elif length <= n_target:
      indices = list(range(fr, to))
    else:
      indices = sorted({int(x) for x in np.linspace(fr, to - 1, num=n_target)})
    for idx in indices:
      try:
        rgb = get_frame(ds, idx, camera_key)
      except Exception as e:  # noqa: BLE001 — skip unreadable frames, keep going
        print(f"  [episodes] {spec.dataset_id} frame {idx}: skipped ({e})",
              file=sys.stderr)
        continue
      yield f"{spec.dataset_id}#{sl.episode_index}:{idx}", rgb


def _frames_from_camera(
  device: int | str, cam_cfg: dict[str, Any], ck_size: tuple[int, int],
  stride: int = 1,
) -> Iterator[tuple[str, np.ndarray]]:
  """Live preview + capture, yielding (label, RGB-uint8) frames from memory.

  Opens a preview window, overlays live checkerboard detections so the
  operator can see whether the board is being picked up, captures frames on
  SPACE/ENTER, and stops on q/ESC. Frames are kept in memory (no encode/decode
  round-trip) and fed straight to the calibrator.
  """
  import cv2

  cap = cv2.VideoCapture(device)
  if not cap.isOpened():
    raise SystemExit(f"could not open camera {device!r}")
  width = int(cam_cfg.get("width", 640))
  height = int(cam_cfg.get("height", 480))
  fps = int(cam_cfg.get("fps", 30))
  flip = bool(cam_cfg.get("flip", False))
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
  cap.set(cv2.CAP_PROP_FPS, fps)

  win = "calibrate-intrinsics (SPACE/ENTER: capture, q/ESC: stop)"
  cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

  captured: list[np.ndarray] = []
  recording = False
  try:
    while True:
      ok, bgr = cap.read()
      if not ok:
        print("  [camera] frame grab failed; stopping.", file=sys.stderr)
        break
      if flip:
        bgr = cv2.flip(bgr, 1)

      # Live corner overlay so the operator sees board pickup.
      gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
      found, corners = cv2.findChessboardCorners(
        gray, ck_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK,
      )
      canvas = bgr.copy()
      if found and corners is not None:
        cv2.drawChessboardCorners(canvas, ck_size, corners, found)
      status = "CAPTURING" if recording else "live"
      color = (0, 0, 255) if recording else (200, 200, 200)
      board = "board OK" if found else "no board"
      cv2.putText(canvas, f"{status}  {board}  frames={len(captured)}",
                  (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
      cv2.imshow(win, canvas)

      if recording and (stride <= 1 or len(captured) % stride == 0):
        captured.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

      key = cv2.waitKey(16) & 0xFF
      if key in (ord("q"), 27):  # q / ESC
        break
      if key in (ord(" "), 13):  # SPACE / ENTER toggles capture
        recording = not recording
        print(f"  [camera] {'capturing' if recording else 'paused'} "
              f"at {len(captured)} frames.")
  finally:
    cap.release()
    cv2.destroyWindow(win)
    cv2.waitKey(1)

  if not captured:
    raise SystemExit("camera: nothing captured (press SPACE/ENTER to capture).")
  print(f"  [camera] captured {len(captured)} frames.")
  for i, rgb in enumerate(captured):
    yield f"camera:{i}", rgb


# ---------------------------------------------------------------------------
# Calibration core (adapted from the pooled calibrator, source-agnostic)
# ---------------------------------------------------------------------------
def _detect_corners(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
  import cv2

  gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
  flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
  found, corners = cv2.findChessboardCornersSB(gray, size, flags=flags)
  if not found or corners is None:
    return None
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
  return cv2.cornerSubPix(gray, corners.astype(np.float32), (11, 11), (-1, -1), criteria)


def _checkerboard_object_points(size: tuple[int, int], square_mm: float) -> np.ndarray:
  cols, rows = size
  objp = np.zeros((cols * rows, 3), dtype=np.float32)
  grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
  objp[:, :2] = grid * float(square_mm)
  return objp


def _centroid_bucket(corners: np.ndarray, image_size: tuple[int, int],
                     grid: tuple[int, int]) -> tuple[int, int]:
  pts = corners.reshape(-1, 2)
  cu, cv = float(pts[:, 0].mean()), float(pts[:, 1].mean())
  W, H = image_size
  gx, gy = max(1, grid[0]), max(1, grid[1])
  bx = min(gx - 1, max(0, int(cu / (W / gx))))
  by = min(gy - 1, max(0, int(cv / (H / gy))))
  return (bx, by)


def _coverage_fill(views: list[DetectedView], image_size: tuple[int, int],
                   grid: tuple[int, int]) -> tuple[np.ndarray, int, int]:
  W, H = image_size
  gx, gy = max(1, grid[0]), max(1, grid[1])
  cell_w, cell_h = W / gx, H / gy
  filled = np.zeros((gy, gx), dtype=bool)
  for v in views:
    pts = v.corners.reshape(-1, 2)
    cx = np.clip((pts[:, 0] / cell_w).astype(int), 0, gx - 1)
    cy = np.clip((pts[:, 1] / cell_h).astype(int), 0, gy - 1)
    filled[cy, cx] = True
  return filled, int(filled.sum()), int(gx * gy)


def _distortion_flags(n_coeffs: int) -> int:
  import cv2

  if n_coeffs == 5:
    return 0
  if n_coeffs == 2:
    return int(cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
  if n_coeffs == 0:
    return int(cv2.CALIB_ZERO_TANGENT_DIST
               | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
  raise ValueError(f"unsupported distortion coefficient count: {n_coeffs}")


def _per_view_rms(obj_points: list[np.ndarray], img_points: list[np.ndarray],
                  rvecs: list[np.ndarray], tvecs: list[np.ndarray],
                  K: np.ndarray, dist: np.ndarray) -> list[float]:
  import cv2

  out: list[float] = []
  for op, ip, rv, tv in zip(obj_points, img_points, rvecs, tvecs):
    proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
    diff = proj.reshape(-1, 2) - ip.reshape(-1, 2)
    out.append(float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))))
  return out


@dataclass
class CalibrationResult:
  intrinsics: Intrinsics
  views: list[DetectedView]
  per_view_rms: list[float]
  coverage_filled: int
  coverage_total: int


def calibrate_from_frames(
  frames: Iterator[tuple[str, np.ndarray]], ol_cfg: ObjectLocalizationConfig,
) -> CalibrationResult:
  """Detect boards in every frame and fit a single intrinsics model.

  Uses every successful detection — no dedup, no coverage/min-frame gate. The
  coverage grid is computed only to populate provenance buckets and the
  --inspect view. Aborts only when zero boards are detected.
  """
  import cv2

  ck_size = (int(ol_cfg.checkerboard_size[0]), int(ol_cfg.checkerboard_size[1]))
  ck_obj = _checkerboard_object_points(ck_size, ol_cfg.checkerboard_square_mm)
  grid = (int(ol_cfg.intrinsics_coverage_grid[0]), int(ol_cfg.intrinsics_coverage_grid[1]))

  views: list[DetectedView] = []
  image_size: tuple[int, int] | None = None
  n_seen = 0
  for label, rgb in frames:
    n_seen += 1
    H, W = rgb.shape[:2]
    if image_size is None:
      image_size = (W, H)
    elif image_size != (W, H):
      print(f"  [detect] {label}: size {(W, H)} != {image_size}; skipped.",
            file=sys.stderr)
      continue
    corners = _detect_corners(rgb, ck_size)
    if corners is None:
      continue
    bucket = _centroid_bucket(corners, image_size, grid)
    views.append(DetectedView(label=label, corners=corners, bucket=bucket))
    print(f"  [detect] HIT {label}  bucket={bucket}  ({len(views)} so far)")

  if image_size is None:
    raise SystemExit("no frames were produced by any source.")
  if not views:
    raise SystemExit(
      f"no checkerboard detected in any of {n_seen} frames "
      f"(expected a {ck_size[0]}x{ck_size[1]} inner-corner board).")

  print(f"\nfitting on {len(views)} views (from {n_seen} frames) ...")
  obj_points = [ck_obj.copy() for _ in views]
  img_points = [v.corners for v in views]

  calib_flags = _distortion_flags(ol_cfg.intrinsics_distortion_model)
  if ol_cfg.intrinsics_fix_aspect_ratio:
    calib_flags |= cv2.CALIB_FIX_ASPECT_RATIO
  if ol_cfg.intrinsics_fix_principal_point:
    calib_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
  K_init: np.ndarray | None = None
  dist_init: np.ndarray | None = None
  if ol_cfg.intrinsics_fix_aspect_ratio or ol_cfg.intrinsics_fix_principal_point:
    W, H = image_size
    K_init = np.array([[float(W), 0.0, W / 2.0],
                       [0.0, float(W), H / 2.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_init = np.zeros(5, dtype=np.float64)
    calib_flags |= cv2.CALIB_USE_INTRINSIC_GUESS

  rms_px, K_raw, dist_raw, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, image_size,
    K_init,  # type: ignore[arg-type]  # None is valid (OpenCV default)
    dist_init,  # type: ignore[arg-type]
    flags=calib_flags, criteria=criteria,
  )
  K = np.asarray(K_raw, dtype=np.float64)
  dist: np.ndarray = np.asarray(dist_raw, dtype=np.float64).reshape(-1)
  if dist.size < 5:
    dist = np.concatenate([dist, np.zeros(5 - dist.size, dtype=np.float64)])
  else:
    dist = dist[:5]

  per_view = _per_view_rms(obj_points, img_points, list(rvecs), list(tvecs), K, dist)
  _filled, filled_n, total_n = _coverage_fill(views, image_size, grid)

  intrinsics = Intrinsics(
    image_size=image_size, K=K, dist=dist,
    rms_px=float(rms_px), n_frames_used=len(views),
  )
  return CalibrationResult(
    intrinsics=intrinsics, views=views, per_view_rms=per_view,
    coverage_filled=filled_n, coverage_total=total_n,
  )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_output(result: CalibrationResult, sources: list[str],
                 ol_cfg: ObjectLocalizationConfig, out_path: Path) -> None:
  blob = result.intrinsics.to_json()
  blob.update({
    "sources": sources,
    "per_view_rms_px": [float(x) for x in result.per_view_rms],
    "rms_threshold_px": float(ol_cfg.intrinsics_rms_threshold_px),
    "view_provenance": [
      {"label": v.label, "bucket": list(v.bucket)} for v in result.views
    ],
  })
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(blob, indent=2))
  print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# --inspect
# ---------------------------------------------------------------------------
def _fov_degrees(K: np.ndarray, image_size: tuple[int, int]) -> tuple[float, float]:
  W, H = image_size
  fx, fy = float(K[0, 0]), float(K[1, 1])
  hfov = 2.0 * np.degrees(np.arctan2(W / 2.0, fx)) if fx > 0 else float("nan")
  vfov = 2.0 * np.degrees(np.arctan2(H / 2.0, fy)) if fy > 0 else float("nan")
  return hfov, vfov


def print_summary(intr: Intrinsics, per_view_rms: list[float],
                  filled: int, total: int) -> None:
  K = intr.K
  hfov, vfov = _fov_degrees(K, intr.image_size)
  rms = np.asarray(per_view_rms, dtype=float) if per_view_rms else np.array([])
  print("\n=== intrinsics ===")
  print(f"  image_size : {intr.image_size[0]} x {intr.image_size[1]}")
  print(f"  fx, fy     : {K[0, 0]:.2f}, {K[1, 1]:.2f}")
  print(f"  cx, cy     : {K[0, 2]:.2f}, {K[1, 2]:.2f}")
  print(f"  FOV (h, v) : {hfov:.1f}deg, {vfov:.1f}deg")
  print(f"  dist       : {np.array2string(intr.dist, precision=5)}")
  print(f"  n_frames   : {intr.n_frames_used}")
  print(f"  RMS (px)   : {intr.rms_px:.4f}  (overall reprojection)")
  if rms.size:
    order = np.argsort(rms)[::-1]
    worst = ", ".join(f"{rms[i]:.3f}" for i in order[:min(5, rms.size)])
    print(f"  per-view   : min={rms.min():.3f} median={np.median(rms):.3f} "
          f"max={rms.max():.3f}")
    print(f"  worst views: {worst}")
  if total:
    print(f"  coverage   : {filled}/{total} grid cells "
          f"({100.0 * filled / total:.0f}%)")


def render_coverage(result: CalibrationResult, ol_cfg: ObjectLocalizationConfig,
                    out_png: Path) -> None:
  """Tiled coverage view: shade unfilled cells, grid lines, per-view corners."""
  import cv2

  W, H = result.intrinsics.image_size
  grid = (int(ol_cfg.intrinsics_coverage_grid[0]), int(ol_cfg.intrinsics_coverage_grid[1]))
  gx, gy = max(1, grid[0]), max(1, grid[1])
  cell_w, cell_h = W / gx, H / gy

  canvas: np.ndarray = np.full((H, W, 3), 32, dtype=np.uint8)
  filled, _, _ = _coverage_fill(result.views, (W, H), grid)

  overlay = canvas.copy()
  for j in range(gy):
    for i in range(gx):
      if not filled[j, i]:
        x0, y0 = int(i * cell_w), int(j * cell_h)
        x1, y1 = int((i + 1) * cell_w), int((j + 1) * cell_h)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 200), -1)
  canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)

  for j in range(gy + 1):
    y = int(j * cell_h)
    cv2.line(canvas, (0, y), (W, y), (140, 140, 140), 1)
  for i in range(gx + 1):
    x = int(i * cell_w)
    cv2.line(canvas, (x, 0), (x, H), (140, 140, 140), 1)

  n = len(result.views)
  for k, v in enumerate(result.views):
    hue = int(180 * (k / max(1, n)))
    hsv = np.array([[[hue, 200, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    for u, vv in v.corners.reshape(-1, 2):
      cv2.circle(canvas, (int(u), int(vv)), 2, color, -1)

  out_png.parent.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(out_png), canvas)
  print(f"wrote {out_png}")


def inspect_existing(out_path: Path, ol_cfg: ObjectLocalizationConfig) -> None:
  """Re-inspect an already-written intrinsics JSON (no recalibration)."""
  if not out_path.exists():
    raise SystemExit(f"--inspect with no sources needs an existing {out_path}.")
  blob = json.loads(out_path.read_text())
  intr = Intrinsics.from_json(blob)
  per_view = [float(x) for x in blob.get("per_view_rms_px", [])]

  # Recover bucket fill from stored provenance, if present.
  grid = tuple(int(x) for x in ol_cfg.intrinsics_coverage_grid)
  gx, gy = max(1, grid[0]), max(1, grid[1])
  filled = np.zeros((gy, gx), dtype=bool)
  for prov in blob.get("view_provenance", []):
    bx, by = prov.get("bucket", [None, None])
    if bx is not None and by is not None:
      filled[min(gy - 1, int(by)), min(gx - 1, int(bx))] = True
  print_summary(intr, per_view, int(filled.sum()), int(gx * gy))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
  ap = argparse.ArgumentParser(
    description="Calibrate camera intrinsics from video / episodes / live camera.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  ap.add_argument("output", type=Path, help="intrinsics JSON to write (mandatory).")
  ap.add_argument("--video", type=Path, action="append", default=[],
                  help="video file (repeatable).")
  ap.add_argument("--episodes", type=str, action="append", default=[],
                  metavar="SPEC", help="EpisodeSpec, e.g. user/cal#1:0-200:5 (repeatable).")
  ap.add_argument("--camera", nargs="?", const="__default__", default=None,
                  metavar="INDEX",
                  help="live preview + capture; optional device index "
                       "(default: real.camera.source). Incompatible with --video/--episodes.")
  ap.add_argument("--resolution", type=str, default=None, metavar="WxH",
                  help="capture resolution for --camera, e.g. 1920x1080 "
                       "(default: real.camera width/height).")
  ap.add_argument("--inspect", action="store_true",
                  help="print a diagnostic summary + write a coverage PNG; "
                       "with no sources, re-inspects the existing output JSON.")
  ap.add_argument("--stride", type=int, default=1,
                  help="use every Nth frame from video/camera sources (default 1).")
  ap.add_argument("--config", type=str, action="append", default=None,
                  help="config overlay yaml (repeatable); defaults to package defaults.")
  ap.add_argument("-o", "--override", action="append", default=[],
                  metavar="KEY=VAL", help="dotted-path config override (repeatable).")
  args = ap.parse_args()

  using_camera = args.camera is not None
  using_files = bool(args.video) or bool(args.episodes)
  if using_camera and using_files:
    ap.error("--camera is incompatible with --video / --episodes.")
  if args.resolution is not None and not using_camera:
    ap.error("--resolution only applies to --camera.")

  cfg = load_config(args.config, overrides=tuple(args.override))
  ol_cfg = init_from_config(cfg)

  # --inspect with no sources: re-inspect the existing JSON and exit.
  if args.inspect and not using_camera and not using_files:
    inspect_existing(args.output, ol_cfg)
    return

  if not using_camera and not using_files:
    ap.error("provide at least one source: --video, --episodes, or --camera.")

  sources: list[str] = []
  frame_iters: list[Iterator[tuple[str, np.ndarray]]] = []

  if using_camera:
    real_cam = _camera_config(cfg)
    if args.resolution is not None:
      w, h = _parse_resolution(args.resolution, ap)
      real_cam["width"], real_cam["height"] = w, h
    device_arg = args.camera
    if device_arg == "__default__":
      device_arg = str(real_cam.get("source", "0"))
    device: int | str = int(device_arg) if device_arg.lstrip("-").isdigit() else device_arg
    sources.append(f"camera:{device}")
    ck_size = (int(ol_cfg.checkerboard_size[0]), int(ol_cfg.checkerboard_size[1]))
    frame_iters.append(_frames_from_camera(device, real_cam, ck_size, stride=args.stride))
  else:
    for vid in args.video:
      sources.append(f"video:{vid}")
      frame_iters.append(_frames_from_video(vid, stride=args.stride))
    for spec in args.episodes:
      sources.append(f"episodes:{spec}")
      frame_iters.append(_frames_from_episodes(
        spec, ol_cfg.camera_key, ol_cfg.intrinsics_n_frames_target))

  def _chain() -> Iterator[tuple[str, np.ndarray]]:
    for it in frame_iters:
      yield from it

  result = calibrate_from_frames(_chain(), ol_cfg)
  write_output(result, sources, ol_cfg, args.output)
  if args.inspect:
    print_summary(result.intrinsics, result.per_view_rms,
                  result.coverage_filled, result.coverage_total)
    render_coverage(result, ol_cfg, args.output.with_suffix(".coverage.png"))


def _parse_resolution(raw: str, ap: argparse.ArgumentParser) -> tuple[int, int]:
  """Parse a ``WxH`` resolution string into ``(width, height)``."""
  parts = raw.lower().replace("X", "x").split("x")
  if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
    ap.error(f"--resolution must be WxH (e.g. 1920x1080), got {raw!r}.")
  return int(parts[0]), int(parts[1])


def _camera_config(cfg: Any) -> dict[str, Any]:
  """Extract real.camera as a plain dict (source/width/height/fps/flip)."""
  from omegaconf import OmegaConf

  real = cfg.get("real", {}) if hasattr(cfg, "get") else {}
  cam = real.get("camera", {}) if real else {}
  out = OmegaConf.to_container(cam, resolve=True) if cam else {}
  if not isinstance(out, dict):
    return {}
  return {str(k): v for k, v in out.items()}


if __name__ == "__main__":
  main()
