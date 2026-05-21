"""`annotate-live` CLI: annotate the mat from the live camera feed.

Opens the configured camera, lets the operator capture a still with
SPACE, runs the standard mat annotation UI, solves extrinsics against
loose intrinsics, and writes a single self-contained calibration bundle
(intrinsics + extrinsics + detection) that ``main.py run --calibration``
can consume directly.
"""
from __future__ import annotations

import logging
from pathlib import Path

import click
import cv2  # type: ignore[import-not-found]
import numpy as np

from chuck_dreamer.config import load_config

from ..calibration.bundle import (
  CalibrationBundle,
  read_loose_intrinsics,
  write_bundle,
)
from ..calibration.extrinsics import (
  render_extrinsics_overlay,
  render_world_axes,
  solve_extrinsics,
)
from ..calibration.interactive import (
  annotate_mat_cv2,
  close_popup_window,
  reset_popup_state,
)
from ..runtime import init_from_config
from .common import override_option

logger = logging.getLogger(__name__)

_PREVIEW_WINDOW = "annotate-live · capture"


@click.command("annotate-live")
@click.option("--output", "output_path", required=True,
              type=click.Path(dir_okay=False, path_type=Path),
              help="Where to write the calibration bundle JSON.")
@click.option("--intrinsics", "intrinsics_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help="Path to an intrinsics.json (defaults to "
                   "<object_localization.cache_dir>/intrinsics.json).")
@click.option("--camera", "camera_source", default=None, type=str,
              help="Camera index or device path. Overrides cfg.real.camera.source.")
@override_option
@click.pass_context
def annotate_live_cmd(ctx, output_path: Path, intrinsics_path: Path | None,
                      camera_source: str | None,
                      overrides: tuple[str, ...]) -> None:
  """Capture a frame from the live camera, annotate the mat, save the bundle."""
  aliases: dict = {}
  if camera_source is not None:
    aliases["real.camera.source"] = camera_source
  cfg = load_config(ctx.obj["config_path"], overrides=overrides, aliases=aliases)
  ol_cfg = init_from_config(cfg)

  if intrinsics_path is None:
    intrinsics_path = Path(ol_cfg.cache_dir) / "intrinsics.json"
  try:
    intrinsics = read_loose_intrinsics(intrinsics_path)
  except Exception as e:
    raise click.ClickException(
      f"failed to read intrinsics from {intrinsics_path}: {e}. "
      f"Pass --intrinsics PATH to point at one explicitly.")
  click.echo(f"intrinsics: {intrinsics_path} (rms={intrinsics.rms_px:.2f}px, "
             f"image_size={intrinsics.image_size})")

  # Lazy-import so headless test environments without a camera-backed
  # CLI dependency can still import the module for inspection.
  from ...run.camera import Camera, CameraConfig

  cam_cfg = CameraConfig(
    source=cfg.real.camera.source,
    width=int(cfg.real.camera.width),
    height=int(cfg.real.camera.height),
    fps=int(cfg.real.camera.fps),
    flip=bool(cfg.real.camera.flip),
  )

  click.echo("Opening camera preview — SPACE captures a frame, "
             "q/ESC aborts.")
  captured_rgb: np.ndarray | None = None
  with Camera(cam_cfg) as camera:
    cv2.namedWindow(_PREVIEW_WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
      while True:
        bgr, rgb = camera.read()
        hud = bgr.copy()
        cv2.putText(hud, "SPACE: capture    q/ESC: cancel",
                    (10, hud.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow(_PREVIEW_WINDOW, hud)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
          captured_rgb = rgb
          break
        if key in (27, ord("q")):
          break
    finally:
      cv2.destroyWindow(_PREVIEW_WINDOW)
      cv2.waitKey(1)

  if captured_rgb is None:
    click.echo("aborted; nothing written.", err=True)
    return

  reset_popup_state()
  try:
    click.echo("Opening annotation UI — see banner for click instructions.")
    detection = annotate_mat_cv2(captured_rgb)
  finally:
    close_popup_window()
  if detection is None:
    click.echo("annotation aborted; nothing written.", err=True)
    return

  try:
    result = solve_extrinsics(detection, intrinsics.K, intrinsics.dist, ol_cfg)
  except Exception as e:
    raise click.ClickException(f"extrinsics solve failed: {e}")

  rms = result.refined_rms_px
  threshold = ol_cfg.extrinsics_rms_threshold_px
  if rms > threshold:
    click.echo(click.style(
      f"WARN reprojection rms={rms:.3f}px > {threshold:.2f}px", fg="yellow"))
  else:
    click.echo(f"rms={rms:.3f}px  phase_offset={result.phase_offset}  "
               f"reverse={result.reverse}")

  overlay = render_extrinsics_overlay(
    captured_rgb, result.extrinsics, intrinsics.K, intrinsics.dist, ol_cfg,
    detection, result,
  )
  axes_overlay = render_world_axes(
    captured_rgb, result.extrinsics, intrinsics.K, intrinsics.dist,
  )
  preview_window = "annotate-live · verify (press any key)"
  cv2.namedWindow(preview_window, cv2.WINDOW_AUTOSIZE)
  try:
    cv2.imshow(preview_window, np.hstack([overlay, axes_overlay]))
    cv2.waitKey(0)
  finally:
    cv2.destroyWindow(preview_window)
    cv2.waitKey(1)

  if not click.confirm(f"Accept this calibration and write to {output_path}?",
                       default=True):
    click.echo("rejected; nothing written.")
    return

  bundle = CalibrationBundle(
    intrinsics=intrinsics,
    extrinsics=result.extrinsics,
    detection=detection,
  )
  write_bundle(output_path, bundle, extra={
    "intrinsics_source": str(intrinsics_path),
    "phase_offset":      int(result.phase_offset),
    "reverse":           bool(result.reverse),
    "seed_rms_px":       float(result.seed_rms_px),
    "camera_source":     str(cam_cfg.source),
    "camera_size":       [int(cam_cfg.width), int(cam_cfg.height)],
  })
  click.echo(f"wrote {output_path}")
