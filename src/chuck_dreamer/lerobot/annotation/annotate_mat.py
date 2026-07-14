"""`annotate-mat` CLI: interactive extrinsics calibration from one frame."""
from __future__ import annotations

import logging
from pathlib import Path

import click

from chuck_dreamer.config import load_config

from .calibration.extrinsics import (
  render_extrinsics_overlay,
  render_world_axes,
  solve_extrinsics,
)
from .calibration.interactive import (
  annotate_mat_cv2,
  close_popup_window,
  reset_popup_state,
)
from .dataset import episode_bounds, get_frame, sample_episode_frames
from chuck_dreamer.perception.config import init_from_config
from chuck_dreamer.perception.types import MatDetection
from chuck_dreamer.store import (
  dataset_cache_dir,
  read_extrinsics,
  read_intrinsics,
  read_mat_annotation,
  write_extrinsics,
  write_mat_annotation,
)
from chuck_dreamer.cli import override_option

logger = logging.getLogger(__name__)


@click.command("annotate-mat")
@click.argument("dataset_ids", nargs=-1, required=True)
@click.option("--force", is_flag=True, default=False,
              help="Re-annotate even if extrinsics.json already exists.")
@click.option("--review", is_flag=True, default=False,
              help="Skip the click UI; reload mat_annotation.json, re-solve, "
                   "re-render verification, prompt to overwrite.")
@override_option
@click.pass_context
def annotate_mat_cmd(ctx, dataset_ids: tuple[str, ...],
                     force: bool, review: bool,
                     overrides: tuple[str, ...]) -> None:
  """Annotate the mat fiducials and solve per-dataset extrinsics."""
  cfg = load_config(ctx.obj["config_path"], overrides=overrides)
  ol_cfg = init_from_config(cfg)
  cache_root = Path(ol_cfg.cache_dir)

  for dataset_id in dataset_ids:
    out_dir   = dataset_cache_dir(cache_root, dataset_id)
    extr_path = out_dir / "extrinsics.json"

    if extr_path.exists() and not force and not review:
      try:
        cached = read_extrinsics(cache_root, dataset_id)
        click.echo(f"{dataset_id}: extrinsics already cached "
                   f"(rms={cached.rms_px:.2f}px), skipping. "
                   f"(use --force or --review)")
        continue
      except Exception as e:
        click.echo(f"{dataset_id}: cached extrinsics unreadable ({e}); re-solving.")

    try:
      intrinsics = read_intrinsics(cache_root, dataset_id)
    except Exception:
      raise click.ClickException(
        f"{dataset_id}: intrinsics.json missing under {out_dir}. "
        f"Run `python main.py import-lerobot calibrate-intrinsics "
        f"{out_dir / 'intrinsics.json'} --episodes {dataset_id}` first.")

    click.echo(f"{dataset_id}: loading LeRobotDataset (episode "
               f"{ol_cfg.episode_empty}, frame {ol_cfg.extrinsics_annotation_frame}) ...")
    try:
      from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except Exception as e:
      raise click.ClickException(f"lerobot not importable: {e}")
    ds = LeRobotDataset(dataset_id)

    fr, to = episode_bounds(ds, ol_cfg.episode_empty)
    target_idx = fr + ol_cfg.extrinsics_annotation_frame
    if target_idx >= to:
      raise click.ClickException(
        f"annotation_frame={ol_cfg.extrinsics_annotation_frame} out of range "
        f"for episode {ol_cfg.episode_empty} (length {to - fr}).")
    frame = get_frame(ds, target_idx, ol_cfg.camera_key)

    detection: MatDetection | None
    if review:
      try:
        detection = read_mat_annotation(cache_root, dataset_id)
      except Exception:
        raise click.ClickException(
          f"{dataset_id}: --review requires mat_annotation.json under {out_dir}.")
      click.echo(f"{dataset_id}: re-using cached mat_annotation.json.")
    else:
      reset_popup_state()
      try:
        click.echo("Opening annotation UI — see banner for click instructions.")
        detection = annotate_mat_cv2(frame)
      finally:
        close_popup_window()
      if detection is None:
        click.echo(f"{dataset_id}: annotation aborted; nothing written.", err=True)
        continue

    try:
      result = solve_extrinsics(detection, intrinsics.K, intrinsics.dist, ol_cfg)
    except Exception as e:
      click.echo(f"{dataset_id}: extrinsics solve failed: {e}", err=True)
      continue

    rms = result.refined_rms_px
    threshold = ol_cfg.extrinsics_rms_threshold_px
    if rms > threshold:
      click.echo(click.style(
        f"{dataset_id}: WARN reprojection rms={rms:.3f}px > {threshold:.2f}px",
        fg="yellow"))
    else:
      click.echo(f"{dataset_id}: rms={rms:.3f}px  "
                 f"phase_offset={result.phase_offset}  "
                 f"reverse={result.reverse}")

    verify_dir = out_dir / "extrinsics_verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    overlay = render_extrinsics_overlay(
      frame, result.extrinsics, intrinsics.K, intrinsics.dist, ol_cfg,
      detection, result,
    )
    axes_overlay = render_world_axes(frame, result.extrinsics, intrinsics.K, intrinsics.dist)

    import cv2
    cv2.imwrite(str(verify_dir / "empty_scene_overlay.png"), overlay)
    cv2.imwrite(str(verify_dir / "world_axes.png"), axes_overlay)

    object_dir = verify_dir / "object_scenes"
    object_dir.mkdir(parents=True, exist_ok=True)
    for ep_label, ep_idx in (
      ("object_left",  ol_cfg.episode_object_left),
      ("object_mid",   ol_cfg.episode_object_mid),
      ("object_right", ol_cfg.episode_object_right),
    ):
      frames = sample_episode_frames(ds, ep_idx, ol_cfg.camera_key, n=1)
      if not frames:
        continue
      img = frames[0]
      ovl = render_extrinsics_overlay(
        img, result.extrinsics, intrinsics.K, intrinsics.dist, ol_cfg,
        detection=None, result=None,
      )
      cv2.imwrite(str(object_dir / f"{ep_label}.png"), ovl)

    _preview([
      verify_dir / "empty_scene_overlay.png",
      verify_dir / "world_axes.png",
    ])

    if not click.confirm(f"Accept this calibration for {dataset_id}?", default=False):
      click.echo(f"{dataset_id}: rejected; nothing written.")
      continue

    write_mat_annotation(cache_root, dataset_id, detection, extra={
      "annotation_frame": int(ol_cfg.extrinsics_annotation_frame),
      "episode_empty":    int(ol_cfg.episode_empty),
      "phase_offset":     int(result.phase_offset),
      "reverse":          bool(result.reverse),
    })
    write_extrinsics(cache_root, dataset_id, result.extrinsics, extra={
      "phase_offset":     int(result.phase_offset),
      "reverse":          bool(result.reverse),
      "seed_rms_px":      float(result.seed_rms_px),
      "annotation_source": "mat_annotation.json",
    })
    click.echo(f"{dataset_id}: wrote {out_dir / 'extrinsics.json'}")


def _preview(paths: list[Path]) -> None:
  """Best-effort matplotlib preview of one or more PNGs."""
  paths = [p for p in paths if p.exists()]
  if not paths:
    return
  try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    n = len(paths)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
      axes = [axes]
    for ax, p in zip(axes, paths):
      ax.imshow(mpimg.imread(str(p)))
      ax.set_title(p.name)
      ax.axis("off")
    plt.show()
  except Exception:
    pass
