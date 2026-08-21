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
from .store_sync import (
  camera_id_from_key,
  get_intrinsics,
  put_extrinsics,
  require_store,
)
from chuck_dreamer.perception.config import init_from_config
from chuck_dreamer.perception.types import Intrinsics, MatDetection
from chuck_dreamer.store import MissingArtifact, for_dataset
from chuck_dreamer.cli import override_option

logger = logging.getLogger(__name__)


@click.command("annotate-mat")
@click.argument("dataset_ids", nargs=-1, required=True)
@click.option("--force", is_flag=True, default=False,
              help="Re-annotate even if extrinsics are already in the store.")
@click.option("--review", is_flag=True, default=False,
              help="Skip the click UI; reload the stored clicks, re-solve, "
                   "re-render verification, prompt to overwrite.")
@override_option
@click.pass_context
def annotate_mat_cmd(ctx, dataset_ids: tuple[str, ...],
                     force: bool, review: bool,
                     overrides: tuple[str, ...]) -> None:
  """Annotate the mat fiducials and solve per-dataset extrinsics."""
  cfg = load_config(ctx.obj["config_path"], overrides=overrides)
  ol_cfg = init_from_config(cfg)
  try:
    store = require_store(cfg, "annotate-mat")
  except RuntimeError as e:
    raise click.ClickException(str(e))
  camera_id = camera_id_from_key(ol_cfg.camera_key)

  for dataset_id in dataset_ids:
    ds_scope = for_dataset(dataset_id)
    stored_extr = (store.get("extrinsics", ds_scope)[0]
                   if store.has("extrinsics", ds_scope) else None)

    if stored_extr is not None and not force and not review:
      rms = stored_extr.get("rms_px")
      click.echo(f"{dataset_id}: extrinsics already in store"
                 f"{f' (rms={rms:.2f}px)' if rms is not None else ''},"
                 f" skipping. (use --force or --review)")
      continue

    try:
      intr_payload, intr_record = get_intrinsics(store, camera_id)
    except MissingArtifact as e:
      raise click.ClickException(str(e))
    intrinsics = Intrinsics.from_json(intr_payload)

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
      mat_blob = (stored_extr or {}).get("mat_annotation")
      if mat_blob is None:
        raise click.ClickException(
          f"{dataset_id}: --review requires stored extrinsics with embedded "
          "clicks; run annotate-mat without --review first.")
      detection = MatDetection.from_json(mat_blob)
      click.echo(f"{dataset_id}: re-using the stored fiducial clicks.")
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

    verify_dir = Path("extrinsics_verify") / dataset_id.replace("/", "__")
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

    payload = result.extrinsics.to_json()
    payload.update({
      "phase_offset": int(result.phase_offset),
      "reverse":      bool(result.reverse),
      "seed_rms_px":  float(result.seed_rms_px),
      # Raw fiducial clicks embedded so the fit can be reviewed / re-run
      # without re-clicking (spec §5).
      "mat_annotation": {
        **detection.to_json(),
        "meta": {
          "annotation_frame": int(ol_cfg.extrinsics_annotation_frame),
          "episode_empty":    int(ol_cfg.episode_empty),
          "phase_offset":     int(result.phase_offset),
          "reverse":          bool(result.reverse),
        },
      },
    })
    # The intrinsics the fit ran against: advisory only — a re-calibration
    # never makes the accepted extrinsics stale, it just suggests re-running
    # this tool (spec §9.3).
    put_extrinsics(store, dataset_id, payload, camera_id=camera_id,
                   advisory={"intrinsics": intr_record.payload_hash})


@click.command("annotate-mat-jpeg")
@click.argument("image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--calib", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=Path("store/camera/front/intrinsics.json"), show_default=True,
              help="intrinsics.json (top-level 'K'/'dist').")
@click.option("--out", type=click.Path(file_okay=False, path_type=Path),
              default=Path("annotate_mat_out"), show_default=True,
              help="Output directory for overlays + extrinsics.json.")
@override_option
@click.pass_context
def annotate_mat_jpeg_cmd(ctx, image: Path, calib: Path, out: Path,
                          overrides: tuple[str, ...]) -> None:
  """Run the mat-annotation UI on a single JPEG and solve extrinsics.

  A calibration scratch tool: unlike ``annotate-mat`` it neither reads a
  LeRobot dataset nor writes the calibration cache / artifact store — the
  solved extrinsics land in --out for inspection."""
  import json

  import cv2
  import numpy as np

  cfg = load_config(ctx.obj["config_path"], overrides=overrides)
  ol_cfg = init_from_config(cfg)

  bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
  if bgr is None:
    raise click.ClickException(f"could not read {image}")
  rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

  blob = json.loads(calib.read_text())
  K = np.asarray(blob["K"], dtype=np.float64)
  dist = np.asarray(blob["dist"], dtype=np.float64)

  reset_popup_state()
  try:
    detection = annotate_mat_cv2(rgb)
  finally:
    close_popup_window()
  if detection is None:
    raise click.ClickException("annotation aborted")

  result = solve_extrinsics(detection, K, dist, ol_cfg)
  click.echo(f"rms={result.refined_rms_px:.3f}px  "
             f"phase_offset={result.phase_offset}  reverse={result.reverse}")

  out.mkdir(parents=True, exist_ok=True)
  overlay = render_extrinsics_overlay(rgb, result.extrinsics, K, dist, ol_cfg,
                                      detection, result)
  axes = render_world_axes(rgb, result.extrinsics, K, dist)
  cv2.imwrite(str(out / "overlay.png"), overlay)
  cv2.imwrite(str(out / "world_axes.png"), axes)
  (out / "extrinsics.json").write_text(json.dumps({
    "R": result.extrinsics.R.tolist(),
    "t": result.extrinsics.t.tolist(),
    "rms_px": float(result.refined_rms_px),
    "phase_offset": int(result.phase_offset),
    "reverse": bool(result.reverse),
  }, indent=2))
  click.echo(f"wrote {out}/")


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
