"""`calibrate-intrinsics` CLI: per-dataset checkerboard calibration."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import click

from chuck_dreamer.config import load_config

from ..calibration.intrinsics import IntrinsicsAbort, calibrate_intrinsics
from ..runtime import init_from_config
from ..types import dataset_cache_dir, read_intrinsics

logger = logging.getLogger(__name__)


@click.command("calibrate-intrinsics")
@click.argument("dataset_ids", nargs=-1, required=True)
@click.option("--config", "config_path", default=None, type=str,
              help="Path to project config (default: configs/default.yaml).")
@click.option("-o", "--override", "overrides", multiple=True, metavar="KEY=VALUE",
              help="Dotted-path config override, repeatable.")
@click.option("--force", is_flag=True, default=False,
              help="Recompute even if intrinsics.json already exists.")
@click.option("--no-show", is_flag=True, default=False,
              help="Suppress interactive display; PNGs are written either way.")
def calibrate_intrinsics_cmd(dataset_ids: tuple[str, ...], config_path: Optional[str],
                             overrides: tuple[str, ...], force: bool, no_show: bool) -> None:
  """Calibrate camera intrinsics from each dataset's checkerboard episode."""
  cfg = load_config(config_path, overrides=overrides)
  ol_cfg = init_from_config(cfg)
  cache_root = Path(ol_cfg.cache_dir)

  any_failed = False
  for dataset_id in dataset_ids:
    out_dir   = dataset_cache_dir(cache_root, dataset_id)
    intr_path = out_dir / "intrinsics.json"
    if intr_path.exists() and not force:
      try:
        cached = read_intrinsics(cache_root, dataset_id)
        click.echo(f"{dataset_id}: intrinsics already cached "
                   f"(rms={cached.rms_px:.2f}px), skipping. "
                   f"(use --force to recompute)")
        verify_dir = out_dir / "intrinsics_verify"
        if not verify_dir.exists():
          click.echo(f"  note: verification artifacts missing at {verify_dir}; "
                     f"re-run with --force to regenerate them.")
        continue
      except Exception as e:
        click.echo(f"{dataset_id}: cached intrinsics unreadable ({e}); recomputing.")

    click.echo(f"{dataset_id}: loading LeRobotDataset (episode "
               f"{ol_cfg.episode_checkerboard}) ...")
    try:
      from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except Exception as e:
      raise click.ClickException(f"lerobot not importable: {e}")
    t_load = time.perf_counter()
    ds = LeRobotDataset(dataset_id)
    click.echo(f"{dataset_id}:   dataset object built in "
               f"{time.perf_counter() - t_load:.1f}s "
               f"(total frames={len(ds)}). Starting corner detection — see INFO logs.")

    try:
      result = calibrate_intrinsics(ds, ol_cfg, cache_root, dataset_id, force=force)
    except IntrinsicsAbort as e:
      click.echo(f"{dataset_id}: ABORT — {e}", err=True)
      any_failed = True
      continue

    rms = result.intrinsics.rms_px
    msg = (f"{dataset_id}: rms={rms:.3f}px  "
           f"frames_used={result.intrinsics.n_frames_used}  "
           f"coverage={result.coverage_filled_cells}/{result.coverage_total_cells} cells")
    if rms > ol_cfg.intrinsics_rms_threshold_px:
      click.echo(click.style(msg + f"  WARN: rms > {ol_cfg.intrinsics_rms_threshold_px:.2f}px",
                              fg="yellow"))
    else:
      click.echo(msg)
    click.echo(f"  wrote: {out_dir / 'intrinsics.json'}")
    click.echo(f"  verify: {out_dir / 'intrinsics_verify'}")

    if not no_show:
      _show_artifacts(out_dir / "intrinsics_verify" / "coverage_mosaic.png")

  if any_failed:
    raise click.ClickException("one or more datasets failed; see messages above.")


def _show_artifacts(path: Path) -> None:
  """Best-effort interactive preview of a single PNG."""
  if not path.exists():
    return
  try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread(str(path))
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(str(path.name))
    plt.axis("off")
    plt.show()
  except Exception:
    pass
