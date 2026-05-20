"""`pick-checkerboard-frames` CLI.

Walks through each provided dataset, opens an interactive cv2 window
to let the user select (frame, bbox) tuples around the checkerboard,
saves the picks to ``<cache>/<slug>/picks.json`` after each dataset,
and finally runs the pooled intrinsics fit using all picks.

The picks survive across invocations, so an interrupted session can
be resumed without losing prior selections.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import click

from chuck_dreamer.config import load_config

from ..calibration.intrinsics import (
  DatasetSpec, IntrinsicsAbort, calibrate_intrinsics_pooled,
)
from ..calibration.picker import pick_frames
from ..calibration.picks import load_picks, save_picks
from ..runtime import init_from_config
from ..types import dataset_cache_dir
from .common import override_option

logger = logging.getLogger(__name__)

_DEFAULT_WORKERS_CAP = 8


@click.command("pick-checkerboard-frames")
@click.argument("dataset_ids", nargs=-1, required=True)
@click.option("--workers", default=None, type=int,
              help=f"Workers for the pooled fit at the end. "
                   f"Default: min(os.cpu_count(), {_DEFAULT_WORKERS_CAP}).")
@click.option("--skip-fit", is_flag=True, default=False,
              help="Run the picker only; skip the pooled fit at the end. "
                   "Picks are still saved to picks.json.")
@click.option("--force", is_flag=True, default=False,
              help="Pass --force to the pooled fit (recompute even if "
                   "intrinsics.json already exists in every dataset's cache).")
@click.option("--no-gates", is_flag=True, default=False,
              help="Skip the centroid-bucket dedupe and the spatial-coverage / "
                   "min-frame abort gates. Use when picks are hand-curated and "
                   "you want every HIT to count toward the fit.")
@override_option
@click.pass_context
def pick_checkerboard_frames_cmd(ctx, dataset_ids: tuple[str, ...],
                                 workers: int | None, skip_fit: bool, force: bool,
                                 no_gates: bool,
                                 overrides: tuple[str, ...]) -> None:
  """Interactively pick checkerboard frames per dataset, then pool-fit intrinsics.

  For each DATASET_ID an OpenCV window opens on the configured
  checkerboard episode. Use arrow keys to scrub frames, draw a bbox
  around the checkerboard, Enter to add it, Tab to advance to the next
  dataset. Prior picks load from ``<cache>/<slug>/picks.json``.
  """
  cfg = load_config(ctx.obj["config_path"], overrides=overrides)
  ol_cfg = init_from_config(cfg)
  cache_root = Path(ol_cfg.cache_dir)
  ids = list(dataset_ids)

  all_picks: dict[str, list] = {}
  for i, dataset_id in enumerate(ids):
    click.echo(f"\n[{i + 1}/{len(ids)}] {dataset_id}")
    existing = load_picks(cache_root, dataset_id)
    if existing:
      click.echo(f"  loaded {len(existing)} previously-saved picks "
                 f"from {dataset_cache_dir(cache_root, dataset_id) / 'picks.json'}")

    result = pick_frames(
      dataset_id, ol_cfg.camera_key, ol_cfg.episode_checkerboard,
      initial_picks=existing,
    )
    save_picks(cache_root, dataset_id, result.picks)
    click.echo(f"  saved {len(result.picks)} picks to "
               f"{dataset_cache_dir(cache_root, dataset_id) / 'picks.json'}")
    all_picks[dataset_id] = result.picks

    if not result.advance and not _last_dataset(i, len(ids)):
      click.echo("  picker exited via q/Esc; remaining datasets skipped. "
                 "(Re-run to continue; picks are persisted.)")
      break

  if skip_fit:
    click.echo("\n--skip-fit set; not running pooled fit. "
               "Re-run without --skip-fit, or:")
    click.echo("  python main.py calibrate-intrinsics " + " ".join(ids) + " --force")
    return

  # Build DatasetSpec list from picks. Skip datasets with no picks.
  specs: list[DatasetSpec] = []
  for did in ids:
    picks = all_picks.get(did) or load_picks(cache_root, did)
    if not picks:
      click.echo(f"  WARN: {did} has no picks; excluding from fit.")
      continue
    picks_sorted = sorted(picks, key=lambda p: p.frame_idx)
    specs.append(DatasetSpec(
      dataset_id    = did,
      frame_indices = [p.frame_idx for p in picks_sorted],
      frame_bboxes  = [p.bbox for p in picks_sorted],
    ))
  if not specs:
    raise click.ClickException("no datasets ended up with any picks; nothing to fit.")

  total_picks = sum(len(s.frame_indices) for s in specs)
  if workers is None:
    workers = min(os.cpu_count() or 1, _DEFAULT_WORKERS_CAP)
  workers = max(1, int(workers))
  click.echo(f"\nPooled fit on {total_picks} picks across {len(specs)} datasets "
             f"({workers} workers) ...")

  t0 = time.perf_counter()
  try:
    result = calibrate_intrinsics_pooled(
      specs, ol_cfg, cache_root, workers=workers, no_gates=no_gates,
    )
  except IntrinsicsAbort as e:
    raise click.ClickException(str(e))

  rms = result.intrinsics.rms_px
  msg = (f"Pooled fit: rms={rms:.3f}px  "
         f"views_used={result.intrinsics.n_frames_used}  "
         f"coverage={result.coverage_filled_cells}/{result.coverage_total_cells} cells  "
         f"elapsed={time.perf_counter() - t0:.1f}s")
  if rms > ol_cfg.intrinsics_rms_threshold_px:
    click.echo(click.style(
      msg + f"  WARN: rms > {ol_cfg.intrinsics_rms_threshold_px:.2f}px",
      fg="yellow"))
  else:
    click.echo(msg)

  click.echo("Per-dataset view contribution:")
  for did, counts in result.per_dataset.items():
    n_used = sum(1 for d in result.detections if d.dataset_id == did)
    click.echo(f"  {did}: views_used={n_used}  "
               f"(attempts={counts.attempts} hits={counts.hits} "
               f"misses={counts.misses} dupes={counts.dupes})")
    out_dir = dataset_cache_dir(cache_root, did)
    click.echo(f"    wrote: {out_dir / 'intrinsics.json'}")
    click.echo(f"    verify: {out_dir / 'intrinsics_verify'}")


def _last_dataset(i: int, n: int) -> bool:
  return i == n - 1
