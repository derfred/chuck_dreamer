"""`calibrate-intrinsics` CLI: pooled checkerboard calibration across datasets.

This command always pools all dataset arguments into one calibration —
the supplied recordings each fix the checkerboard in a single pose for
the whole episode, so per-dataset calibration is degenerate. The same
``intrinsics.json`` is written into each dataset's cache dir so the
downstream per-dataset loader contract still works.
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
from ..runtime import init_from_config
from ..types import dataset_cache_dir, read_intrinsics
from .common import override_option

logger = logging.getLogger(__name__)

_DEFAULT_WORKERS_CAP = 8


@click.command("calibrate-intrinsics")
@click.argument("dataset_specs", nargs=-1, required=True, metavar="DATASET_ID[#FRAME,FRAME,...]")
@click.option("--force", is_flag=True, default=False,
              help="Recompute even if intrinsics.json already exists in every "
                   "dataset's cache dir.")
@click.option("--workers", default=None, type=int,
              help=f"Parallel worker count for frame decode + corner detection. "
                   f"Default: min(os.cpu_count(), {_DEFAULT_WORKERS_CAP}).")
@click.option("--no-show", is_flag=True, default=False,
              help="Suppress interactive display; PNGs are written either way.")
@click.option("--no-gates", is_flag=True, default=False,
              help="Skip the centroid-bucket dedupe and the spatial-coverage / "
                   "min-frame abort gates. Use when frames are hand-curated "
                   "and you want every HIT to count toward the fit.")
@override_option
@click.pass_context
def calibrate_intrinsics_cmd(ctx, dataset_specs: tuple[str, ...],
                             force: bool, workers: int | None, no_show: bool,
                             no_gates: bool,
                             overrides: tuple[str, ...]) -> None:
  """Pool the checkerboard episodes of all DATASET_SPECS into one calibration.

  Each DATASET_SPEC is either a bare ``user/dataset`` (the episode is
  enumerated normally) or ``user/dataset#f1,f2,f3`` to skip enumeration
  and use exactly those global frame indices. The ``#`` form is the
  resume-after-crash shortcut: each successful run writes
  ``<cache>/<slug>/intrinsics_hits.jsonl``; grep the frame_idx values
  out and feed them back in next time.
  """
  cfg = load_config(ctx.obj["config_path"], overrides=overrides)
  ol_cfg = init_from_config(cfg)
  cache_root = Path(ol_cfg.cache_dir)
  try:
    specs = [_parse_dataset_spec(s) for s in dataset_specs]
  except ValueError as e:
    raise click.ClickException(str(e))
  ids = [s.dataset_id for s in specs]

  if workers is None:
    workers = min(os.cpu_count() or 1, _DEFAULT_WORKERS_CAP)
  workers = max(1, int(workers))

  # Idempotent short-circuit: if every dataset already has a cached
  # intrinsics.json and --force isn't set, just confirm and exit.
  if not force:
    cached_all = True
    for did in ids:
      p = dataset_cache_dir(cache_root, did) / "intrinsics.json"
      if not p.exists():
        cached_all = False
        break
    if cached_all:
      sample = read_intrinsics(cache_root, ids[0])
      click.echo(f"All {len(ids)} datasets already have intrinsics.json "
                 f"(rms={sample.rms_px:.3f}px from {ids[0]}). "
                 f"Use --force to recompute.")
      return

  click.echo(f"Pooling {len(ids)} datasets with {workers} workers. "
             f"Target: {ol_cfg.intrinsics_n_frames_target} distinct viewpoint "
             f"buckets (min {ol_cfg.intrinsics_n_frames_min}). "
             f"Streaming progress via INFO logs.")
  for spec in specs:
    if spec.frame_indices is not None:
      click.echo(f"  - {spec.dataset_id}  (using {len(spec.frame_indices)} "
                 f"explicit frames: {_summarize_frames(spec.frame_indices)})")
    else:
      click.echo(f"  - {spec.dataset_id}")

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

  if not no_show:
    _show_artifacts(dataset_cache_dir(cache_root, ids[0]) / "intrinsics_verify" / "coverage_mosaic.png")


def _parse_dataset_spec(raw: str) -> DatasetSpec:
  """Parse ``user/dataset[#f1,f2,...]`` into a DatasetSpec.

  Frame indices, when given, must be comma-separated non-negative
  integers. Duplicates are de-duped (order preserved). Anything that
  isn't a number raises ValueError with a message naming the bad token.
  """
  if "#" not in raw:
    return DatasetSpec(dataset_id=raw, frame_indices=None)
  dataset_id, _, frames_str = raw.partition("#")
  if not dataset_id:
    raise ValueError(f"dataset spec {raw!r}: empty dataset id before '#'.")
  if not frames_str.strip():
    raise ValueError(f"dataset spec {raw!r}: '#' present but no frame indices follow.")
  indices: list[int] = []
  seen: set[int] = set()
  for tok in frames_str.split(","):
    tok = tok.strip()
    if not tok:
      continue
    try:
      n = int(tok)
    except ValueError:
      raise ValueError(f"dataset spec {raw!r}: frame index {tok!r} is not an integer.")
    if n < 0:
      raise ValueError(f"dataset spec {raw!r}: frame index {n} is negative.")
    if n in seen:
      continue
    seen.add(n)
    indices.append(n)
  if not indices:
    raise ValueError(f"dataset spec {raw!r}: no usable frame indices.")
  return DatasetSpec(dataset_id=dataset_id, frame_indices=indices)


def _summarize_frames(indices: list[int]) -> str:
  if len(indices) <= 6:
    return ", ".join(str(i) for i in indices)
  return (", ".join(str(i) for i in indices[:3])
          + f", ... ({len(indices) - 6} more), "
          + ", ".join(str(i) for i in indices[-3:]))


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
    plt.title(str(path))
    plt.axis("off")
    plt.show()
  except Exception:
    pass
