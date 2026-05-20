"""`calibrate-intrinsics` CLI: fit intrinsics from dedicated calibration recordings.

Takes one or more CALIBRATION_SOURCE specs of the form
``DATASET_ID[#EPISODES[:FRAMES]]``. Sources are pooled into a single
``cv2.calibrateCamera`` fit; the resulting ``intrinsics.json`` is
written into every ``--apply-to`` dataset's cache dir so downstream
pose code keeps using the same per-dataset loader contract.

If auto-sampling produces fewer than the configured ``n_frames_min``
HITs, the command offers to launch the bbox picker on each source;
the picker's selections then feed back into the fit.

Example: pool two takes of the same camera, restrict the second to
episodes 0-1, and apportion the default 64-frame budget across them:

  python main.py calibrate-intrinsics \\
    user/cal_takeA  user/cal_takeB#0-1  --force

Spec grammar (full reference in calibration/spec_parser.py). Frame
indices in the ``:FRAMES`` part are **episode-relative**: `0-80` means
"frames 0 through 80 of each selected episode", not global indices.

  DATASET_ID                          - all episodes, auto-sampled
  DATASET_ID#0,2-4                    - episodes 0, 2, 3, 4
  DATASET_ID#all:0-200                - all eps, frames 0..200 of each
  DATASET_ID#1:0-200:5                - episode 1, frames 0,5,...,200
  DATASET_ID#0-1:0,50-100:10          - eps 0+1, frame 0 + 50,60,...,100 of each
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
from ..calibration.spec_parser import (
  CalibrationSource, parse_calibration_source,
)
from ..dataset import all_episode_bounds_from_meta
from ..runtime import init_from_config
from ..types import dataset_cache_dir, dataset_slug, read_intrinsics
from .common import override_option

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_WORKERS_CAP = 8


@click.command("calibrate-intrinsics")
@click.argument("calibration_sources", required=True, nargs=-1,
                metavar="CALIBRATION_SOURCE...")
@click.option("--apply-to", "apply_to", multiple=True, metavar="DATASET_ID",
              help="Target dataset id to receive the intrinsics. Repeatable. "
                   "If omitted, applies to every directory already under "
                   "cache_dir (i.e. every dataset you've already touched).")
@click.option("--n-frames", default=64, type=int,
              help="How many frames to sample, total, across all selected "
                   "calibration sources (default 64). Apportioned by source "
                   "and within each source by episode length. Sources that "
                   "specify an explicit ``:FRAMES`` range bypass this budget.")
@click.option("--force", is_flag=True, default=False,
              help="Recompute even if intrinsics.json already exists in every "
                   "target dataset's cache dir.")
@click.option("--workers", default=None, type=int,
              help=f"Parallel worker count for frame decode + corner detection. "
                   f"Default: min(os.cpu_count(), {_DEFAULT_WORKERS_CAP}).")
@click.option("--no-show", is_flag=True, default=False,
              help="Suppress interactive display; PNGs are written either way.")
@click.option("--no-gates", is_flag=True, default=False,
              help="Skip the centroid-bucket dedupe and the spatial-coverage / "
                   "min-frame abort gates.")
@click.option("--picker/--no-picker", default=True,
              help="If auto-sampling yields fewer than n_frames_min HITs, "
                   "launch the bbox picker on each source and re-fit "
                   "(default: enabled). Disable to fail loudly.")
@override_option
@click.pass_context
def calibrate_intrinsics_cmd(ctx, calibration_sources: tuple[str, ...],
                             apply_to: tuple[str, ...],
                             n_frames: int,
                             force: bool, workers: int | None, no_show: bool,
                             no_gates: bool, picker: bool,
                             overrides: tuple[str, ...]) -> None:
  """Pool checkerboard frames from one or more CALIBRATION_SOURCE specs
  into a single intrinsics fit.

  Each CALIBRATION_SOURCE is ``DATASET_ID[#EPISODES[:FRAMES]]``. See the
  module docstring for the full grammar. Multiple sources are accepted
  positionally; their HITs are pooled before ``cv2.calibrateCamera``.
  """
  cfg = load_config(ctx.obj["config_path"], overrides=overrides)
  ol_cfg = init_from_config(cfg)
  cache_root = Path(ol_cfg.cache_dir)

  # Parse all CLI specs up front so syntax errors fire before any work.
  try:
    sources = [parse_calibration_source(s) for s in calibration_sources]
  except ValueError as e:
    raise click.ClickException(str(e))

  targets = list(apply_to) if apply_to else _discover_existing_targets(cache_root)
  if not targets:
    raise click.ClickException(
      "no --apply-to targets provided and no cached datasets found under "
      f"{cache_root}. Pass --apply-to USER/DATASET for each target.")

  if workers is None:
    workers = min(os.cpu_count() or 1, _DEFAULT_WORKERS_CAP)
  workers = max(1, int(workers))

  # Resolve each source: read its episodes from metadata, intersect with
  # the explicit episode/frame filters in the spec, and apportion the
  # global n_frames budget across sources without explicit frame lists.
  click.echo(f"Calibration sources ({len(sources)}):")
  resolved = _resolve_sources(sources, n_frames)
  for did, info in resolved.items():
    eps   = info["episodes"]
    n     = len(info["frame_indices"])
    label = "explicit" if info["explicit_frames"] else "auto"
    click.echo(f"  {did}  episodes={eps}  frames={n} ({label})")

  total_frames = sum(len(info["frame_indices"]) for info in resolved.values())
  if total_frames == 0:
    raise click.ClickException(
      "no frames available across any source after resolving spec.")
  click.echo(f"  total queue: {total_frames} frames across {len(resolved)} sources")

  ol_cfg_calib = _ol_cfg_with_episode(
    ol_cfg, _first_episode(resolved, sources[0].dataset_id), n_frames,
  )

  # Idempotency: short-circuit only if every target already has cached
  # intrinsics and --force isn't set.
  if not force:
    if all((dataset_cache_dir(cache_root, did) / "intrinsics.json").exists()
           for did in targets):
      sample = read_intrinsics(cache_root, targets[0])
      click.echo(f"All {len(targets)} target datasets already have "
                 f"intrinsics.json (rms={sample.rms_px:.3f}px from "
                 f"{targets[0]}). Use --force to recompute.")
      return

  click.echo(f"Applying to {len(targets)} dataset(s):")
  for did in targets:
    click.echo(f"  - {did}")

  specs = [
    DatasetSpec(dataset_id=did, frame_indices=info["frame_indices"])
    for did, info in resolved.items()
  ]
  result = _try_fit(specs, ol_cfg_calib, cache_root, workers, no_gates,
                    targets=targets)

  # Picker fallback: too few HITs to fit.
  if result is None:
    if not picker:
      raise click.ClickException(
        "auto-sampling yielded too few HITs and --no-picker disabled "
        "the fallback. Add more sources, or re-record with more "
        "board diversity.")
    click.echo()
    click.echo(click.style(
      "auto-sampling did not produce enough usable detections.",
      fg="yellow"))
    if not click.confirm("Launch the bbox picker on each source?",
                          default=True):
      raise click.ClickException("aborted; re-run with --no-picker to skip the prompt.")

    specs = _picker_fallback_specs(resolved, ol_cfg, cache_root)
    if not specs:
      raise click.ClickException(
        "picker returned no picks; aborting calibration.")
    click.echo(f"  re-fitting with picker-curated specs ...")
    result = _try_fit(specs, ol_cfg_calib, cache_root, workers, no_gates,
                      targets=targets)
    if result is None:
      raise click.ClickException(
        "fit still failed after picker; check that the calibration "
        "sources actually show the checkerboard.")

  _report(result, ol_cfg, cache_root, targets)
  if not no_show:
    _show_artifacts(dataset_cache_dir(cache_root, targets[0])
                    / "intrinsics_verify" / "coverage_mosaic.png")


def _resolve_sources(sources: list[CalibrationSource], n_frames_budget: int
                     ) -> dict[str, dict]:
  """Resolve each CalibrationSource into a per-dataset frame list.

  Returns a dict keyed by dataset_id (preserving CLI order) with:
    - 'episodes':         list of episode indices selected (for display)
    - 'frame_indices':    sorted list of global frame indices to sample
    - 'explicit_frames':  bool, True iff the spec gave an explicit
                          FRAMES range (so n_frames_budget is ignored
                          for this source)

  Sources without explicit FRAMES share the global ``n_frames_budget``
  proportional to their total selected-episode length.
  """
  # First pass: discover bounds + episodes per source.
  per_source_episodes: dict[str, list[tuple[int, tuple[int, int]]]] = {}
  per_source_explicit: dict[str, list[int] | None] = {}
  for src in sources:
    try:
      ep_bounds = all_episode_bounds_from_meta(src.dataset_id)
    except Exception as e:
      raise click.ClickException(
        f"could not read episode metadata for {src.dataset_id}: {e}")
    if src.episodes is None:
      selected = list(ep_bounds)
    else:
      wanted = set(src.episodes)
      selected = [(ep, bounds) for ep, bounds in ep_bounds if ep in wanted]
      missing = wanted - {ep for ep, _ in selected}
      if missing:
        click.echo(click.style(
          f"  WARN {src.dataset_id}: episodes {sorted(missing)} not in "
          f"dataset metadata; skipping.", fg="yellow"))
    if not selected:
      raise click.ClickException(
        f"{src.dataset_id}: no episodes resolved from spec.")
    per_source_episodes[src.dataset_id] = selected
    per_source_explicit[src.dataset_id] = src.frames

  # Compute per-source length only for the sources that need to share the
  # budget (no explicit frames given).
  unconstrained_lengths = {
    did: sum(to - fr for _, (fr, to) in eps)
    for did, eps in per_source_episodes.items()
    if per_source_explicit[did] is None
  }
  total_unconstrained = sum(unconstrained_lengths.values())

  out: dict[str, dict] = {}
  for did, eps in per_source_episodes.items():
    explicit = per_source_explicit[did]
    if explicit is not None:
      # Episode-relative semantics: each entry in `explicit` is an offset
      # from each selected episode's start. Translate to global indices
      # and clip to that episode's bounds.
      keep: set[int] = set()
      for ep, (fr, to) in eps:
        ep_length = to - fr
        for rel in explicit:
          if 0 <= rel < ep_length:
            keep.add(fr + rel)
      frame_indices = sorted(keep)
    else:
      length = unconstrained_lengths[did]
      if total_unconstrained <= 0 or length <= 0:
        per_source_share = 0
      else:
        per_source_share = max(1,
          int(round(n_frames_budget * length / total_unconstrained)))
      per_ep = _apportion_frames(eps, per_source_share)
      frame_indices = sorted({i for v in per_ep.values() for i in v})
    out[did] = {
      "episodes":        [ep for ep, _ in eps],
      "frame_indices":   frame_indices,
      "explicit_frames": explicit is not None,
    }
  return out


def _first_episode(resolved: dict[str, dict], first_did: str) -> int:
  eps = resolved[first_did]["episodes"]
  return int(eps[0]) if eps else 0


def _picker_fallback_specs(resolved: dict[str, dict],
                            ol_cfg, cache_root: Path) -> list[DatasetSpec]:
  """Loop the picker over every (source, episode) tuple; build new
  DatasetSpecs from the accumulated picks.
  """
  specs: list[DatasetSpec] = []
  for did, info in resolved.items():
    existing = load_picks(cache_root, did)
    if existing:
      click.echo(f"  loaded {len(existing)} previously-saved picks for {did}")
    picks_by_frame = {p.frame_idx: p for p in existing}
    abort = False
    for ep in info["episodes"]:
      click.echo(f"  picker: {did} episode {ep} ...")
      pr = pick_frames(
        did, ol_cfg.camera_key, ep,
        initial_picks=list(picks_by_frame.values()),
      )
      for p in pr.picks:
        picks_by_frame[p.frame_idx] = p
      save_picks(cache_root, did, list(picks_by_frame.values()))
      if not pr.advance:
        click.echo("  picker exited via q/Esc; skipping remaining episodes.")
        abort = True
        break
    if not picks_by_frame:
      continue
    picks_sorted = sorted(picks_by_frame.values(), key=lambda p: p.frame_idx)
    specs.append(DatasetSpec(
      dataset_id    = did,
      frame_indices = [p.frame_idx for p in picks_sorted],
      frame_bboxes  = [p.bbox for p in picks_sorted],
    ))
    if abort:
      break
  return specs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_fit(specs, ol_cfg, cache_root, workers, no_gates, targets):
  """Run the pooled fit; on IntrinsicsAbort (min-frames or coverage), return
  None so the caller can decide what to do (picker fallback / give up).
  """
  t0 = time.perf_counter()
  try:
    result = calibrate_intrinsics_pooled(
      specs, ol_cfg, cache_root, workers=workers, no_gates=no_gates,
    )
  except IntrinsicsAbort as e:
    logger.warning("[calibrate-intrinsics] fit aborted: %s", e)
    return None
  # Pooled fit writes intrinsics.json only into the calibration dataset's
  # cache dir (which is the only entry in `specs`). Copy it into every
  # `--apply-to` target so downstream load_calibration() works there too.
  from ..types import write_intrinsics
  for did in targets:
    if did == specs[0].dataset_id:
      continue
    write_intrinsics(cache_root, did, result.intrinsics, extra={
      "source_dataset":   specs[0].dataset_id,
      "per_view_rms_px":  [float(x) for x in result.per_view_rms],
      "rms_threshold_px": float(ol_cfg.intrinsics_rms_threshold_px),
      "view_provenance":  [
        {"dataset_id": d.dataset_id, "frame_idx": d.frame_idx,
         "bucket": list(d.bucket)} for d in result.detections
      ],
    })
  result._elapsed_s = time.perf_counter() - t0  # used by _report
  return result


def _report(result, ol_cfg, cache_root, targets):
  rms = result.intrinsics.rms_px
  msg = (f"Pooled fit: rms={rms:.3f}px  "
         f"views_used={result.intrinsics.n_frames_used}  "
         f"coverage={result.coverage_filled_cells}/{result.coverage_total_cells} cells  "
         f"elapsed={getattr(result, '_elapsed_s', 0.0):.1f}s")
  if rms > ol_cfg.intrinsics_rms_threshold_px:
    click.echo(click.style(
      msg + f"  WARN: rms > {ol_cfg.intrinsics_rms_threshold_px:.2f}px",
      fg="yellow"))
  else:
    click.echo(msg)
  click.echo("Applied to:")
  for did in targets:
    out_dir = dataset_cache_dir(cache_root, did)
    click.echo(f"  {did}  -> {out_dir / 'intrinsics.json'}")


def _apportion_frames(selected: list[tuple[int, tuple[int, int]]],
                      n_frames: int) -> dict[int, list[int]]:
  """Apportion ``n_frames`` evenly across ``selected`` episodes,
  proportional to each episode's length, and pick that many globally
  indexed frames per episode via np.linspace.
  """
  total_length = sum(to - fr for _, (fr, to) in selected)
  out: dict[int, list[int]] = {}
  if total_length == 0:
    return out
  # First pass: float share per episode.
  shares: dict[int, float] = {
    ep: n_frames * (to - fr) / total_length
    for ep, (fr, to) in selected
  }
  # Round to integers, then redistribute leftover frames largest-fraction first.
  assigned: dict[int, int] = {ep: int(s) for ep, s in shares.items()}
  leftover = n_frames - sum(assigned.values())
  ordered_by_frac = sorted(shares.items(), key=lambda kv: kv[1] - int(kv[1]),
                            reverse=True)
  for i in range(leftover):
    ep, _ = ordered_by_frac[i % len(ordered_by_frac)]
    assigned[ep] += 1
  for ep, (fr, to) in selected:
    k = max(1, min(to - fr, assigned[ep]))
    if to - fr <= k:
      out[ep] = list(range(fr, to))
    else:
      out[ep] = sorted({int(x) for x in np.linspace(fr, to - 1, num=k)})
  return out


def _discover_existing_targets(cache_root: Path) -> list[str]:
  """Discover datasets already present under cache_root.

  Sources, in order of reliability:
    - picks.json (always carries dataset_id)
    - intrinsics.json's source_datasets list (legacy pooled fits)
  The cache dir's slugged name can't be inverted reliably, so we ignore
  it. Returns a deduped, sorted list.
  """
  import json
  out: list[str] = []
  if not cache_root.exists():
    return out
  for sub in sorted(cache_root.iterdir()):
    if not sub.is_dir():
      continue
    picks = sub / "picks.json"
    intr  = sub / "intrinsics.json"
    if picks.exists():
      try:
        blob = json.loads(picks.read_text())
        did  = blob.get("dataset_id")
        if did and did not in out:
          out.append(did)
          continue
      except Exception:
        pass
    if intr.exists():
      try:
        blob = json.loads(intr.read_text())
        for did in (blob.get("source_datasets") or []):
          if did and did not in out:
            out.append(did)
      except Exception:
        continue
  return sorted(out)


def _ol_cfg_with_episode(ol_cfg, calibration_episode: int, n_frames_target: int):
  """Return a shallow copy of ol_cfg with episode_checkerboard +
  intrinsics_n_frames_target overridden for one fit.
  """
  from dataclasses import replace
  return replace(
    ol_cfg,
    episode_checkerboard       = int(calibration_episode),
    intrinsics_n_frames_target = int(n_frames_target),
  )


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
