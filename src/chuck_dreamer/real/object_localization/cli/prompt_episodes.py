"""`prompt-episodes` CLI: batch-collect first-frame object prompts.

Loops every (dataset, episode) implied by one or more CALIBRATION_SOURCE
specs, opens an OpenCV click window on each episode's first frame, and
saves the click to ``calibration_cache/<slug>/object_prompts.json`` via
``save_prompt``. Subsequent ``import-lerobot`` runs (with the same
processor) find the cached prompts and don't block on clicks.

The spec syntax is the same one used by ``calibrate-intrinsics`` (see
``calibration/spec_parser.py``); the ``:FRAMES`` part is ignored because
the prompt is always taken from the *first frame of each episode*.
"""
from __future__ import annotations

import logging
from pathlib import Path

import click

from chuck_dreamer.config import load_config

from ..calibration.spec_parser import (
  CalibrationSource, parse_calibration_source,
)
from ..dataset import all_episode_bounds_from_meta, get_frame
from ..prompts import click_object, load_prompt, save_prompt, sidecar_path
from ..runtime import init_from_config
from .common import override_option

logger = logging.getLogger(__name__)


@click.command("prompt-episodes")
@click.argument("sources", required=True, nargs=-1, metavar="DATASET_ID[#EPISODES]...")
@click.option("--frame", "frame_offset", default=0, type=int,
              help="Episode-relative frame index to use for the click "
                   "(default 0 = first frame). Use e.g. --frame 10 if the "
                   "object isn't visible in the very first frame of every "
                   "episode.")
@click.option("--force", is_flag=True, default=False,
              help="Re-click episodes that already have a cached prompt "
                   "(default: skip them).")
@click.option("--skip-empty", is_flag=True, default=True,
              help="Skip episodes whose metadata says length 0 (default on).")
@override_option
@click.pass_context
def prompt_episodes_cmd(ctx, sources: tuple[str, ...],
                        frame_offset: int, force: bool, skip_empty: bool,
                        overrides: tuple[str, ...]) -> None:
  """Click each episode's object once, save to object_prompts.json.

  Run this before a batch import so the import itself runs non-interactively
  in the background. Picks up where you left off if interrupted (already-
  clicked episodes are skipped unless --force).
  """
  cfg = load_config(ctx.obj["config_path"], overrides=overrides)
  ol_cfg = init_from_config(cfg)
  cache_root = Path(ol_cfg.cache_dir)

  try:
    parsed = [parse_calibration_source(s) for s in sources]
  except ValueError as e:
    raise click.ClickException(str(e))

  # Resolve each source into the list of (dataset_id, episode_index) tuples
  # to prompt for. Episode-relative FRAMES from the spec are ignored.
  pending: list[tuple[str, int]] = []
  for src in parsed:
    try:
      ep_bounds = all_episode_bounds_from_meta(src.dataset_id)
    except Exception as e:
      raise click.ClickException(
        f"could not read episode metadata for {src.dataset_id}: {e}")
    if src.episodes is None:
      eps = [ep for ep, _ in ep_bounds]
    else:
      ep_set = set(src.episodes)
      eps = [ep for ep, _ in ep_bounds if ep in ep_set]
      missing = ep_set - set(eps)
      if missing:
        click.echo(click.style(
          f"  WARN {src.dataset_id}: episodes {sorted(missing)} not in "
          f"dataset metadata; skipping.", fg="yellow"))
    for ep in eps:
      pending.append((src.dataset_id, ep))

  if not pending:
    raise click.ClickException("no (dataset, episode) pairs to prompt for.")

  # Filter to those that need a click.
  todo: list[tuple[str, int]] = []
  for did, ep in pending:
    if force:
      todo.append((did, ep)); continue
    if load_prompt(cache_root, did, ep) is None:
      todo.append((did, ep))

  skipped = len(pending) - len(todo)
  click.echo(f"\n{len(pending)} episodes total; {skipped} already cached, "
             f"{len(todo)} need a click.")
  if not todo:
    click.echo("nothing to do; --force to re-click.")
    return

  # Group consecutive same-dataset entries so we only construct each
  # LeRobotDataset once. Within a group we just step episodes.
  from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

  current_ds_id: str | None = None
  ds = None
  for i, (did, ep) in enumerate(todo, start=1):
    if did != current_ds_id:
      click.echo(f"\n[{i}/{len(todo)}] loading LeRobotDataset for {did} ...")
      ds = LeRobotDataset(did)
      current_ds_id = did
    try:
      bounds = next(b for e, b in all_episode_bounds_from_meta(did) if e == ep)
    except StopIteration:
      click.echo(f"  episode {ep}: no metadata; skipping.")
      continue
    ep_fr, ep_to = bounds
    ep_length = ep_to - ep_fr
    if skip_empty and ep_length <= 0:
      click.echo(f"  episode {ep}: empty; skipping.")
      continue
    global_idx = ep_fr + max(0, min(ep_length - 1, int(frame_offset)))
    rgb = get_frame(ds, global_idx, ol_cfg.camera_key)

    banner = f"[{i}/{len(todo)}]  {did}  episode {ep}  (frame {global_idx})"
    click.echo(f"  {banner}: click the object  (q to skip this episode)")
    prompt = click_object(rgb, banner=banner)
    if prompt is None:
      click.echo("    skipped.")
      continue
    save_prompt(cache_root, did, ep, prompt)
    click.echo(f"    saved: u={prompt[0]} v={prompt[1]}  -> "
               f"{sidecar_path(cache_root, did)}")

  click.echo("\nAll prompts gathered. Next step: run `import-lerobot` for these "
             "datasets; the processor will read the cached prompts and run "
             "non-interactively.")
