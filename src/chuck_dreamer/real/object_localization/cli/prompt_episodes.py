"""`prompt-episodes` CLI: batch-collect object prompts at episode keyframes.

For every (dataset, episode) implied by one or more CALIBRATION_SOURCE
specs, opens an OpenCV click window at each configured keyframe (by
default: first frame and last frame of the episode) and saves the click
to ``calibration_cache/<slug>/object_prompts.json``. Subsequent
``import-lerobot`` runs (with the same processor) read the cached
prompts and use them as cold-start anchors for the pose estimator.

Why multiple keyframes per episode: the warm-started pose tracker
drifts over long episodes when the robot arm intrudes on the
segmentation mask. Anchoring the trajectory at both ends (or more
keyframes if needed) lets the smoother pull the tracker back to a
known-good position at each anchor.
"""
from __future__ import annotations

import logging
from pathlib import Path

import click

from chuck_dreamer.config import load_config

from ..dataset import all_episode_bounds_from_meta, get_frame
from ..prompts import (
  click_object, load_keyframe_prompts, save_keyframe_prompt, sidecar_path,
)
from ..runtime import init_from_config
from chuck_dreamer.cli import override_option
from chuck_dreamer.common.episode_spec import EpisodeSpec

logger = logging.getLogger(__name__)


@click.command("prompt-episodes")
@click.argument("sources", required=True, nargs=-1, metavar="DATASET_ID[#EPISODES]...")
@click.option("--keyframes", "keyframes_spec", default="start,end",
              help="Comma-separated list of episode-relative keyframes to "
                   "prompt for, where 'start' = frame 0 and 'end' = the "
                   "last frame. Integers are episode-relative frame indices "
                   "(positive from start, negative from end). Default "
                   "'start,end' = first + last frame of every episode. "
                   "Examples: 'start' (legacy first-frame only), "
                   "'start,400,end' (3 keyframes including frame 400).")
@click.option("--force", is_flag=True, default=False,
              help="Re-click keyframes that already have a cached prompt "
                   "(default: skip them).")
@click.option("--skip-empty", is_flag=True, default=True,
              help="Skip episodes whose metadata says length 0 (default on).")
@override_option
@click.pass_context
def prompt_episodes_cmd(ctx, sources: tuple[str, ...],
                        keyframes_spec: str,
                        force: bool, skip_empty: bool,
                        overrides: tuple[str, ...]) -> None:
  """Click each keyframe's object, save to object_prompts.json.

  Run this before a batch import so the import itself runs
  non-interactively in the background. Picks up where you left off if
  interrupted (already-clicked keyframes are skipped unless --force).
  """
  cfg = load_config(ctx.obj["config_path"], overrides=overrides)
  ol_cfg = init_from_config(cfg)
  cache_root = Path(ol_cfg.cache_dir)

  parsed = EpisodeSpec.parse_many(sources, command="prompt-episodes")

  try:
    keyframe_offsets = _parse_keyframes(keyframes_spec)
  except ValueError as e:
    raise click.ClickException(f"--keyframes: {e}")

  # Resolve each source into a list of (dataset_id, episode_index, length).
  pending: list[tuple[str, int, int]] = []
  for src in parsed:
    try:
      slices, _ = src.read_episodes()
    except Exception as e:
      raise click.ClickException(
        f"could not read episode metadata for {src.dataset_id}: {e}")
    if src.episodes is not None:
      missing = set(src.episodes) - {s.episode_index for s in slices}
      if missing:
        click.echo(click.style(
          f"  WARN {src.dataset_id}: episodes {sorted(missing)} not in "
          f"dataset metadata; skipping.", fg="yellow"))
    for s in slices:
      pending.append((src.dataset_id, s.episode_index, s.length))

  if not pending:
    raise click.ClickException("no (dataset, episode) pairs to prompt for.")

  # For each (dataset, episode), expand the keyframe offsets against the
  # episode length to get concrete episode-relative frame indices.
  todo: list[tuple[str, int, int, int]] = []   # (did, ep, ep_len, rel_frame)
  for did, ep, ep_len in pending:
    if skip_empty and ep_len <= 0:
      continue
    existing = load_keyframe_prompts(cache_root, did, ep) if not force else {}
    for offset in keyframe_offsets:
      rel = _resolve_keyframe_offset(offset, ep_len)
      if rel is None:
        continue
      if rel in existing:
        continue
      todo.append((did, ep, ep_len, rel))

  total_kfs = sum(len(_kf_for_len(keyframe_offsets, ep_len))
                  for _, _, ep_len in pending if (not skip_empty) or ep_len > 0)
  skipped = total_kfs - len(todo)
  click.echo(f"\n{total_kfs} keyframes total ({len(pending)} episodes × "
             f"~{len(keyframe_offsets)} kf each); {skipped} already cached, "
             f"{len(todo)} need a click.")
  if not todo:
    click.echo("nothing to do; --force to re-click.")
    return

  from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

  current_ds_id: str | None = None
  ds = None
  for i, (did, ep, ep_len, rel) in enumerate(todo, start=1):
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
    global_idx = ep_fr + max(0, min(ep_to - ep_fr - 1, rel))
    rgb = get_frame(ds, global_idx, ol_cfg.camera_key)

    banner = (f"[{i}/{len(todo)}]  {did}  ep {ep}  "
              f"kf rel={rel} (global {global_idx}, ep_len={ep_len})")
    click.echo(f"  {banner}: click the object  (q to skip this keyframe)")
    prompt = click_object(rgb, banner=banner)
    if prompt is None:
      click.echo("    skipped.")
      continue
    save_keyframe_prompt(cache_root, did, ep, rel, prompt)
    click.echo(f"    saved: u={prompt[0]} v={prompt[1]}  -> "
               f"{sidecar_path(cache_root, did)}")

  click.echo("\nAll keyframes gathered. Next step: run `import-lerobot` for "
             "these datasets; the processor reads cached keyframes and "
             "anchors the pose trajectory at each one.")


# ---------------------------------------------------------------------------
# Keyframe spec parsing
# ---------------------------------------------------------------------------

# A keyframe offset is either an int (episode-relative frame, positive
# from start, negative from end), the sentinel 'start' (== 0), or
# 'end' (== -1). We store them in this normalized form and resolve to
# concrete frame indices once we know the episode length.
KeyframeOffset = int | str   # int or one of {'start', 'end'}


def _parse_keyframes(spec: str) -> list[KeyframeOffset]:
  tokens = [t.strip().lower() for t in (spec or "").split(",") if t.strip()]
  if not tokens:
    raise ValueError(f"empty keyframes spec {spec!r}")
  out: list[KeyframeOffset] = []
  for t in tokens:
    if t in ("start", "end"):
      out.append(t)
    else:
      try:
        out.append(int(t))
      except ValueError:
        raise ValueError(f"token {t!r} is not an integer or 'start'/'end'.")
  # Dedupe while preserving order.
  seen: set[KeyframeOffset] = set()
  unique: list[KeyframeOffset] = []
  for o in out:
    if o not in seen:
      seen.add(o)
      unique.append(o)
  return unique


def _resolve_keyframe_offset(offset: KeyframeOffset, ep_len: int) -> int | None:
  """Map a keyframe spec entry to a concrete episode-relative frame index,
  or None if it falls outside the episode.
  """
  if offset == "start":
    return 0
  if offset == "end":
    return ep_len - 1
  n = int(offset)
  if n < 0:
    n = ep_len + n
  if 0 <= n < ep_len:
    return n
  return None


def _kf_for_len(offsets: list[KeyframeOffset], ep_len: int) -> list[int]:
  out: list[int] = []
  seen: set[int] = set()
  for o in offsets:
    rel = _resolve_keyframe_offset(o, ep_len)
    if rel is None or rel in seen:
      continue
    seen.add(rel)
    out.append(rel)
  return out
