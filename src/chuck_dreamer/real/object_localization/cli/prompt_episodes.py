"""`prompt-episodes` CLI: collect object keyframes for a dataset.

For every (dataset, episode) implied by one or more CALIBRATION_SOURCE
specs, opens an OpenCV click window at each configured keyframe and saves
the click to ``calibration_cache/<slug>/object_prompts.json``. Subsequent
``import-lerobot`` runs (with the same processor) read the cached prompts
and use them as cold-start anchors for the pose estimator.

Why multiple keyframes per episode: the warm-started pose tracker drifts
over long episodes when the robot arm intrudes on the segmentation mask.
Anchoring the trajectory at several keyframes lets the smoother pull the
tracker back to a known-good position at each anchor.

Two ways to declare keyframes via ``--keyframes``:

  * A plain token (``start``, ``end``, or a signed int) is a **manual
    click** keyframe — you click the object yourself.
  * A token of the form ``A:B:step`` is an **augment range**: its
    endpoints ``A`` and ``B`` are *also* manual clicks, and after the
    clicking pass the estimator proposes ``step``-spaced keyframes
    strictly between them, which you accept/reject in a review UI.

So ``--keyframes "start:end:30"`` clicks start + end and then augments
every 30 frames in between — the densify workflow that used to be the
separate ``augment-keyframes`` command. With no ``:`` token, no augment
pass runs (pure manual clicking).

The augment pass walks each range with a bidirectional pose-estimator
chain (forward from the left anchor, backward from the right) plus a
cheap background-subtraction blob search to reject arm intrusions — see
``docs/object_labeling_problem.md`` approach (2).
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np

from chuck_dreamer.config import load_config

from ..dataset import all_episode_bounds_from_meta, get_frame
from ..prompts import (
  click_object, load_keyframe_prompts, save_keyframe_prompt, sidecar_path,
)
from ..runtime import init_from_config
from ..scene_bg import ensure_scene_bg
from ..types import CameraCalibration
from chuck_dreamer.cli import override_option
from chuck_dreamer.common.episode_spec import EpisodeSpec

logger = logging.getLogger(__name__)


@click.command("prompt-episodes")
@click.argument("sources", required=True, nargs=-1, metavar="DATASET_ID[#EPISODES]...")
@click.option("--keyframes", "keyframes_spec", default="start,end",
              help="Comma-separated keyframes to collect. A plain token is "
                   "a manual click: 'start' = frame 0, 'end' = last frame, "
                   "or a signed int (episode-relative; negative counts from "
                   "the end). A token 'A:B:step' is an augment range: A and B "
                   "are clicked manually, then the estimator proposes "
                   "step-spaced keyframes strictly between them (reviewed in a "
                   "UI). Default 'start,end' = click first + last frame, no "
                   "augmentation. Examples: 'start:end:30' (click start+end, "
                   "augment every 30 frames between); 'start:400:20,400:end:40' "
                   "(per-range strides).")
@click.option("--force", is_flag=True, default=False,
              help="Re-click manual keyframes that already have a cached "
                   "prompt (default: skip them).")
@click.option("--search-radius-px", default=60, type=int,
              help="[augment] Pixel radius around the previous accepted (u, v) "
                   "inside which the bg-foreground blob must lie (default 60).")
@click.option("--area-ratio-min", default=0.4, type=float,
              help="[augment] Reject if (candidate area / previous area) is "
                   "outside [ratio, 1/ratio]. Default 0.4.")
@click.option("--min-confidence", default=0.3, type=float,
              help="[augment] Reject poses with confidence below this "
                   "(default 0.3).")
@click.option("--max-residual-px", default=2.0, type=float,
              help="[augment] Reject poses with reprojection residual above "
                   "this (default 2.0 px).")
@click.option("--ee-guard-px", default=80, type=int,
              help="[augment] Reject candidates where the EE projects within "
                   "this pixel radius of the search window. Set to 0 to "
                   "disable (default 80).")
@click.option("--bg-samples", default=16, type=int,
              help="[augment] Frames sampled from the empty episode to build "
                   "the background model (default 16). Reused from cache if "
                   "available.")
@click.option("--rebuild-bg", is_flag=True, default=False,
              help="[augment] Force recompute of the scene background "
                   "(default: reuse cached scene_bg.npz if present).")
@click.option("--ee-key", default="ee_pos", type=str,
              help="[augment] Sample key to read the end-effector position "
                   "from for the EE-guard. Set to '' to skip the EE-guard.")
@override_option
@click.pass_context
def prompt_episodes_cmd(ctx, sources: tuple[str, ...],
                        keyframes_spec: str, force: bool,
                        search_radius_px: int, area_ratio_min: float,
                        min_confidence: float, max_residual_px: float,
                        ee_guard_px: int, bg_samples: int, rebuild_bg: bool,
                        ee_key: str,
                        overrides: tuple[str, ...]) -> None:
  """Click each keyframe's object, save to object_prompts.json.

  Runs in two phases. First, a manual click pass over every keyframe
  named in ``--keyframes`` (plain tokens and the endpoints of every
  augment range). Then, if ``--keyframes`` contains any ``A:B:step``
  range, an augment pass that proposes extra keyframes inside each range
  and reviews them interactively. With no range token, only the click
  pass runs.

  Picks up where you left off if interrupted (already-clicked manual
  keyframes are skipped unless --force).
  """
  cfg = load_config(ctx.obj["config_path"], overrides=overrides)
  ol_cfg = init_from_config(cfg)
  cache_root = Path(ol_cfg.cache_dir)

  parsed = EpisodeSpec.parse_many(
    sources, allow_frames=False, command="prompt-episodes")

  try:
    manual_offsets, ranges = _parse_keyframes(keyframes_spec)
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

  # For each (dataset, episode), expand the manual keyframe offsets against
  # the episode length to get concrete episode-relative frame indices.
  todo: list[tuple[str, int, int, int]] = []   # (did, ep, ep_len, rel_frame)
  for did, ep, ep_len in pending:
    if ep_len <= 0:
      raise click.ClickException(
        f"{did} ep {ep}: episode too short (len={ep_len}); this command "
        f"only makes sense on episodes of non-zero length.")
    existing = load_keyframe_prompts(cache_root, did, ep) if not force else {}
    for offset in manual_offsets:
      rel = _resolve_keyframe_offset(offset, ep_len)
      if rel is None:
        continue
      if rel in existing:
        continue
      todo.append((did, ep, ep_len, rel))

  total_kfs = sum(len(_kf_for_len(manual_offsets, ep_len))
                  for _, _, ep_len in pending)
  skipped = total_kfs - len(todo)
  click.echo(f"\n{total_kfs} manual keyframes total ({len(pending)} episodes "
             f"× ~{len(manual_offsets)} kf each); {skipped} already cached, "
             f"{len(todo)} need a click.")

  if todo:
    _run_click_pass(todo, cache_root, ol_cfg.camera_key)
  else:
    click.echo("nothing to click; --force to re-click.")

  if ranges:
    _run_augment_pass(
      pending, ranges, cache_root, ol_cfg,
      search_radius_px=search_radius_px, area_ratio_min=area_ratio_min,
      min_confidence=min_confidence, max_residual_px=max_residual_px,
      ee_guard_px=ee_guard_px, bg_samples=bg_samples, rebuild_bg=rebuild_bg,
      ee_key=ee_key,
    )
  else:
    click.echo("\nAll keyframes gathered. Next step: run `import-lerobot` for "
               "these datasets; the processor reads cached keyframes and "
               "anchors the pose trajectory at each one.")


def _run_click_pass(todo: list[tuple[str, int, int, int]],
                    cache_root: Path, camera_key: str) -> None:
  """Open a click window for each (dataset, episode, rel) keyframe."""
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
    rgb = get_frame(ds, global_idx, camera_key)

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


# ---------------------------------------------------------------------------
# Keyframe spec parsing
# ---------------------------------------------------------------------------

# A keyframe offset is either an int (episode-relative frame, positive
# from start, negative from end), the sentinel 'start' (== 0), or
# 'end' (== -1). We store them in this normalized form and resolve to
# concrete frame indices once we know the episode length.
KeyframeOffset = int | str   # int or one of {'start', 'end'}


@dataclass
class _AugmentRange:
  """An ``A:B:step`` range from the keyframes spec.

  ``a``/``b`` are keyframe offsets (the same normalized form as a manual
  keyframe); ``step`` is the positive stride to propose between them.
  """
  a: KeyframeOffset
  b: KeyframeOffset
  step: int


def _parse_offset_token(t: str) -> KeyframeOffset:
  if t in ("start", "end"):
    return t
  try:
    return int(t)
  except ValueError:
    raise ValueError(f"token {t!r} is not an integer or 'start'/'end'.")


def _parse_keyframes(spec: str) -> tuple[list[KeyframeOffset],
                                         list[_AugmentRange]]:
  """Parse the keyframes spec into manual offsets and augment ranges.

  Returns ``(manual_offsets, ranges)``. Range endpoints are added to
  ``manual_offsets`` too (they're clicked manually). Manual offsets are
  deduped preserving order.
  """
  tokens = [t.strip().lower() for t in (spec or "").split(",") if t.strip()]
  if not tokens:
    raise ValueError(f"empty keyframes spec {spec!r}")

  manual: list[KeyframeOffset] = []
  ranges: list[_AugmentRange] = []
  for t in tokens:
    if ":" in t:
      parts = t.split(":")
      if len(parts) != 3:
        raise ValueError(
          f"range {t!r} must be 'A:B:step' (got {len(parts)} parts).")
      a = _parse_offset_token(parts[0])
      b = _parse_offset_token(parts[1])
      try:
        step = int(parts[2])
      except ValueError:
        raise ValueError(f"range {t!r}: step {parts[2]!r} is not an integer.")
      if step <= 0:
        raise ValueError(f"range {t!r}: step must be positive (got {step}).")
      ranges.append(_AugmentRange(a=a, b=b, step=step))
      manual.append(a)
      manual.append(b)
    else:
      manual.append(_parse_offset_token(t))

  # Dedupe while preserving order.
  seen: set[KeyframeOffset] = set()
  unique: list[KeyframeOffset] = []
  for o in manual:
    if o not in seen:
      seen.add(o)
      unique.append(o)
  return unique, ranges


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


# ---------------------------------------------------------------------------
# Augment pass
# ---------------------------------------------------------------------------


@dataclass
class _Proposal:
  """One candidate keyframe under review."""
  ep_rel: int                 # episode-relative frame index
  global_idx: int             # dataset-global frame index
  uv: tuple[int, int]         # back-projected mesh centroid in pixels
  pose: object                # ObjectPose; opaque here, used only for the overlay
  confidence: float
  residual_px: float
  rgb: np.ndarray             # frame for the review UI


@dataclass
class _ChainState:
  """Mutable per-gap state: where the chain currently believes the object is."""
  pose: object              # ObjectPose
  uv:   tuple[int, int]
  area: float               # rendered silhouette area in px
  rel:  int                 # episode-relative frame index this state came from


def _run_augment_pass(pending: list[tuple[str, int, int]],
                      ranges: list[_AugmentRange],
                      cache_root: Path, ol_cfg, *,
                      search_radius_px: int, area_ratio_min: float,
                      min_confidence: float, max_residual_px: float,
                      ee_guard_px: int, bg_samples: int, rebuild_bg: bool,
                      ee_key: str) -> None:
  """Propose extra keyframes inside each declared range; review and save.

  Groups episodes by dataset so calibration + estimator + scene_bg are
  built once per dataset and reused across its episodes.
  """
  from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
  from ..estimator import ObjectPoseEstimator

  by_ds: dict[str, list[tuple[int, int]]] = {}
  for did, ep, ep_len in pending:
    by_ds.setdefault(did, []).append((ep, ep_len))

  total_proposed = 0
  total_accepted = 0
  for did, episodes in by_ds.items():
    click.echo(f"\n=== augment {did} ({len(episodes)} episodes) ===")
    try:
      cal = CameraCalibration.load(cache_root, did)
    except FileNotFoundError as e:
      raise click.ClickException(f"{did}: {e}")

    mesh_path = Path(ol_cfg.mesh_path)
    if not mesh_path.exists():
      raise click.ClickException(
        f"mesh not found at {mesh_path}; set object_localization.object.mesh_path.")

    click.echo(f"  preparing scene background from episode "
               f"{ol_cfg.episode_empty} ({bg_samples} samples) ...")
    scene_bg = ensure_scene_bg(
      cache_root, did, ol_cfg.camera_key,
      ol_cfg.episode_empty, n_samples=bg_samples, rebuild=rebuild_bg,
      calibration=cal, ol_cfg=ol_cfg,
    )

    est_cfg = dict(ol_cfg.raw.get("object") or {})
    est_cfg.setdefault("sam2_checkpoint", ol_cfg.sam2_checkpoint)
    click.echo(f"  loading mesh from {mesh_path} ...")
    estimator = ObjectPoseEstimator(
      calibration = cal,
      mesh_path   = mesh_path,
      config      = est_cfg,
      device      = ol_cfg.device,
      use_sam2    = ol_cfg.use_sam2,
      scene_bg    = scene_bg,
    )

    click.echo(f"  loading LeRobotDataset({did}) ...")
    ds = LeRobotDataset(did)

    for ep, ep_len in episodes:
      ep_count, ep_accept = _process_episode(
        ds, did, ep, ep_len, cache_root, estimator, ol_cfg.camera_key,
        cal, ranges,
        search_radius_px=search_radius_px,
        area_ratio_min=area_ratio_min,
        min_confidence=min_confidence,
        max_residual_px=max_residual_px,
        ee_guard_px=ee_guard_px,
        ee_key=ee_key,
      )
      total_proposed += ep_count
      total_accepted += ep_accept

  click.echo(f"\nDone. Proposed {total_proposed} keyframes, accepted "
             f"{total_accepted}.")


def _process_episode(ds, dataset_id: str, episode_idx: int, ep_len: int,
                     cache_root: Path, estimator, camera_key: str,
                     calibration: CameraCalibration,
                     ranges: list[_AugmentRange],
                     *,
                     search_radius_px: int, area_ratio_min: float,
                     min_confidence: float, max_residual_px: float,
                     ee_guard_px: int, ee_key: str,
                     ) -> tuple[int, int]:
  """Augment one episode inside every declared range. Returns (n_proposed, n_accepted).

  For each range ``[A_rel, B_rel]`` (resolved against this episode's
  length):

    1. Run a forward chain from the left anchor and a backward chain
       from the right anchor. Each chain marches step frames at a time,
       skipping frames the cheap pre-checks reject, and warm-starting
       the pose estimator from the previous accepted fit.

    2. Merge the two chains per stride-aligned frame: if both produced
       a fit at the same rel, keep the one with lower residual (tie-
       break: higher confidence). If only one direction fitted that
       frame, keep it.

    3. Stream the merged proposals through the review UI; save each
       accepted one immediately.

  The bidirectional pass exists because the forward chain tends to
  drift the further it gets from its anchor. With two chains, drift-
  prone middle frames of long gaps usually have at least one direction
  that produces a clean fit.
  """
  manual = load_keyframe_prompts(cache_root, dataset_id, episode_idx)

  try:
    ep_fr, ep_to = next(
      b for e, b in all_episode_bounds_from_meta(dataset_id) if e == episode_idx)
  except StopIteration:
    raise click.ClickException(f"ep {episode_idx}: no metadata.")

  # Resolve every range's endpoints against this episode length.
  resolved: list[tuple[int, int, int]] = []
  for r in ranges:
    a_rel = _resolve_keyframe_offset(r.a, ep_len)
    b_rel = _resolve_keyframe_offset(r.b, ep_len)
    if a_rel is None or b_rel is None:
      raise click.ClickException(
        f"ep {episode_idx}: range {r.a}:{r.b}:{r.step} resolves outside the "
        f"episode (len={ep_len}).")
    if a_rel == b_rel:
      raise click.ClickException(
        f"ep {episode_idx}: range {r.a}:{r.b}:{r.step} has identical "
        f"endpoints (rel={a_rel}).")
    if a_rel > b_rel:
      a_rel, b_rel = b_rel, a_rel
    resolved.append((a_rel, b_rel, r.step))

  click.echo(f"\n  ep {episode_idx}: len={ep_len}, augment ranges "
             f"(rel)={[(a, b, s) for a, b, s in resolved]}")

  n_proposed = 0
  n_accepted = 0
  quit_episode = False

  for a_rel, b_rel, step in resolved:
    if quit_episode:
      break
    if b_rel - a_rel <= step:
      click.echo(f"    range [{a_rel}, {b_rel}] step={step}: gap too small; "
                 f"nothing to augment.")
      continue

    if a_rel not in manual or b_rel not in manual:
      raise click.ClickException(
        f"ep {episode_idx}: range endpoint(s) at rel {a_rel}/{b_rel} were "
        f"not clicked (skipped in the click pass?). Re-run and click them.")

    a_prompt = manual[a_rel]
    b_prompt = manual[b_rel]
    a_anchor = _anchor_pose(ds, ep_fr, ep_to, a_rel, a_prompt,
                            camera_key, estimator, calibration)
    if a_anchor is None:
      click.echo(f"    anchor rel={a_rel}: estimator failed to fit; "
                 f"skipping range [{a_rel}, {b_rel}].")
      continue
    b_anchor = _anchor_pose(ds, ep_fr, ep_to, b_rel, b_prompt,
                            camera_key, estimator, calibration)
    if b_anchor is None:
      click.echo(f"    anchor rel={b_rel}: estimator failed to fit; "
                 f"falling back to forward-only over [{a_rel}, {b_rel}].")

    fwd_chain = _ChainState(
      pose = a_anchor[0], uv = a_anchor[1], area = a_anchor[2], rel = a_rel,
    )
    fwd_proposals = _chain_proposals(
      ds, ep_fr, ep_to, camera_key, estimator, calibration, fwd_chain,
      start_rel = a_rel + step,
      end_rel   = b_rel,
      step      = step,
      search_radius_px=search_radius_px, area_ratio_min=area_ratio_min,
      min_confidence=min_confidence, max_residual_px=max_residual_px,
      ee_guard_px=ee_guard_px, ee_key=ee_key, direction="forward",
    )

    bwd_proposals: list[_Proposal] = []
    if b_anchor is not None:
      # March backwards on the *same* stride grid the forward chain
      # used, so the two chains' proposals share rel values and we can
      # merge by rel without rounding gymnastics.
      bwd_chain = _ChainState(
        pose = b_anchor[0], uv = b_anchor[1], area = b_anchor[2], rel = b_rel,
      )
      # Start one stride before the right anchor and walk down to the
      # last stride point strictly after a_rel.
      last_fwd_rel = a_rel + ((b_rel - 1 - a_rel) // step) * step
      if last_fwd_rel <= a_rel:
        last_fwd_rel = a_rel + step
      bwd_start = last_fwd_rel
      bwd_end   = a_rel
      bwd_proposals = _chain_proposals(
        ds, ep_fr, ep_to, camera_key, estimator, calibration, bwd_chain,
        start_rel = bwd_start,
        end_rel   = bwd_end,
        step      = -step,
        search_radius_px=search_radius_px, area_ratio_min=area_ratio_min,
        min_confidence=min_confidence, max_residual_px=max_residual_px,
        ee_guard_px=ee_guard_px, ee_key=ee_key, direction="backward",
      )

    merged = _merge_bidirectional(fwd_proposals, bwd_proposals)
    click.echo(f"    range [{a_rel}, {b_rel}]: forward={len(fwd_proposals)}  "
               f"backward={len(bwd_proposals)}  merged={len(merged)} "
               f"proposals.")

    for proposal in merged:
      n_proposed += 1
      decision, kept = _review_one_proposal(
        proposal, dataset_id, episode_idx, ep_len,
        estimator, calibration)

      if decision == 'q':
        quit_episode = True
        break
      if decision == 'a' and kept is not None:
        save_keyframe_prompt(cache_root, dataset_id, episode_idx,
                             kept.ep_rel, kept.uv)
        click.echo(f"    saved rel={kept.ep_rel} u={kept.uv[0]} "
                   f"v={kept.uv[1]} -> "
                   f"{sidecar_path(cache_root, dataset_id)}")
        n_accepted += 1

  return n_proposed, n_accepted


def _anchor_pose(ds, ep_fr: int, ep_to: int, anchor_rel: int, prompt,
                 camera_key: str, estimator,
                 calibration: CameraCalibration,
                 ) -> tuple[object, tuple[int, int], float] | None:
  """Cold-start the estimator at a manual keyframe; return (pose, uv, area)."""
  from ..estimator import _EpisodeContext

  global_idx = ep_fr + max(0, min(ep_to - ep_fr - 1, anchor_rel))
  rgb = get_frame(ds, global_idx, camera_key)
  estimator._episode = _EpisodeContext()
  t0 = time.perf_counter()
  pose = estimator.estimate(rgb, prompt=prompt)
  if pose is None:
    return None
  uv   = _project_pose_uv(pose, calibration)
  area = float(_pose_silhouette_area(pose, estimator))
  click.echo(f"    anchor rel={anchor_rel}: pose recovered in "
             f"{time.perf_counter()-t0:.1f}s "
             f"(conf={pose.confidence:.2f}, "
             f"resid={pose.reprojection_error_px:.2f}px)")
  return pose, uv, area


def _chain_proposals(ds, ep_fr: int, ep_to: int, camera_key: str,
                     estimator, calibration: CameraCalibration,
                     chain: _ChainState, *,
                     start_rel: int, end_rel: int, step: int,
                     search_radius_px: int, area_ratio_min: float,
                     min_confidence: float, max_residual_px: float,
                     ee_guard_px: int, ee_key: str,
                     direction: str,
                     ) -> list[_Proposal]:
  """Walk a stride-spaced chain from ``start_rel`` toward ``end_rel``,
  advancing ``chain`` after every successful fit.

  ``step`` is signed (positive for forward, negative for backward) and
  must match the sign of ``end_rel - start_rel``. The walk stops at
  the first stride point that crosses ``end_rel`` (exclusive). Frames
  that fail the cheap pre-checks or the pose check are skipped but
  do not abort the walk — the chain just stays anchored on the last
  successful fit and tries the next stride point.

  This is the building block used twice per range: once forward from
  the left anchor, once backward from the right anchor.
  """
  out: list[_Proposal] = []
  if step == 0:
    return out
  rel = start_rel
  going_forward = step > 0
  while (going_forward and rel < end_rel) or (not going_forward and rel > end_rel):
    if rel < 0 or rel >= (ep_to - ep_fr):
      rel += step
      continue
    global_idx = ep_fr + rel
    rgb = get_frame(ds, global_idx, camera_key)

    ee_uv = _ee_projected_uv(ds, global_idx, ee_key, calibration)
    if (ee_guard_px > 0 and ee_uv is not None
        and _within_radius(ee_uv, chain.uv, search_radius_px + ee_guard_px)):
      logger.info("    rel=%d [%s]: skip — EE in guard radius.", rel, direction)
      rel += step
      continue

    blob = _bg_blob_near(rgb, chain.uv, search_radius_px, estimator.scene_bg)
    if blob is None:
      logger.info("    rel=%d [%s]: skip — no bg blob.", rel, direction)
      rel += step
      continue
    blob_uv, blob_area = blob
    if chain.area > 0 and blob_area > 0:
      ratio = blob_area / chain.area
      if ratio < area_ratio_min or ratio > 1.0 / area_ratio_min:
        logger.info("    rel=%d [%s]: skip — area ratio %.2f.",
                    rel, direction, ratio)
        rel += step
        continue

    pose = estimator.estimate(rgb, prev_pose=chain.pose, prompt=blob_uv)
    if pose is None:
      logger.info("    rel=%d [%s]: skip — estimator None.", rel, direction)
      rel += step
      continue
    if pose.confidence < min_confidence:
      logger.info("    rel=%d [%s]: skip — confidence %.2f.",
                  rel, direction, pose.confidence)
      rel += step
      continue
    if (not math.isfinite(pose.reprojection_error_px)
        or pose.reprojection_error_px > max_residual_px):
      logger.info("    rel=%d [%s]: skip — residual %.2fpx.",
                  rel, direction, pose.reprojection_error_px)
      rel += step
      continue

    pose_uv = _project_pose_uv(pose, calibration)
    out.append(_Proposal(
      ep_rel       = rel,
      global_idx   = global_idx,
      uv           = pose_uv,
      pose         = pose,
      confidence   = float(pose.confidence),
      residual_px  = float(pose.reprojection_error_px),
      rgb          = rgb,
    ))
    # Advance the chain: the just-fitted pose seeds the next warm-start.
    chain.pose = pose
    chain.uv   = pose_uv
    chain.area = float(blob_area)
    chain.rel  = rel
    rel += step
  return out


def _merge_bidirectional(fwd: list[_Proposal],
                          bwd: list[_Proposal]) -> list[_Proposal]:
  """Combine forward + backward proposals by ``ep_rel``.

  When both chains produced a fit at the same rel, keep the one with
  the lower reprojection residual; tie-break on higher confidence.
  Returns the combined list sorted by ep_rel so the review UI sees
  proposals in temporal order regardless of which chain produced them.
  """
  by_rel: dict[int, _Proposal] = {}
  for p in fwd:
    by_rel[p.ep_rel] = p
  for p in bwd:
    existing = by_rel.get(p.ep_rel)
    if existing is None:
      by_rel[p.ep_rel] = p
      continue
    if _proposal_score(p) < _proposal_score(existing):
      by_rel[p.ep_rel] = p
  return sorted(by_rel.values(), key=lambda q: q.ep_rel)


def _proposal_score(p: _Proposal) -> float:
  """Lower is better. Residual dominates; confidence breaks ties.

  Residual is the direct optimizer objective and tracks how well the
  rendered silhouette/edges align with the image. Confidence is
  noisier (it folds in area-ratio heuristics) but useful when two
  fits have similar residuals.
  """
  resid = p.residual_px if math.isfinite(p.residual_px) else 1e6
  # Subtract a small confidence term so higher confidence wins on
  # near-equal residuals. 0.1px-per-confidence-point is a soft enough
  # weighting that confidence never overrides a clearly-better fit.
  return resid - 0.1 * p.confidence


# ---------------------------------------------------------------------------
# Cheap candidate-finder helpers
# ---------------------------------------------------------------------------


def _bg_blob_near(rgb: np.ndarray, center_uv: tuple[int, int] | None,
                  radius_px: int, scene_bg
                  ) -> tuple[tuple[int, int], int] | None:
  """Return ``((u, v), area_px)`` of the bg-foreground blob near ``center_uv``.

  Builds the full-frame foreground mask, restricts it to a square
  window around ``center_uv``, and picks the connected component
  closest to the center. Returns ``None`` if nothing of substance
  is in the window.
  """
  import cv2

  if center_uv is None:
    return None
  fg = scene_bg.foreground_mask(rgb)
  if not fg.any():
    return None

  H, W = fg.shape
  u0, v0 = center_uv
  y0 = max(0, v0 - radius_px); y1 = min(H, v0 + radius_px + 1)
  x0 = max(0, u0 - radius_px); x1 = min(W, u0 + radius_px + 1)
  window = fg[y0:y1, x0:x1].astype(np.uint8)
  if window.sum() < 20:
    return None

  num, labels, stats, centroids = cv2.connectedComponentsWithStats(
    window, connectivity=8)
  if num <= 1:
    return None

  # Pick the largest component in the window; tie-break by proximity to
  # the window center. Largest is usually correct (the object dwarfs
  # mat-fiducial bleed-through, etc.); proximity is the fallback.
  areas = stats[1:, cv2.CC_STAT_AREA]
  best_local = int(np.argmax(areas)) + 1
  cy_local, cx_local = centroids[best_local][1], centroids[best_local][0]
  u_full = int(round(x0 + cx_local))
  v_full = int(round(y0 + cy_local))
  return (u_full, v_full), int(areas[best_local - 1])


def _project_pose_uv(pose, calibration: CameraCalibration) -> tuple[int, int]:
  """Project the object's world-frame centroid down to image pixels."""
  import cv2
  K    = np.asarray(calibration.intrinsics.K, dtype=np.float64)
  dist = np.asarray(calibration.intrinsics.dist, dtype=np.float64)
  R    = np.asarray(calibration.extrinsics.R, dtype=np.float64)
  t    = np.asarray(calibration.extrinsics.t, dtype=np.float64).reshape(3)
  rvec, _ = cv2.Rodrigues(R)
  uv, _ = cv2.projectPoints(pose.xyz_mm.reshape(1, 3), rvec, t, K, dist)
  u, v = uv.reshape(2)
  return int(round(float(u))), int(round(float(v)))


def _pose_silhouette_area(pose, estimator) -> int:
  """Render the silhouette implied by the pose and return its pixel area.

  We use this as the reference for the area-ratio guard on the *next*
  candidate's bg blob: a sudden 5x change in foreground area between
  adjacent stride frames is almost always an arm intrusion, not the
  object.
  """
  from ..render import project_silhouette, transform_object
  verts = transform_object(estimator.mesh.vertices_mm, pose.R_world_obj,
                           pose.xyz_mm)
  ren_mask, _ = project_silhouette(verts, estimator.mesh.triangles,
                                   estimator.camera)
  if ren_mask is None:
    return 0
  return int((ren_mask > 0).sum())


def _ee_projected_uv(ds, global_idx: int, ee_key: str,
                     calibration: CameraCalibration
                     ) -> tuple[int, int] | None:
  """Project the EE position at ``global_idx`` to image pixels, or None.

  Returns None if ``ee_key`` is missing/empty, the dataset doesn't
  carry it, or the value isn't a 3-vector. This is a guard, not a
  hard requirement — episodes without an EE channel just skip the
  EE proximity check.
  """
  if not ee_key:
    return None
  try:
    sample = ds[global_idx]
    if ee_key not in sample:
      return None
    raw = sample[ee_key]
  except (KeyError, IndexError, TypeError):
    return None
  try:
    import torch
    if isinstance(raw, torch.Tensor):
      raw = raw.detach().cpu().numpy()
  except Exception:
    pass
  arr = np.asarray(raw, dtype=np.float64).reshape(-1)
  if arr.size < 3:
    return None
  ee_world = arr[:3] * 1000.0   # ee_pos is metres; calibration is mm.

  import cv2
  K    = np.asarray(calibration.intrinsics.K, dtype=np.float64)
  dist = np.asarray(calibration.intrinsics.dist, dtype=np.float64)
  R    = np.asarray(calibration.extrinsics.R, dtype=np.float64)
  t    = np.asarray(calibration.extrinsics.t, dtype=np.float64).reshape(3)
  rvec, _ = cv2.Rodrigues(R)
  uv, _ = cv2.projectPoints(ee_world.reshape(1, 3), rvec, t, K, dist)
  u, v = uv.reshape(2)
  return int(round(float(u))), int(round(float(v)))


def _within_radius(a: tuple[int, int], b: tuple[int, int] | None,
                    r: int) -> bool:
  if b is None:
    return False
  return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= r * r


# ---------------------------------------------------------------------------
# Review UI
# ---------------------------------------------------------------------------

_WIN_NAME = "prompt-episodes augment"
_BANNER_H = 64
_FOOTER_H = 40
_MAX_W    = 1280
_MAX_H    = 720


def _review_one_proposal(proposal: _Proposal,
                         dataset_id: str, episode_idx: int, ep_len: int,
                         estimator,
                         calibration: CameraCalibration,
                         ) -> tuple[str, _Proposal | None]:
  """Review one proposal interactively.

  Returns ``(decision, proposal_to_save)``:
    * ``('a', p)`` — accept ``p`` (which may be the original auto-fit
      or a re-fit produced by a click on the image).
    * ``('r', None)`` — reject; don't save anything for this frame.
    * ``('q', None)`` — quit the rest of the episode.

  Key bindings:
    a / SPACE  accept the current displayed fit and advance to the
               next stride frame in the gap.
    r / x      reject; don't save; chain anchor stays at the previous
               accept so the next frame's bg-search uses the last
               known-good (u, v).
    LEFT-CLICK on the image → treated as a new prompt at that pixel.
               The estimator cold-starts at the click; the canvas
               updates with the new wireframe in <1s. Click again
               to refine, press 'a' to accept the displayed fit.
    q / ESC    quit the rest of this episode.
  """
  import cv2

  current = proposal
  # Per-frame mouse state. Every left-click on the image area is
  # consumed as a re-prompt — no modal `c` key required any more.
  click_state: dict = {"click": None, "scale": 1.0,
                       "image_size": (0, 0), "banner_h": _BANNER_H,
                       "busy": False}

  def on_mouse(event, x, y, _flags, _ud):
    if event != cv2.EVENT_LBUTTONDOWN:
      return
    if click_state["busy"]:
      return
    W, H = click_state["image_size"]
    s = click_state["scale"]
    bh = click_state["banner_h"]
    if y < bh or y >= bh + int(H * s):
      return
    u = x / s
    v = (y - bh) / s
    if not (0 <= u < W and 0 <= v < H):
      return
    click_state["click"] = (int(round(u)), int(round(v)))

  cv2.namedWindow(_WIN_NAME, cv2.WINDOW_AUTOSIZE)
  cv2.setMouseCallback(_WIN_NAME, on_mouse)
  try:
    while True:
      H, W = current.rgb.shape[:2]
      scale = min(_MAX_W / W, _MAX_H / H, 1.0)
      click_state["scale"] = scale
      click_state["image_size"] = (W, H)

      canvas = _render_review_canvas(
        current, dataset_id, episode_idx, ep_len, estimator,
        is_refit=current is not proposal)
      cv2.imshow(_WIN_NAME, canvas)

      key = cv2.waitKey(16) & 0xFFFF

      # Click handling runs every tick so the user doesn't have to
      # press anything first.
      if click_state["click"] is not None:
        new_uv = click_state["click"]
        click_state["click"] = None
        click_state["busy"] = True
        click.echo(f"    rel={current.ep_rel}: re-clicked at "
                   f"u={new_uv[0]} v={new_uv[1]} — re-fitting "
                   f"(cold-start) ...")
        new_p = _refit_at_click(current, new_uv, estimator, calibration)
        click_state["busy"] = False
        if new_p is None:
          click.echo(f"    rel={current.ep_rel}: re-fit FAILED "
                     f"(segmenter or rest-class search returned nothing); "
                     f"click again or press 'r' to skip.")
        else:
          current = new_p
          click.echo(f"    rel={new_p.ep_rel}: re-fit OK "
                     f"(conf={new_p.confidence:.2f}, "
                     f"resid={new_p.residual_px:.2f}px). "
                     f"Press 'a' to accept, click again to refine, "
                     f"'r' to skip.")
        continue

      if key == 0xFFFF:
        continue
      if key in (ord('q'), 27):
        return 'q', None
      if key in (ord('a'), ord(' ')):
        return 'a', current
      if key in (ord('r'), ord('x')):
        return 'r', None
      # Anything else is ignored; loop and wait.
  finally:
    cv2.destroyWindow(_WIN_NAME)
    cv2.waitKey(1)


def _refit_at_click(orig: _Proposal, new_uv: tuple[int, int],
                    estimator, calibration: CameraCalibration
                    ) -> _Proposal | None:
  """Cold-start the estimator at ``new_uv`` and return the updated proposal.

  Mirrors what a manual keyframe click would do: reset the episode
  context so the rest-class search runs again, then call
  ``estimate(prompt=...)``. Returns None if the segmenter or rest-class
  fit fails.
  """
  from ..estimator import _EpisodeContext
  estimator._episode = _EpisodeContext()
  try:
    pose = estimator.estimate(orig.rgb, prompt=new_uv)
  except Exception as e:
    logger.warning("re-fit at %s failed: %s", new_uv, e)
    return None
  if pose is None:
    return None
  pose_uv = _project_pose_uv(pose, calibration)
  return _Proposal(
    ep_rel       = orig.ep_rel,
    global_idx   = orig.global_idx,
    uv           = pose_uv,
    pose         = pose,
    confidence   = float(pose.confidence),
    residual_px  = float(pose.reprojection_error_px),
    rgb          = orig.rgb,
  )


def _render_review_canvas(p: _Proposal,
                          dataset_id: str, episode_idx: int, ep_len: int,
                          estimator,
                          is_refit: bool = False) -> np.ndarray:
  import cv2
  from ..render import project_visible_edges, transform_object

  H, W = p.rgb.shape[:2]
  scale = min(_MAX_W / W, _MAX_H / H, 1.0)
  win_w = int(W * scale)
  canvas_h = int(H * scale)
  win_h = _BANNER_H + canvas_h + _FOOTER_H

  bgr = cv2.cvtColor(p.rgb, cv2.COLOR_RGB2BGR)

  # Mesh wireframe overlay (green); same path as test_object_pose.
  verts_world = transform_object(estimator.mesh.vertices_mm,
                                 p.pose.R_world_obj, p.pose.xyz_mm)
  edges = project_visible_edges(verts_world, estimator.mesh.triangles,
                                estimator.camera)
  for q0, q1 in edges:
    cv2.line(bgr,
             (int(round(q0[0])), int(round(q0[1]))),
             (int(round(q1[0])), int(round(q1[1]))),
             (0, 255, 0), 2, cv2.LINE_AA)

  # Click target as a magenta crosshair.
  cv2.drawMarker(bgr, p.uv, (255, 0, 255), cv2.MARKER_CROSS, 18, 2)

  if scale != 1.0:
    disp = cv2.resize(bgr, (win_w, canvas_h), interpolation=cv2.INTER_AREA)
  else:
    disp = bgr

  canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
  canvas[:_BANNER_H, :] = (32, 32, 32)
  canvas[_BANNER_H + canvas_h:, :] = (48, 48, 48)
  canvas[_BANNER_H:_BANNER_H + canvas_h, :, :] = disp

  banner = (f"{dataset_id}  ep={episode_idx} (len={ep_len})  "
            f"rel={p.ep_rel}  (global={p.global_idx})")
  if is_refit:
    banner += "  — RE-FIT (human click)"
  cv2.putText(canvas, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
              (80, 220, 220) if is_refit else (230, 230, 230),
              1, cv2.LINE_AA)
  stats = (f"u={p.uv[0]} v={p.uv[1]}  conf={p.confidence:.2f}  "
           f"resid={p.residual_px:.2f}px  "
           f"xy_mm=({p.pose.xy_mm[0]:+.1f},{p.pose.xy_mm[1]:+.1f})  "
           f"yaw={math.degrees(p.pose.yaw_rad):+.1f}deg")
  cv2.putText(canvas, stats, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
              (180, 220, 255), 1, cv2.LINE_AA)

  footer_y = _BANNER_H + canvas_h
  help_str = ("a/SPACE accept (save + advance)   r/x reject (advance)   "
              "click on object to re-fit   q quit episode")
  cv2.putText(canvas, help_str,
              (10, footer_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
              (200, 200, 200), 1, cv2.LINE_AA)
  return canvas
