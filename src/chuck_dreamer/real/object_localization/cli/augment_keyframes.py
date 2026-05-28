"""`augment-keyframes` CLI: propose extra keyframes between manual ones.

For each (dataset, episode) named on the command line, walk the gaps
between consecutive manual keyframes (from ``object_prompts.json``) and
propose new keyframes by:

  1. Sampling candidate frame indices at a fixed stride inside the gap.
  2. Running a cheap background-subtraction search inside a small window
     around the previous accepted (u, v) to find the object centroid.
     Rejected if the EE projects into that window, if the blob area is
     far off the previous one, or if no blob is found.
  3. Running the existing ``ObjectPoseEstimator`` in warm-start mode
     against the surviving candidate. Rejected if the recovered pose
     has low confidence or high residual.
  4. After the whole episode, opening a small OpenCV review UI: each
     proposal is shown with the mesh wireframe overlay and the operator
     accepts (``a``) or rejects (``r``) it; accepted clicks are saved
     back into ``object_prompts.json`` via ``save_keyframe_prompt``.

Designed as a cheap "1.5x" between fully-manual keyframing and a full
smoother — see ``docs/object_labeling_problem.md`` approach (2).
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
  load_keyframe_prompts, save_keyframe_prompt, sidecar_path,
)
from ..runtime import init_from_config
from ..scene_bg import ensure_scene_bg
from ..types import CameraCalibration
from chuck_dreamer.cli import override_option
from chuck_dreamer.common.episode_spec import EpisodeSpec

logger = logging.getLogger(__name__)


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


@click.command("augment-keyframes")
@click.argument("sources", required=True, nargs=-1,
                metavar="DATASET_ID[#EPISODES]...")
@click.option("--stride", default=30, type=int,
              help="Sample one candidate every STRIDE frames inside each gap "
                   "between consecutive manual keyframes (default 30).")
@click.option("--search-radius-px", default=60, type=int,
              help="Pixel radius around the previous accepted (u, v) inside "
                   "which the bg-foreground blob must lie (default 60). The "
                   "object can't travel further than this between adjacent "
                   "candidate frames at typical fps.")
@click.option("--area-ratio-min", default=0.4, type=float,
              help="Reject if (candidate area / previous area) is outside "
                   "[ratio, 1/ratio]. Default 0.4 (so blob can grow/shrink "
                   "by 2.5x before we reject).")
@click.option("--min-confidence", default=0.3, type=float,
              help="Reject poses with confidence below this (default 0.5).")
@click.option("--max-residual-px", default=2.0, type=float,
              help="Reject poses with reprojection residual above this "
                   "(default 6.0 px).")
@click.option("--ee-guard-px", default=80, type=int,
              help="Reject candidates where the EE projects within this "
                   "pixel radius of the search window. The EE blob is the "
                   "most common false positive. Set to 0 to disable.")
@click.option("--bg-samples", default=16, type=int,
              help="Frames sampled from the empty episode to build the "
                   "background model (default 16). Reused from the cache "
                   "if available.")
@click.option("--rebuild-bg", is_flag=True, default=False,
              help="Force recompute of the scene background (default: "
                   "reuse cached scene_bg.npz if present).")
@click.option("--auto-accept", is_flag=True, default=False,
              help="Skip the review UI: write every surviving proposal "
                   "straight to the sidecar. Use only when you've already "
                   "validated thresholds for this dataset.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Compute and review proposals but do not write the "
                   "sidecar.")
@click.option("--ee-key", default="ee_pos", type=str,
              help="Sample key to read the end-effector position from for "
                   "the EE-guard. Set to '' to skip the EE-guard even if "
                   "--ee-guard-px > 0.")
@override_option
@click.pass_context
def augment_keyframes_cmd(ctx, sources: tuple[str, ...],
                          stride: int, search_radius_px: int,
                          area_ratio_min: float, min_confidence: float,
                          max_residual_px: float, ee_guard_px: int,
                          bg_samples: int, rebuild_bg: bool,
                          auto_accept: bool, dry_run: bool,
                          ee_key: str,
                          overrides: tuple[str, ...]) -> None:
  """Propose extra keyframes between manual ones; review and save.

  Run ``prompt-episodes`` first to seed the start/end clicks. This
  command only generates *between* existing manual keyframes, never
  before the first one or after the last.
  """
  cfg = load_config(ctx.obj["config_path"], overrides=overrides)
  ol_cfg = init_from_config(cfg)
  cache_root = Path(ol_cfg.cache_dir)

  parsed = EpisodeSpec.parse_many(
    sources, allow_frames=False, command="augment-keyframes")

  # Resolve sources into a flat list of (dataset_id, episode_idx, length).
  pending: list[tuple[str, int, int]] = []
  for src in parsed:
    try:
      slices, _ = src.read_episodes()
    except Exception as e:
      raise click.ClickException(
        f"could not read episode metadata for {src.dataset_id}: {e}")
    for s in slices:
      pending.append((src.dataset_id, s.episode_index, s.length))

  if not pending:
    raise click.ClickException("no (dataset, episode) pairs to process.")

  # Group by dataset so we build calibration + estimator + scene_bg once
  # per dataset and reuse it across all its episodes.
  by_ds: dict[str, list[tuple[int, int]]] = {}
  for did, ep, ep_len in pending:
    by_ds.setdefault(did, []).append((ep, ep_len))

  from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
  from ..estimator import ObjectPoseEstimator

  total_proposed = 0
  total_accepted = 0
  for did, episodes in by_ds.items():
    click.echo(f"\n=== {did} ({len(episodes)} episodes) ===")
    try:
      cal = CameraCalibration.load(cache_root, did)
    except FileNotFoundError as e:
      click.echo(click.style(f"  skip: {e}", fg="yellow"))
      continue

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
        cal,
        stride=stride,
        search_radius_px=search_radius_px,
        area_ratio_min=area_ratio_min,
        min_confidence=min_confidence,
        max_residual_px=max_residual_px,
        ee_guard_px=ee_guard_px,
        ee_key=ee_key,
        auto_accept=auto_accept,
        dry_run=dry_run,
      )
      total_proposed += ep_count
      total_accepted += ep_accept

  click.echo(f"\nDone. Proposed {total_proposed} keyframes, accepted "
             f"{total_accepted}{' (dry-run; nothing written)' if dry_run else ''}.")


# ---------------------------------------------------------------------------
# Per-episode processing
# ---------------------------------------------------------------------------


def _process_episode(ds, dataset_id: str, episode_idx: int, ep_len: int,
                     cache_root: Path, estimator, camera_key: str,
                     calibration: CameraCalibration,
                     *,
                     stride: int, search_radius_px: int,
                     area_ratio_min: float,
                     min_confidence: float, max_residual_px: float,
                     ee_guard_px: int, ee_key: str,
                     auto_accept: bool, dry_run: bool,
                     ) -> tuple[int, int]:
  """Run augmentation for one episode. Returns (n_proposed, n_accepted).

  For each gap between consecutive manual keyframes:

    1. Run a forward chain from the left anchor and a backward chain
       from the right anchor. Each chain marches stride frames at a
       time, skipping frames the cheap pre-checks reject, and warm-
       starting the pose estimator from the previous accepted fit.

    2. Merge the two chains per stride-aligned frame: if both produced
       a fit at the same rel, keep the one with lower residual (tie-
       break: higher confidence). If only one direction fitted that
       frame, keep it.

    3. Stream the merged proposals through the review UI; save each
       accepted one immediately.

  The bidirectional pass exists because the forward chain tends to
  drift the further it gets from its anchor, and the right manual
  keyframe contains no information in a one-directional chain. With
  two chains, drift-prone middle frames of long gaps usually have at
  least one direction that produces a clean fit.
  """
  manual = load_keyframe_prompts(cache_root, dataset_id, episode_idx)
  if len(manual) < 2:
    click.echo(f"  ep {episode_idx}: <2 manual keyframes ({len(manual)}); "
               f"need at least start+end. Skipping.")
    return 0, 0

  try:
    ep_fr, ep_to = next(
      b for e, b in all_episode_bounds_from_meta(dataset_id) if e == episode_idx)
  except StopIteration:
    click.echo(f"  ep {episode_idx}: no metadata; skipping.")
    return 0, 0

  manual_rel = sorted(int(k) for k in manual.keys())
  click.echo(f"\n  ep {episode_idx}: len={ep_len}, manual KFs at "
             f"rel={manual_rel}")

  from ..estimator import _EpisodeContext

  n_proposed = 0
  n_accepted = 0
  quit_episode = False

  for i in range(len(manual_rel) - 1):
    if quit_episode:
      break
    a_rel = manual_rel[i]
    b_rel = manual_rel[i + 1]
    if b_rel - a_rel <= stride:
      continue

    a_prompt = manual[a_rel]
    b_prompt = manual[b_rel]
    a_anchor = _anchor_pose(ds, ep_fr, ep_to, a_rel, a_prompt,
                            camera_key, estimator, calibration)
    if a_anchor is None:
      click.echo(f"    anchor rel={a_rel}: estimator failed to fit; "
                 f"skipping gap [{a_rel}, {b_rel}].")
      continue
    b_anchor = _anchor_pose(ds, ep_fr, ep_to, b_rel, b_prompt,
                            camera_key, estimator, calibration)
    if b_anchor is None:
      click.echo(f"    anchor rel={b_rel}: estimator failed to fit; "
                 f"falling back to forward-only over gap [{a_rel}, {b_rel}].")

    fwd_chain = _ChainState(
      pose = a_anchor[0], uv = a_anchor[1], area = a_anchor[2], rel = a_rel,
    )
    fwd_proposals = _chain_proposals(
      ds, ep_fr, ep_to, camera_key, estimator, calibration, fwd_chain,
      start_rel = a_rel + stride,
      end_rel   = b_rel,
      step      = stride,
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
      last_fwd_rel = a_rel + ((b_rel - 1 - a_rel) // stride) * stride
      if last_fwd_rel <= a_rel:
        last_fwd_rel = a_rel + stride
      bwd_start = last_fwd_rel
      bwd_end   = a_rel
      bwd_proposals = _chain_proposals(
        ds, ep_fr, ep_to, camera_key, estimator, calibration, bwd_chain,
        start_rel = bwd_start,
        end_rel   = bwd_end,
        step      = -stride,
        search_radius_px=search_radius_px, area_ratio_min=area_ratio_min,
        min_confidence=min_confidence, max_residual_px=max_residual_px,
        ee_guard_px=ee_guard_px, ee_key=ee_key, direction="backward",
      )

    merged = _merge_bidirectional(fwd_proposals, bwd_proposals)
    click.echo(f"    gap [{a_rel}, {b_rel}]: forward={len(fwd_proposals)}  "
               f"backward={len(bwd_proposals)}  merged={len(merged)} "
               f"proposals.")

    for proposal in merged:
      n_proposed += 1
      if auto_accept:
        decision = 'a'
        kept = proposal
      else:
        decision, kept = _review_one_proposal(
          proposal, dataset_id, episode_idx, ep_len,
          estimator, calibration)

      if decision == 'q':
        quit_episode = True
        break
      if decision == 'a' and kept is not None:
        if not dry_run:
          save_keyframe_prompt(cache_root, dataset_id, episode_idx,
                               kept.ep_rel, kept.uv)
          click.echo(f"    saved rel={kept.ep_rel} u={kept.uv[0]} "
                     f"v={kept.uv[1]} -> "
                     f"{sidecar_path(cache_root, dataset_id)}")
        else:
          click.echo(f"    dry-run: would save rel={kept.ep_rel}.")
        n_accepted += 1

  return n_proposed, n_accepted


@dataclass
class _ChainState:
  """Mutable per-gap state: where the chain currently believes the object is."""
  pose: object              # ObjectPose
  uv:   tuple[int, int]
  area: float               # rendered silhouette area in px
  rel:  int                 # episode-relative frame index this state came from


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

  This is the building block used twice per gap: once forward from
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

_WIN_NAME = "augment-keyframes"
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

  Mirrors what ``prompt-episodes`` would do if the user had clicked
  this frame as a manual keyframe: reset the episode context so the
  rest-class search runs again, then call ``estimate(prompt=...)``.
  Returns None if the segmenter or rest-class fit fails.
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
