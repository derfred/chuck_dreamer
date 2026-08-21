"""`import-lerobot` CLI group: the import pipeline and its supporting steps.

``import-lerobot run`` converts a LeRobot v3 HF dataset into the repo's
episode files, optionally running the EE-FK, object-pose, and table-frame
pipeline nodes. ``import-lerobot doctor`` (see ``doctor.py``) derives every
leaf input from the node declarations and checks the artifact store /
assets per resolved scope key, printing remediation for anything missing.
The calibration / annotation commands (``calibrate-intrinsics``,
``annotate-mat``, ``prompt-episodes``) and the artifact-store commands are
wired onto the group at the bottom of this module.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import click

from chuck_dreamer.config import load_config
from chuck_dreamer.lerobot.episode_spec import EpisodeSpec
from .pipeline import Run

logger = logging.getLogger(__name__)


def _import_dataset(
  run: Run,
  output_dir: str,
  format: str = "hdf5",
  tags: tuple[str, ...] = (),
  name_prefix: str | None = None,
  jobs: int = 1,
  devices: list[str] | None = None,
) -> int:
  """Run the importer over ``run`` with a tqdm progress bar; return the count
  of episodes written. Surfaces missing-artifact errors as ``ClickException``."""
  from tqdm import tqdm

  from .importer import import_dataset

  episode_filter = run.spec.episode_filter
  click.echo(
    f"Importing {run.spec.dataset_id} → {output_dir}  (format={format}, "
    f"with_ee_pos={run.params.get('with_ee_pos', False)}, "
    f"with_object_pose={run.params.get('with_object_pose', False)}, "
    f"tags={list(tags)}, name_prefix={name_prefix!r}, "
    f"jobs={jobs}, devices={devices or 'configured'}, "
    f"episodes={'all' if episode_filter is None else sorted(episode_filter)})"
  )

  count = 0
  try:
    for ep_idx, out_path in tqdm(
      import_dataset(run, output_dir, format=format, tags=tags,
                     name_prefix=name_prefix,
                     jobs=jobs, devices=devices),
      desc="Episodes",
      total=len(episode_filter) if episode_filter is not None else None,
    ):
      count += 1
      logger.info("episode %d → %s", ep_idx, out_path)
  except FileNotFoundError as e:
    raise click.ClickException(str(e))
  click.echo(f"Done. Wrote {count} episodes.")
  return count


def _resolve_jobs_and_devices(
  jobs: int, gpus: str | None, n_episodes: int,
) -> tuple[int, list[str] | None]:
  """Turn the ``--jobs`` / ``--gpus`` CLI flags into ``(jobs, devices)``.

  ``jobs == 0`` ⇒ auto: ``min(cpu_count - 2, n_episodes)`` (leave headroom for
  the producer + decode threads), floored at 1. ``--gpus`` is either an integer
  count (``"2"`` → ``["cuda:0", "cuda:1"]``) or an explicit comma-separated
  device list (``"cuda:0,cuda:3"``). ``None`` leaves the configured device
  untouched (one producer)."""
  import os

  if jobs == 0:
    jobs = max(1, min((os.cpu_count() or 2) - 2, n_episodes))

  devices: list[str] | None
  if gpus is None:
    devices = None
  elif gpus.strip().isdigit():
    devices = [f"cuda:{i}" for i in range(int(gpus))]
  else:
    devices = [d.strip() for d in gpus.split(",") if d.strip()]
  return jobs, devices


# Shared stage flags: both `run` and `doctor` resolve the same stage set.
def _stage_options(fn):
  fn = click.option(
    "--with-ee-pos/--no-ee-pos", default=True,
    help="Convert joint_qpos to radians and run the MuJoCo FK "
         "(chuck_dreamer.common.fk) to fill ee_pos_arm / ee_quat_arm "
         "/ ee_action_arm in mm (default: on). Required for "
         "EE-mode training; needs assets/mujoco/so101_arm.xml.")(fn)
  fn = click.option(
    "--with-object-pose/--no-object-pose", default=True,
    help="Run SAM2 segmentation + per-frame analysis-by-synthesis pose "
         "fit to fill obj_pos_table (+ its validity sibling) "
         "(default: on). Requires the calibration artifacts "
         "(`main.py import-lerobot calibrate-intrinsics` + "
         "`main.py import-lerobot annotate-mat`) and a stored frame-0 "
         "prompt (`main.py import-lerobot prompt-episodes`).")(fn)
  fn = click.option(
    "--with-table-frame/--no-table-frame", default=False,
    help="Also emit the table-frame coordinate tracks (ee_pos_table / "
         "ee_quat_table / ee_action_table / obj_pos_arm). Needs the "
         "dataset's table_to_arm transform in the store "
         "(`main.py import-lerobot extract-table-to-arm`).")(fn)
  fn = click.option(
    "--video-key", default=None, type=str,
    help="Video feature key (e.g. observation.images.wrist). "
         "Defaults to the first one.")(fn)
  return fn


@click.group("import-lerobot")
def import_lerobot_cmd():
  """The teleop import pipeline: convert LeRobot datasets to episode files,
  plus the calibration / annotation steps the pipeline depends on.

  Typical order: `calibrate-intrinsics` (once per camera) → `annotate-mat`
  and `prompt-episodes` (once per dataset) → `doctor` to verify → `run`.
  """


@import_lerobot_cmd.command("run")
@click.argument("repo_id", type=str, metavar="REPO_ID[#EPISODES]")
@click.option("--output", required=True, type=str,
              help="Output directory for the written episode files.")
@click.option("--format", "fmt", default="rerun", type=click.Choice(["hdf5", "rerun"]),
              help="Episode output format (hdf5 or rerun)")
@_stage_options
@click.option("--tag", "tags", multiple=True, metavar="TAG",
              help="Tag to stamp onto each written episode's metadata. Repeatable. "
                   "The replay buffer reads these for protected_tags (no-evict) and "
                   "tag_weights (sample upweighting). Typical use: --tag real for "
                   "physical-robot recordings.")
@click.option("--name-prefix", default=None, type=str,
              help="String prepended to each written episode's filename. "
                   "Use to import multiple datasets into the same --output "
                   "directory without filename collisions (default "
                   "names are episode-00000, etc. — identical across "
                   "datasets). Typical: --name-prefix task_2_1805_2")
@click.option("--jobs", "-j", default=1, type=int, metavar="N",
              help="Number of CPU worker processes for the object-pose fit "
                   "(default 1 = serial, byte-identical output). N>1 pipelines "
                   "GPU segmentation against the CPU pose fit across episodes; "
                   "N=0 auto-picks min(cpu_count-2, n_episodes). Per-episode "
                   "results are identical regardless of N.")
@click.option("--gpus", "gpus", default=None, type=str, metavar="N|LIST",
              help="GPU producer devices for parallel import (with --jobs>1). "
                   "Either a count (2 → cuda:0,cuda:1) or an explicit list "
                   "(cuda:0,cuda:3). One producer thread is pinned per device, "
                   "raising the segmentation throughput ceiling ~Nx. Default: "
                   "the configured object_localization.device, one producer.")
@click.pass_context
def import_lerobot_run_cmd(ctx, repo_id, output, fmt, video_key,
                           with_ee_pos, with_object_pose, with_table_frame,
                           tags, name_prefix, jobs, gpus):
  """Convert a LeRobot v3 HF dataset (REPO_ID) into the repo's episode files.

  Loads the dataset through ``LeRobotDataset`` (which decodes video frames
  and exposes every feature per frame), groups frames by episode, and
  writes one episode file per LeRobot episode under --output.

  Pipeline stages applied to each imported episode:

    1. Raw import (always): image, joint_action, joint_qpos, timestamp.
    2. ``--with-ee-pos`` (default on): joint rescaling + FK to ee_pos_arm /
       ee_quat_arm / ee_action_arm (mm).
    3. ``--with-object-pose`` (default on): SAM2 segmentation + per-frame
       analysis-by-synthesis pose fit to obj_pos_table (+ validity).
    4. ``--with-table-frame``: the frame-crossing nodes (ee_pos_table /
       ee_quat_table / ee_action_table / obj_pos_arm) via the dataset's
       table_to_arm transform.

  ``REPO_ID`` accepts the same ``DATASET[#EPISODES]`` syntax used by
  ``calibrate-intrinsics`` and ``prompt-episodes`` — e.g.
  ``user/dataset#8-99`` to import only episodes 8 onward. ``:FRAMES``
  syntax is rejected here (frame-range filtering doesn't make sense
  for the importer).
  """
  spec   = EpisodeSpec.parse(repo_id, allow_frames=False, command="import-lerobot run")
  cfg    = load_config(ctx.obj["config_path"])
  params = dict(with_ee_pos=with_ee_pos, with_object_pose=with_object_pose, with_table_frame=with_table_frame)
  run    = Run(cfg, spec, params, video_key=video_key)

  n_episodes             = len(run.slices)
  resolved_jobs, devices = _resolve_jobs_and_devices(jobs, gpus, n_episodes)
  _import_dataset(run, output, format=fmt, tags=tuple(tags),
                  name_prefix=name_prefix,
                  jobs=resolved_jobs, devices=devices)


@import_lerobot_cmd.command("doctor")
@click.argument("repo_id", type=str, metavar="REPO_ID[#EPISODES]")
@_stage_options
@click.pass_context
def import_lerobot_doctor_cmd(ctx, repo_id, video_key,
                              with_ee_pos, with_object_pose,
                              with_table_frame):
  """Check that every artifact the requested pipeline needs is present,
  without importing anything.

  Derives the leaf inputs from the node declarations of the same
  per-episode node list ``run`` would execute, checks each against the
  artifact store / assets at its resolved scope key, and prints the exact
  remediation for anything missing (spec §10.3). Exits non-zero if
  anything is missing.
  """
  from .doctor import doctor_run

  spec   = EpisodeSpec.parse(repo_id, allow_frames=False, command="import-lerobot doctor")
  cfg    = load_config(ctx.obj["config_path"])
  params = dict(with_ee_pos=with_ee_pos, with_object_pose=with_object_pose, with_table_frame=with_table_frame)
  run    = Run(cfg, spec, params, video_key=video_key)

  ok = doctor_run(run)
  ctx.exit(0 if ok else 1)


@import_lerobot_cmd.command("extract-table-to-arm")
@click.argument("repo_id", type=str, metavar="REPO_ID")
@click.option("--variant", type=click.Choice(["3", "7"]), default=None,
              help="Touchpoint protocol variant. Also written into the "
                   "dataset's dataset_config (it is the annotation that "
                   "selects the protocol); omit to use the value already "
                   "in dataset_config.")
@click.option("--force", is_flag=True, default=False,
              help="Recompute even when a transform with matching input "
                   "versions is already in the store, and persist even if "
                   "the residual reject gate fails.")
@click.pass_context
def extract_table_to_arm_cmd(ctx, repo_id, variant, force):
  """Derive the table↔arm transform from REPO_ID's touchpoint episodes
  (spec §7.2) and cache it in the artifact store as ``table_to_arm``.

  The touchpoint variant (3 or 7) names the capture protocol: which episode
  indices are touchpoints and which known mat point each one touches. Joint
  values are read from the data parquet only (no video decode), medianed
  per episode, run through FK, and rigidly aligned against the table-frame
  targets. Collinear touch sets fall back to a planar heuristic (arm z
  assumed parallel to table z) — the residual report says when that
  happened.
  """
  from chuck_dreamer.store import MissingArtifact, annotation_record, for_dataset

  from .nodes.table_to_arm import ResidualGateError, extract_table_to_arm

  spec = EpisodeSpec.parse(repo_id, allow_frames=False, command="extract-table-to-arm")
  if spec.episode_filter is not None or spec.episode_open_from is not None:
    raise click.ClickException(
      "extract-table-to-arm operates on the whole dataset; drop the "
      f"#EPISODES portion of {repo_id!r} (the touchpoint episodes are "
      "selected by the variant).")

  cfg   = load_config(ctx.obj["config_path"])
  run   = Run(cfg, spec, params={})
  store = run.ctx.store
  if store is None:
    raise click.ClickException(
      "extract-table-to-arm needs the artifact store; set store.root in "
      "the config")

  if variant is not None:
    scope    = for_dataset(run.dataset_id)
    existing = (store.get("dataset_config", scope)[0] if store.has("dataset_config", scope) else {})
    payload  = {**existing, "touchpoint_variant": int(variant)}
    if payload != existing:
      store.put("dataset_config", scope, payload,
                annotation_record("dataset_config", scope, "extract-table-to-arm"))
      click.echo(f"store: dataset_config @ {run.dataset_id} touchpoint_variant={variant}")

  try:
    result = extract_table_to_arm(run, force=force)
  except (MissingArtifact, ResidualGateError, ValueError, RuntimeError) as e:
    raise click.ClickException(str(e))

  p = result.payload
  if result.cached:
    click.echo("table_to_arm already in store with matching input versions "
               "(use --force to recompute):")
  for tp in p["touchpoints"]:
    click.echo(f"  ep{tp['episode']:>3}  target {tp['target_mm'][:2]} mm  "
               f"residual {tp['residual_mm']:6.2f} mm  "
               f"spread {max(tp['q_spread_deg']):.2f}°")
  click.echo(f"method={p['method']}  variant={p['variant']}  "
             f"max_residual={p['max_residual_mm']:.2f} mm  "
             f"rms={p['rms_residual_mm']:.2f} mm")
  for w in result.warnings:
    click.echo(click.style(f"warning: {w}", fg="yellow"))


# ---------------------------------------------------------------------------
# Artifact-store subcommands
# ---------------------------------------------------------------------------
@import_lerobot_cmd.group("store")
def store_group():
  """Inspect and manage the artifact store (docs/trainer/artifact_store.md)."""


_STORE_ROOT_OPT = click.option(
  "--store", "store_root", default="store", type=click.Path(path_type=Path),
  show_default=True, help="Artifact-store root directory.")


@store_group.command("ls")
@_STORE_ROOT_OPT
def store_ls_cmd(store_root: Path):
  """List every artifact in the store with its scope and producing node."""
  from chuck_dreamer.store import ArtifactStore

  infos = ArtifactStore(store_root).ls()
  if not infos:
    click.echo(f"store {store_root}: empty")
    return
  for info in infos:
    click.echo(f"{info.artifact:16s} {json.dumps(info.scope.to_json()):48s} "
               f"{info.node:24s} {info.created_at}")


@store_group.command("show")
@click.argument("artifact")
@click.option("--camera", default=None, help="CAMERA scope key.")
@click.option("--dataset", default=None, help="DATASET scope key.")
@click.option("--episode", default=None, type=int, help="EPISODE scope key (with --dataset).")
@_STORE_ROOT_OPT
def store_show_cmd(artifact: str, camera: str | None, dataset: str | None,
                   episode: int | None, store_root: Path):
  """Print one artifact's provenance record (and JSON payloads)."""
  from chuck_dreamer.store import ArtifactStore, MissingArtifact, Scope

  scope = Scope(camera=camera, dataset=dataset, episode=episode)
  store = ArtifactStore(store_root)
  try:
    payload, record = store.get(artifact, scope)
  except (MissingArtifact, ValueError, KeyError) as e:
    raise click.ClickException(str(e))
  click.echo(json.dumps(record.to_json(), indent=2))
  if store.registry[artifact].codec == "json":
    click.echo(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Calibration / annotation subcommands (defined in lerobot.annotation, next
# to the code they drive; wired here so the whole import surface lives on
# this one group).
# ---------------------------------------------------------------------------
def _register_annotation_commands() -> None:
  from .annotation import (
    annotate_mat_cmd,
    annotate_mat_jpeg_cmd,
    calibrate_intrinsics_cmd,
    prompt_episodes_cmd,
  )
  for cmd in (calibrate_intrinsics_cmd, annotate_mat_cmd, annotate_mat_jpeg_cmd, prompt_episodes_cmd):
    import_lerobot_cmd.add_command(cmd)


_register_annotation_commands()
