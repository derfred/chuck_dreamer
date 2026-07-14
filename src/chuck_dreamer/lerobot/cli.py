"""`import-lerobot` CLI group: the import pipeline and its supporting steps.

``import-lerobot run`` converts a LeRobot v3 HF dataset into the repo's
episode files, optionally running the EE-FK and object-pose pipeline
stages. ``import-lerobot doctor`` checks that every artifact the requested
stages need is present and prints remediation for anything missing. The
calibration / annotation commands (``calibrate-intrinsics``,
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


def _doctor_import_lerobot(run: Run, episode_config_path: Path | None) -> bool:
  """Check calibration / model files required by import-lerobot.

  Drives the same :class:`Run` the importer uses: it iterates each
  selected episode's stage list and prints every stage's
  :meth:`Stage.requirements`, so adding a stage surfaces its prerequisites
  here automatically. Requirements may be episode-specific (e.g. the
  frame-0 segmentation prompt), so this iterates *every* selected episode
  rather than the dataset alone. The ``--episode-config`` / ``T_world_arm``
  check is importer-level (not a stage) and is handled separately at the end.
  """
  from ..common import FK_MODEL_PATH
  from .stages import Requirement

  dataset_id = run.spec.dataset_id
  click.echo(f"Doctor check for import-lerobot {dataset_id}")
  click.echo(f"  with_ee_pos={run.params.get('with_ee_pos', False)}  "
             f"with_object_pose={run.params.get('with_object_pose', False)}  "
             f"episode_config={'yes' if episode_config_path else 'no'}")

  missing: list[str] = []
  seen: set[tuple[str, Path]] = set()

  def check(req: Requirement) -> None:
    key = (req.label, req.path)
    if key in seen:
      return
    seen.add(key)
    if req.satisfied():
      click.echo(click.style("  OK    ", fg="green") + f"{req.label}: {req.path}")
    else:
      click.echo(click.style("  MISS  ", fg="red") + f"{req.label}: {req.path}")
      click.echo(f"        run: {req.remediation}")
      missing.append(req.label)

  try:
    slices, _ = run.read_slices()
  except Exception as e:  # noqa: BLE001 - surface metadata-read failures as doctor output
    click.echo(click.style(f"  ERROR  could not read episodes: {e}", fg="red"))
    return False
  episode_indices = [sl.episode_index for sl in slices]
  click.echo(f"  episodes: {episode_indices or 'none'}")

  for ep_idx in episode_indices:
    for stage in run.pipeline(ep_idx):
      for req in stage.requirements():
        check(req)

  if episode_config_path is not None:
    check(Requirement(
      "fk_episode_config.json", episode_config_path,
      f"create {episode_config_path} listing touch episodes for {dataset_id}"))
    # T_world_arm derivation runs the FK regardless of --with-ee-pos.
    check(Requirement(
      "FK MuJoCo model", FK_MODEL_PATH,
      "restore assets/mujoco/so101_arm.xml from git"))

  if missing:
    click.echo(click.style(
      f"\nDoctor: {len(missing)} missing artifact(s).", fg="red"))
    return False
  click.echo(click.style("\nDoctor: all required artifacts present.", fg="green"))
  return True


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
         "(chuck_dreamer.common.fk) to fill ee_pos "
         "/ ee_quat / ee_action (default: on). Required for "
         "EE-mode training; needs assets/mujoco/so101_arm.xml.")(fn)
  fn = click.option(
    "--with-object-pose/--no-object-pose", default=True,
    help="Run SAM2 segmentation + per-frame analysis-by-synthesis pose "
         "fit to fill object_xy / object_gap_too_long (default: on). "
         "Requires a per-dataset calibration cache entry "
         "(`main.py import-lerobot calibrate-intrinsics` + "
         "`main.py import-lerobot annotate-mat`) and a cached frame-0 "
         "prompt (`main.py import-lerobot prompt-episodes`).")(fn)
  fn = click.option(
    "--video-key", default=None, type=str,
    help="Video feature key (e.g. observation.images.wrist). "
         "Defaults to the first one.")(fn)
  fn = click.option(
    "--episode-config", "episode_config_path", default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a fk_episode_config.json. When the imported "
         "repo_id appears in it, derive T_world_arm from the "
         "touch episodes (Umeyama on joint medians + FK) and "
         "stamp it onto each written episode's metadata.")(fn)
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
                           with_ee_pos, with_object_pose, tags, name_prefix,
                           episode_config_path, jobs, gpus):
  """Convert a LeRobot v3 HF dataset (REPO_ID) into the repo's episode files.

  Loads the dataset through ``LeRobotDataset`` (which decodes video frames
  and exposes every feature per frame), groups frames by episode, and
  writes one episode file per LeRobot episode under --output.

  Pipeline stages applied to each imported episode:

    1. Raw import (always): image, joint_action, joint_qpos, timestamp.
    2. ``--with-ee-pos`` (default on): joint rescaling + FK to ee_pos /
       ee_quat / ee_action.
    3. ``--with-object-pose`` (default on): SAM2 segmentation + per-frame
       analysis-by-synthesis pose fit to object_xy / object_gap_too_long.

  ``REPO_ID`` accepts the same ``DATASET[#EPISODES]`` syntax used by
  ``calibrate-intrinsics`` and ``prompt-episodes`` — e.g.
  ``user/dataset#8-99`` to import only episodes 8 onward. ``:FRAMES``
  syntax is rejected here (frame-range filtering doesn't make sense
  for the importer).
  """
  spec = EpisodeSpec.parse(repo_id, allow_frames=False, command="import-lerobot run")

  cfg    = load_config(ctx.obj["config_path"])
  params = dict(with_ee_pos=with_ee_pos, with_object_pose=with_object_pose)
  run    = Run(cfg, spec, params, video_key=video_key)

  n_episodes = len(run.slices)
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
                              episode_config_path):
  """Check that all calibration / model files required for the requested
  pipeline stages are present, without importing anything.

  Walks the same per-episode stage list ``run`` would execute and prints
  every stage requirement with the exact remediation command for each
  missing artifact. Exits non-zero if anything is missing.
  """
  spec = EpisodeSpec.parse(repo_id, allow_frames=False,
                           command="import-lerobot doctor")

  cfg    = load_config(ctx.obj["config_path"])
  params = dict(with_ee_pos=with_ee_pos, with_object_pose=with_object_pose)
  run    = Run(cfg, spec, params, video_key=video_key)

  ok = _doctor_import_lerobot(run, episode_config_path=episode_config_path)
  ctx.exit(0 if ok else 1)


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


@store_group.command("migrate")
@click.option("--cache", "cache_root", default="calibration_cache",
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              show_default=True, help="Legacy calibration_cache directory.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print the migration plan without writing.")
@_STORE_ROOT_OPT
def store_migrate_cmd(cache_root: Path, dry_run: bool, store_root: Path):
  """One-time, idempotent migration of calibration_cache/ into the store.

  Prompts once per dataset for the camera id (the legacy layout has no
  camera identity); the answer also seeds each dataset's dataset_config.
  The source cache is never modified or deleted.
  """
  from chuck_dreamer.store import ArtifactStore
  from chuck_dreamer.store.migrate import migrate_calibration_cache

  answers: dict[str, str] = {}

  def camera_id_for(dataset_id: str) -> str:
    if dataset_id not in answers:
      default = next(iter(answers.values()), None)
      answers[dataset_id] = click.prompt(
        f"camera id for {dataset_id}", default=default)
    return answers[dataset_id]

  actions = migrate_calibration_cache(
    cache_root, ArtifactStore(store_root), camera_id_for, dry_run=dry_run)
  for a in actions:
    click.echo(str(a))
  n = sum(1 for a in actions if a.kind == "migrated")
  conflicts = sum(1 for a in actions if a.kind == "conflict")
  click.echo(f"\n{'would migrate' if dry_run else 'migrated'} {n} artifact(s); "
             f"{conflicts} conflict(s).")
  if conflicts:
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Calibration / annotation subcommands (defined in lerobot.annotation, next
# to the code they drive; wired here so the whole import surface lives on
# this one group).
# ---------------------------------------------------------------------------
def _register_annotation_commands() -> None:
  from .annotation import (
    annotate_mat_cmd,
    calibrate_intrinsics_cmd,
    prompt_episodes_cmd,
  )
  for cmd in (calibrate_intrinsics_cmd, annotate_mat_cmd, prompt_episodes_cmd):
    import_lerobot_cmd.add_command(cmd)


_register_annotation_commands()
