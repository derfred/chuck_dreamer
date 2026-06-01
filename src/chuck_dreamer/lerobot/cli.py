"""`import-lerobot` CLI: convert a LeRobot v3 HF dataset into the repo's
episode files, optionally running the EE-FK and object-pose pipeline
stages and stamping a derived ``T_world_arm`` onto each episode.
"""
from __future__ import annotations

import logging
from pathlib import Path
import click

from chuck_dreamer.common.episode_spec import EpisodeSpec
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
  click.echo(f"  with_ee_pos={run.with_ee_pos}  with_object_pose={run.with_object_pose}  "
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
  *,
  format: str = "hdf5",
  tags: tuple[str, ...] = (),
  name_prefix: str | None = None,
) -> int:
  """Run the importer over ``run`` with a tqdm progress bar; return the count
  of episodes written. Surfaces missing-artifact errors as ``ClickException``."""
  from tqdm import tqdm

  from .importer import import_dataset

  episode_filter = run.spec.episode_filter
  count = 0
  try:
    for ep_idx, out_path in tqdm(
      import_dataset(run, output_dir, format=format, tags=tags, name_prefix=name_prefix),
      desc="Episodes",
      total=len(episode_filter) if episode_filter is not None else None,
    ):
      count += 1
      logger.info("episode %d → %s", ep_idx, out_path)
  except FileNotFoundError as e:
    raise click.ClickException(str(e))
  return count


@click.command("import-lerobot")
@click.argument("repo_id", type=str, metavar="REPO_ID[#EPISODES]")
@click.option("--output", default=None, type=str,
              help="Output directory. Required unless --doctor is passed.")
@click.option("--format", "fmt", default="rerun", type=click.Choice(["hdf5", "rerun"]),
              help="Episode output format (hdf5 or rerun)")
@click.option("--video-key", default=None, type=str,
              help="Video feature key (e.g. observation.images.wrist). Defaults to the first one.")
@click.option("--with-ee-pos/--no-ee-pos", default=True,
              help="Convert joint_qpos to radians and run the MuJoCo FK "
                   "(chuck_dreamer.real.fk_calibration.fk) to fill ee_pos "
                   "/ ee_quat / ee_action (default: on). Required for "
                   "EE-mode training; needs assets/mujoco/so101_arm.xml.")
@click.option("--with-object-pose/--no-object-pose", default=True,
              help="Run SAM2 segmentation + per-frame pose fit + RTS smoothing to "
                   "fill object_xy / object_gap_too_long (default: on). Requires a "
                   "per-dataset calibration cache entry (`main.py analyze-cameras`) "
                   "and a cached frame-0 prompt (`main.py prompt-episodes`).")
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
@click.option("--episode-config", "episode_config_path", default=None,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a fk_episode_config.json. When the imported "
                   "repo_id appears in it, derive T_world_arm from the "
                   "touch episodes (Umeyama on joint medians + FK) and "
                   "stamp it onto each written episode's metadata.")
@click.option("--doctor", "doctor", is_flag=True, default=False,
              help="Check that all calibration / model files required for "
                   "the requested pipeline stages are present, then exit "
                   "without importing. Prints the exact remediation command "
                   "for each missing artifact.")
@click.pass_context
def import_lerobot_cmd(ctx, repo_id, output, fmt, video_key,
                       with_ee_pos, with_object_pose, tags, name_prefix,
                       episode_config_path, doctor):
  """Convert a LeRobot v3 HF dataset (REPO_ID) into the repo's episode files.

  Loads the dataset through ``LeRobotDataset`` (which decodes video frames
  and exposes every feature per frame), groups frames by episode, and
  writes one episode file per LeRobot episode under --output.

  Pipeline stages applied to each imported episode:

    1. Raw import (always): image, joint_action, joint_qpos, timestamp.
    2. ``--with-ee-pos`` (default on): joint rescaling + FK to ee_pos /
       ee_quat / ee_action.
    3. ``--with-object-pose`` (default on): SAM2 segmentation + per-frame
       pose fit + RTS smoothing to object_xy / object_gap_too_long.

  ``REPO_ID`` accepts the same ``DATASET[#EPISODES]`` syntax used by
  ``calibrate-intrinsics`` and ``prompt-episodes`` — e.g.
  ``user/dataset#8-99`` to import only episodes 8 onward. ``:FRAMES``
  syntax is rejected here (frame-range filtering doesn't make sense
  for the importer).
  """

  spec = EpisodeSpec.parse(repo_id, allow_frames=False, command="import-lerobot")

  if not doctor and not output:
    raise click.UsageError("--output is required (unless --doctor is passed).")

  run = Run(spec, with_ee_pos=with_ee_pos, with_object_pose=with_object_pose,
            video_key=video_key)

  if doctor:
    ok = _doctor_import_lerobot(run, episode_config_path=episode_config_path)
    ctx.exit(0 if ok else 1)
  else:
    episode_filter = spec.episode_filter
    click.echo(
      f"Importing {spec.dataset_id} → {output}  (format={fmt}, "
      f"with_ee_pos={with_ee_pos}, with_object_pose={with_object_pose}, "
      f"tags={list(tags)}, name_prefix={name_prefix!r}, "
      f"episodes={'all' if episode_filter is None else sorted(episode_filter)})"
    )
    count = _import_dataset(run, output, format=fmt, tags=tuple(tags), name_prefix=name_prefix)
    click.echo(f"Done. Wrote {count} episodes.")
