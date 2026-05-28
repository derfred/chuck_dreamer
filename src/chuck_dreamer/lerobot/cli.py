"""`import-lerobot` CLI: convert a LeRobot v3 HF dataset into the repo's
episode files, optionally running the EE-FK and object-pose pipeline
stages and stamping a derived ``T_world_arm`` onto each episode.
"""
from __future__ import annotations

import logging
from pathlib import Path

import click

from chuck_dreamer.config import load_config

logger = logging.getLogger(__name__)


def _build_arm_calibration(ctx, dataset_id: str, episode_config_path: Path):
  """Compute a T_world_arm block for ``dataset_id`` from an fk_episode_config.

  Returns the JSON block to stamp onto each episode's metadata, or
  ``None`` if the dataset isn't in the config / Umeyama rejected the
  alignment. Mirrors `calibrate-live`'s 4-touch flow; the only
  difference is that joint medians come from the LeRobot parquet
  instead of the live arm.
  """
  from chuck_dreamer.real.object_localization.calibration.arm import (
    calibrate_from_lerobot_dataset,
  )
  from chuck_dreamer.real.object_localization.runtime import init_from_config
  from chuck_dreamer.real.fk_calibration.fk import FK, load_fk_dq
  from chuck_dreamer.real.fk_calibration.extract_joints import load_episode_config

  from .pipeline import FK_MODEL_PATH

  episode_to_label_by_dataset = load_episode_config(episode_config_path)
  if dataset_id not in episode_to_label_by_dataset:
    click.echo(f"{dataset_id} not in {episode_config_path}; skipping "
               "T_world_arm derivation.")
    return None
  episode_to_label = episode_to_label_by_dataset[dataset_id]

  cfg = load_config(ctx.obj["config_path"])
  ol_cfg = init_from_config(cfg)

  if not FK_MODEL_PATH.exists():
    raise click.ClickException(f"FK model not found at {FK_MODEL_PATH}")
  fk = FK(FK_MODEL_PATH)
  fk_dq = load_fk_dq(ol_cfg.cache_dir)

  click.echo(f"Deriving T_world_arm from {dataset_id} "
             f"({len(episode_to_label)} touches: "
             f"{sorted(episode_to_label.values())})")
  block, residuals, max_res, rms_res = calibrate_from_lerobot_dataset(
    dataset_id=dataset_id,
    episode_to_label=episode_to_label,
    fk=fk,
    L_mm=ol_cfg.mat_line_length_mm,
    D_mm=ol_cfg.mat_line_separation_mm,
    fk_dq=fk_dq,
    fk_model_path=FK_MODEL_PATH,
  )
  click.echo(f"  max_residual = {max_res:.2f} mm   "
             f"rms_residual = {rms_res:.2f} mm")
  if block is None:
    click.echo(click.style(
      f"  REJECT: max_residual {max_res:.2f}mm exceeds threshold. "
      "T_world_arm NOT stamped onto episodes.", fg="red"))
    return None
  click.echo(click.style(f"  OK — stamping T_world_arm on every episode.",
                         fg="green"))
  return block


def _doctor_import_lerobot(ctx, dataset_id: str, *,
                           with_ee_pos: bool, with_object_pose: bool,
                           episode_config_path: Path | None) -> bool:
  """Check calibration / model files required by import-lerobot.

  Generic over the enabled stages: it resolves the same stage list the
  importer would run and prints each stage's :meth:`Stage.requirements`,
  so adding a stage surfaces its prerequisites here automatically. The
  ``--episode-config`` / ``T_world_arm`` check is importer-level (not a
  stage) and is handled separately at the end.
  """
  from .pipeline import FK_MODEL_PATH
  from .stages import (
    Requirement, StageContext, enabled_from_flags, resolve_stages,
  )

  click.echo(f"Doctor check for import-lerobot {dataset_id}")
  click.echo(f"  with_ee_pos={with_ee_pos}  with_object_pose={with_object_pose}  "
             f"episode_config={'yes' if episode_config_path else 'no'}")

  missing: list[str] = []
  seen: set[Path] = set()

  def check(req: Requirement) -> None:
    if req.path in seen:
      return
    seen.add(req.path)
    if req.satisfied():
      click.echo(click.style("  OK    ", fg="green") + f"{req.label}: {req.path}")
    else:
      click.echo(click.style("  MISS  ", fg="red") + f"{req.label}: {req.path}")
      click.echo(f"        run: {req.remediation}")
      missing.append(req.label)

  sctx = StageContext(source_repo=dataset_id)
  enabled = enabled_from_flags(
    with_ee_pos=with_ee_pos, with_object_pose=with_object_pose)
  for stage in resolve_stages(enabled):
    for req in stage.requirements(sctx):
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


@click.command("import-lerobot")
@click.argument("repo_id", type=str, metavar="REPO_ID[#EPISODES]")
@click.option("--output", default=None, type=str,
              help="Output directory. Required unless --doctor is passed.")
@click.option("--format", "fmt", default="rerun", type=click.Choice(["hdf5", "rerun"]),
              help="Episode output format (hdf5 or rerun)")
@click.option("--video-key", default=None, type=str,
              help="Video feature key (e.g. observation.images.wrist). Defaults to the first one.")
@click.option("--max-episodes", default=None, type=int,
              help="Cap on number of episodes to convert (default: all)")
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
def import_lerobot_cmd(ctx, repo_id, output, fmt, video_key, max_episodes,
                       with_ee_pos, with_object_pose, tags, name_prefix,
                       episode_config_path, doctor):
  """Convert a LeRobot v3 HF dataset (REPO_ID) into the repo's episode files.

  Pulls the parquet + MP4 from Hugging Face, slices per episode using
  ``meta/episodes/*.parquet``, decodes the matching video range, and
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
  from tqdm import tqdm

  from chuck_dreamer.real.object_localization.calibration.spec_parser import (
    parse_calibration_source,
  )
  from chuck_dreamer.lerobot.importer import import_dataset

  try:
    spec = parse_calibration_source(repo_id)
  except ValueError as e:
    raise click.ClickException(str(e))
  if spec.frames is not None:
    raise click.ClickException(
      "import-lerobot doesn't accept the :FRAMES portion of the spec "
      "(only #EPISODES is meaningful). Got: "
      f"{repo_id!r}")
  parsed_repo_id = spec.dataset_id
  episode_filter: set[int] | None = (
    set(spec.episodes) if spec.episodes is not None else None
  )

  if not doctor and not output:
    raise click.UsageError("--output is required (unless --doctor is passed).")

  if doctor:
    ok = _doctor_import_lerobot(
      ctx, parsed_repo_id,
      with_ee_pos=with_ee_pos,
      with_object_pose=with_object_pose,
      episode_config_path=episode_config_path,
    )
    ctx.exit(0 if ok else 1)

  arm_calibration = None
  if episode_config_path is not None:
    arm_calibration = _build_arm_calibration(
      ctx, parsed_repo_id, episode_config_path)

  click.echo(
    f"Importing {parsed_repo_id} → {output}  (format={fmt}, "
    f"with_ee_pos={with_ee_pos}, with_object_pose={with_object_pose}, "
    f"tags={list(tags)}, name_prefix={name_prefix!r}, "
    f"episodes={'all' if episode_filter is None else sorted(episode_filter)}, "
    f"T_world_arm={'yes' if arm_calibration else 'no'})"
  )
  count = 0
  try:
    for ep_idx, out_path in tqdm(
      import_dataset(
        parsed_repo_id, output,
        format=fmt, video_key=video_key, max_episodes=max_episodes,
        tags=tuple(tags),
        with_ee_pos=with_ee_pos,
        with_object_pose=with_object_pose,
        name_prefix=name_prefix,
        episode_filter=episode_filter,
        arm_calibration=arm_calibration,
      ),
      desc="Episodes",
      total=(len(episode_filter) if episode_filter is not None else max_episodes),
    ):
      count += 1
      logger.info("episode %d → %s", ep_idx, out_path)
  except FileNotFoundError as e:
    raise click.ClickException(str(e))
  click.echo(f"Done. Wrote {count} episodes.")
