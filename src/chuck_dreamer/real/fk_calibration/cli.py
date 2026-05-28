"""`extract-joints` CLI: aggregate per-touch joint vectors from LeRobot
recordings and drop them into ``calibration_cache/<slug>/joint_extraction/``.

Each episode in the touch-recordings is one fiducial-mat touch (the
arm rests on a labelled point for the whole episode). The CLI takes
one ``episode_config`` JSON that names every (dataset, episode) ->
touch label, runs the per-episode median, writes per-episode JSONs to
the cache, and renders three sanity-check figures to the same place.

Example config — name the two touch prototypes once at the top, then
reference one per dataset and list the episode indices in prototype
order. ``overrides`` patches single-episode mislabels::

    {
      "prototypes": {
        "three_touch": ["left-middle", "origin", "right-middle"],
        "seven_touch": ["left-middle", "origin", "right-middle",
                        "left-far", "left-near",
                        "right-near", "right-far"]
      },
      "datasets": [
        {"dataset_id": "chorfiyoussef/task_1_1805_6",
         "prototype": "three_touch", "episodes": [5, 6, 7]},
        {"dataset_id": "chorfiyoussef/task1_2005_2_20260520_191951",
         "prototype": "seven_touch", "episodes": [4, 5, 6, 7, 8, 9, 10]}
      ]
    }

Usage::

    python main.py extract-joints path/to/episode_config.json \\
        [--no-visualize]
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import click

from chuck_dreamer.config import load_config

from chuck_dreamer.cli import override_option

from ..object_localization.runtime import init_from_config
from ..object_localization.types import dataset_cache_dir

logger = logging.getLogger(__name__)

@click.command("calibrate-arm-from-episodes")
@click.argument("episode_config", type=click.Path(exists=True, dir_okay=False,
                                                  path_type=Path))
@click.option("--force/--no-force", default=False,
              help="Write T_world_arm.json even if Umeyama rejects (max "
                   "residual ≥ 10 mm). Diagnostics are always written.")
@override_option
@click.pass_context
def calibrate_arm_from_episodes_cmd(ctx, episode_config: Path, force: bool,
                                    overrides: tuple[str, ...]) -> None:
  """Derive T_world_arm per dataset from EPISODE_CONFIG and cache it.

  For each dataset listed in EPISODE_CONFIG, reads the touch episodes,
  takes per-episode joint medians, runs FK + Umeyama against the
  canonical mat-frame targets, and writes the result to
  ``calibration_cache/<dataset_slug>/T_world_arm.json``. Same data
  contract as the live `annotate-live` arm step.
  """
  from ..object_localization.calibration.arm import (
    calibrate_from_lerobot_dataset,
  )
  from .fk import FK, load_fk_dq

  cfg = load_config(ctx.obj["config_path"], overrides=overrides)
  ol_cfg = init_from_config(cfg)
  cache_root = Path(ol_cfg.cache_dir)
  fk_dq = load_fk_dq(cache_root)

  try:
    config = ej.load_episode_config(episode_config)
  except ValueError as e:
    raise click.ClickException(str(e))
  if not config:
    raise click.ClickException(f"{episode_config}: no datasets configured.")

  fk_model_path = (Path(__file__).resolve().parents[3]
                   / "assets" / "mujoco" / "so101_arm.xml")
  if not fk_model_path.exists():
    raise click.ClickException(f"FK model not found at {fk_model_path}")
  fk = FK(fk_model_path)

  click.echo(f"Episode config: {episode_config}")
  click.echo(f"Cache root:     {cache_root}")
  click.echo(f"FK model:       {fk_model_path}")
  click.echo(f"Mat geometry:   L={ol_cfg.mat_line_length_mm}mm "
             f"D={ol_cfg.mat_line_separation_mm}mm")
  import numpy as _np
  if _np.any(fk_dq != 0):
    click.echo(f"FK joint offsets (dq): "
               f"{_np.round(fk_dq, 3).tolist()}")
  else:
    click.echo("FK joint offsets (dq): zeros (no calibration_cache/fk_dq.json)")
  click.echo(f"Datasets:       {len(config)}")

  n_ok = 0
  n_warn = 0
  n_rejected = 0
  for did, mapping in config.items():
    click.echo()
    click.echo(click.style(f"[{did}]", fg="cyan"))
    click.echo(f"  {len(mapping)} touches: "
               f"{', '.join(f'{e}={l}' for e, l in sorted(mapping.items()))}")

    try:
      block, residuals, max_res, rms_res = calibrate_from_lerobot_dataset(
        dataset_id=did,
        episode_to_label=mapping,
        fk=fk,
        L_mm=ol_cfg.mat_line_length_mm,
        D_mm=ol_cfg.mat_line_separation_mm,
        fk_dq=fk_dq,
        fk_model_path=fk_model_path,
      )
    except (ValueError, FileNotFoundError) as e:
      raise click.ClickException(f"{did}: {e}")

    click.echo(f"  max_residual = {max_res:.2f} mm   "
               f"rms_residual = {rms_res:.2f} mm")
    for (ep, lab), r in zip(sorted(mapping.items()), residuals):
      click.echo(f"    ep{ep:>3d}  {lab:>13s}  res={r:6.2f} mm")

    out_dir = dataset_cache_dir(cache_root, did)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "T_world_arm.json"

    if block is None:
      n_rejected += 1
      click.echo(click.style(
        f"  REJECT: max_residual {max_res:.2f}mm exceeds 10 mm. "
        f"Three-touch prototypes are collinear and will hit this; use a "
        f"seven_touch dataset for a usable transform.", fg="red"))
      if not force:
        click.echo(f"  not written (pass --force to write anyway).")
        continue
      # Build a diagnostics-only block so callers can still inspect.
      block = {
        "T_world_arm":   None,
        "diagnostics":   {
          "touches": [
            {
              "name":     lab,
              "p_world":  None,
              "residual": float(r),
            }
            for (_, lab), r in zip(sorted(mapping.items()), residuals)
          ],
          "max_residual_mm": float(max_res),
          "rms_residual_mm": float(rms_res),
          "n_touches":       int(len(residuals)),
          "rejected":        True,
        },
        "metadata":      {
          "source":     "lerobot_dataset",
          "dataset_id": did,
        },
      }
    else:
      if max_res >= 5.0:
        n_warn += 1
        click.echo(click.style(f"  WARN ({max_res:.2f} mm in [5, 10) mm)",
                               fg="yellow"))
      else:
        n_ok += 1
        click.echo(click.style(f"  OK ({max_res:.2f} mm < 5 mm)", fg="green"))

    out_path.write_text(json.dumps(block, indent=2))
    click.echo(f"  wrote {out_path}")

  click.echo()
  click.echo(f"Done. ok={n_ok}  warn={n_warn}  rejected={n_rejected}  "
             f"(total={len(config)})")


def _show_figures(paths: list[Path]) -> None:
  try:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
  except Exception:
    return
  for p in paths:
    if not p.exists():
      continue
    img = mpimg.imread(str(p))
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(str(p.name))
    plt.axis("off")
  plt.show()
