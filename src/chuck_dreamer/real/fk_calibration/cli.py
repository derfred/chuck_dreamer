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

import logging
import time
from pathlib import Path

import click

from chuck_dreamer.config import load_config

from ..object_localization.cli.common import override_option
from ..object_localization.runtime import init_from_config
from ..object_localization.types import dataset_cache_dir
from . import extract_joints as ej
from . import visualize as viz

logger = logging.getLogger(__name__)


@click.command("extract-joints")
@click.argument("episode_config", type=click.Path(exists=True, dir_okay=False,
                                                  path_type=Path))
@click.option("--visualize/--no-visualize", default=True,
              help="Render sanity-check figures after extraction (default).")
@click.option("--show/--no-show", default=False,
              help="Pop up the matplotlib figures interactively in addition "
                   "to writing PNGs.")
@override_option
@click.pass_context
def extract_joints_cmd(ctx, episode_config: Path, visualize: bool,
                       show: bool, overrides: tuple[str, ...]) -> None:
  """Extract median joint vectors per touch episode and cache them.

  EPISODE_CONFIG is a JSON file mapping ``{dataset_id: {episode_index:
  touch_label}}``. See the module docstring for the schema.
  """
  cfg = load_config(ctx.obj["config_path"], overrides=overrides)
  ol_cfg = init_from_config(cfg)
  cache_root = Path(ol_cfg.cache_dir)

  try:
    config = ej.load_episode_config(episode_config)
  except ValueError as e:
    raise click.ClickException(str(e))
  if not config:
    raise click.ClickException(f"{episode_config}: no datasets configured.")

  click.echo(f"Episode config: {episode_config}")
  click.echo(f"Cache root:     {cache_root}")
  click.echo(f"Datasets:       {len(config)}")
  for did, mapping in config.items():
    click.echo(f"  {did}  ({len(mapping)} episodes: "
               f"{sorted(mapping.keys())})")

  t0 = time.perf_counter()
  results: list[ej.DatasetExtraction] = []
  total_touches = 0
  for did, mapping in config.items():
    click.echo()
    click.echo(click.style(f"[{did}]", fg="cyan"))
    try:
      result = ej.extract_dataset(did, mapping)
    except (ValueError, FileNotFoundError) as e:
      raise click.ClickException(f"{did}: {e}")
    written = ej.write_dataset_extraction(cache_root, result)
    out_dir = ej.joint_extraction_dir(cache_root, did)
    click.echo(f"  wrote {len(written) - 1} episode JSONs + metadata "
               f"to {out_dir}")
    results.append(result)
    total_touches += len(result.episodes)

  click.echo()
  click.echo(f"Total touches extracted: {total_touches} "
             f"across {len(results)} dataset(s) "
             f"in {time.perf_counter() - t0:.1f}s")

  flagged = [(r.dataset_id, e) for r in results for e in r.episodes if e.flags]
  if flagged:
    click.echo(click.style(
      f"  {len(flagged)} flagged episode(s) — review by eye:",
      fg="yellow"))
    for did, e in flagged:
      click.echo(f"    {did}  ep{e.episode_index}  ({e.touch_label})  "
                 f"flags={e.flags}")

  if visualize:
    viz_dir = _shared_viz_dir(cache_root, results)
    click.echo()
    click.echo(f"Rendering sanity-check figures to {viz_dir} ...")
    paths = viz.render(results, viz_dir)
    for p in paths:
      click.echo(f"  {p}")
    if show:
      _show_figures(paths)


def _shared_viz_dir(cache_root: Path,
                    results: list[ej.DatasetExtraction]) -> Path:
  """Pick a directory for cross-dataset figures.

  Single dataset: write next to that dataset's per-episode JSONs.
  Multiple datasets: write under ``<cache>/joint_extraction_summary/``
  so the cross-dataset view doesn't hide inside any one dataset folder.
  """
  if len(results) == 1:
    return ej.joint_extraction_dir(cache_root, results[0].dataset_id)
  return cache_root / "joint_extraction_summary"


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
