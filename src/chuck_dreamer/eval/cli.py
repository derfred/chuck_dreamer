"""`eval` CLI: run deep eval (notebooks + rerun recordings) on a trained
checkpoint via :class:`~chuck_dreamer.eval.DeepEvalRunner`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import click

from chuck_dreamer.config import load_config

from ..cli import override_option

logger = logging.getLogger(__name__)


@click.command("eval")
@click.argument("name", required=False)
@click.option("--all", "run_all", is_flag=True,
              help="Run every notebook listed under cfg.eval.deep.notebooks. "
                   "Mutually exclusive with NAME.")
@click.option("--recordings/--no-recordings", default=None,
              help="Also dump per-episode rerun recordings of the rollout into "
                   "<output-dir>/recordings/. Default: follow cfg.eval.deep.recordings. "
                   "Can be used WITHOUT a notebook to only dump recordings.")
@click.option("--checkpoint", "checkpoint_path", default=None, type=str,
              help="Path to a trained checkpoint .safetensors file. "
                   "Defaults to {save_dir}/{experiment}/latest.safetensors.")
@click.option("--cleanup/--no-cleanup", default=None,
              help="Whether to clean up intermediate files after evaluation. Defaults to cfg.eval.deep.cleanup.")
@click.option("--num-episodes", default=None, type=int,
              help="Episodes to evaluate on. Defaults to cfg.eval.deep.num_episodes.")
@click.option("--burn-in", default=None, type=int,
              help="Closed-loop burn-in steps. Defaults to cfg.eval.burn_in.")
@click.option("--horizon", default=None, type=int,
              help="Open-loop horizon length. Defaults to cfg.eval.deep.horizon.")
@click.option("--seed", default=None, type=int, help="Random seed (random if omitted).")
@click.option("--output-dir", "output_dir", default=None, type=str,
              help="Directory to write executed notebooks + recordings. "
                   "Default: ./eval_runs/<ckpt-stem>/.")
@click.option("-p", "--param", "extra_params", multiple=True, metavar="KEY=VALUE",
              help="Additional papermill parameter override (repeatable).")
@override_option
@click.pass_context
def eval_cmd(ctx, name, run_all, recordings, checkpoint_path, num_episodes,
             burn_in, horizon, seed, output_dir, extra_params, cleanup, overrides):
  """Run deep eval on a trained checkpoint.

  Three modes:

    * ``main.py eval``                          — list available notebooks.
    * ``main.py eval <notebook>``               — execute one notebook.
    * ``main.py eval --all``                    — execute every notebook
                                                  in ``cfg.eval.deep.notebooks``.
    * Add ``--recordings`` to also dump per-episode rerun rollout files.
      ``--recordings`` works without a notebook too — useful for just
      generating viewer-ready recordings against a checkpoint.
  """
  import os

  from chuck_dreamer.eval import DeepEvalRunner, available_notebooks

  notebooks = available_notebooks()
  if not name and not run_all and recordings is None:
    click.echo("Available notebooks:")
    for n, p in notebooks.items():
      click.echo(f"  {n:<30s} {p}")
    click.echo("\nUse `eval <name>` to run one, `eval --all` to run every configured "
               "notebook, or `eval --recordings` to dump recordings without notebooks.")
    return
  if name and run_all:
    raise click.BadParameter("--all is mutually exclusive with a notebook NAME argument.")
  if name and name not in notebooks:
    raise click.BadParameter(f"unknown notebook {name!r}. Available: {sorted(notebooks)}")

  cfg = load_config(ctx.obj["config_path"], overrides=overrides, aliases={"seed": seed})

  if checkpoint_path is None:
    experiment = cfg.logging.experiment_name or "default"
    checkpoint_path = os.path.join(cfg.logging.save_dir, experiment, "latest.safetensors")
  if not Path(checkpoint_path).exists():
    raise click.ClickException(f"checkpoint not found: {checkpoint_path}")

  if name:
    chosen_notebooks: tuple[str, ...] = (name,)
  elif run_all:
    chosen_notebooks = tuple(cfg.eval.deep.notebooks or ())
    if not chosen_notebooks:
      click.echo("cfg.eval.deep.notebooks is empty — only recordings will be dumped.")
  else:
    chosen_notebooks = ()

  effective_recordings = (
    bool(cfg.eval.deep.recordings) if recordings is None else bool(recordings)
  )
  if not chosen_notebooks and not effective_recordings:
    raise click.BadParameter(
      "nothing to do: pass a notebook NAME, --all, or --recordings."
    )

  if output_dir is None:
    output_dir = os.path.join("eval_runs", Path(checkpoint_path).stem)
  Path(output_dir).mkdir(parents=True, exist_ok=True)

  extras: dict = {}
  for kv in extra_params:
    if "=" not in kv:
      raise click.BadParameter(f"--param expects KEY=VALUE, got {kv!r}")
    k, v = kv.split("=", 1)
    extras[k.strip()] = v

  click.echo(f"Deep eval on checkpoint: {checkpoint_path}")
  click.echo(f"Output dir:              {output_dir}")
  click.echo(f"Notebooks:               {list(chosen_notebooks) or '—'}")
  click.echo(f"Recordings:              {'on' if effective_recordings else 'off'}")

  runner = DeepEvalRunner(cfg)
  summary = runner.run_sync(
    checkpoint_path=str(checkpoint_path),
    output_dir=output_dir,
    notebooks=chosen_notebooks,
    recordings=effective_recordings,
    num_episodes=num_episodes,
    burn_in=burn_in,
    horizon=horizon,
    extra_params=extras,
  )
  click.echo("Done.")
  for nb in summary["executed"]:
    click.echo(f"  notebook → {nb}")
  if summary["recordings_path"]:
    click.echo(f"  recordings → {summary['recordings_path']}")

  if cleanup:
    click.echo("Cleaning up checkpoint...")
    os.unlink(checkpoint_path)
    click.echo("Cleanup done.")
