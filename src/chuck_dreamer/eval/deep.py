"""Deep-eval runner — slow notebooks + rerun-episode dumps for one checkpoint.

Single entry point for the "in-depth analysis" tier. Two trigger paths:

* From the training loop: :meth:`DeepEvalRunner.maybe_fire` checks the
  ``cfg.eval.deep.every_n_evals`` cadence against the running eval phase
  counter, and on a hit it spawns a detached subprocess running
  ``python main.py eval --all`` against the latest checkpoint. Training
  never blocks on the subprocess.

* From the CLI: :meth:`DeepEvalRunner.run_sync` does the same work in
  the current process — used by ``main.py eval``. Same code path, same
  config knobs, just synchronous.

Both paths execute zero or more notebooks via papermill and optionally
dump per-episode rerun recordings via :class:`EpisodeWriter`.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Notebook discovery
# ---------------------------------------------------------------------------


def _evals_dir() -> Path:
  """Directory holding the deep-eval notebooks."""
  return Path(__file__).resolve().parent.parent / "evals"


def available_notebooks() -> dict[str, Path]:
  """Return a mapping of notebook-name → path under ``chuck_dreamer/evals/``."""
  d = _evals_dir()
  if not d.exists():
    return {}
  return {p.stem: p for p in sorted(d.glob("*.ipynb"))}


# ---------------------------------------------------------------------------
# Per-checkpoint config snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeepEvalConfig:
  """The deep-eval knobs pulled out of ``cfg.eval.deep``."""

  every_n_evals: int | None
  notebooks:     tuple[str, ...]
  recordings:    bool
  output_dir:    str
  burn_in:       int
  horizon:       int
  num_episodes:  int

  @classmethod
  def from_config(cls, config) -> "DeepEvalConfig":
    deep = config.eval.deep
    return cls(
      every_n_evals=(int(deep.every_n_evals) if deep.every_n_evals else None),
      notebooks=tuple(deep.notebooks or ()),
      recordings=bool(deep.recordings),
      output_dir=str(deep.output_dir),
      burn_in=int(config.eval.burn_in),
      horizon=int(deep.horizon),
      num_episodes=int(deep.num_episodes),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class DeepEvalRunner:
  """Schedules + executes deep evals for a single training run.

  One instance lives on the :class:`OnlineEvaluator` and tracks the
  internal phase counter for cadence gating. Construction is cheap —
  notebooks aren't loaded until :meth:`run_sync` actually runs.
  """

  def __init__(self, config):
    self.config = config
    self._cfg   = DeepEvalConfig.from_config(config)
    self._phase_count = 0
    # Resolve where the project root is. The trainer is launched from
    # the repo root, so cwd works for the subprocess invocation. We
    # cache it here so reading is cheap.
    self._project_root = Path.cwd()
    self._main_py      = self._project_root / "main.py"

  # ---------- async cadence ----------

  def maybe_fire(self, *, iteration: int, checkpoint_path: str | None, cleanup: bool = False) -> None:
    """Bump the phase counter and, on a deep-eval hit, fire the subprocess.

    ``checkpoint_path`` is the snapshot to load. ``None`` (no checkpoint
    written yet) skips firing — the first deep-eval pass needs weights
    to load. A missing path on disk is also a skip.
    """
    if self._cfg.every_n_evals is None:
      return
    self._phase_count += 1
    if self._phase_count % self._cfg.every_n_evals != 0:
      return
    if not checkpoint_path or not os.path.exists(checkpoint_path):
      logger.warning(
        f"Deep eval scheduled at iteration {iteration} but no checkpoint "
        f"snapshot is available yet (got {checkpoint_path!r}); skipping."
      )
      return
    self._fire_subprocess(iteration=iteration, checkpoint_path=checkpoint_path, cleanup=cleanup)

  def _fire_subprocess(self, *, iteration: int, checkpoint_path: str, cleanup: bool = False) -> None:
    """Spawn a detached ``main.py eval`` subprocess. Does not wait."""
    if not self._main_py.exists():
      logger.warning(f"Cannot locate main.py at {self._main_py}; skipping deep eval.")
      return

    output_dir = Path(self._cfg.output_dir) / f"iteration{iteration:06d}"
    cmd = [
      sys.executable, str(self._main_py), "eval",
      "--all",
      "--checkpoint", str(checkpoint_path),
      "--output-dir", str(output_dir),
      "--num-episodes", str(self._cfg.num_episodes),
      "--burn-in",      str(self._cfg.burn_in),
      "--horizon",      str(self._cfg.horizon),
      "--cleanup"       if cleanup else "--no-cleanup",
    ]
    if self._cfg.recordings:
      cmd.append("--recordings")
    logger.info(
      f"Firing deep eval subprocess at iteration {iteration} → {output_dir}: "
      f"{' '.join(cmd)}"
    )
    # Detach: discard stdio so a hanging notebook can't backpressure us.
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "deep_eval.log"
    log_file = open(log_path, "w")  # closed by the child's exit
    subprocess.Popen(  # noqa: S603 — args are project-controlled
      cmd,
      cwd=str(self._project_root),
      stdout=log_file,
      stderr=subprocess.STDOUT,
      stdin=subprocess.DEVNULL,
      start_new_session=True,
    )

  # ---------- synchronous (CLI) ----------

  def run_sync(
    self,
    checkpoint_path: str,
    *,
    output_dir:      str | Path,
    notebooks:       tuple[str, ...] | None = None,
    recordings:      bool | None = None,
    num_episodes:    int | None = None,
    burn_in:         int | None = None,
    horizon:         int | None = None,
    extra_params:    dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    """Run the deep-eval bundle synchronously. Returns a summary dict.

    Each argument falls back to the corresponding ``cfg.eval.deep.*``
    value when ``None`` so the CLI can selectively override.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chosen_notebooks = (
      tuple(notebooks) if notebooks is not None else self._cfg.notebooks
    )
    do_recordings   = self._cfg.recordings if recordings   is None else bool(recordings)
    n_episodes      = self._cfg.num_episodes if num_episodes is None else int(num_episodes)
    chosen_burn_in  = self._cfg.burn_in      if burn_in      is None else int(burn_in)
    chosen_horizon  = self._cfg.horizon      if horizon      is None else int(horizon)

    summary: dict[str, Any] = {
      "checkpoint":  str(checkpoint_path),
      "output_dir":  str(output_dir),
      "notebooks":   list(chosen_notebooks),
      "recordings":  do_recordings,
      "executed":    [],
      "recordings_path": None,
    }

    if not chosen_notebooks and not do_recordings:
      logger.warning(
        "Deep eval has no notebooks selected and --recordings is off — nothing to do."
      )
      return summary

    if chosen_notebooks:
      self._run_notebooks(
        chosen_notebooks,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        num_episodes=n_episodes,
        burn_in=chosen_burn_in,
        horizon=chosen_horizon,
        extra_params=extra_params or {},
        summary=summary,
      )

    if do_recordings:
      rec_dir = output_dir / "recordings"
      self._dump_recordings(
        checkpoint_path=checkpoint_path,
        output_dir=rec_dir,
        num_episodes=n_episodes,
        burn_in=chosen_burn_in,
        horizon=chosen_horizon,
      )
      summary["recordings_path"] = str(rec_dir)

    return summary

  # ---------- notebook execution ----------

  def _run_notebooks(
    self,
    notebooks:    tuple[str, ...],
    *,
    checkpoint_path: str,
    output_dir:      Path,
    num_episodes:    int,
    burn_in:         int,
    horizon:         int,
    extra_params:    dict[str, Any],
    summary:         dict[str, Any],
  ) -> None:
    import papermill as pm

    notebooks_dir = _evals_dir()
    available     = available_notebooks()

    # Pick the first eval source for the data path the notebook expects.
    sources = list(self.config.eval.source)
    if not sources:
      raise RuntimeError("cfg.eval.source is empty — cannot run deep eval notebooks.")
    primary = sources[0]

    base_parameters: dict[str, Any] = {
      "checkpoint_path": str(checkpoint_path),
      "data_path":       str(primary["path"]),
      "data_format":     str(primary["format"]),
      "num_episodes":    int(num_episodes),
      "burn_in":         int(burn_in),
      "horizon":         int(horizon),
      "seed":            int(self.config.seed),
    }
    base_parameters.update(extra_params)

    for nb_name in notebooks:
      if nb_name not in available:
        raise ValueError(
          f"unknown deep-eval notebook {nb_name!r}; available: {sorted(available)}"
        )
      in_path  = available[nb_name]
      out_path = output_dir / f"{nb_name}.ipynb"
      logger.info(f"Executing notebook {in_path} → {out_path}")
      pm.execute_notebook(
        input_path=str(in_path),
        output_path=str(out_path),
        parameters=base_parameters,
        cwd=str(notebooks_dir.parent.parent.parent),  # project root
      )
      summary["executed"].append(str(out_path))

  # ---------- rerun recordings ----------

  def _dump_recordings(
    self,
    *,
    checkpoint_path: str,
    output_dir:      Path,
    num_episodes:    int,
    burn_in:         int,
    horizon:         int,
  ) -> None:
    """Dump per-episode rerun recordings of (posterior vs prior) rollouts."""
    from ..sim.episode_writer import EpisodeWriter
    from .checkpoint import load_checkpoint
    from .core import load_eval_episodes, run_split_rollout

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = load_checkpoint(checkpoint_path)

    fmt = str(self.config.logging.episodes.dump_format)
    writer = EpisodeWriter(str(output_dir), format=fmt)
    obs_mode = str(self.config.env.obs_mode)
    min_len  = burn_in + horizon

    written = 0
    for episode in load_eval_episodes(
      ckpt.config, min_len=min_len, max_episodes=num_episodes,
    ):
      result = run_split_rollout(
        ckpt.model, episode,
        burn_in=burn_in,
        horizon=horizon,
        obs_mode=obs_mode,
        decode=True,
        predict_reward=False,
      )
      if result is None:
        continue
      writer.write_eval_episode(
        {
          "obs":             result.target_obs,
          "recon_posterior": result.recon_posterior,
          "recon_prior":     result.recon_prior,
          "h_posterior":     result.h_posterior,
          "s_posterior":     result.s_posterior,
          "h_prior":         result.h_prior,
          "s_prior":         result.s_prior,
        },
        metadata={
          "seed":       int(ckpt.config.seed),
          "source":     "deep_eval",
          "checkpoint": str(checkpoint_path),
          "source_dir": episode.source_dir,
          "burn_in":    burn_in,
        },
        name_suffix=f"{episode.source_dir}__{episode.stem}",
      )
      written += 1
    logger.info(f"Wrote {written} deep-eval recordings → {output_dir}")
