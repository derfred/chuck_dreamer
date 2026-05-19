"""Evaluation infrastructure.

Two tiers:

* :class:`OnlineEvaluator` — fast scalar metrics that run on the main
  training thread every ``training.eval_every`` iterations. Logs to the
  :class:`~chuck_dreamer.training.tracker.Tracker`.
* :class:`DeepEvalRunner` — slow, in-depth analysis (notebooks executed
  via papermill + rerun-episode dumps) for a single checkpoint. Invoked
  by ``main.py eval`` and by :class:`OnlineEvaluator` when a deep eval
  phase fires; the latter spawns a detached subprocess so training never
  waits on it.

The :mod:`.core`, :mod:`.metrics`, and :mod:`.checkpoint` modules are
the building blocks shared by both tiers and by the notebooks under
``src/chuck_dreamer/evals/``.
"""

from .checkpoint import LoadedCheckpoint, load_checkpoint
from .core import EvalEpisode, RolloutOutputs, load_eval_episodes, run_split_rollout
from .deep import DeepEvalRunner, available_notebooks
from .metrics import (
  recon_l2,
  reward_abs_error,
  stack_curves,
  zero_dynamics_prediction,
)
from .online import OnlineEvaluator

__all__ = [
  "LoadedCheckpoint",
  "load_checkpoint",
  "EvalEpisode",
  "RolloutOutputs",
  "load_eval_episodes",
  "run_split_rollout",
  "OnlineEvaluator",
  "DeepEvalRunner",
  "available_notebooks",
  "recon_l2",
  "reward_abs_error",
  "stack_curves",
  "zero_dynamics_prediction",
]
