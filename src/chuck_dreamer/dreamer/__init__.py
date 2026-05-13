"""Model definitions and utilities."""

from .cem_policy import CEMPolicy
from .policy import DreamerPolicy
from .world_model import (
  Distribution,
  EvalRollout,
  State,
  Trajectory,
  WorldModel,
  eval_split_rollout,
  rollout,
)

from .mlx_model import DreamerMLXModel


def build_model(config, obs_shape, action_dim: int):
  """Build the Dreamer model. ``obs_shape`` may be a tuple (state/image)
  or a dict with ``image``/``proprio`` keys (image_proprio)."""
  if config.hardware.device == "mlx":
    return DreamerMLXModel(config, obs_shape=obs_shape, action_dim=action_dim)
  else:
    raise ValueError(f"Unsupported library specified in config: {config.hardware.device}")


__all__ = [
  "build_model",
  "DreamerPolicy",
  "CEMPolicy",
  "WorldModel",
  "State",
  "Distribution",
  "Trajectory",
  "rollout",
  "EvalRollout",
  "eval_split_rollout",
]
