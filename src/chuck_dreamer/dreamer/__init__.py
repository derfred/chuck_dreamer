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
from .torch_model import DreamerTorchModel


# Device names that route to the PyTorch backend. ``"auto"`` lets the
# torch backend pick cuda → mps → cpu in that order.
_TORCH_DEVICES = {"cuda", "mps", "cpu", "auto"}


def build_model(config, obs_shape, action_dim: int):
  """Build the Dreamer model. ``obs_shape`` may be a tuple (state/image)
  or a dict with ``image``/``proprio`` keys (image_proprio).

  Dispatches on ``config.hardware.device``:
    - ``"mlx"`` → :class:`DreamerMLXModel`
    - ``"cuda"`` / ``"mps"`` / ``"cpu"`` / ``"auto"`` → :class:`DreamerTorchModel`
  """
  device = config.hardware.device
  if device == "mlx":
    return DreamerMLXModel(config, obs_shape=obs_shape, action_dim=action_dim)
  if device in _TORCH_DEVICES:
    return DreamerTorchModel(config, obs_shape=obs_shape, action_dim=action_dim)
  raise ValueError(f"Unsupported library specified in config: {device}")


__all__ = [
  "build_model",
  "DreamerMLXModel",
  "DreamerTorchModel",
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
