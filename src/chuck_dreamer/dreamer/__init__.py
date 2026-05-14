"""Model definitions and utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

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


if TYPE_CHECKING:
  # Type-only imports so static checkers see the symbols without forcing
  # mlx/torch to be importable at module load time on the other backend.
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

  Backend modules are imported lazily so a CUDA-only machine never needs
  mlx installed (and vice versa).
  """
  device = config.hardware.device
  if device == "mlx":
    from .mlx_model import DreamerMLXModel
    return DreamerMLXModel(config, obs_shape=obs_shape, action_dim=action_dim)
  if device in _TORCH_DEVICES:
    from .torch_model import DreamerTorchModel
    return DreamerTorchModel(config, obs_shape=obs_shape, action_dim=action_dim)
  raise ValueError(f"Unsupported library specified in config: {device}")


def __getattr__(name: str):
  """Lazy attribute access for the backend classes.

  Importing ``DreamerMLXModel`` / ``DreamerTorchModel`` from this package
  still works, but the corresponding backend (mlx or torch) is only
  imported on first access.
  """
  if name == "DreamerMLXModel":
    from .mlx_model import DreamerMLXModel
    return DreamerMLXModel
  if name == "DreamerTorchModel":
    from .torch_model import DreamerTorchModel
    return DreamerTorchModel
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
