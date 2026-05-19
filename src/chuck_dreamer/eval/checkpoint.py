"""Single entry point for "checkpoint path → (config, env, model)".

The trainer embeds the resolved training config in the safetensors
metadata, so a checkpoint is self-describing — we don't need a separate
config file to reload it. Both ``main.py eval`` and the eval notebooks
go through :func:`load_checkpoint` so the rebuild path matches
:class:`~chuck_dreamer.trainer.Trainer.__init__`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LoadedCheckpoint:
  """A checkpoint reloaded into a runnable model.

  ``config`` is the resolved training config that was embedded in the
  safetensors metadata block. ``env`` is the :class:`PushingEnv` used to
  derive obs/action shapes (and to expose ``obs_mode`` / ``act_mode``).
  ``model`` is in eval mode with weights loaded; do not call
  ``wm_update`` on it.
  """

  config:    Any
  env:       Any
  model:     Any
  path:      str


def load_checkpoint(path: str) -> LoadedCheckpoint:
  """Reload a safetensors checkpoint into a runnable model.

  Pulls the embedded config from the file, rebuilds the env to derive
  obs/action shapes, builds the model with those shapes, switches it to
  eval mode, and loads the weights.

  Raises ``RuntimeError`` if the checkpoint has no embedded config — old
  checkpoints from before the metadata block was added would need to be
  re-saved by the current trainer first.
  """
  from ..dreamer import build_model
  from ..dreamer.mlx_model import load_config_from_checkpoint
  from ..sim.pushing_env import PushingEnv

  config = load_config_from_checkpoint(path)
  if config is None:
    raise RuntimeError(
      f"checkpoint {path!r} has no embedded config metadata; "
      "re-save with a current Trainer to embed the config."
    )

  env        = PushingEnv(config)
  obs_shape  = env.model_obs_shape
  action_dim = int(env.action_space.shape[0])

  model = build_model(config, obs_shape=obs_shape, action_dim=action_dim)
  model.training = False
  model.load(path)

  return LoadedCheckpoint(config=config, env=env, model=model, path=path)
