"""Configuration management using OmegaConf."""

from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf


# Spatial dim at the CNN bottleneck (after all encoder strided convs).
# Combined with the per-layer strides this fixes the input resolution:
#   image_size = CNN_BOTTLENECK_HW * prod(strides)
CNN_BOTTLENECK_HW = 4


def derive_image_size(strides: tuple[int, ...]) -> int:
  """Image side length implied by an encoder's per-layer strides.

  Each conv strides spatial dims down and we want the post-encoder feature
  map to be ``CNN_BOTTLENECK_HW`` per side, so the input must be
  ``CNN_BOTTLENECK_HW * prod(strides)`` per side. Padding is chosen
  per-layer in the encoder so each layer divides cleanly.
  """
  prod = 1
  for s in strides:
    prod *= int(s)
  return CNN_BOTTLENECK_HW * prod


# Idempotent: ``replace=True`` keeps re-imports during testing from raising.
OmegaConf.register_new_resolver(
  "derive_image_size",
  lambda strides: derive_image_size(tuple(int(s) for s in strides)),
  replace=True,
)


_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "default.yaml"


def load_config(
  config_path: Optional[str] = None,
  overrides: tuple[str, ...] = (),
  aliases: Optional[dict[str, Any]] = None,
) -> DictConfig:
  """Load a YAML config, apply dotted-path overrides, normalize, return.

  ``load_config`` is the single entry point: parses YAML, applies CLI
  ``aliases`` first (``{dotted_key: value}``, ``None`` values skipped),
  merges any ``-o key=value`` ``overrides`` on top, randomizes ``seed``
  if unset, and normalizes ``training.data.warmup_source`` into a
  uniform list of dicts. After this returns, callers can assume
  ``cfg.training.data.warmup_source`` is a ``list[dict]`` whose entries
  each carry ``path``, ``format``, ``num_episodes``.

  ``aliases`` are applied before ``overrides``, so an explicit ``-o``
  flag always wins over a CLI shorthand for the same key.
  """
  path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
  cfg  = cast(DictConfig, OmegaConf.load(path))
  merged: list[str] = []
  if aliases:
    merged.extend(f"{k}={v}" for k, v in aliases.items() if v is not None)
  if overrides:
    merged.extend(overrides)
  if merged:
    cfg = cast(DictConfig, OmegaConf.merge(cfg, OmegaConf.from_dotlist(merged)))
  if cfg.get("seed") is None:
    cfg.seed = int(np.random.default_rng().integers(0, 2**31))
  _normalize_warmup_source(cfg)
  _normalize_eval_source(cfg)
  _normalize_obs_mode(cfg)
  return cfg


def _normalize_episode_source(
  raw: Any,
  default_fmt: Any,
  default_n: Any,
) -> list[dict]:
  """Expand a path / dict / list-of-either spec into a uniform ``list[dict]``.

  Accepts three forms:
    * ``str``                  — single path, inherits the supplied
                                 ``default_fmt`` / ``default_n``.
    * ``list`` of str | dict   — mixed entries; dict entries may override
                                 ``format`` / ``num_episodes`` per source.
    * ``dict``                 — single entry, same override rules as a
                                 list-of-one dict.

  Each output entry carries fully-resolved ``path``, ``format``,
  ``num_episodes`` keys.
  """
  if isinstance(raw, str):
    items: list[Any] = [raw]
  elif isinstance(raw, ListConfig):
    items = list(raw)
  else:
    items = [raw]

  normalized: list[dict] = []
  for item in items:
    if isinstance(item, str):
      normalized.append({
        "path":         item,
        "format":       default_fmt,
        "num_episodes": default_n,
      })
    else:
      normalized.append({
        "path":         item["path"],
        "format":       item.get("format",       default_fmt),
        "num_episodes": item.get("num_episodes", default_n),
      })
  return normalized


def _normalize_warmup_source(cfg: DictConfig) -> None:
  """Normalize ``training.data.warmup_source`` into a ``list[dict]``.

  After normalization the top-level ``warmup_format`` /
  ``warmup_num_episodes`` keys are deleted so consumers can't accidentally
  read stale defaults.
  """
  data_cfg = cfg.training.data
  data_cfg.warmup_source = _normalize_episode_source(
    data_cfg.warmup_source,
    data_cfg.warmup_format,
    data_cfg.warmup_num_episodes,
  )
  del data_cfg.warmup_format
  del data_cfg.warmup_num_episodes


def _normalize_obs_mode(cfg: DictConfig) -> None:
  """Normalize ``env.obs_mode`` into a canonical list of component names.

  Accepts a string (legacy single-mode form) or a list of strings, and
  rewrites both to a list under ``cfg.env.obs_mode``. Delegates the
  vocabulary check to :func:`Observation.normalize_obs_mode` — typos
  trip here rather than at the encoder.
  """
  from .training.observation import normalize_obs_mode

  raw = cfg.env.obs_mode
  cfg.env.obs_mode = list(normalize_obs_mode(raw))


def _normalize_eval_source(cfg: DictConfig) -> None:
  """Normalize ``eval.source`` into a ``list[dict]``.

  After normalization the top-level ``eval.data_format`` /
  ``eval.num_episodes`` keys are deleted so consumers can't accidentally
  read stale defaults.
  """
  eval_cfg = cfg.eval
  eval_cfg.source = _normalize_episode_source(
    eval_cfg.source,
    eval_cfg.data_format,
    eval_cfg.num_episodes,
  )
  del eval_cfg.data_format
  del eval_cfg.num_episodes


if __name__ == "__main__":
  print(OmegaConf.to_yaml(load_config()))
