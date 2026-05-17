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
  return cfg


def _normalize_warmup_source(cfg: DictConfig) -> None:
  """Expand ``training.data.warmup_source`` into a uniform list of dicts.

  Accepts three forms in the raw config:
    * ``str``                  — single path, inherits the top-level
                                 ``warmup_format`` / ``warmup_num_episodes``.
    * ``list`` of str | dict   — mixed entries; dict entries may override
                                 ``format`` / ``num_episodes`` per source.
    * ``dict``                 — single entry, same override rules as a
                                 list-of-one dict.

  After normalization the top-level ``warmup_format`` /
  ``warmup_num_episodes`` keys are deleted so consumers can't accidentally
  read stale defaults.
  """
  data_cfg    = cfg.training.data
  raw         = data_cfg.warmup_source
  default_fmt = data_cfg.warmup_format
  default_n   = data_cfg.warmup_num_episodes

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

  data_cfg.warmup_source = normalized
  del data_cfg.warmup_format
  del data_cfg.warmup_num_episodes


if __name__ == "__main__":
  print(OmegaConf.to_yaml(load_config()))
