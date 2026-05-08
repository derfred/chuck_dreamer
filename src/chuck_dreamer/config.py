"""Configuration management using OmegaConf."""

from pathlib import Path
from typing import Optional, cast

from omegaconf import DictConfig, OmegaConf


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


def load_config(config_path: Optional[str] = None) -> DictConfig:
  """Load a YAML config, falling back to ``configs/default.yaml``."""
  path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
  return cast(DictConfig, OmegaConf.load(path))


if __name__ == "__main__":
  print(OmegaConf.to_yaml(load_config()))
