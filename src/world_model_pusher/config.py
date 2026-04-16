"""Configuration management using OmegaConf."""

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> DictConfig:
    """
    Load configuration from file or return default config.

    Args:
        config_path: Path to configuration file. If None, returns
            default config.

    Returns:
        Configuration object
    """
    if config_path is not None and Path(config_path).exists():
        config = OmegaConf.load(config_path)
    else:
        config = get_default_config()

    return config


def get_default_config() -> DictConfig:
    """Get default configuration from YAML file."""

    # Try to load from default.yaml first
    default_config_path = Path(
        __file__).parent.parent.parent / "configs" / "default.yaml"
    return OmegaConf.load(default_config_path)


def save_config(config: DictConfig, save_path: str) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration object to save
        save_path: Path where to save the configuration
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, save_path)
    logger.info(f"Configuration saved to {save_path}")


def merge_configs(
        base_config: DictConfig,
        override_config: DictConfig) -> DictConfig:
    """
    Merge two configurations with override taking precedence.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    return OmegaConf.merge(base_config, override_config)


def validate_config(config: DictConfig) -> bool:
    """
    Validate configuration parameters.

    Args:
        config: Configuration to validate

    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Check required fields
        required_fields = [
            "model.observation_dim",
            "model.action_dim",
            "training.batch_size",
            "training.learning_rate"
        ]

        for field in required_fields:
            if OmegaConf.select(config, field) is None:
                logger.error(
                    f"Required field '{field}' is missing from configuration")
                return False

        # Check value ranges
        if config.training.learning_rate <= 0:
            logger.error("Learning rate must be positive")
            return False

        if config.training.batch_size <= 0:
            logger.error("Batch size must be positive")
            return False

        if config.model.latent_dim <= 0:
            logger.error("Latent dimension must be positive")
            return False

        logger.info("Configuration validation passed")
        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    config = get_default_config()
    print("Default configuration:")
    print(OmegaConf.to_yaml(config))

    # Validate configuration
    is_valid = validate_config(config)
    print(f"Configuration is valid: {is_valid}")

    # Save configuration
    save_config(config, "configs/default.yaml")
