"""Training utilities and main training loop."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path
import logging
from typing import Dict, Any, Optional, cast
import time
from tqdm import tqdm  # type: ignore[import-untyped]
import wandb

from ..dreamer.mlx_models import WorldModelEncoder, mse_loss
from ..data.tfrecord_utils import TFRecordReader
from ..config import DictConfig

logger = logging.getLogger(__name__)


class Trainer:
    """Training manager for MLX models."""

    def __init__(self,
                 config: DictConfig,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 train_loader: Any,
                 val_loader: Optional[Any] = None):
        """
        Initialize trainer.

        Args:
            config: Configuration object
            model: MLX model to train
            optimizer: MLX optimizer
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Setup logging
        self.setup_logging()

        # Setup save directory
        self.save_dir = Path(config.logging.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Setup logging and experiment tracking."""
        if self.config.logging.use_wandb:
            experiment_name = self.config.logging.experiment_name
            if experiment_name is None:
                experiment_name = f"world_model_{int(time.time())}"

            wandb.init(
                project=self.config.logging.project_name,
                name=experiment_name,
                config=dict(self.config)
            )

    def loss_fn(self, batch: Dict[str, mx.array]) -> mx.array:
        """
        Compute loss for a batch.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Loss value
        """
        observations = batch["observations"]
        actions = batch["actions"]
        targets = batch["targets"]

        # Forward pass
        predictions = self.model(observations, actions)

        # Compute loss
        loss = mse_loss(predictions, targets)

        return cast(mx.array, loss)

    def train_step(self, batch: Dict[str, mx.array]) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: Training batch

        Returns:
            Dictionary containing metrics
        """
        # Compute loss and gradients
        loss, grads = mx.value_and_grad(self.loss_fn)(batch)

        # Gradient clipping
        if self.config.training.gradient_clipping > 0:
            grads, _ = optim.clip_grad_norm(
                grads, self.config.training.gradient_clipping)

        # Update parameters
        self.optimizer.update(self.model, grads)

        # Update step counter
        self.global_step += 1

        return {"train_loss": float(loss)}

    def validate(self) -> Dict[str, float]:
        """
        Perform validation.

        Returns:
            Dictionary containing validation metrics
        """
        if self.val_loader is None:
            return {}

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            loss = self.loss_fn(batch)
            total_loss += float(loss)
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {"val_loss": avg_loss}

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Current metrics
        """
        model_weights = dict(self.model.parameters())
        optimizer_state = self.optimizer.state

        # Save latest checkpoint
        checkpoint_path = self.save_dir / "latest_checkpoint.safetensors"
        mx.save_safetensors(str(checkpoint_path), model_weights)
        mx.savez(
            str(self.save_dir / "latest_checkpoint_meta.npz"),
            epoch=mx.array(epoch),
            global_step=mx.array(self.global_step),
        )

        # Save best checkpoint if validation loss improved
        val_loss = metrics.get("val_loss", float('inf'))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.save_dir / "best_checkpoint.safetensors"
            mx.save_safetensors(str(best_path), model_weights)
            logger.info(
                f"New best checkpoint saved with val_loss: {val_loss:.6f}")
        _ = optimizer_state  # retained for future use

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        weights = cast(Dict[str, mx.array], mx.load(checkpoint_path))
        self.model.load_weights(list(weights.items()))

        meta_path = str(checkpoint_path).replace(
            ".safetensors", "_meta.npz")
        meta = cast(Dict[str, mx.array], mx.load(meta_path))
        self.current_epoch = int(meta["epoch"].item())
        self.global_step = int(meta["global_step"].item())

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(
            f"Resuming from epoch {self.current_epoch},"
            f" step {self.global_step}")

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Training for {self.config.training.num_epochs} epochs")

        for epoch in range(
                self.current_epoch,
                self.config.training.num_epochs):
            epoch_start_time = time.time()

            # Training
            self.model.train()
            epoch_metrics = {"epoch": epoch}

            # Progress bar for training batches
            train_pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}/{self.config.training.num_epochs}",
                leave=False
            )

            for batch_idx, batch in enumerate(train_pbar):
                # Training step
                step_metrics = self.train_step(batch)
                epoch_metrics.update(step_metrics)

                # Update progress bar
                train_pbar.set_postfix(step_metrics)

                # Log metrics
                if self.global_step % self.config.logging.log_every == 0:
                    if self.config.logging.use_wandb:
                        wandb.log({
                            **step_metrics,
                            "epoch": epoch,
                            "global_step": self.global_step
                        })

            # Validation
            if (epoch % self.config.training.eval_every == 0
                    and self.val_loader is not None):
                self.model.eval()
                val_metrics = self.validate()
                epoch_metrics.update(val_metrics)

                metrics_str = " - ".join(
                    [f"{k}: {v:.6f}" for k, v in epoch_metrics.items()])
                logger.info(f"Epoch {epoch} - " + metrics_str)

            # Save checkpoint
            if epoch % self.config.training.save_every == 0:
                self.save_checkpoint(epoch, epoch_metrics)

            # Log epoch metrics
            if self.config.logging.use_wandb:
                wandb.log({
                    **epoch_metrics,
                    "epoch_time": time.time() - epoch_start_time
                })

            self.current_epoch = epoch + 1

        logger.info("Training completed!")

        # Final checkpoint save
        self.save_checkpoint(self.current_epoch, epoch_metrics)


def create_optimizer(model: nn.Module, config: DictConfig) -> optim.Optimizer:
    """
    Create optimizer from configuration.

    Args:
        model: Model to optimize
        config: Configuration object

    Returns:
        MLX optimizer
    """
    optimizer_type = config.optimizer.type.lower()
    lr = config.training.learning_rate

    if optimizer_type == "adamw":
        return optim.AdamW(
            learning_rate=lr,
            weight_decay=config.optimizer.weight_decay,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps
        )
    elif optimizer_type == "adam":
        return optim.Adam(
            learning_rate=lr,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps
        )
    elif optimizer_type == "sgd":
        return optim.SGD(learning_rate=lr)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_data_loader(
        tfrecord_path: str,
        config: DictConfig,
        is_training: bool = True):
    """
    Create data loader from TFRecord file.

    Args:
        tfrecord_path: Path to TFRecord file
        config: Configuration object
        is_training: Whether this is for training (affects shuffling)

    Returns:
        Data loader
    """
    reader = TFRecordReader()

    dataset = reader.read_tfrecord(
        filepath=tfrecord_path,
        batch_size=config.training.batch_size,
        shuffle=is_training and config.data.shuffle,
        buffer_size=config.data.buffer_size
    )

    return dataset


if __name__ == "__main__":
    # Example training script
    from ..config import load_config

    # Load configuration
    config = load_config()

    # Create model
    model = WorldModelEncoder(
        observation_dim=config.model.observation_dim,
        action_dim=config.model.action_dim,
        latent_dim=config.model.latent_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads
    )

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # Create data loaders
    train_loader = create_data_loader(
        config.data.tfrecord_path, config, is_training=True)
    val_loader = None
    if config.data.val_tfrecord_path:
        val_loader = create_data_loader(
            config.data.val_tfrecord_path, config, is_training=False)

    # Create trainer
    trainer = Trainer(config, model, optimizer, train_loader, val_loader)

    # Start training
    trainer.train()
