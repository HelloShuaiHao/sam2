"""Tensorboard logger for training metrics.

Provides enhanced Tensorboard logging beyond HuggingFace's built-in support.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class TensorboardLogger:
    """Enhanced Tensorboard logger with custom metrics support.

    Wraps torch.utils.tensorboard.SummaryWriter with convenience methods
    for logging training metrics, images, and histograms.

    Args:
        log_dir: Directory to save Tensorboard logs
        comment: Optional comment to append to log directory name
        flush_secs: How often to flush events to disk (seconds)
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        comment: str = "",
        flush_secs: int = 120
    ):
        """Initialize Tensorboard logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(
                log_dir=str(self.log_dir),
                comment=comment,
                flush_secs=flush_secs
            )
            logger.info(f"Tensorboard logger initialized: {self.log_dir}")
        except ImportError:
            logger.warning(
                "Tensorboard not available. Install with: pip install tensorboard"
            )
            self.writer = None

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int
    ) -> None:
        """Log a scalar value.

        Args:
            tag: Name of the scalar (e.g., "train/loss")
            value: Scalar value
            step: Global step
        """
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        tag: str,
        values: Dict[str, float],
        step: int
    ) -> None:
        """Log multiple scalars under one tag.

        Args:
            tag: Main tag (e.g., "loss")
            values: Dictionary of scalar values (e.g., {"train": 0.5, "val": 0.6})
            step: Global step
        """
        if self.writer:
            self.writer.add_scalars(tag, values, step)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = ""
    ) -> None:
        """Log a dictionary of metrics.

        Args:
            metrics: Dictionary of metric name to value
            step: Global step
            prefix: Optional prefix for all metric names (e.g., "train/")
        """
        if not self.writer:
            return

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                tag = f"{prefix}{name}" if prefix else name
                self.log_scalar(tag, float(value), step)

    def log_learning_rate(
        self,
        lr: float,
        step: int
    ) -> None:
        """Log learning rate.

        Args:
            lr: Learning rate value
            step: Global step
        """
        self.log_scalar("learning_rate", lr, step)

    def log_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, list],
        step: int,
        bins: str = 'tensorflow'
    ) -> None:
        """Log a histogram of values.

        Args:
            tag: Histogram name (e.g., "weights/layer1")
            values: Array or list of values
            step: Global step
            bins: Binning method ('tensorflow', 'auto', 'fd', etc.)
        """
        if self.writer:
            self.writer.add_histogram(tag, values, step, bins=bins)

    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        step: int,
        dataformats: str = 'CHW'
    ) -> None:
        """Log an image.

        Args:
            tag: Image name
            image: Image array
            step: Global step
            dataformats: Format of image ('CHW', 'HWC', etc.)
        """
        if self.writer:
            self.writer.add_image(tag, image, step, dataformats=dataformats)

    def log_images(
        self,
        tag: str,
        images: np.ndarray,
        step: int,
        dataformats: str = 'NCHW'
    ) -> None:
        """Log multiple images as a grid.

        Args:
            tag: Grid name
            images: Batch of images
            step: Global step
            dataformats: Format of images batch
        """
        if self.writer:
            self.writer.add_images(tag, images, step, dataformats=dataformats)

    def log_text(
        self,
        tag: str,
        text: str,
        step: int
    ) -> None:
        """Log text.

        Args:
            tag: Text tag
            text: Text content
            step: Global step
        """
        if self.writer:
            self.writer.add_text(tag, text, step)

    def log_hparams(
        self,
        hparams: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> None:
        """Log hyperparameters and metrics.

        Args:
            hparams: Dictionary of hyperparameters
            metrics: Dictionary of metrics
        """
        if self.writer:
            self.writer.add_hparams(hparams, metrics)

    def log_graph(
        self,
        model: Any,
        input_to_model: Any
    ) -> None:
        """Log model graph.

        Args:
            model: PyTorch model
            input_to_model: Example input tensor
        """
        if self.writer:
            try:
                self.writer.add_graph(model, input_to_model)
            except Exception as e:
                logger.warning(f"Failed to log model graph: {e}")

    def close(self) -> None:
        """Close the Tensorboard writer."""
        if self.writer:
            self.writer.close()
            logger.info("Tensorboard logger closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class TrainingMetricsLogger:
    """Convenience logger for common training metrics.

    Provides pre-configured logging for standard training metrics.

    Args:
        tensorboard_logger: TensorboardLogger instance
        log_frequency: Log metrics every N steps
    """

    def __init__(
        self,
        tensorboard_logger: TensorboardLogger,
        log_frequency: int = 10
    ):
        """Initialize metrics logger."""
        self.tb_logger = tensorboard_logger
        self.log_frequency = log_frequency
        self.step_count = 0

    def log_training_step(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        **extra_metrics
    ) -> None:
        """Log metrics for a training step.

        Args:
            step: Global step
            loss: Training loss
            learning_rate: Current learning rate
            **extra_metrics: Additional metrics to log
        """
        # Log every N steps
        if step % self.log_frequency == 0:
            self.tb_logger.log_scalar("train/loss", loss, step)
            self.tb_logger.log_learning_rate(learning_rate, step)

            # Log extra metrics
            for name, value in extra_metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_logger.log_scalar(f"train/{name}", value, step)

        self.step_count += 1

    def log_validation_step(
        self,
        step: int,
        metrics: Dict[str, float]
    ) -> None:
        """Log validation metrics.

        Args:
            step: Global step
            metrics: Dictionary of validation metrics
        """
        self.tb_logger.log_metrics(metrics, step, prefix="val/")

    def log_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Log summary metrics for an epoch.

        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Optional validation metrics
        """
        # Log training metrics
        for name, value in train_metrics.items():
            self.tb_logger.log_scalar(f"epoch/train_{name}", value, epoch)

        # Log validation metrics
        if val_metrics:
            for name, value in val_metrics.items():
                self.tb_logger.log_scalar(f"epoch/val_{name}", value, epoch)

    def log_model_weights(
        self,
        model: Any,
        step: int
    ) -> None:
        """Log model weight histograms.

        Args:
            model: PyTorch model
            step: Global step
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.tb_logger.log_histogram(
                    f"weights/{name}",
                    param.data.cpu().numpy(),
                    step
                )
                if param.grad is not None:
                    self.tb_logger.log_histogram(
                        f"gradients/{name}",
                        param.grad.cpu().numpy(),
                        step
                    )


# Convenience function
def create_tensorboard_logger(
    log_dir: Union[str, Path],
    comment: str = "",
    log_frequency: int = 10
) -> tuple[TensorboardLogger, TrainingMetricsLogger]:
    """Create Tensorboard loggers.

    Args:
        log_dir: Directory for Tensorboard logs
        comment: Optional comment
        log_frequency: Logging frequency

    Returns:
        Tuple of (TensorboardLogger, TrainingMetricsLogger)

    Example:
        >>> tb_logger, metrics_logger = create_tensorboard_logger("./runs")
        >>> metrics_logger.log_training_step(step=100, loss=0.5, learning_rate=1e-4)
    """
    tb_logger = TensorboardLogger(log_dir, comment=comment)
    metrics_logger = TrainingMetricsLogger(tb_logger, log_frequency=log_frequency)

    return tb_logger, metrics_logger
