"""Best model tracker for selecting best checkpoint based on metrics."""

from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class BestModelTracker:
    """Track and identify the best model checkpoint based on metrics.

    Supports both minimization (loss) and maximization (accuracy) metrics.
    """

    def __init__(
        self,
        metric_name: str = "eval_loss",
        mode: str = "min"
    ):
        """Initialize best model tracker.

        Args:
            metric_name: Name of metric to track
            mode: "min" for loss-like metrics, "max" for accuracy-like metrics
        """
        self.metric_name = metric_name
        self.mode = mode

        if mode not in ["min", "max"]:
            raise ValueError(f"Mode must be 'min' or 'max', got: {mode}")

        self.best_value: Optional[float] = None
        self.best_step: Optional[int] = None
        self.best_epoch: Optional[int] = None

        logger.info(f"Tracking best model by {mode}imizing '{metric_name}'")

    def update(self, step: int, epoch: int, metrics: dict) -> bool:
        """Update tracker with new metrics.

        Args:
            step: Current training step
            epoch: Current epoch
            metrics: Dictionary of metrics

        Returns:
            True if this is a new best, False otherwise
        """
        if self.metric_name not in metrics:
            logger.warning(f"Metric '{self.metric_name}' not found in metrics")
            return False

        current_value = metrics[self.metric_name]

        # Check if this is the best so far
        is_best = self._is_better(current_value)

        if is_best:
            self.best_value = current_value
            self.best_step = step
            self.best_epoch = epoch

            logger.info(
                f"New best {self.metric_name}: {current_value:.4f} "
                f"(step={step}, epoch={epoch})"
            )

        return is_best

    def _is_better(self, value: float) -> bool:
        """Check if value is better than current best.

        Args:
            value: Value to compare

        Returns:
            True if better, False otherwise
        """
        if self.best_value is None:
            return True

        if self.mode == "min":
            return value < self.best_value
        else:  # mode == "max"
            return value > self.best_value

    def get_best(self) -> Optional[dict]:
        """Get best checkpoint information.

        Returns:
            Dict with best model info, or None if no checkpoints yet
        """
        if self.best_value is None:
            return None

        return {
            "metric_name": self.metric_name,
            "value": self.best_value,
            "step": self.best_step,
            "epoch": self.best_epoch,
            "mode": self.mode
        }

    def reset(self) -> None:
        """Reset tracker to initial state."""
        self.best_value = None
        self.best_step = None
        self.best_epoch = None
        logger.info("Best model tracker reset")
