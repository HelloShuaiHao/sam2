"""Base trainer interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base class for model trainers.

    All trainer implementations must inherit from this class and
    implement the required methods.
    """

    def __init__(self, config: Any):
        """Initialize trainer with configuration.

        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.current_step = 0
        self.current_epoch = 0

    @abstractmethod
    def setup(self) -> None:
        """Set up trainer (load model, tokenizer, etc.)."""
        pass

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Execute training loop.

        Returns:
            Dictionary with training results and metrics
        """
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on validation set.

        Returns:
            Dictionary with evaluation metrics
        """
        pass

    @abstractmethod
    def save_checkpoint(self, output_dir: Path) -> None:
        """Save model checkpoint.

        Args:
            output_dir: Directory to save checkpoint
        """
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        pass

    def get_progress(self) -> Dict[str, Any]:
        """Get current training progress.

        Returns:
            Dictionary with progress information
        """
        return {
            "current_step": self.current_step,
            "current_epoch": self.current_epoch
        }
