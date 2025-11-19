"""Configuration for dataset splitting."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import json


class SplitStrategy(Enum):
    """Dataset splitting strategies."""
    STRATIFIED = "stratified"  # Maintain class balance
    TEMPORAL = "temporal"       # Split by time/video segments
    RANDOM = "random"           # Simple random split


@dataclass
class SplitConfig:
    """Configuration for dataset splitting.

    Attributes:
        strategy: Splitting strategy to use
        train_ratio: Proportion for training set (0.0-1.0)
        val_ratio: Proportion for validation set (0.0-1.0)
        test_ratio: Proportion for test set (0.0-1.0)
        seed: Random seed for reproducibility
        metadata: Additional metadata
    """
    strategy: SplitStrategy
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int = 42
    metadata: Optional[dict] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Check ratios sum to 1.0 (with some tolerance for float precision)
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )

        # Check ratios are non-negative
        if self.train_ratio < 0 or self.val_ratio < 0 or self.test_ratio < 0:
            raise ValueError("Split ratios must be non-negative")

        # Validate strategy
        if not isinstance(self.strategy, SplitStrategy):
            raise ValueError(f"Invalid strategy: {self.strategy}")

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "strategy": self.strategy.value,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
            "metadata": self.metadata or {}
        }

    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file.

        Args:
            filepath: Path to save configuration
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> 'SplitConfig':
        """Create configuration from dictionary.

        Args:
            data: Dictionary with configuration

        Returns:
            SplitConfig instance
        """
        strategy = SplitStrategy(data["strategy"])
        return cls(
            strategy=strategy,
            train_ratio=data["train_ratio"],
            val_ratio=data["val_ratio"],
            test_ratio=data["test_ratio"],
            seed=data.get("seed", 42),
            metadata=data.get("metadata")
        )

    @classmethod
    def from_json(cls, filepath: str) -> 'SplitConfig':
        """Load configuration from JSON file.

        Args:
            filepath: Path to configuration file

        Returns:
            SplitConfig instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
