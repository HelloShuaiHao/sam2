"""Simple random dataset splitter."""

from typing import Any, Dict, List, Tuple
import random
import logging

from .split_config import SplitConfig

logger = logging.getLogger(__name__)


class RandomSplitter:
    """Perform simple random split of dataset.

    Randomly assigns samples to train/val/test splits according to
    configured ratios. No class balancing or temporal considerations.
    """

    def __init__(self, config: SplitConfig):
        """Initialize random splitter.

        Args:
            config: Split configuration
        """
        self.config = config
        random.seed(config.seed)

    def split(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Perform random split of data.

        Args:
            data: List of data samples

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Shuffle data
        shuffled = data.copy()
        random.shuffle(shuffled)

        # Calculate split indices
        n = len(shuffled)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)

        # Split data
        train_data = shuffled[:train_end]
        val_data = shuffled[train_end:val_end]
        test_data = shuffled[val_end:]

        logger.info(
            f"Random split complete: "
            f"train={len(train_data)} ({len(train_data)/n*100:.1f}%), "
            f"val={len(val_data)} ({len(val_data)/n*100:.1f}%), "
            f"test={len(test_data)} ({len(test_data)/n*100:.1f}%)"
        )

        return train_data, val_data, test_data

    def validate_split(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        test_data: List[Dict]
    ) -> Dict[str, Any]:
        """Validate split ratios match configuration.

        Args:
            train_data: Training split
            val_data: Validation split
            test_data: Test split

        Returns:
            Dictionary with validation results
        """
        total = len(train_data) + len(val_data) + len(test_data)

        if total == 0:
            return {
                "is_valid": False,
                "message": "Empty dataset"
            }

        actual_ratios = {
            "train": len(train_data) / total,
            "val": len(val_data) / total,
            "test": len(test_data) / total
        }

        expected_ratios = {
            "train": self.config.train_ratio,
            "val": self.config.val_ratio,
            "test": self.config.test_ratio
        }

        # Calculate deviations
        deviations = {
            "train": abs(actual_ratios["train"] - expected_ratios["train"]),
            "val": abs(actual_ratios["val"] - expected_ratios["val"]),
            "test": abs(actual_ratios["test"] - expected_ratios["test"])
        }

        max_deviation = max(deviations.values())

        # Allow 2% deviation due to rounding
        is_valid = max_deviation <= 0.02

        return {
            "is_valid": is_valid,
            "actual_ratios": actual_ratios,
            "expected_ratios": expected_ratios,
            "deviations": deviations,
            "max_deviation": round(max_deviation, 4),
            "sizes": {
                "train": len(train_data),
                "val": len(val_data),
                "test": len(test_data),
                "total": total
            }
        }
