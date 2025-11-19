"""Stratified dataset splitter that maintains class balance."""

from collections import defaultdict
from typing import Any, Dict, List, Tuple
import random
import logging

from .split_config import SplitConfig

logger = logging.getLogger(__name__)


class StratifiedSplitter:
    """Split dataset while maintaining class distribution across splits.

    Ensures that train/val/test splits have similar class proportions
    to the original dataset.
    """

    def __init__(self, config: SplitConfig):
        """Initialize stratified splitter.

        Args:
            config: Split configuration
        """
        self.config = config
        random.seed(config.seed)

    def split(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Perform stratified split of data.

        Args:
            data: List of data samples, each with a 'label' or 'labels' field

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Group samples by class
        class_groups = self._group_by_class(data)

        train_data = []
        val_data = []
        test_data = []

        # Split each class proportionally
        for label, samples in class_groups.items():
            # Shuffle samples within class
            shuffled = samples.copy()
            random.shuffle(shuffled)

            # Calculate split indices
            n = len(shuffled)
            train_end = int(n * self.config.train_ratio)
            val_end = train_end + int(n * self.config.val_ratio)

            # Split samples
            train_data.extend(shuffled[:train_end])
            val_data.extend(shuffled[train_end:val_end])
            test_data.extend(shuffled[val_end:])

            logger.debug(
                f"Class '{label}': {len(shuffled)} samples -> "
                f"train={len(shuffled[:train_end])}, "
                f"val={len(shuffled[train_end:val_end])}, "
                f"test={len(shuffled[val_end:])}"
            )

        # Final shuffle to mix classes
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

        logger.info(
            f"Stratified split complete: "
            f"train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
        )

        return train_data, val_data, test_data

    def _group_by_class(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Group samples by their class labels.

        Args:
            data: List of data samples

        Returns:
            Dictionary mapping labels to lists of samples
        """
        groups = defaultdict(list)

        for sample in data:
            # Handle different label formats
            if "label" in sample:
                label = sample["label"]
                groups[label].append(sample)
            elif "labels" in sample:
                # For multi-label samples, use first label for stratification
                labels = sample["labels"]
                if labels:
                    primary_label = labels[0] if isinstance(labels, list) else str(labels)
                    groups[primary_label].append(sample)
                else:
                    groups["unlabeled"].append(sample)
            else:
                # No label field, group as unlabeled
                groups["unlabeled"].append(sample)

        return dict(groups)

    def validate_split(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        test_data: List[Dict],
        tolerance: float = 0.05
    ) -> Dict[str, Any]:
        """Validate that splits maintain class distribution.

        Args:
            train_data: Training split
            val_data: Validation split
            test_data: Test split
            tolerance: Acceptable deviation from original distribution (default: 5%)

        Returns:
            Dictionary with validation results
        """
        # Calculate distributions
        all_data = train_data + val_data + test_data
        original_dist = self._calculate_distribution(all_data)
        train_dist = self._calculate_distribution(train_data)
        val_dist = self._calculate_distribution(val_data)
        test_dist = self._calculate_distribution(test_data)

        # Check deviations
        deviations = {}
        max_deviation = 0.0

        for label in original_dist:
            orig_pct = original_dist[label]
            train_pct = train_dist.get(label, 0)
            val_pct = val_dist.get(label, 0)
            test_pct = test_dist.get(label, 0)

            deviations[label] = {
                "original": round(orig_pct, 2),
                "train": round(train_pct, 2),
                "val": round(val_pct, 2),
                "test": round(test_pct, 2),
                "train_dev": round(abs(train_pct - orig_pct), 2),
                "val_dev": round(abs(val_pct - orig_pct), 2),
                "test_dev": round(abs(test_pct - orig_pct), 2),
            }

            max_deviation = max(
                max_deviation,
                abs(train_pct - orig_pct),
                abs(val_pct - orig_pct),
                abs(test_pct - orig_pct)
            )

        # Determine if split is valid
        is_valid = max_deviation <= tolerance * 100

        return {
            "is_valid": is_valid,
            "max_deviation_percent": round(max_deviation, 2),
            "tolerance_percent": tolerance * 100,
            "deviations_by_class": deviations
        }

    def _calculate_distribution(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate class distribution as percentages.

        Args:
            data: List of samples

        Returns:
            Dictionary mapping labels to percentage (0-100)
        """
        if not data:
            return {}

        groups = self._group_by_class(data)
        total = len(data)

        distribution = {
            label: (len(samples) / total * 100)
            for label, samples in groups.items()
        }

        return distribution
