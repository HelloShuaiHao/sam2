"""Dataset balance analysis and class distribution checks."""

from typing import Any, Dict
import logging

from .validator import ValidationRule, ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


class BalanceAnalyzer(ValidationRule):
    """Analyze dataset class balance and detect imbalances.

    Flags severe class imbalance that may affect training quality.
    """

    def __init__(self, imbalance_threshold: float = 10.0):
        """Initialize balance analyzer.

        Args:
            imbalance_threshold: Ratio threshold for severe imbalance (default: 10.0)
                                If max_class/min_class > threshold, flag as severe
        """
        super().__init__("class_balance_analysis")
        self.imbalance_threshold = imbalance_threshold

    def validate(self, data: Any) -> ValidationResult:
        """Analyze class distribution and detect imbalances.

        Args:
            data: Dataset with class annotations

        Returns:
            ValidationResult with balance analysis
        """
        # Calculate class distribution
        class_counts = self._calculate_distribution(data)

        if not class_counts:
            return ValidationResult(
                name=self.name,
                status=ValidationStatus.ERROR,
                message="No class labels found in dataset"
            )

        # Analyze balance
        analysis = self._analyze_balance(class_counts)

        return analysis

    def _calculate_distribution(self, data: Any) -> Dict[str, int]:
        """Calculate class distribution across dataset.

        Args:
            data: Dataset with annotations

        Returns:
            Dictionary mapping class labels to counts
        """
        distribution: Dict[str, int] = {}

        frames = data.get("frames", [])
        for frame in frames:
            objects = frame.get("objects", [])
            for obj in objects:
                label = obj.get("label", "unknown")
                distribution[label] = distribution.get(label, 0) + 1

        return distribution

    def _analyze_balance(self, class_counts: Dict[str, int]) -> ValidationResult:
        """Analyze class balance and generate result.

        Args:
            class_counts: Dictionary of class counts

        Returns:
            ValidationResult with balance analysis
        """
        if not class_counts:
            return ValidationResult(
                name=self.name,
                status=ValidationStatus.ERROR,
                message="No classes found"
            )

        # Get stats
        total_instances = sum(class_counts.values())
        num_classes = len(class_counts)
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        max_class = max(class_counts, key=class_counts.get)
        min_class = min(class_counts, key=class_counts.get)

        # Calculate imbalance ratio
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        # Calculate percentages
        percentages = {
            label: (count / total_instances * 100)
            for label, count in class_counts.items()
        }

        # Determine status
        if imbalance_ratio >= self.imbalance_threshold:
            status = ValidationStatus.WARNING
            message = (
                f"Severe class imbalance detected: "
                f"{max_class} has {imbalance_ratio:.1f}x more samples than {min_class}"
            )
        elif imbalance_ratio >= 5.0:
            status = ValidationStatus.WARNING
            message = (
                f"Moderate class imbalance detected: "
                f"{max_class} has {imbalance_ratio:.1f}x more samples than {min_class}"
            )
        else:
            status = ValidationStatus.PASSED
            message = f"Classes are reasonably balanced across {num_classes} classes"

        return ValidationResult(
            name=self.name,
            status=status,
            message=message,
            details={
                "total_instances": total_instances,
                "num_classes": num_classes,
                "class_distribution": class_counts,
                "class_percentages": percentages,
                "imbalance_ratio": round(imbalance_ratio, 2),
                "most_common": max_class,
                "least_common": min_class
            }
        )
