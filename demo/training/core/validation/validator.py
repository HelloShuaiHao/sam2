"""Core validator framework for dataset quality validation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation status levels."""
    PASSED = "passed"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationResult:
    """Result from a single validation check.

    Attributes:
        name: Name of the validation check
        status: Status level (PASSED, WARNING, ERROR)
        message: Human-readable message
        details: Optional detailed information
    """
    name: str
    status: ValidationStatus
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details or {}
        }


class ValidationRule(ABC):
    """Abstract base class for validation rules.

    Each validation rule implements a specific check on the dataset.
    """

    def __init__(self, name: str):
        """Initialize validation rule.

        Args:
            name: Name of this validation rule
        """
        self.name = name

    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """Execute validation check.

        Args:
            data: Data to validate

        Returns:
            ValidationResult with check outcome
        """
        pass


class Validator:
    """Main validator that orchestrates multiple validation rules.

    The validator runs a suite of validation rules and aggregates results.
    """

    def __init__(self):
        """Initialize validator."""
        self.rules: List[ValidationRule] = []

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule.

        Args:
            rule: ValidationRule to add
        """
        self.rules.append(rule)
        logger.debug(f"Added validation rule: {rule.name}")

    def validate(self, data: Any) -> Dict[str, Any]:
        """Run all validation rules on data.

        Args:
            data: Data to validate

        Returns:
            Dictionary containing:
                - status: Overall status (passed/passed_with_warnings/failed)
                - summary: Summary statistics
                - checks: List of individual check results
                - recommendations: List of recommended actions
        """
        results = []
        errors = 0
        warnings = 0

        # Run all validation rules
        for rule in self.rules:
            try:
                result = rule.validate(data)
                results.append(result)

                if result.status == ValidationStatus.ERROR:
                    errors += 1
                elif result.status == ValidationStatus.WARNING:
                    warnings += 1

                logger.info(f"Validation rule '{rule.name}': {result.status.value}")

            except Exception as e:
                logger.error(f"Validation rule '{rule.name}' failed with exception: {e}")
                results.append(ValidationResult(
                    name=rule.name,
                    status=ValidationStatus.ERROR,
                    message=f"Validation failed: {str(e)}",
                    details={"exception": str(e)}
                ))
                errors += 1

        # Determine overall status
        if errors > 0:
            overall_status = "failed"
        elif warnings > 0:
            overall_status = "passed_with_warnings"
        else:
            overall_status = "passed"

        # Generate recommendations
        recommendations = self._generate_recommendations(results)

        return {
            "status": overall_status,
            "summary": {
                "total_checks": len(results),
                "passed": len(results) - errors - warnings,
                "warnings": warnings,
                "errors": errors
            },
            "checks": [r.to_dict() for r in results],
            "recommendations": recommendations
        }

    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate actionable recommendations based on validation results.

        Args:
            results: List of validation results

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check for specific issues and provide recommendations
        for result in results:
            if result.status == ValidationStatus.ERROR:
                if "missing" in result.message.lower():
                    recommendations.append("Remove or fix frames with missing annotations")
                elif "corrupt" in result.message.lower():
                    recommendations.append("Regenerate corrupted masks from source")

            elif result.status == ValidationStatus.WARNING:
                if "imbalance" in result.message.lower():
                    recommendations.append("Consider data augmentation for minority classes")
                elif "small" in result.message.lower():
                    recommendations.append("Review annotations for unusually small objects")

        # Add general recommendation if dataset is valid
        if not any(r.status == ValidationStatus.ERROR for r in results):
            recommendations.append("Dataset is ready for training")

        return recommendations
