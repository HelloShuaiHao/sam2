"""Generate human-readable validation reports."""

from datetime import datetime
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate Markdown validation reports from validation results."""

    @staticmethod
    def generate_report(validation_results: Dict[str, Any], dataset_name: str = "Dataset") -> str:
        """Generate Markdown report from validation results.

        Args:
            validation_results: Results from Validator.validate()
            dataset_name: Name of the dataset being validated

        Returns:
            Markdown-formatted report string
        """
        lines = []

        # Header
        lines.append(f"# Dataset Quality Validation Report")
        lines.append(f"")
        lines.append(f"**Dataset:** {dataset_name}")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Status:** {validation_results['status'].upper()}")
        lines.append(f"")

        # Summary
        summary = validation_results.get("summary", {})
        lines.append(f"## Summary")
        lines.append(f"")
        lines.append(f"- **Total Checks:** {summary.get('total_checks', 0)}")
        lines.append(f"- **Passed:** ✅ {summary.get('passed', 0)}")
        lines.append(f"- **Warnings:** ⚠️  {summary.get('warnings', 0)}")
        lines.append(f"- **Errors:** ❌ {summary.get('errors', 0)}")
        lines.append(f"")

        # Detailed checks
        checks = validation_results.get("checks", [])

        # Passed checks
        passed = [c for c in checks if c["status"] == "passed"]
        if passed:
            lines.append(f"## ✅ Passed Checks")
            lines.append(f"")
            for check in passed:
                lines.append(f"- **{check['name']}:** {check['message']}")
            lines.append(f"")

        # Warnings
        warnings = [c for c in checks if c["status"] == "warning"]
        if warnings:
            lines.append(f"## ⚠️  Warnings")
            lines.append(f"")
            for check in warnings:
                lines.append(f"### {check['name']}")
                lines.append(f"{check['message']}")
                lines.append(f"")

                # Add details if available
                details = check.get("details", {})
                if details:
                    lines.append(f"**Details:**")
                    for key, value in details.items():
                        if isinstance(value, (list, dict)):
                            continue  # Skip complex types for main report
                        lines.append(f"- {key}: {value}")
                    lines.append(f"")

        # Errors
        errors = [c for c in checks if c["status"] == "error"]
        if errors:
            lines.append(f"## ❌ Errors")
            lines.append(f"")
            for check in errors:
                lines.append(f"### {check['name']}")
                lines.append(f"{check['message']}")
                lines.append(f"")

                # Add details
                details = check.get("details", {})
                if details:
                    lines.append(f"**Details:**")
                    for key, value in details.items():
                        if isinstance(value, list) and key.endswith("_frames") or key.endswith("_boxes"):
                            lines.append(f"- {key}: {len(value)} items")
                            if value:
                                lines.append(f"  - First few: {value[:5]}")
                        elif not isinstance(value, (list, dict)):
                            lines.append(f"- {key}: {value}")
                    lines.append(f"")

        # Recommendations
        recommendations = validation_results.get("recommendations", [])
        if recommendations:
            lines.append(f"## 💡 Recommendations")
            lines.append(f"")
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append(f"")

        # Add class distribution if available
        for check in checks:
            if check["name"] == "class_balance_analysis":
                details = check.get("details", {})
                if "class_distribution" in details:
                    lines.append(f"## 📊 Class Distribution")
                    lines.append(f"")
                    dist = details["class_distribution"]
                    percentages = details.get("class_percentages", {})
                    lines.append(f"| Class | Count | Percentage |")
                    lines.append(f"|-------|-------|------------|")
                    for label in sorted(dist.keys(), key=lambda x: dist[x], reverse=True):
                        count = dist[label]
                        pct = percentages.get(label, 0)
                        lines.append(f"| {label} | {count} | {pct:.1f}% |")
                    lines.append(f"")

        return "\n".join(lines)

    @staticmethod
    def save_report(report: str, output_path: str) -> None:
        """Save report to file.

        Args:
            report: Report content
            output_path: Path to save report
        """
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Validation report saved to {output_path}")
