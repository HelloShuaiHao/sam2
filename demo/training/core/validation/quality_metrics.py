"""Quality metrics calculation for dataset annotations."""

from typing import Any, Dict, List
import logging
import statistics

from .validator import ValidationRule, ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


class QualityMetrics(ValidationRule):
    """Calculate and validate annotation quality metrics.

    Checks for unusually small objects, annotation density, and other
    quality indicators.
    """

    def __init__(self, min_object_area_percent: float = 0.1):
        """Initialize quality metrics calculator.

        Args:
            min_object_area_percent: Minimum object area as % of image (default: 0.1%)
        """
        super().__init__("quality_metrics")
        self.min_object_area_percent = min_object_area_percent

    def validate(self, data: Any) -> ValidationResult:
        """Calculate quality metrics for dataset.

        Args:
            data: Dataset with annotations

        Returns:
            ValidationResult with quality metrics
        """
        metrics = self._calculate_metrics(data)

        return self._analyze_metrics(metrics)

    def _calculate_metrics(self, data: Any) -> Dict[str, Any]:
        """Calculate various quality metrics.

        Args:
            data: Dataset

        Returns:
            Dictionary of calculated metrics
        """
        frames = data.get("frames", [])
        metadata = data.get("metadata", {})
        video_info = metadata.get("video", {})

        img_width = video_info.get("width", 1920)
        img_height = video_info.get("height", 1080)
        image_area = img_width * img_height

        # Collect metrics
        object_areas = []
        object_counts_per_frame = []
        small_objects = []

        for frame in frames:
            objects = frame.get("objects", [])
            object_counts_per_frame.append(len(objects))

            for obj in objects:
                bbox = obj.get("bbox", [0, 0, 0, 0])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    area = w * h
                    area_percent = (area / image_area * 100) if image_area > 0 else 0

                    object_areas.append(area)

                    # Flag small objects
                    if area_percent < self.min_object_area_percent:
                        small_objects.append({
                            "frame": frame.get("frame_index"),
                            "object_id": obj.get("object_id"),
                            "area_percent": round(area_percent, 3),
                            "label": obj.get("label")
                        })

        # Calculate statistics
        metrics = {
            "total_frames": len(frames),
            "total_objects": sum(object_counts_per_frame),
            "avg_objects_per_frame": statistics.mean(object_counts_per_frame) if object_counts_per_frame else 0,
            "min_objects_per_frame": min(object_counts_per_frame) if object_counts_per_frame else 0,
            "max_objects_per_frame": max(object_counts_per_frame) if object_counts_per_frame else 0,
            "avg_object_area": statistics.mean(object_areas) if object_areas else 0,
            "median_object_area": statistics.median(object_areas) if object_areas else 0,
            "small_objects_count": len(small_objects),
            "small_objects": small_objects
        }

        return metrics

    def _analyze_metrics(self, metrics: Dict[str, Any]) -> ValidationResult:
        """Analyze metrics and generate validation result.

        Args:
            metrics: Calculated metrics

        Returns:
            ValidationResult
        """
        warnings = []
        small_count = metrics["small_objects_count"]
        total_objects = metrics["total_objects"]

        # Check for small objects
        if small_count > 0:
            percentage = (small_count / total_objects * 100) if total_objects > 0 else 0
            if percentage > 5:
                warnings.append(
                    f"{small_count} objects ({percentage:.1f}%) have area < "
                    f"{self.min_object_area_percent}% of image"
                )

        # Check for low annotation density
        avg_objects = metrics["avg_objects_per_frame"]
        if avg_objects < 1.0:
            warnings.append(
                f"Low annotation density: {avg_objects:.1f} objects per frame on average"
            )

        # Determine status
        if warnings:
            status = ValidationStatus.WARNING
            message = f"Quality issues found: {len(warnings)} warnings"
        else:
            status = ValidationStatus.PASSED
            message = (
                f"Good annotation quality: {total_objects} objects across "
                f"{metrics['total_frames']} frames"
            )

        # Remove small_objects list from details for brevity, keep summary
        details = metrics.copy()
        details["small_objects"] = details["small_objects"][:10]  # First 10 only
        details["warnings"] = warnings

        return ValidationResult(
            name=self.name,
            status=status,
            message=message,
            details=details
        )
