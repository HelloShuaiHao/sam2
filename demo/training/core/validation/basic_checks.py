"""Basic validation checks for dataset integrity."""

from typing import Any, Dict, List, Set
import logging

from .validator import ValidationRule, ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


class MissingAnnotationsCheck(ValidationRule):
    """Check for frames with missing or zero annotations."""

    def __init__(self):
        super().__init__("missing_annotations_check")

    def validate(self, data: Any) -> ValidationResult:
        """Check for missing annotations.

        Args:
            data: Dataset containing frames with annotations

        Returns:
            ValidationResult
        """
        frames = data.get("frames", [])
        missing_frames = []

        for frame in frames:
            objects = frame.get("objects", [])
            if not objects or len(objects) == 0:
                missing_frames.append(frame.get("frame_index", "unknown"))

        total_frames = len(frames)
        missing_count = len(missing_frames)

        if missing_count == 0:
            return ValidationResult(
                name=self.name,
                status=ValidationStatus.PASSED,
                message=f"All {total_frames} frames have annotations"
            )
        else:
            percentage = (missing_count / total_frames * 100) if total_frames > 0 else 0

            return ValidationResult(
                name=self.name,
                status=ValidationStatus.ERROR,
                message=f"{missing_count} frames ({percentage:.1f}%) have no annotations",
                details={
                    "missing_frames": missing_frames[:10],  # First 10 for brevity
                    "total_missing": missing_count
                }
            )


class DuplicateFramesCheck(ValidationRule):
    """Check for duplicate frames in the dataset."""

    def __init__(self):
        super().__init__("duplicate_frames_check")

    def validate(self, data: Any) -> ValidationResult:
        """Check for duplicate frame indices.

        Args:
            data: Dataset containing frames

        Returns:
            ValidationResult
        """
        frames = data.get("frames", [])
        frame_indices = [f.get("frame_index") for f in frames]

        # Find duplicates
        seen: Set[int] = set()
        duplicates = []

        for idx in frame_indices:
            if idx in seen:
                duplicates.append(idx)
            else:
                seen.add(idx)

        if not duplicates:
            return ValidationResult(
                name=self.name,
                status=ValidationStatus.PASSED,
                message=f"No duplicate frames found ({len(frames)} unique frames)"
            )
        else:
            return ValidationResult(
                name=self.name,
                status=ValidationStatus.WARNING,
                message=f"Found {len(duplicates)} duplicate frame indices",
                details={"duplicates": duplicates[:10]}
            )


class MaskIntegrityCheck(ValidationRule):
    """Check RLE mask format and integrity."""

    def __init__(self):
        super().__init__("mask_integrity_check")

    def validate(self, data: Any) -> ValidationResult:
        """Validate mask RLE encoding.

        Args:
            data: Dataset with mask annotations

        Returns:
            ValidationResult
        """
        frames = data.get("frames", [])
        invalid_masks = []
        total_masks = 0

        for frame in frames:
            objects = frame.get("objects", [])
            for obj in objects:
                total_masks += 1
                mask_rle = obj.get("mask_rle")

                # Basic validation: check if mask_rle exists and is non-empty
                if not mask_rle or (isinstance(mask_rle, str) and len(mask_rle) == 0):
                    invalid_masks.append({
                        "frame": frame.get("frame_index"),
                        "object_id": obj.get("object_id"),
                        "reason": "Empty or missing RLE"
                    })
                    continue

                # For dict format (common RLE format)
                if isinstance(mask_rle, dict):
                    if "counts" not in mask_rle or "size" not in mask_rle:
                        invalid_masks.append({
                            "frame": frame.get("frame_index"),
                            "object_id": obj.get("object_id"),
                            "reason": "Invalid RLE dict format"
                        })

        invalid_count = len(invalid_masks)
        percentage = (invalid_count / total_masks * 100) if total_masks > 0 else 0

        if invalid_count == 0:
            return ValidationResult(
                name=self.name,
                status=ValidationStatus.PASSED,
                message=f"All {total_masks} masks have valid RLE encoding"
            )
        else:
            return ValidationResult(
                name=self.name,
                status=ValidationStatus.ERROR,
                message=f"{invalid_count} masks ({percentage:.1f}%) have invalid RLE encoding",
                details={
                    "invalid_masks": invalid_masks[:10],
                    "total_invalid": invalid_count
                }
            )


class BoundingBoxCheck(ValidationRule):
    """Check bounding box validity."""

    def __init__(self, image_width: int = 1920, image_height: int = 1080):
        """Initialize bounding box check.

        Args:
            image_width: Expected image width
            image_height: Expected image height
        """
        super().__init__("bounding_box_check")
        self.image_width = image_width
        self.image_height = image_height

    def validate(self, data: Any) -> ValidationResult:
        """Validate bounding boxes are within image bounds.

        Args:
            data: Dataset with bounding box annotations

        Returns:
            ValidationResult
        """
        frames = data.get("frames", [])

        # Get image dimensions from metadata if available
        metadata = data.get("metadata", {})
        video_info = metadata.get("video", {})
        img_width = video_info.get("width", self.image_width)
        img_height = video_info.get("height", self.image_height)

        invalid_boxes = []
        total_boxes = 0

        for frame in frames:
            objects = frame.get("objects", [])
            for obj in objects:
                total_boxes += 1
                bbox = obj.get("bbox", [])

                if len(bbox) != 4:
                    invalid_boxes.append({
                        "frame": frame.get("frame_index"),
                        "object_id": obj.get("object_id"),
                        "reason": "Invalid bbox format (not 4 values)"
                    })
                    continue

                x, y, w, h = bbox

                # Check bounds
                if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
                    invalid_boxes.append({
                        "frame": frame.get("frame_index"),
                        "object_id": obj.get("object_id"),
                        "reason": f"Bbox out of bounds: [{x}, {y}, {w}, {h}]"
                    })

                # Check for zero-size boxes
                if w <= 0 or h <= 0:
                    invalid_boxes.append({
                        "frame": frame.get("frame_index"),
                        "object_id": obj.get("object_id"),
                        "reason": f"Zero or negative size: w={w}, h={h}"
                    })

        invalid_count = len(invalid_boxes)

        if invalid_count == 0:
            return ValidationResult(
                name=self.name,
                status=ValidationStatus.PASSED,
                message=f"All {total_boxes} bounding boxes are valid"
            )
        else:
            percentage = (invalid_count / total_boxes * 100) if total_boxes > 0 else 0
            return ValidationResult(
                name=self.name,
                status=ValidationStatus.ERROR,
                message=f"{invalid_count} bounding boxes ({percentage:.1f}%) are invalid",
                details={
                    "invalid_boxes": invalid_boxes[:10],
                    "total_invalid": invalid_count
                }
            )


class BasicChecks:
    """Convenience class to run all basic validation checks."""

    @staticmethod
    def get_all_checks(image_width: int = 1920, image_height: int = 1080) -> List[ValidationRule]:
        """Get list of all basic validation checks.

        Args:
            image_width: Expected image width
            image_height: Expected image height

        Returns:
            List of ValidationRule instances
        """
        return [
            MissingAnnotationsCheck(),
            DuplicateFramesCheck(),
            MaskIntegrityCheck(),
            BoundingBoxCheck(image_width, image_height)
        ]
