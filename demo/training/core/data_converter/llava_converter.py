"""LLaVA instruction format converter."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List
import logging

from .base_converter import BaseConverter
from .sam2_parser import SAM2Parser

logger = logging.getLogger(__name__)


class LLaVAConverter(BaseConverter):
    """Convert SAM2 exports to LLaVA instruction-tuning format.

    Creates JSONL files compatible with LLaVA training pipeline,
    with varied instruction-response pairs for better generalization.
    """

    # Instruction templates for variety
    INSTRUCTION_TEMPLATES = [
        "Identify and segment all objects in this image.",
        "Segment all visible objects in the image.",
        "Find and segment each object present in this image.",
        "Detect and segment all objects shown.",
        "Locate and segment every object in the image.",
    ]

    SPECIFIC_TEMPLATES = {
        "person": [
            "Segment the person in this image.",
            "Identify and segment all people.",
            "Find all persons in the image.",
        ],
        "car": [
            "Segment all cars in this image.",
            "Identify and segment the vehicles.",
            "Find all cars present.",
        ],
        "dog": [
            "Segment the dog in this image.",
            "Identify all dogs shown.",
            "Find and segment each dog.",
        ]
    }

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize LLaVA converter.

        Args:
            config: Optional configuration including:
                - vary_instructions: Whether to use varied templates (default: True)
                - custom_templates: Additional instruction templates
        """
        super().__init__(config)
        self.parser = SAM2Parser()
        self.vary_instructions = config.get("vary_instructions", True) if config else True

    def convert(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """Convert SAM2 export to LLaVA JSONL format.

        Args:
            input_path: Path to SAM2 export ZIP or extracted directory
            output_path: Path to save LLaVA dataset

        Returns:
            Conversion statistics
        """
        # Parse SAM2 export
        if input_path.suffix == '.zip':
            data = self.parser.parse_zip(input_path)
        else:
            data = self.parser.parse_directory(input_path)

        logger.info(f"Parsed SAM2 export from {input_path}")

        # Convert to LLaVA format
        frames = self.parser.get_frames()
        samples = []

        for frame in frames:
            if not frame["objects"]:  # Skip frames with no annotations
                continue

            sample = self._create_llava_sample(frame)
            if sample:
                samples.append(sample)

        # Save as JSONL
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / "llava_dataset.jsonl"

        with open(output_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

        # Create manifest file
        manifest = self._create_manifest(samples, output_file)
        with open(output_path / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Saved {len(samples)} samples to {output_file}")

        return {
            "total_samples": len(samples),
            "total_classes": len(self.parser.get_class_distribution()),
            "class_distribution": self.parser.get_class_distribution(),
            "output_format": "llava",
            "success": True,
            "output_path": str(output_path)
        }

    def validate(self, data: Any) -> bool:
        """Validate LLaVA format.

        Args:
            data: Sample to validate

        Returns:
            True if valid
        """
        required_fields = ["id", "image", "conversations"]
        if not all(field in data for field in required_fields):
            return False

        # Validate conversations structure
        convs = data.get("conversations", [])
        if not convs or len(convs) < 2:
            return False

        # Check conversation format
        return all(
            "from" in turn and "value" in turn
            for turn in convs
        )

    def _create_llava_sample(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        """Create a LLaVA dataset sample from a frame.

        Args:
            frame: Frame data from SAM2 parser

        Returns:
            LLaVA sample dictionary
        """
        frame_idx = frame["frame_index"]
        objects = frame["objects"]

        # Extract object information
        labels = [obj.get("label", "object") for obj in objects]
        masks = [obj.get("mask_rle", "") for obj in objects]
        bboxes = [obj.get("bbox", [0, 0, 0, 0]) for obj in objects]

        # Generate conversations
        conversations = self._generate_llava_conversations(labels, bboxes)

        sample = {
            "id": f"sam2_frame_{frame_idx:04d}",
            "image": str(frame["image_path"]) if frame["image_path"] else f"frame_{frame_idx:04d}.jpg",
            "conversations": conversations,
            "masks": masks,
            "bounding_boxes": bboxes
        }

        return sample

    def _generate_llava_conversations(
        self,
        labels: List[str],
        bboxes: List[List[int]]
    ) -> List[Dict[str, str]]:
        """Generate LLaVA-style conversation pairs.

        Args:
            labels: List of object labels
            bboxes: List of bounding boxes

        Returns:
            List of conversation turns
        """
        # Get unique labels and counts
        unique_labels = list(set(labels))
        label_counts = {label: labels.count(label) for label in unique_labels}

        # Choose instruction template
        if self.vary_instructions:
            if len(unique_labels) == 1 and unique_labels[0] in self.SPECIFIC_TEMPLATES:
                instruction = random.choice(self.SPECIFIC_TEMPLATES[unique_labels[0]])
            else:
                instruction = random.choice(self.INSTRUCTION_TEMPLATES)
        else:
            instruction = "Identify and segment all objects in this image."

        # Generate response with object details
        response = self._generate_response(label_counts, bboxes)

        return [
            {
                "from": "human",
                "value": f"<image>\n{instruction}"
            },
            {
                "from": "gpt",
                "value": response
            }
        ]

    def _generate_response(
        self,
        label_counts: Dict[str, int],
        bboxes: List[List[int]]
    ) -> str:
        """Generate assistant response text.

        Args:
            label_counts: Dictionary of label to count
            bboxes: List of bounding boxes

        Returns:
            Response string
        """
        if not label_counts:
            return "I don't see any objects in the image."

        # Build object description
        if len(label_counts) == 1:
            label, count = list(label_counts.items())[0]
            if count == 1:
                desc = f"1 {label}"
            else:
                plural = f"{label}s" if not label.endswith('s') else label
                desc = f"{count} {plural}"
        else:
            parts = []
            for label, count in sorted(label_counts.items()):
                plural = f"{label}s" if count > 1 and not label.endswith('s') else label
                parts.append(f"{count} {plural}")
            desc = ", ".join(parts[:-1]) + f" and {parts[-1]}"

        # Include location information
        response = f"I can see {desc} in the image. "

        # Add bounding box information for first few objects
        if bboxes and len(bboxes) <= 3:
            response += "The objects are located at coordinates: "
            coords = [f"[{b[0]}, {b[1]}, {b[2]}, {b[3]}]" for b in bboxes]
            response += ", ".join(coords) + ". "

        response += "I've provided segmentation masks for each object."

        return response

    def _create_manifest(self, samples: List[Dict[str, Any]], dataset_file: Path) -> Dict[str, Any]:
        """Create manifest file with dataset information.

        Args:
            samples: List of all samples
            dataset_file: Path to dataset JSONL file

        Returns:
            Manifest dictionary
        """
        # Extract all labels
        all_labels = []
        for sample in samples:
            # Count objects per sample
            num_objects = len(sample.get("masks", []))
            all_labels.extend([f"object_{i}" for i in range(num_objects)])

        return {
            "dataset_name": "SAM2 LLaVA Format",
            "version": "1.0.0",
            "format": "llava_jsonl",
            "total_samples": len(samples),
            "dataset_file": str(dataset_file.name),
            "description": "Video segmentation annotations in LLaVA instruction-tuning format",
            "features": {
                "id": "Unique sample identifier",
                "image": "Path to image file",
                "conversations": "Instruction-response pairs",
                "masks": "RLE-encoded segmentation masks",
                "bounding_boxes": "Object bounding boxes [x, y, w, h]"
            }
        }

    def get_supported_formats(self) -> List[str]:
        """Get supported output formats.

        Returns:
            List containing "llava"
        """
        return ["llava"]
