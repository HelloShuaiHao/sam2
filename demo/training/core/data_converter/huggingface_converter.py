"""HuggingFace dataset format converter."""

import json
from pathlib import Path
from typing import Any, Dict, List
import logging

from .base_converter import BaseConverter
from .sam2_parser import SAM2Parser

logger = logging.getLogger(__name__)


class HuggingFaceConverter(BaseConverter):
    """Convert SAM2 exports to HuggingFace Datasets format.

    Creates a dataset compatible with HuggingFace's datasets library,
    including images, conversations, masks, and bounding boxes.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize HuggingFace converter.

        Args:
            config: Optional configuration including:
                - generate_instructions: Whether to generate instruction pairs (default: True)
                - instruction_templates: List of instruction templates to use
        """
        super().__init__(config)
        self.parser = SAM2Parser()

    def convert(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """Convert SAM2 export to HuggingFace dataset format.

        Args:
            input_path: Path to SAM2 export ZIP or extracted directory
            output_path: Path to save HuggingFace dataset

        Returns:
            Conversion statistics
        """
        # Parse SAM2 export
        if input_path.suffix == '.zip':
            data = self.parser.parse_zip(input_path)
        else:
            data = self.parser.parse_directory(input_path)

        logger.info(f"Parsed SAM2 export from {input_path}")

        # Convert to HuggingFace format
        frames = self.parser.get_frames()
        samples = []

        for frame in frames:
            if not frame["objects"]:  # Skip frames with no annotations
                continue

            sample = self._create_sample(frame)
            if sample:
                samples.append(sample)

        # Save as JSONL (compatible with datasets.load_dataset)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / "dataset.jsonl"

        with open(output_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

        # Create dataset info file
        info = self._create_dataset_info(samples)
        with open(output_path / "dataset_info.json", 'w') as f:
            json.dump(info, f, indent=2)

        logger.info(f"Saved {len(samples)} samples to {output_file}")

        return {
            "total_samples": len(samples),
            "total_classes": len(self.parser.get_class_distribution()),
            "class_distribution": self.parser.get_class_distribution(),
            "output_format": "huggingface",
            "success": True,
            "output_path": str(output_path)
        }

    def validate(self, data: Any) -> bool:
        """Validate HuggingFace dataset format.

        Args:
            data: Dataset sample to validate

        Returns:
            True if valid
        """
        required_fields = ["id", "image", "conversations"]
        return all(field in data for field in required_fields)

    def _create_sample(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        """Create a HuggingFace dataset sample from a frame.

        Args:
            frame: Frame data from SAM2 parser

        Returns:
            Dataset sample dictionary
        """
        frame_idx = frame["frame_index"]
        objects = frame["objects"]

        # Extract object information
        labels = [obj.get("label", "object") for obj in objects]
        masks = [obj.get("mask_rle", "") for obj in objects]
        bboxes = [obj.get("bbox", [0, 0, 0, 0]) for obj in objects]

        # Generate conversations
        conversations = self._generate_conversations(labels, len(objects))

        sample = {
            "id": f"frame_{frame_idx:04d}",
            "image": str(frame["image_path"]) if frame["image_path"] else "",
            "conversations": conversations,
            "masks": masks,
            "bounding_boxes": bboxes,
            "frame_index": frame_idx,
            "timestamp_sec": frame["timestamp_sec"]
        }

        return sample

    def _generate_conversations(self, labels: List[str], num_objects: int) -> List[Dict[str, str]]:
        """Generate instruction-response conversation pairs.

        Args:
            labels: List of object labels
            num_objects: Number of objects

        Returns:
            List of conversation turns
        """
        # Get unique labels
        unique_labels = list(set(labels))
        label_counts = {label: labels.count(label) for label in unique_labels}

        # Generate human instruction
        if len(unique_labels) == 1:
            label = unique_labels[0]
            count = label_counts[label]
            if count == 1:
                human_msg = f"<image>\nIdentify and segment the {label} in this image."
            else:
                human_msg = f"<image>\nIdentify and segment all {label}s in this image."
        else:
            human_msg = "<image>\nIdentify and segment all objects in this image."

        # Generate assistant response
        if len(unique_labels) == 1:
            label = unique_labels[0]
            count = label_counts[label]
            if count == 1:
                gpt_msg = f"I can see 1 {label} in the image with segmentation mask and bounding box."
            else:
                gpt_msg = f"I can see {count} {label}s in the image with segmentation masks and bounding boxes."
        else:
            parts = [f"{count} {label}{'s' if count > 1 else ''}" for label, count in label_counts.items()]
            gpt_msg = f"I can see {', '.join(parts)} in the image with segmentation masks and bounding boxes."

        return [
            {"from": "human", "value": human_msg},
            {"from": "gpt", "value": gpt_msg}
        ]

    def _create_dataset_info(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create dataset information file.

        Args:
            samples: List of all dataset samples

        Returns:
            Dataset info dictionary
        """
        all_labels = []
        for sample in samples:
            # Extract labels from conversations or metadata
            for obj_labels in [sample.get("labels", [])]:
                all_labels.extend(obj_labels)

        unique_labels = list(set(all_labels)) if all_labels else []

        return {
            "name": "SAM2 Vision-Language Dataset",
            "version": "1.0.0",
            "description": "Dataset converted from SAM2 annotations for vision-language model fine-tuning",
            "total_samples": len(samples),
            "num_classes": len(unique_labels),
            "classes": unique_labels,
            "format": "huggingface",
            "features": {
                "id": "string",
                "image": "image_path",
                "conversations": "list[dict]",
                "masks": "list[string]",
                "bounding_boxes": "list[list[int]]"
            }
        }

    def get_supported_formats(self) -> List[str]:
        """Get supported output formats.

        Returns:
            List containing "huggingface"
        """
        return ["huggingface"]
