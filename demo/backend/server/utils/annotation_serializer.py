# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
JSON annotation serializer for exporting video tracking results.

Generates structured JSON output with video metadata, export configuration,
and per-frame annotations with RLE-encoded masks.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from utils.rle_encoder import RLEEncoder

logger = logging.getLogger(__name__)


class AnnotationSerializer:
    """
    Serializes video annotation data to JSON format.
    """

    def __init__(self, video_metadata: Dict[str, Any]):
        """
        Initialize serializer with video metadata.

        Args:
            video_metadata: Dictionary containing video information
                Required keys: filename, width, height, fps, total_frames, duration_sec
        """
        self.video_metadata = video_metadata
        self.annotations = []

    def add_frame_annotation(
        self,
        frame_index: int,
        timestamp_sec: float,
        objects: List[Dict[str, Any]]
    ):
        """
        Add annotations for a single frame.

        Args:
            frame_index: Frame index (0-based)
            timestamp_sec: Frame timestamp in seconds
            objects: List of object dictionaries, each containing:
                - object_id: int
                - label: str (optional)
                - mask: np.ndarray (binary mask)
                - confidence: float (optional)
        """
        frame_annotation = {
            "frame_index": frame_index,
            "timestamp_sec": round(timestamp_sec, 3),
            "objects": []
        }

        for obj in objects:
            # Encode mask to RLE
            mask_rle = RLEEncoder.encode(obj["mask"])

            # Calculate bounding box and area
            bbox = RLEEncoder.rle_to_bbox(mask_rle)
            area = RLEEncoder.calculate_mask_area(mask_rle)

            obj_annotation = {
                "object_id": obj["object_id"],
                "label": obj.get("label", f"object_{obj['object_id']}"),
                "mask_rle": mask_rle["counts"],  # Just the counts string
                "bbox": bbox,  # [x, y, width, height]
                "area": area,
                "confidence": obj.get("confidence", 1.0)
            }

            frame_annotation["objects"].append(obj_annotation)

        self.annotations.append(frame_annotation)

    def serialize(
        self,
        target_fps: float,
        frame_indices: List[int],
        export_timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate the complete JSON export structure.

        Args:
            target_fps: Export frame rate
            frame_indices: List of frame indices that were exported
            export_timestamp: ISO timestamp of export (defaults to current time)

        Returns:
            Dictionary containing complete export data
        """
        if export_timestamp is None:
            export_timestamp = datetime.utcnow().isoformat() + "Z"

        export_data = {
            "video": {
                "filename": self.video_metadata.get("filename", "unknown.mp4"),
                "width": self.video_metadata["width"],
                "height": self.video_metadata["height"],
                "source_fps": self.video_metadata["fps"],
                "total_frames": self.video_metadata["total_frames"],
                "duration_sec": round(self.video_metadata["duration_sec"], 3)
            },
            "export_config": {
                "target_fps": target_fps,
                "total_exported_frames": len(frame_indices),
                "frame_indices": frame_indices,
                "export_timestamp": export_timestamp,
                "exporter_version": "1.0.0"
            },
            "annotations": sorted(self.annotations, key=lambda x: x["frame_index"])
        }

        return export_data

    def to_json_string(
        self,
        target_fps: float,
        frame_indices: List[int],
        indent: int = 2
    ) -> str:
        """
        Serialize to JSON string.

        Args:
            target_fps: Export frame rate
            frame_indices: List of exported frame indices
            indent: JSON indentation (default 2)

        Returns:
            JSON string
        """
        data = self.serialize(target_fps, frame_indices)
        return json.dumps(data, indent=indent)

    def to_json_file(
        self,
        filepath: str,
        target_fps: float,
        frame_indices: List[int],
        indent: int = 2
    ):
        """
        Save annotations to JSON file.

        Args:
            filepath: Output file path
            target_fps: Export frame rate
            frame_indices: List of exported frame indices
            indent: JSON indentation
        """
        data = self.serialize(target_fps, frame_indices)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)

        logger.info(f"Saved annotations to {filepath}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the annotations.

        Returns:
            Dictionary with annotation statistics
        """
        if not self.annotations:
            return {
                "total_frames": 0,
                "total_objects": 0,
                "unique_objects": set(),
                "objects_per_frame": 0
            }

        total_objects = sum(len(frame["objects"]) for frame in self.annotations)
        unique_objects = set()

        for frame in self.annotations:
            for obj in frame["objects"]:
                unique_objects.add(obj["object_id"])

        return {
            "total_frames": len(self.annotations),
            "total_objects": total_objects,
            "unique_objects": len(unique_objects),
            "objects_per_frame": total_objects / len(self.annotations) if self.annotations else 0
        }


def create_metadata_file(
    video_metadata: Dict[str, Any],
    export_config: Dict[str, Any],
    stats: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a separate metadata file with export information.

    Args:
        video_metadata: Video information
        export_config: Export configuration
        stats: Annotation statistics

    Returns:
        Metadata dictionary
    """
    return {
        "export_info": {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "format_version": "1.0",
            "exporter": "SAM2 Demo Export Tool"
        },
        "video": video_metadata,
        "export_config": export_config,
        "statistics": stats
    }
