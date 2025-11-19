"""Parser for SAM2 annotation exports."""

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SAM2Parser:
    """Parse SAM2 annotation export files.

    Handles extraction and parsing of SAM2 export ZIP files containing:
    - annotations.json: Frame-level annotations with object masks
    - metadata.json: Video metadata (fps, dimensions, etc.)
    - Frame images (referenced in annotations)
    """

    def __init__(self):
        """Initialize SAM2 parser."""
        self.annotations: Optional[Dict[str, Any]] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.frames_dir: Optional[Path] = None

    def parse_zip(self, zip_path: Path, extract_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Parse SAM2 export ZIP file.

        Args:
            zip_path: Path to SAM2 export ZIP file
            extract_dir: Optional directory to extract files to.
                        If None, uses temp directory.

        Returns:
            Dictionary containing:
                - annotations: Parsed annotations data
                - metadata: Video metadata
                - frames_dir: Path to extracted frames directory

        Raises:
            ValueError: If ZIP file is invalid or missing required files
            FileNotFoundError: If ZIP file doesn't exist
        """
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")

        # Create extraction directory
        if extract_dir is None:
            extract_dir = zip_path.parent / f"{zip_path.stem}_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Extract ZIP file
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            logger.info(f"Extracted {zip_path} to {extract_dir}")
        except zipfile.BadZipFile as e:
            raise ValueError(f"Invalid ZIP file: {e}")

        # Parse extracted files
        return self.parse_directory(extract_dir)

    def parse_directory(self, dir_path: Path) -> Dict[str, Any]:
        """Parse SAM2 export directory.

        Args:
            dir_path: Path to directory containing SAM2 export files

        Returns:
            Dictionary containing parsed data

        Raises:
            ValueError: If required files are missing
        """
        # Handle nested directory structure (ZIP may extract to subdirectory)
        actual_dir = dir_path
        if not (dir_path / "annotations.json").exists():
            # Look for subdirectory containing the export (skip __MACOSX and hidden dirs)
            subdirs = [
                d for d in dir_path.iterdir()
                if d.is_dir()
                and not d.name.startswith('.')
                and not d.name.startswith('__')
            ]
            if subdirs:
                # Try first non-hidden, non-metadata subdirectory
                potential_dir = subdirs[0]
                if (potential_dir / "annotations.json").exists():
                    actual_dir = potential_dir
                    logger.info(f"Found export files in subdirectory: {actual_dir.name}")

        annotations_file = actual_dir / "annotations.json"
        metadata_file = actual_dir / "metadata.json"

        # Check for required files
        if not annotations_file.exists():
            raise ValueError(f"Missing annotations.json in {actual_dir}")

        # Parse annotations
        with open(annotations_file, 'r') as f:
            full_data = json.load(f)

        # Extract annotations array
        self.annotations = {"annotations": full_data.get("annotations", [])}

        # Parse metadata if available in annotations file or separate metadata file
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        elif "video" in full_data:
            # Video metadata is in the annotations file
            self.metadata = {"video": full_data.get("video", {})}
        else:
            logger.warning(f"metadata.json not found and no video info in annotations, using defaults")
            self.metadata = self._create_default_metadata()

        # Find frames directory
        possible_frame_dirs = ["frames", "images", "data"]
        for frame_dir_name in possible_frame_dirs:
            frame_dir = actual_dir / frame_dir_name
            if frame_dir.exists() and frame_dir.is_dir():
                self.frames_dir = frame_dir
                break

        if self.frames_dir is None:
            logger.warning(f"No frames directory found in {actual_dir}")
            self.frames_dir = actual_dir

        return {
            "annotations": self.annotations,
            "metadata": self.metadata,
            "frames_dir": self.frames_dir
        }

    def get_frames(self) -> List[Dict[str, Any]]:
        """Get list of all annotated frames.

        Returns:
            List of frame dictionaries containing:
                - frame_index: Frame number
                - timestamp_sec: Timestamp in seconds
                - objects: List of annotated objects in frame
                - image_path: Path to frame image

        Raises:
            RuntimeError: If parser hasn't been initialized with parse_zip/parse_directory
        """
        if self.annotations is None:
            raise RuntimeError("Parser not initialized. Call parse_zip() or parse_directory() first.")

        frames = []
        annotations_list = self.annotations.get("annotations", [])

        for frame_data in annotations_list:
            frame = {
                "frame_index": frame_data.get("frame_index"),
                "timestamp_sec": frame_data.get("timestamp_sec", 0.0),
                "objects": frame_data.get("objects", []),
                "image_path": self._get_frame_image_path(frame_data.get("frame_index"))
            }
            frames.append(frame)

        return frames

    def get_objects_by_frame(self, frame_index: int) -> List[Dict[str, Any]]:
        """Get all objects in a specific frame.

        Args:
            frame_index: Frame number

        Returns:
            List of object dictionaries containing:
                - object_id: Unique object identifier
                - label: Object class label
                - mask_rle: RLE-encoded segmentation mask
                - bbox: Bounding box [x, y, width, height]

        Raises:
            RuntimeError: If parser not initialized
            ValueError: If frame not found
        """
        if self.annotations is None:
            raise RuntimeError("Parser not initialized.")

        for frame_data in self.annotations.get("annotations", []):
            if frame_data.get("frame_index") == frame_index:
                return frame_data.get("objects", [])

        raise ValueError(f"Frame {frame_index} not found in annotations")

    def get_class_distribution(self) -> Dict[str, int]:
        """Calculate class distribution across all frames.

        Returns:
            Dictionary mapping class labels to occurrence counts
        """
        if self.annotations is None:
            return {}

        distribution: Dict[str, int] = {}

        for frame_data in self.annotations.get("annotations", []):
            for obj in frame_data.get("objects", []):
                label = obj.get("label", "unknown")
                distribution[label] = distribution.get(label, 0) + 1

        return distribution

    def get_video_metadata(self) -> Dict[str, Any]:
        """Get video metadata.

        Returns:
            Dictionary containing video metadata:
                - filename: Video filename
                - width: Video width in pixels
                - height: Video height in pixels
                - source_fps: Original video FPS
                - total_frames: Total number of frames
        """
        if self.metadata is None:
            return self._create_default_metadata()

        return self.metadata.get("video", {})

    def _get_frame_image_path(self, frame_index: int) -> Optional[Path]:
        """Get path to frame image file.

        Args:
            frame_index: Frame number

        Returns:
            Path to frame image, or None if not found
        """
        if self.frames_dir is None:
            return None

        # Try common frame naming patterns (prioritize PNG as that's common in SAM2 exports)
        patterns = [
            f"frame_{frame_index:06d}.png",
            f"frame_{frame_index:06d}.jpg",
            f"frame_{frame_index:04d}.png",
            f"frame_{frame_index:04d}.jpg",
            f"{frame_index:06d}.png",
            f"{frame_index:06d}.jpg",
            f"{frame_index:04d}.png",
            f"{frame_index:04d}.jpg",
        ]

        for pattern in patterns:
            frame_path = self.frames_dir / pattern
            if frame_path.exists():
                return frame_path

        return None

    def _create_default_metadata(self) -> Dict[str, Any]:
        """Create default metadata when metadata.json is missing.

        Returns:
            Dictionary with default metadata values
        """
        return {
            "video": {
                "filename": "unknown.mp4",
                "width": 1920,
                "height": 1080,
                "source_fps": 30,
                "total_frames": 0
            }
        }
