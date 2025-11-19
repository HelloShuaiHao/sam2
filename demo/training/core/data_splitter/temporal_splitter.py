"""Temporal dataset splitter for video data to prevent data leakage."""

from typing import Any, Dict, List, Tuple
import random
import logging

from .split_config import SplitConfig

logger = logging.getLogger(__name__)


class TemporalSplitter:
    """Split video dataset by temporal segments to prevent data leakage.

    Ensures that frames from the same video or temporal segment don't
    appear in multiple splits, which could lead to inflated performance.
    """

    def __init__(self, config: SplitConfig):
        """Initialize temporal splitter.

        Args:
            config: Split configuration
        """
        self.config = config
        random.seed(config.seed)

    def split(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Perform temporal split of video data.

        Args:
            data: List of frame samples with 'video_id' or 'frame_index' fields

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Group frames by video/segment
        video_groups = self._group_by_video(data)

        logger.info(f"Found {len(video_groups)} video segments with {len(data)} total frames")

        # Sort video groups by size for better distribution
        sorted_videos = sorted(
            video_groups.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        # Shuffle to randomize which videos go to which split
        random.shuffle(sorted_videos)

        # Allocate videos to splits
        train_data = []
        val_data = []
        test_data = []

        total_frames = len(data)
        target_train = int(total_frames * self.config.train_ratio)
        target_val = int(total_frames * self.config.val_ratio)

        # Greedy allocation to match ratios as closely as possible
        for video_id, frames in sorted_videos:
            current_train = len(train_data)
            current_val = len(val_data)

            # Decide which split gets this video
            if current_train < target_train:
                train_data.extend(frames)
            elif current_val < target_val:
                val_data.extend(frames)
            else:
                test_data.extend(frames)

        logger.info(
            f"Temporal split complete: "
            f"train={len(train_data)} ({len(train_data)/total_frames*100:.1f}%), "
            f"val={len(val_data)} ({len(val_data)/total_frames*100:.1f}%), "
            f"test={len(test_data)} ({len(test_data)/total_frames*100:.1f}%)"
        )

        return train_data, val_data, test_data

    def _group_by_video(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Group frames by video ID or temporal segments.

        Args:
            data: List of frame samples

        Returns:
            Dictionary mapping video IDs to lists of frames
        """
        groups = {}

        for sample in data:
            # Try to extract video ID from different possible fields
            video_id = self._extract_video_id(sample)

            if video_id not in groups:
                groups[video_id] = []
            groups[video_id].append(sample)

        return groups

    def _extract_video_id(self, sample: Dict[str, Any]) -> str:
        """Extract video ID from sample.

        Args:
            sample: Data sample

        Returns:
            Video ID or generated segment ID
        """
        # Try explicit video_id field
        if "video_id" in sample:
            return str(sample["video_id"])

        # Try to extract from image path
        if "image" in sample:
            image_path = str(sample["image"])
            # Extract video name from path (e.g., "video1/frame_001.jpg" -> "video1")
            parts = image_path.split("/")
            if len(parts) > 1:
                return parts[-2]  # Directory name before filename

        # Try frame_index with temporal segmentation
        # Group every N frames as a segment to prevent leakage
        if "frame_index" in sample:
            frame_idx = sample["frame_index"]
            segment_size = 30  # Group frames in 30-frame segments
            segment_id = frame_idx // segment_size
            return f"segment_{segment_id}"

        # Fallback: use sample ID if available
        if "id" in sample:
            # Extract video part from ID (e.g., "video1_frame_001" -> "video1")
            sample_id = str(sample["id"])
            if "_frame_" in sample_id:
                return sample_id.split("_frame_")[0]
            return sample_id

        # Last resort: all in one group
        return "unknown_video"

    def validate_split(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        test_data: List[Dict]
    ) -> Dict[str, Any]:
        """Validate that there's no video overlap between splits.

        Args:
            train_data: Training split
            val_data: Validation split
            test_data: Test split

        Returns:
            Dictionary with validation results
        """
        # Get video IDs from each split
        train_videos = set(self._extract_video_id(s) for s in train_data)
        val_videos = set(self._extract_video_id(s) for s in val_data)
        test_videos = set(self._extract_video_id(s) for s in test_data)

        # Check for overlaps
        train_val_overlap = train_videos & val_videos
        train_test_overlap = train_videos & test_videos
        val_test_overlap = val_videos & test_videos

        has_overlap = bool(train_val_overlap or train_test_overlap or val_test_overlap)

        return {
            "has_overlap": has_overlap,
            "train_videos": len(train_videos),
            "val_videos": len(val_videos),
            "test_videos": len(test_videos),
            "overlaps": {
                "train_val": list(train_val_overlap),
                "train_test": list(train_test_overlap),
                "val_test": list(val_test_overlap)
            },
            "is_valid": not has_overlap
        }
