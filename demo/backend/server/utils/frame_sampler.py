# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Frame sampling utility for exporting annotations at specified frame rates.

This module provides time-based frame sampling to ensure consistent results
across videos with variable frame rates or dropped frames.
"""

from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class FrameSampler:
    """
    Samples video frames based on target FPS using time-based intervals.
    """

    def __init__(self, source_fps: float, total_frames: int, duration_sec: float):
        """
        Initialize the frame sampler.

        Args:
            source_fps: Source video frame rate
            total_frames: Total number of frames in the video
            duration_sec: Video duration in seconds
        """
        self.source_fps = source_fps
        self.total_frames = total_frames
        self.duration_sec = duration_sec

    def calculate_frame_indices(self, target_fps: float) -> List[int]:
        """
        Calculate which frame indices to export based on target FPS.

        Uses time-based sampling: calculates target timestamps at regular intervals
        (1/target_fps) and finds the closest frame to each timestamp.

        Args:
            target_fps: Target export frame rate

        Returns:
            List of frame indices to export (sorted, unique)

        Raises:
            ValueError: If target_fps is invalid or exceeds source_fps
        """
        if target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {target_fps}")

        if target_fps > self.source_fps:
            logger.warning(
                f"target_fps ({target_fps}) exceeds source_fps ({self.source_fps}), "
                f"capping at source_fps"
            )
            target_fps = self.source_fps

        # Calculate time interval between exported frames
        time_interval = 1.0 / target_fps

        # Generate target timestamps
        timestamps = []
        current_time = 0.0
        while current_time < self.duration_sec:
            timestamps.append(current_time)
            current_time += time_interval

        # Convert timestamps to frame indices
        frame_indices = []
        for timestamp in timestamps:
            # Find closest frame to this timestamp
            frame_index = self._timestamp_to_frame_index(timestamp)
            if frame_index < self.total_frames:
                frame_indices.append(frame_index)

        # Remove duplicates and sort
        frame_indices = sorted(set(frame_indices))

        logger.info(
            f"Sampled {len(frame_indices)} frames from {self.total_frames} "
            f"(target: {target_fps} FPS, source: {self.source_fps} FPS)"
        )

        return frame_indices

    def _timestamp_to_frame_index(self, timestamp: float) -> int:
        """
        Convert timestamp to nearest frame index.

        Args:
            timestamp: Time in seconds

        Returns:
            Frame index (0-based)
        """
        # Calculate frame index: timestamp * fps
        frame_index = round(timestamp * self.source_fps)
        # Clamp to valid range
        return max(0, min(frame_index, self.total_frames - 1))

    def estimate_export_count(self, target_fps: float) -> int:
        """
        Estimate how many frames will be exported at target FPS.

        Args:
            target_fps: Target export frame rate

        Returns:
            Estimated number of frames
        """
        if target_fps <= 0:
            return 0
        if target_fps > self.source_fps:
            target_fps = self.source_fps

        # Estimated count is duration * target_fps
        estimated = int(self.duration_sec * target_fps)
        # Cap at total frames
        return min(estimated, self.total_frames)

    def get_sampling_stats(self, target_fps: float) -> dict:
        """
        Get detailed statistics about the sampling.

        Args:
            target_fps: Target export frame rate

        Returns:
            Dictionary with sampling statistics
        """
        frame_indices = self.calculate_frame_indices(target_fps)

        return {
            "source_fps": self.source_fps,
            "target_fps": target_fps,
            "total_frames": self.total_frames,
            "exported_frames": len(frame_indices),
            "duration_sec": self.duration_sec,
            "sampling_ratio": len(frame_indices) / self.total_frames if self.total_frames > 0 else 0,
            "time_interval_sec": 1.0 / min(target_fps, self.source_fps),
        }
