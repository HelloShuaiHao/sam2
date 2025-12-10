# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
import shutil
import subprocess
from glob import glob
from pathlib import Path
from typing import Dict, Optional

import imagesize
from app_conf import GALLERY_PATH, POSTERS_PATH, POSTERS_PREFIX
from data.data_types import Video
from tqdm import tqdm

logger = logging.getLogger(__name__)


def validate_video_file(filepath: os.PathLike) -> bool:
    """
    Validate if a video file is readable and not corrupted.
    Returns True if valid, False otherwise.
    """
    try:
        # Check if file exists and has size > 0
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            logger.warning(f"Video file {filepath} does not exist or is empty")
            return False

        # Try to probe the video file with ffprobe
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-count_packets",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    str(filepath)
                ],
                capture_output=True,
                timeout=5
            )
            # If ffprobe returns non-zero or output is empty, video is likely corrupted
            if result.returncode != 0 or not result.stdout:
                logger.warning(f"Video file {filepath} failed ffprobe validation")
                return False

        return True
    except Exception as e:
        logger.warning(f"Error validating video file {filepath}: {e}")
        return False


def preload_data() -> Dict[str, Video]:
    """
    Preload data including gallery videos and their posters.
    """
    # Dictionaries for videos and datasets on the backend.
    # Note that since Python 3.7, dictionaries preserve their insert order, so
    # when looping over its `.values()`, elements inserted first also appear first.
    # https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6
    all_videos = {}

    video_path_pattern = os.path.join(GALLERY_PATH, "**/*.mp4")
    video_paths = glob(video_path_pattern, recursive=True)

    for p in tqdm(video_paths):
        # Validate video file before processing
        if not validate_video_file(p):
            logger.warning(f"Skipping corrupted or invalid video: {p}")
            continue

        try:
            video = get_video(p, GALLERY_PATH)
            all_videos[video.code] = video
        except Exception as e:
            logger.warning(f"Failed to load video {p}: {e}. Skipping this video.")
            continue

    return all_videos


def get_video(
    filepath: os.PathLike,
    absolute_path: Path,
    file_key: Optional[str] = None,
    generate_poster: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None,
    verbose: Optional[bool] = False,
) -> Video:
    """
    Get video object given
    """
    # Use absolute_path to include the parent directory in the video
    video_path = os.path.relpath(filepath, absolute_path.parent)

    # Extract date from folder structure (e.g., gallery/2025-01-15/video.mp4)
    date = None
    path_parts = Path(video_path).parts
    for part in path_parts:
        # Match YYYY-MM-DD format
        if re.match(r'^\d{4}-\d{2}-\d{2}$', part):
            date = part
            break

    poster_path = None
    if generate_poster:
        poster_id = os.path.splitext(os.path.basename(filepath))[0]
        poster_filename = f"{str(poster_id)}.jpg"
        poster_path = f"{POSTERS_PREFIX}/{poster_filename}"

        # Extract the first frame from video
        poster_output_path = os.path.join(POSTERS_PATH, poster_filename)
        ffmpeg = shutil.which("ffmpeg")
        result = subprocess.call(
            [
                ffmpeg,
                "-y",
                "-i",
                str(filepath),
                "-pix_fmt",
                "yuv420p",
                "-frames:v",
                "1",
                "-update",
                "1",
                "-strict",
                "unofficial",
                str(poster_output_path),
            ],
            stdout=None if verbose else subprocess.DEVNULL,
            stderr=None if verbose else subprocess.DEVNULL,
        )

        # Extract video width and height from poster. This is important to optimize
        # rendering previews in the mosaic video preview.
        if result == 0 and os.path.exists(poster_output_path):
            width, height = imagesize.get(poster_output_path)
        else:
            # If poster generation failed, try to get dimensions from video directly
            logger.warning(f"Failed to generate poster for {filepath}, using default dimensions")
            width, height = 1920, 1080  # Default dimensions

    return Video(
        code=video_path,
        path=video_path if file_key is None else file_key,
        poster_path=poster_path,
        width=width,
        height=height,
        date=date,
    )
