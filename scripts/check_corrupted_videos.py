#!/usr/bin/env python3
"""
Script to check for and optionally remove corrupted video files in the gallery.
Usage:
    python check_corrupted_videos.py --check  # Only check, don't delete
    python check_corrupted_videos.py --remove # Check and remove corrupted files
"""

import argparse
import os
import shutil
import subprocess
from glob import glob
from pathlib import Path


def validate_video_file(filepath):
    """
    Validate if a video file is readable and not corrupted.
    Returns (is_valid, error_message)
    """
    try:
        # Check if file exists and has size > 0
        if not os.path.exists(filepath):
            return False, "File does not exist"

        file_size = os.path.getsize(filepath)
        if file_size == 0:
            return False, "File is empty (0 bytes)"

        # Try to probe the video file with ffprobe
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return True, "Warning: ffprobe not found, cannot validate"

        result = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=codec_type,stream=width,stream=height",
                "-of", "csv=p=0",
                str(filepath)
            ],
            capture_output=True,
            timeout=10,
            text=True
        )

        # If ffprobe returns non-zero, video is corrupted
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return False, f"ffprobe error: {error_msg}"

        # Check if we got valid output
        if not result.stdout.strip():
            return False, "No video stream found"

        return True, f"Valid ({file_size / 1024 / 1024:.2f} MB)"

    except subprocess.TimeoutExpired:
        return False, "Timeout while validating"
    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Check for corrupted video files')
    parser.add_argument('--gallery-path', default='/data/gallery',
                        help='Path to gallery directory (default: /data/gallery)')
    parser.add_argument('--remove', action='store_true',
                        help='Remove corrupted files (default: only check)')
    parser.add_argument('--check', action='store_true',
                        help='Only check files, do not remove (default)')

    args = parser.parse_args()

    # Default to check mode if neither is specified
    if not args.remove and not args.check:
        args.check = True

    gallery_path = args.gallery_path

    if not os.path.exists(gallery_path):
        print(f"Gallery path does not exist: {gallery_path}")
        return

    print(f"Scanning for video files in: {gallery_path}")
    video_pattern = os.path.join(gallery_path, "**/*.mp4")
    video_paths = glob(video_pattern, recursive=True)

    print(f"Found {len(video_paths)} video files\n")

    corrupted_files = []
    valid_files = []

    for video_path in video_paths:
        is_valid, message = validate_video_file(video_path)

        status = "✓ VALID" if is_valid else "✗ CORRUPTED"
        print(f"{status}: {video_path}")
        print(f"  → {message}")

        if is_valid:
            valid_files.append(video_path)
        else:
            corrupted_files.append((video_path, message))

    print(f"\n{'=' * 80}")
    print(f"Summary:")
    print(f"  Valid files: {len(valid_files)}")
    print(f"  Corrupted files: {len(corrupted_files)}")
    print(f"{'=' * 80}\n")

    if corrupted_files:
        print("Corrupted files:")
        for path, reason in corrupted_files:
            print(f"  - {path}")
            print(f"    Reason: {reason}")

        if args.remove:
            print(f"\nRemoving {len(corrupted_files)} corrupted files...")
            for path, _ in corrupted_files:
                try:
                    os.remove(path)
                    print(f"  Removed: {path}")

                    # Also try to remove associated poster
                    poster_path = path.replace('/gallery/', '/posters/').replace('.mp4', '.jpg')
                    if os.path.exists(poster_path):
                        os.remove(poster_path)
                        print(f"  Removed poster: {poster_path}")
                except Exception as e:
                    print(f"  Failed to remove {path}: {e}")
            print("\nCleanup complete!")
        else:
            print("\nTo remove these files, run with --remove flag")
    else:
        print("All video files are valid!")


if __name__ == '__main__':
    main()
