# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
RLE (Run-Length Encoding) utility for compressing binary masks.

Provides COCO-compatible RLE encoding to reduce mask file sizes in exports.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class RLEEncoder:
    """
    Encodes binary masks to RLE format for efficient storage.
    """

    @staticmethod
    def encode(mask: np.ndarray) -> Dict[str, any]:
        """
        Encode a binary mask to RLE format (COCO-compatible).

        Args:
            mask: Binary mask as numpy array (H, W) with values 0 or 1

        Returns:
            Dictionary with 'counts' (RLE string) and 'size' ([H, W])

        Raises:
            ValueError: If mask is not 2D or not binary
        """
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, got shape {mask.shape}")

        # Flatten mask in Fortran order (column-major) to match COCO format
        flat_mask = np.asfortranarray(mask).flatten()

        # Ensure binary
        flat_mask = (flat_mask > 0).astype(np.uint8)

        # Run-length encoding
        # Find where values change
        diff = np.diff(flat_mask, prepend=0)
        change_indices = np.where(diff != 0)[0]

        # Calculate run lengths
        run_lengths = np.diff(change_indices, append=len(flat_mask))

        # COCO RLE format starts with count of 0s
        # If first pixel is 1, prepend a 0-length run
        if len(change_indices) > 0 and flat_mask[change_indices[0]] == 1:
            run_lengths = np.concatenate([[change_indices[0]], run_lengths])
        elif len(change_indices) == 0:
            # All zeros or all ones
            if flat_mask[0] == 0:
                run_lengths = np.array([len(flat_mask)])
            else:
                run_lengths = np.array([0, len(flat_mask)])

        # Convert to COCO RLE string format
        counts_str = " ".join(map(str, run_lengths.tolist()))

        result = {
            "counts": counts_str,
            "size": [int(mask.shape[0]), int(mask.shape[1])],  # [H, W]
        }

        # Log compression ratio
        original_size = mask.size
        compressed_size = len(counts_str)
        compression_ratio = original_size / max(compressed_size, 1)

        logger.debug(
            f"RLE encoded mask {mask.shape}: "
            f"{original_size} bytes -> {compressed_size} bytes "
            f"(compression: {compression_ratio:.1f}x)"
        )

        return result

    @staticmethod
    def decode(rle: Dict[str, any]) -> np.ndarray:
        """
        Decode RLE format back to binary mask.

        Args:
            rle: Dictionary with 'counts' and 'size'

        Returns:
            Binary mask as numpy array (H, W)
        """
        counts_str = rle["counts"]
        size = rle["size"]  # [H, W]

        # Parse run lengths
        run_lengths = list(map(int, counts_str.split()))

        # Reconstruct flat mask
        flat_mask = np.zeros(size[0] * size[1], dtype=np.uint8)

        current_idx = 0
        current_val = 0  # Start with 0

        for run_length in run_lengths:
            flat_mask[current_idx:current_idx + run_length] = current_val
            current_idx += run_length
            current_val = 1 - current_val  # Toggle between 0 and 1

        # Reshape to 2D in Fortran order
        mask = flat_mask.reshape(size, order='F')

        return mask

    @staticmethod
    def encode_to_coco_format(mask: np.ndarray) -> str:
        """
        Encode mask to COCO RLE string format (compact).

        This uses the more compact binary encoding used by COCO tools.

        Args:
            mask: Binary mask array

        Returns:
            RLE counts string
        """
        rle = RLEEncoder.encode(mask)
        return rle["counts"]

    @staticmethod
    def calculate_mask_area(rle: Dict[str, any]) -> int:
        """
        Calculate the area (number of 1-pixels) from RLE without decoding.

        Args:
            rle: RLE dictionary

        Returns:
            Number of pixels with value 1
        """
        counts_str = rle["counts"]
        run_lengths = list(map(int, counts_str.split()))

        # Sum every other run (the 1-valued runs)
        # First run is 0s, second is 1s, third is 0s, etc.
        area = sum(run_lengths[1::2])  # Start from index 1, step by 2

        return area

    @staticmethod
    def rle_to_bbox(rle: Dict[str, any]) -> List[int]:
        """
        Calculate bounding box from RLE mask.

        Args:
            rle: RLE dictionary

        Returns:
            Bounding box as [x, y, width, height]
        """
        # Decode mask to find bounding box
        mask = RLEEncoder.decode(rle)

        # Find rows and columns with any 1s
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            # Empty mask
            return [0, 0, 0, 0]

        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]

        y_min, y_max = row_indices[0], row_indices[-1]
        x_min, x_max = col_indices[0], col_indices[-1]

        # COCO bbox format: [x, y, width, height]
        bbox = [
            int(x_min),
            int(y_min),
            int(x_max - x_min + 1),
            int(y_max - y_min + 1)
        ]

        return bbox
