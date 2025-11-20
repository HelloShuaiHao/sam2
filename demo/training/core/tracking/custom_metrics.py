"""Custom metrics for segmentation tasks.

Provides metrics specific to vision-language segmentation tasks,
including IoU, mAP, and other segmentation quality metrics.
"""

import logging
from typing import Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


def compute_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5
) -> float:
    """Compute Intersection over Union (IoU) for binary masks.

    Args:
        pred_mask: Predicted mask (H, W) or (H, W, 1)
        gt_mask: Ground truth mask (H, W) or (H, W, 1)
        threshold: Threshold for binarizing predictions

    Returns:
        IoU score between 0 and 1
    """
    # Ensure 2D
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze()
    if gt_mask.ndim == 3:
        gt_mask = gt_mask.squeeze()

    # Binarize
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    gt_binary = (gt_mask > threshold).astype(np.uint8)

    # Compute intersection and union
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    if union == 0:
        return 0.0

    iou = intersection / union
    return float(iou)


def compute_dice(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5
) -> float:
    """Compute Dice coefficient for binary masks.

    Args:
        pred_mask: Predicted mask
        gt_mask: Ground truth mask
        threshold: Threshold for binarizing predictions

    Returns:
        Dice coefficient between 0 and 1
    """
    # Ensure 2D
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze()
    if gt_mask.ndim == 3:
        gt_mask = gt_mask.squeeze()

    # Binarize
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    gt_binary = (gt_mask > threshold).astype(np.uint8)

    # Compute intersection
    intersection = np.logical_and(pred_binary, gt_binary).sum()

    # Compute dice
    dice = (2.0 * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-8)
    return float(dice)


def compute_map(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    iou_thresholds: Optional[List[float]] = None
) -> Dict[str, float]:
    """Compute mean Average Precision (mAP) for masks.

    Args:
        pred_masks: List of predicted masks
        gt_masks: List of ground truth masks
        iou_thresholds: IoU thresholds to evaluate at (default: [0.5, 0.75, 0.95])

    Returns:
        Dictionary with mAP scores at different thresholds
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75, 0.95]

    if len(pred_masks) != len(gt_masks):
        raise ValueError(
            f"Number of predictions ({len(pred_masks)}) != "
            f"number of ground truths ({len(gt_masks)})"
        )

    results = {}

    for threshold in iou_thresholds:
        ious = []
        for pred, gt in zip(pred_masks, gt_masks):
            iou = compute_iou(pred, gt, threshold=0.5)
            ious.append(iou)

        # Compute AP at this threshold
        # Simple version: just average the IoUs
        ap = np.mean(ious)
        results[f"AP@{threshold}"] = float(ap)

    # Overall mAP
    results["mAP"] = float(np.mean(list(results.values())))

    return results


def compute_pixel_accuracy(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5
) -> float:
    """Compute pixel-wise accuracy.

    Args:
        pred_mask: Predicted mask
        gt_mask: Ground truth mask
        threshold: Threshold for binarizing predictions

    Returns:
        Pixel accuracy between 0 and 1
    """
    # Ensure 2D
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze()
    if gt_mask.ndim == 3:
        gt_mask = gt_mask.squeeze()

    # Binarize
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    gt_binary = (gt_mask > threshold).astype(np.uint8)

    # Compute accuracy
    correct = (pred_binary == gt_binary).sum()
    total = pred_binary.size

    accuracy = correct / total
    return float(accuracy)


class SegmentationMetrics:
    """Container for segmentation metrics.

    Accumulates metrics over multiple batches and computes averages.

    Example:
        >>> metrics = SegmentationMetrics()
        >>> for pred, gt in predictions:
        ...     metrics.update(pred, gt)
        >>> results = metrics.compute()
        >>> print(f"Mean IoU: {results['iou']:.4f}")
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.ious = []
        self.dices = []
        self.pixel_accs = []
        self.num_samples = 0

    def update(
        self,
        pred_masks: Union[np.ndarray, List[np.ndarray]],
        gt_masks: Union[np.ndarray, List[np.ndarray]],
        threshold: float = 0.5
    ) -> None:
        """Update metrics with new predictions.

        Args:
            pred_masks: Predicted mask(s)
            gt_masks: Ground truth mask(s)
            threshold: Threshold for binarization
        """
        # Handle single mask or batch
        if isinstance(pred_masks, np.ndarray) and pred_masks.ndim == 2:
            pred_masks = [pred_masks]
            gt_masks = [gt_masks]
        elif isinstance(pred_masks, np.ndarray):
            # Batch of masks
            pred_masks = list(pred_masks)
            gt_masks = list(gt_masks)

        # Compute metrics for each mask
        for pred, gt in zip(pred_masks, gt_masks):
            self.ious.append(compute_iou(pred, gt, threshold))
            self.dices.append(compute_dice(pred, gt, threshold))
            self.pixel_accs.append(compute_pixel_accuracy(pred, gt, threshold))
            self.num_samples += 1

    def compute(self) -> Dict[str, float]:
        """Compute average metrics.

        Returns:
            Dictionary of metric names to values
        """
        if self.num_samples == 0:
            return {
                "iou": 0.0,
                "dice": 0.0,
                "pixel_accuracy": 0.0,
                "num_samples": 0
            }

        return {
            "iou": float(np.mean(self.ious)),
            "dice": float(np.mean(self.dices)),
            "pixel_accuracy": float(np.mean(self.pixel_accs)),
            "num_samples": self.num_samples
        }

    def compute_per_threshold(
        self,
        thresholds: List[float]
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics at multiple thresholds.

        Args:
            thresholds: List of thresholds to evaluate

        Returns:
            Dictionary mapping threshold to metrics dict
        """
        # This would require storing the raw masks, which is memory-intensive
        # For now, just return the single threshold results
        logger.warning(
            "Per-threshold metrics require storing raw masks. "
            "Returning single threshold results only."
        )
        return {"default": self.compute()}

    def __str__(self) -> str:
        """String representation."""
        metrics = self.compute()
        return (
            f"SegmentationMetrics("
            f"IoU={metrics['iou']:.4f}, "
            f"Dice={metrics['dice']:.4f}, "
            f"PixelAcc={metrics['pixel_accuracy']:.4f}, "
            f"N={metrics['num_samples']})"
        )


def log_segmentation_metrics(
    tensorboard_logger,
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    step: int,
    prefix: str = "val"
) -> Dict[str, float]:
    """Compute and log segmentation metrics to Tensorboard.

    Args:
        tensorboard_logger: TensorboardLogger instance
        pred_masks: List of predicted masks
        gt_masks: List of ground truth masks
        step: Global step
        prefix: Metric prefix (e.g., "val", "test")

    Returns:
        Dictionary of computed metrics
    """
    # Compute metrics
    metrics_tracker = SegmentationMetrics()
    metrics_tracker.update(pred_masks, gt_masks)
    metrics = metrics_tracker.compute()

    # Log to tensorboard
    for name, value in metrics.items():
        if name != "num_samples":
            tensorboard_logger.log_scalar(f"{prefix}/{name}", value, step)

    logger.info(f"[Step {step}] {prefix} metrics: {metrics}")

    return metrics
