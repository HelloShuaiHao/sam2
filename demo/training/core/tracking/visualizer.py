"""Training visualization tools.

Provides utilities for visualizing training progress, predictions,
and metrics.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """Visualizer for training metrics and predictions.

    Creates plots and visualizations during training.

    Args:
        output_dir: Directory to save visualizations
        use_matplotlib: Whether to use matplotlib for plotting
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        use_matplotlib: bool = True
    ):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_matplotlib = use_matplotlib
        if use_matplotlib:
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                import matplotlib.pyplot as plt
                self.plt = plt
                logger.info("Matplotlib visualizer initialized")
            except ImportError:
                logger.warning("Matplotlib not available. Visualizations disabled.")
                self.use_matplotlib = False
                self.plt = None
        else:
            self.plt = None

    def plot_loss_curve(
        self,
        steps: List[int],
        losses: List[float],
        title: str = "Training Loss",
        save_name: str = "loss_curve.png"
    ) -> Optional[Path]:
        """Plot training loss curve.

        Args:
            steps: List of step numbers
            losses: List of loss values
            title: Plot title
            save_name: Filename to save plot

        Returns:
            Path to saved plot, or None if plotting failed
        """
        if not self.use_matplotlib or self.plt is None:
            return None

        try:
            fig, ax = self.plt.subplots(figsize=(10, 6))
            ax.plot(steps, losses, linewidth=2)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            # Save plot
            save_path = self.output_dir / save_name
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.plt.close(fig)

            logger.info(f"Loss curve saved to {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to plot loss curve: {e}")
            return None

    def plot_multi_metric(
        self,
        steps: List[int],
        metrics: Dict[str, List[float]],
        title: str = "Training Metrics",
        save_name: str = "metrics.png"
    ) -> Optional[Path]:
        """Plot multiple metrics on the same plot.

        Args:
            steps: List of step numbers
            metrics: Dictionary of metric name to values
            title: Plot title
            save_name: Filename to save plot

        Returns:
            Path to saved plot, or None
        """
        if not self.use_matplotlib or self.plt is None:
            return None

        try:
            fig, ax = self.plt.subplots(figsize=(10, 6))

            for name, values in metrics.items():
                ax.plot(steps, values, label=name, linewidth=2)

            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Save plot
            save_path = self.output_dir / save_name
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.plt.close(fig)

            logger.info(f"Metrics plot saved to {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to plot metrics: {e}")
            return None

    def visualize_prediction(
        self,
        image: np.ndarray,
        pred_mask: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
        title: str = "Prediction",
        save_name: str = "prediction.png"
    ) -> Optional[Path]:
        """Visualize a single prediction.

        Args:
            image: Input image (H, W, 3) or (3, H, W)
            pred_mask: Predicted mask (H, W)
            gt_mask: Optional ground truth mask (H, W)
            title: Plot title
            save_name: Filename to save

        Returns:
            Path to saved visualization, or None
        """
        if not self.use_matplotlib or self.plt is None:
            return None

        try:
            # Normalize image if needed
            if image.ndim == 3:
                if image.shape[0] == 3:  # CHW -> HWC
                    image = np.transpose(image, (1, 2, 0))

                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)

            # Create subplots
            if gt_mask is not None:
                fig, axes = self.plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(image)
                axes[0].set_title("Input Image")
                axes[0].axis('off')

                axes[1].imshow(pred_mask, cmap='gray')
                axes[1].set_title("Predicted Mask")
                axes[1].axis('off')

                axes[2].imshow(gt_mask, cmap='gray')
                axes[2].set_title("Ground Truth")
                axes[2].axis('off')
            else:
                fig, axes = self.plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(image)
                axes[0].set_title("Input Image")
                axes[0].axis('off')

                axes[1].imshow(pred_mask, cmap='gray')
                axes[1].set_title("Predicted Mask")
                axes[1].axis('off')

            fig.suptitle(title)

            # Save
            save_path = self.output_dir / save_name
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.plt.close(fig)

            logger.info(f"Prediction visualization saved to {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to visualize prediction: {e}")
            return None

    def create_comparison_grid(
        self,
        images: List[np.ndarray],
        pred_masks: List[np.ndarray],
        gt_masks: Optional[List[np.ndarray]] = None,
        max_images: int = 8,
        save_name: str = "comparison_grid.png"
    ) -> Optional[Path]:
        """Create a grid comparing multiple predictions.

        Args:
            images: List of input images
            pred_masks: List of predicted masks
            gt_masks: Optional list of ground truth masks
            max_images: Maximum number of images to show
            save_name: Filename to save

        Returns:
            Path to saved grid, or None
        """
        if not self.use_matplotlib or self.plt is None:
            return None

        try:
            n = min(len(images), max_images)
            cols = 3 if gt_masks is not None else 2
            rows = n

            fig, axes = self.plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))

            if rows == 1:
                axes = axes.reshape(1, -1)

            for i in range(n):
                # Image
                img = images[i]
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)

                axes[i, 0].imshow(img)
                axes[i, 0].set_title(f"Image {i+1}")
                axes[i, 0].axis('off')

                # Prediction
                axes[i, 1].imshow(pred_masks[i], cmap='gray')
                axes[i, 1].set_title(f"Prediction {i+1}")
                axes[i, 1].axis('off')

                # Ground truth
                if gt_masks is not None:
                    axes[i, 2].imshow(gt_masks[i], cmap='gray')
                    axes[i, 2].set_title(f"GT {i+1}")
                    axes[i, 2].axis('off')

            fig.suptitle("Prediction Comparison", fontsize=16)
            self.plt.tight_layout()

            # Save
            save_path = self.output_dir / save_name
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.plt.close(fig)

            logger.info(f"Comparison grid saved to {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to create comparison grid: {e}")
            return None

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        save_name: str = "confusion_matrix.png"
    ) -> Optional[Path]:
        """Plot confusion matrix.

        Args:
            confusion_matrix: Confusion matrix array (N, N)
            class_names: Optional list of class names
            title: Plot title
            save_name: Filename to save

        Returns:
            Path to saved plot, or None
        """
        if not self.use_matplotlib or self.plt is None:
            return None

        try:
            fig, ax = self.plt.subplots(figsize=(10, 8))

            # Plot heatmap
            im = ax.imshow(confusion_matrix, cmap='Blues')

            # Set ticks
            n_classes = confusion_matrix.shape[0]
            if class_names is None:
                class_names = [f"Class {i}" for i in range(n_classes)]

            ax.set_xticks(np.arange(n_classes))
            ax.set_yticks(np.arange(n_classes))
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_yticklabels(class_names)

            # Add text annotations
            for i in range(n_classes):
                for j in range(n_classes):
                    text = ax.text(j, i, int(confusion_matrix[i, j]),
                                 ha="center", va="center", color="black")

            ax.set_title(title)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            fig.colorbar(im, ax=ax)
            self.plt.tight_layout()

            # Save
            save_path = self.output_dir / save_name
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.plt.close(fig)

            logger.info(f"Confusion matrix saved to {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {e}")
            return None


# Convenience function
def create_visualizer(output_dir: Union[str, Path]) -> TrainingVisualizer:
    """Create a training visualizer.

    Args:
        output_dir: Directory for saving visualizations

    Returns:
        TrainingVisualizer instance

    Example:
        >>> viz = create_visualizer("./output/visualizations")
        >>> viz.plot_loss_curve(steps, losses)
    """
    return TrainingVisualizer(output_dir)
