"""Checkpoint manager for saving and loading model checkpoints."""

from pathlib import Path
from typing import Dict, List, Optional
import json
import shutil
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage model checkpoints with automatic pruning.

    Keeps track of checkpoints and automatically removes old ones
    based on configuration.
    """

    def __init__(
        self,
        output_dir: Path,
        save_total_limit: int = 3,
        keep_best: bool = True
    ):
        """Initialize checkpoint manager.

        Args:
            output_dir: Directory to save checkpoints
            save_total_limit: Maximum number of checkpoints to keep
            keep_best: Whether to always keep the best checkpoint
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_total_limit = save_total_limit
        self.keep_best = keep_best

        self.checkpoints: List[Dict] = []
        self.best_checkpoint: Optional[Dict] = None

        # Load existing checkpoint metadata
        self._load_metadata()

    def save_checkpoint(
        self,
        step: int,
        epoch: int,
        metrics: Dict[str, float],
        checkpoint_dir: Path,
        is_best: bool = False
    ) -> Path:
        """Save checkpoint and manage pruning.

        Args:
            step: Training step
            epoch: Training epoch
            metrics: Metrics at this checkpoint
            checkpoint_dir: Directory containing checkpoint files
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        checkpoint_info = {
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "path": str(checkpoint_dir),
            "timestamp": datetime.now().isoformat(),
            "is_best": is_best
        }

        self.checkpoints.append(checkpoint_info)

        if is_best:
            self.best_checkpoint = checkpoint_info
            logger.info(f"New best checkpoint at step {step} with metrics: {metrics}")

        # Save metadata
        self._save_metadata()

        # Prune old checkpoints
        self._prune_checkpoints()

        logger.info(f"Checkpoint saved: step={step}, epoch={epoch}")

        return checkpoint_dir

    def load_checkpoint(self, checkpoint_path: Optional[Path] = None) -> Optional[Dict]:
        """Load checkpoint information.

        Args:
            checkpoint_path: Specific checkpoint to load, or None for latest

        Returns:
            Checkpoint info dict, or None if not found
        """
        if checkpoint_path is None:
            # Load latest checkpoint
            if not self.checkpoints:
                logger.warning("No checkpoints available")
                return None
            return self.checkpoints[-1]

        # Find specific checkpoint
        checkpoint_path = str(checkpoint_path)
        for ckpt in self.checkpoints:
            if ckpt["path"] == checkpoint_path:
                return ckpt

        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None

    def get_best_checkpoint(self) -> Optional[Dict]:
        """Get best checkpoint info.

        Returns:
            Best checkpoint dict, or None
        """
        return self.best_checkpoint

    def get_latest_checkpoint(self) -> Optional[Dict]:
        """Get latest checkpoint info.

        Returns:
            Latest checkpoint dict, or None
        """
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]

    def list_checkpoints(self) -> List[Dict]:
        """List all checkpoints.

        Returns:
            List of checkpoint info dicts
        """
        return self.checkpoints.copy()

    def _prune_checkpoints(self) -> None:
        """Remove old checkpoints based on save_total_limit."""
        if len(self.checkpoints) <= self.save_total_limit:
            return

        # Sort by step
        sorted_ckpts = sorted(self.checkpoints, key=lambda x: x["step"])

        # Determine which to keep
        to_keep = []
        to_remove = []

        # Always keep the best checkpoint
        if self.keep_best and self.best_checkpoint:
            to_keep.append(self.best_checkpoint)

        # Keep the most recent checkpoints
        recent = sorted_ckpts[-(self.save_total_limit - len(to_keep)):]
        to_keep.extend([ckpt for ckpt in recent if ckpt not in to_keep])

        # Mark others for removal
        to_remove = [ckpt for ckpt in sorted_ckpts if ckpt not in to_keep]

        # Remove old checkpoints
        for ckpt in to_remove:
            ckpt_path = Path(ckpt["path"])
            if ckpt_path.exists():
                shutil.rmtree(ckpt_path)
                logger.info(f"Removed old checkpoint: {ckpt_path}")

        # Update checkpoint list
        self.checkpoints = to_keep

    def _save_metadata(self) -> None:
        """Save checkpoint metadata to disk."""
        metadata = {
            "checkpoints": self.checkpoints,
            "best_checkpoint": self.best_checkpoint,
            "save_total_limit": self.save_total_limit
        }

        metadata_path = self.output_dir / "checkpoints_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self) -> None:
        """Load checkpoint metadata from disk."""
        metadata_path = self.output_dir / "checkpoints_metadata.json"

        if not metadata_path.exists():
            return

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self.checkpoints = metadata.get("checkpoints", [])
            self.best_checkpoint = metadata.get("best_checkpoint")

            logger.info(f"Loaded {len(self.checkpoints)} checkpoint(s) from metadata")
        except Exception as e:
            logger.error(f"Failed to load checkpoint metadata: {e}")
