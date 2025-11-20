"""Training callbacks for HuggingFace Trainer.

Provides custom callbacks for checkpoint management, early stopping,
and progress reporting during training.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import time

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from ..checkpoints.checkpoint_manager import CheckpointManager
from ..checkpoints.best_model_tracker import BestModelTracker

logger = logging.getLogger(__name__)


class EnhancedCheckpointCallback(TrainerCallback):
    """Enhanced checkpoint saving with best model tracking and pruning.

    Features:
    - Automatic checkpoint pruning
    - Best model tracking
    - Metadata storage
    - Custom save conditions

    Args:
        output_dir: Directory to save checkpoints
        save_total_limit: Maximum number of checkpoints to keep
        metric_for_best: Metric name to determine best model
        mode: "min" for loss-like, "max" for accuracy-like metrics
        keep_best: Always keep the best checkpoint
    """

    def __init__(
        self,
        output_dir: str,
        save_total_limit: int = 3,
        metric_for_best: str = "eval_loss",
        mode: str = "min",
        keep_best: bool = True
    ):
        """Initialize checkpoint callback."""
        self.checkpoint_manager = CheckpointManager(
            output_dir=Path(output_dir),
            save_total_limit=save_total_limit,
            keep_best=keep_best
        )
        self.best_tracker = BestModelTracker(
            metric_name=metric_for_best,
            mode=mode
        )
        self.metric_for_best = metric_for_best

        logger.info(f"EnhancedCheckpointCallback initialized")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"  Save limit: {save_total_limit}")
        logger.info(f"  Best metric: {metric_for_best} ({mode})")

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called when a checkpoint is saved.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
        """
        # Get checkpoint directory (just saved by Trainer)
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

        # Get current metrics
        metrics = {}
        if state.log_history:
            # Get latest logged metrics
            latest_log = state.log_history[-1]
            metrics = {k: v for k, v in latest_log.items() if isinstance(v, (int, float))}

        # Check if this is the best checkpoint
        is_best = False
        if self.metric_for_best in metrics:
            is_best = self.best_tracker.update(
                step=state.global_step,
                epoch=int(state.epoch) if state.epoch else 0,
                metrics=metrics
            )

        # Register checkpoint with manager
        self.checkpoint_manager.save_checkpoint(
            step=state.global_step,
            epoch=int(state.epoch) if state.epoch else 0,
            metrics=metrics,
            checkpoint_dir=checkpoint_dir,
            is_best=is_best
        )

        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called at the end of training.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
        """
        best_info = self.best_tracker.get_best()
        if best_info:
            logger.info("=" * 80)
            logger.info("Training Complete - Best Model Summary")
            logger.info("=" * 80)
            logger.info(f"Best {best_info['metric_name']}: {best_info['value']:.4f}")
            logger.info(f"  Step: {best_info['step']}")
            logger.info(f"  Epoch: {best_info['epoch']}")
            logger.info("=" * 80)

        return control


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping based on metric plateau.

    Stops training if the monitored metric doesn't improve for
    a specified number of evaluations.

    Args:
        early_stopping_patience: Number of evaluations without improvement
        early_stopping_threshold: Minimum change to count as improvement
        metric_for_best: Metric to monitor
        mode: "min" for loss-like, "max" for accuracy-like
    """

    def __init__(
        self,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.0,
        metric_for_best: str = "eval_loss",
        mode: str = "min"
    ):
        """Initialize early stopping callback."""
        self.patience = early_stopping_patience
        self.threshold = early_stopping_threshold
        self.metric_name = metric_for_best
        self.mode = mode

        self.best_value: Optional[float] = None
        self.patience_counter = 0
        self.should_stop = False

        logger.info(f"EarlyStoppingCallback initialized")
        logger.info(f"  Metric: {metric_for_best} ({mode})")
        logger.info(f"  Patience: {early_stopping_patience}")
        logger.info(f"  Threshold: {early_stopping_threshold}")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs
    ):
        """Called after evaluation.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            metrics: Evaluation metrics
        """
        if self.metric_name not in metrics:
            logger.warning(f"Metric '{self.metric_name}' not found, skipping early stopping check")
            return control

        current_value = metrics[self.metric_name]

        # Check if improved
        improved = self._check_improvement(current_value)

        if improved:
            self.best_value = current_value
            self.patience_counter = 0
            logger.info(f"Metric improved: {self.metric_name}={current_value:.4f}")
        else:
            self.patience_counter += 1
            logger.info(
                f"No improvement in {self.metric_name} "
                f"({self.patience_counter}/{self.patience})"
            )

            if self.patience_counter >= self.patience:
                logger.info("=" * 80)
                logger.info("EARLY STOPPING TRIGGERED")
                logger.info("=" * 80)
                logger.info(f"No improvement for {self.patience} evaluations")
                logger.info(f"Best {self.metric_name}: {self.best_value:.4f}")
                logger.info("Stopping training...")
                logger.info("=" * 80)

                control.should_training_stop = True
                self.should_stop = True

        return control

    def _check_improvement(self, value: float) -> bool:
        """Check if value represents an improvement.

        Args:
            value: Current metric value

        Returns:
            True if improved, False otherwise
        """
        if self.best_value is None:
            return True

        if self.mode == "min":
            improvement = self.best_value - value
        else:  # mode == "max"
            improvement = value - self.best_value

        return improvement > self.threshold


class ProgressReportCallback(TrainerCallback):
    """Progress reporting with ETA and throughput metrics.

    Provides detailed progress updates during training including:
    - Steps per second
    - Estimated time to completion
    - Current loss
    - Learning rate

    Args:
        report_frequency: Report every N steps (default: 10)
        log_to_file: Optional file path to log progress
    """

    def __init__(
        self,
        report_frequency: int = 10,
        log_to_file: Optional[str] = None
    ):
        """Initialize progress callback."""
        self.report_frequency = report_frequency
        self.log_file = Path(log_to_file) if log_to_file else None

        self.start_time: Optional[float] = None
        self.last_report_time: Optional[float] = None
        self.last_report_step: int = 0

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Progress will be logged to: {self.log_file}")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called at the beginning of training.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
        """
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.last_report_step = 0

        logger.info("=" * 80)
        logger.info("Training Started")
        logger.info("=" * 80)
        logger.info(f"Total steps: {state.max_steps}")
        logger.info(f"Epochs: {args.num_train_epochs}")
        logger.info(f"Batch size: {args.per_device_train_batch_size}")
        logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        logger.info("=" * 80)

        self._log_to_file(f"Training started at {datetime.now().isoformat()}")

        return control

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, Any],
        **kwargs
    ):
        """Called when logging.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            logs: Log dictionary
        """
        # Only report at specified frequency
        if state.global_step % self.report_frequency != 0:
            return control

        current_time = time.time()

        # Calculate progress
        progress = state.global_step / state.max_steps * 100
        elapsed = current_time - self.start_time

        # Calculate throughput
        steps_since_last = state.global_step - self.last_report_step
        time_since_last = current_time - self.last_report_time
        steps_per_sec = steps_since_last / time_since_last if time_since_last > 0 else 0

        # Estimate time remaining
        remaining_steps = state.max_steps - state.global_step
        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

        # Format report
        report = self._format_report(
            step=state.global_step,
            max_steps=state.max_steps,
            progress=progress,
            elapsed=elapsed,
            eta=eta_seconds,
            steps_per_sec=steps_per_sec,
            logs=logs
        )

        logger.info(report)
        self._log_to_file(report)

        # Update last report time
        self.last_report_time = current_time
        self.last_report_step = state.global_step

        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called at the end of training.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
        """
        total_time = time.time() - self.start_time
        avg_steps_per_sec = state.global_step / total_time

        logger.info("=" * 80)
        logger.info("Training Completed")
        logger.info("=" * 80)
        logger.info(f"Total steps: {state.global_step}")
        logger.info(f"Total time: {self._format_time(total_time)}")
        logger.info(f"Average speed: {avg_steps_per_sec:.2f} steps/sec")
        logger.info("=" * 80)

        self._log_to_file(f"Training completed at {datetime.now().isoformat()}")
        self._log_to_file(f"Total time: {self._format_time(total_time)}")

        return control

    def _format_report(
        self,
        step: int,
        max_steps: int,
        progress: float,
        elapsed: float,
        eta: float,
        steps_per_sec: float,
        logs: Dict[str, Any]
    ) -> str:
        """Format progress report.

        Args:
            step: Current step
            max_steps: Total steps
            progress: Progress percentage
            elapsed: Elapsed time
            eta: Estimated time remaining
            steps_per_sec: Steps per second
            logs: Log dictionary

        Returns:
            Formatted report string
        """
        # Extract key metrics
        loss = logs.get("loss", 0.0)
        lr = logs.get("learning_rate", 0.0)

        report = (
            f"Step {step}/{max_steps} ({progress:.1f}%) | "
            f"Loss: {loss:.4f} | "
            f"LR: {lr:.2e} | "
            f"Speed: {steps_per_sec:.2f} steps/s | "
            f"Elapsed: {self._format_time(elapsed)} | "
            f"ETA: {self._format_time(eta)}"
        )

        return report

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string (e.g., "1h 23m 45s")
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            secs = int(seconds % 60)
            return f"{hours}h {minutes}m {secs}s"

    def _log_to_file(self, message: str) -> None:
        """Log message to file.

        Args:
            message: Message to log
        """
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] {message}\n")


# Convenience function
def create_default_callbacks(
    output_dir: str,
    enable_early_stopping: bool = False,
    early_stopping_patience: int = 3,
    save_total_limit: int = 3,
    metric_for_best: str = "eval_loss",
    mode: str = "min"
) -> list:
    """Create default set of callbacks.

    Args:
        output_dir: Checkpoint output directory
        enable_early_stopping: Whether to enable early stopping
        early_stopping_patience: Patience for early stopping
        save_total_limit: Max checkpoints to keep
        metric_for_best: Metric to track for best model
        mode: "min" or "max" for the metric

    Returns:
        List of callback instances

    Example:
        >>> callbacks = create_default_callbacks(
        ...     output_dir="./checkpoints",
        ...     enable_early_stopping=True,
        ...     early_stopping_patience=3
        ... )
        >>> trainer = Trainer(..., callbacks=callbacks)
    """
    callbacks = [
        EnhancedCheckpointCallback(
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            metric_for_best=metric_for_best,
            mode=mode
        ),
        ProgressReportCallback(
            report_frequency=10,
            log_to_file=f"{output_dir}/progress.log"
        )
    ]

    if enable_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                metric_for_best=metric_for_best,
                mode=mode
            )
        )

    return callbacks
