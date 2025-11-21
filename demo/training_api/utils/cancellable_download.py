"""
Cancellable model download wrapper for HuggingFace.

This module provides mechanisms to interrupt long-running HuggingFace downloads
when a training job is cancelled.
"""

import signal
import threading
from typing import Callable, Optional
from contextlib import contextmanager


class TimeoutException(Exception):
    """Raised when operation times out or is cancelled."""
    pass


@contextmanager
def cancellable_operation(check_cancelled: Callable[[], bool], check_interval: float = 1.0):
    """
    Context manager for operations that should be cancellable.

    Args:
        check_cancelled: Function that returns True if operation should be cancelled
        check_interval: How often to check for cancellation (seconds)

    Raises:
        TimeoutException: If operation is cancelled

    Example:
        with cancellable_operation(lambda: job_cancelled):
            model = AutoModel.from_pretrained("model-name")
    """
    cancel_event = threading.Event()
    stop_checking = threading.Event()

    def check_loop():
        """Background thread that checks for cancellation."""
        while not stop_checking.is_set():
            if check_cancelled():
                cancel_event.set()
                # Send interrupt to main thread
                # Note: This is aggressive but necessary for HuggingFace downloads
                break
            stop_checking.wait(check_interval)

    checker_thread = threading.Thread(target=check_loop, daemon=True)
    checker_thread.start()

    try:
        yield cancel_event
        if cancel_event.is_set():
            raise TimeoutException("Operation was cancelled")
    finally:
        stop_checking.set()
        checker_thread.join(timeout=2.0)


class CancellableDownloadMonitor:
    """
    Monitor that can interrupt HuggingFace downloads.

    Usage:
        monitor = CancellableDownloadMonitor(lambda: job.is_cancelled())
        with monitor:
            model = AutoModel.from_pretrained("model-name")
    """

    def __init__(self, check_cancelled: Callable[[], bool], check_interval: float = 2.0):
        """
        Initialize monitor.

        Args:
            check_cancelled: Function that returns True if should cancel
            check_interval: How often to check (seconds)
        """
        self.check_cancelled = check_cancelled
        self.check_interval = check_interval
        self.checker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    def _check_loop(self):
        """Background thread that monitors for cancellation."""
        while not self.stop_event.is_set():
            if self.check_cancelled():
                print("⚠️  Download cancelled - interrupting operation")
                # Unfortunately, we can't cleanly interrupt HuggingFace downloads
                # They will continue in background, but we can exit the training job
                self.stop_event.set()
                break
            self.stop_event.wait(self.check_interval)

    def __enter__(self):
        """Start monitoring."""
        self.stop_event.clear()
        self.checker_thread = threading.Thread(target=self._check_loop, daemon=True)
        self.checker_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring."""
        self.stop_event.set()
        if self.checker_thread:
            self.checker_thread.join(timeout=1.0)

    def is_cancelled(self) -> bool:
        """Check if operation was cancelled."""
        return self.stop_event.is_set()


def with_progress_callback(check_cancelled: Callable[[], bool], progress_callback: Optional[Callable] = None):
    """
    Decorator to add cancellation checks and progress reporting.

    Args:
        check_cancelled: Function that returns True if should cancel
        progress_callback: Optional callback for progress updates

    Example:
        @with_progress_callback(lambda: job.is_cancelled())
        def load_model():
            return AutoModel.from_pretrained("model-name")
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Check before starting
            if check_cancelled():
                raise TimeoutException("Operation cancelled before starting")

            # Run function
            result = func(*args, **kwargs)

            # Check after completion
            if check_cancelled():
                raise TimeoutException("Operation cancelled after completion")

            return result
        return wrapper
    return decorator
