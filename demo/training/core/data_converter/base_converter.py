"""Base converter interface for data format conversion."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


class BaseConverter(ABC):
    """Abstract base class for data format converters.

    All converters must implement the convert() and validate() methods
    to transform SAM2 exports into training-compatible formats.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize converter with optional configuration.

        Args:
            config: Optional configuration dictionary for converter behavior
        """
        self.config = config or {}

    @abstractmethod
    def convert(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """Convert data from input format to target format.

        Args:
            input_path: Path to input data (SAM2 export ZIP or directory)
            output_path: Path where converted data should be saved

        Returns:
            Dictionary containing conversion statistics and metadata:
                - total_samples: Number of samples converted
                - total_classes: Number of unique classes
                - class_distribution: Dict mapping class names to counts
                - output_format: Name of the output format
                - success: Boolean indicating conversion success
        """
        pass

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate that data conforms to expected format.

        Args:
            data: Data to validate

        Returns:
            True if data is valid, False otherwise
        """
        pass

    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats for this converter.

        Returns:
            List of format names (e.g., ["huggingface", "llava"])
        """
        return []
