"""Utility functions for model inference and deployment.

Provides helper functions for image processing, result formatting,
and common inference operations.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import json

import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Image Processing Utilities
# =============================================================================

def load_and_preprocess_image(
    image_path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Load and preprocess an image for inference.

    Args:
        image_path: Path to image file
        target_size: Optional target size (width, height)
        normalize: Whether to normalize pixel values

    Returns:
        Tuple of (processed_image, metadata)

    Example:
        >>> img, meta = load_and_preprocess_image("scan.jpg", target_size=(336, 336))
        >>> print(meta)
        {'original_size': (512, 512), 'preprocessed_size': (336, 336)}
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    metadata = {
        "original_size": original_size,
        "preprocessed_size": original_size
    }

    # Resize if needed
    if target_size:
        image = image.resize(target_size, Image.LANCZOS)
        metadata["preprocessed_size"] = target_size

    return image, metadata


def validate_image(image: Image.Image) -> bool:
    """Validate that an image is suitable for inference.

    Args:
        image: PIL Image

    Returns:
        True if valid, raises ValueError otherwise

    Raises:
        ValueError: If image is invalid
    """
    # Check format
    if image.mode not in ["RGB", "RGBA", "L"]:
        raise ValueError(f"Unsupported image mode: {image.mode}. Expected RGB, RGBA, or L")

    # Check size
    width, height = image.size
    if width < 16 or height < 16:
        raise ValueError(f"Image too small: {width}x{height}. Minimum size is 16x16")

    if width > 4096 or height > 4096:
        raise ValueError(f"Image too large: {width}x{height}. Maximum size is 4096x4096")

    return True


def batch_images(
    image_paths: List[Union[str, Path]],
    target_size: Optional[Tuple[int, int]] = None
) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
    """Load and preprocess a batch of images.

    Args:
        image_paths: List of image paths
        target_size: Optional target size for all images

    Returns:
        Tuple of (images, metadata_list)
    """
    images = []
    metadata_list = []

    for path in image_paths:
        img, meta = load_and_preprocess_image(path, target_size)
        images.append(img)
        metadata_list.append(meta)

    return images, metadata_list


# =============================================================================
# Result Processing Utilities
# =============================================================================

def format_prediction_result(
    text: str,
    image_path: Union[str, Path],
    prompt: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Format a prediction result into a standardized dictionary.

    Args:
        text: Generated text from model
        image_path: Path to input image
        prompt: Input prompt
        metadata: Optional additional metadata

    Returns:
        Formatted result dictionary
    """
    result = {
        "text": text,
        "prompt": prompt,
        "image_path": str(image_path),
    }

    if metadata:
        result["metadata"] = metadata

    return result


def extract_segmentation_from_text(
    text: str,
    format: str = "json"
) -> Optional[Dict[str, Any]]:
    """Extract segmentation information from model output text.

    Args:
        text: Model output text
        format: Expected format ("json", "markdown", "plain")

    Returns:
        Extracted segmentation data or None

    Example:
        >>> text = "The image shows a tumor at coordinates [100, 200] with size 50x30."
        >>> seg = extract_segmentation_from_text(text)
    """
    # This is a placeholder for actual segmentation extraction logic
    # The implementation depends on your model's output format

    if format == "json":
        # Try to parse JSON from text
        import re
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text)

        for match in matches:
            try:
                data = json.loads(match)
                return data
            except json.JSONDecodeError:
                continue

    return None


def parse_medical_entities(text: str) -> Dict[str, List[str]]:
    """Parse medical entities from model output.

    Args:
        text: Model output text

    Returns:
        Dictionary with entity types and values

    Example:
        >>> text = "Identified: heart, lungs. Abnormalities: tumor in left lung."
        >>> entities = parse_medical_entities(text)
        >>> print(entities)
        {'organs': ['heart', 'lungs'], 'abnormalities': ['tumor']}
    """
    entities = {
        "organs": [],
        "abnormalities": [],
        "regions": [],
        "findings": []
    }

    # Simple keyword-based extraction (can be enhanced with NER)
    text_lower = text.lower()

    # Common medical terms
    organ_keywords = ["heart", "lung", "liver", "kidney", "brain", "spine"]
    abnormality_keywords = ["tumor", "lesion", "mass", "nodule", "cyst"]

    for keyword in organ_keywords:
        if keyword in text_lower:
            entities["organs"].append(keyword)

    for keyword in abnormality_keywords:
        if keyword in text_lower:
            entities["abnormalities"].append(keyword)

    return entities


# =============================================================================
# Model Utilities
# =============================================================================

def estimate_inference_memory(
    model_size_gb: float,
    batch_size: int = 1,
    use_4bit: bool = True
) -> float:
    """Estimate GPU memory required for inference.

    Args:
        model_size_gb: Model size in GB
        batch_size: Batch size
        use_4bit: Whether using 4-bit quantization

    Returns:
        Estimated memory in GB

    Example:
        >>> memory_gb = estimate_inference_memory(13.5, batch_size=1, use_4bit=True)
        >>> print(f"Estimated: {memory_gb:.1f} GB")
    """
    # Base model memory
    if use_4bit:
        model_memory = model_size_gb * 0.3  # 4-bit reduces to ~30%
    else:
        model_memory = model_size_gb

    # Activation memory (rough estimate)
    activation_memory = 0.5 * batch_size

    # Add overhead
    overhead = 0.5

    total = model_memory + activation_memory + overhead

    return total


def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices.

    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": []
    }

    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()

        for i in range(torch.cuda.device_count()):
            device_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_gb": torch.cuda.get_device_properties(i).total_memory / 1024**3,
                "memory_allocated_gb": torch.cuda.memory_allocated(i) / 1024**3,
                "memory_reserved_gb": torch.cuda.memory_reserved(i) / 1024**3
            }
            info["devices"].append(device_info)

    return info


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared")


# =============================================================================
# Configuration Utilities
# =============================================================================

def load_deployment_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load deployment configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def validate_deployment_config(config: Dict[str, Any]) -> bool:
    """Validate deployment configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = ["model_path", "device", "max_batch_size"]

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    return True


# =============================================================================
# Logging Utilities
# =============================================================================

def setup_inference_logging(
    log_file: Optional[Union[str, Path]] = None,
    level: str = "INFO"
) -> logging.Logger:
    """Set up logging for inference.

    Args:
        log_file: Optional log file path
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger("inference")
    logger.setLevel(getattr(logging, level))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_inference_stats(
    start_time: float,
    end_time: float,
    image_path: Union[str, Path],
    result: Dict[str, Any]
) -> None:
    """Log inference statistics.

    Args:
        start_time: Start timestamp
        end_time: End timestamp
        image_path: Input image path
        result: Inference result
    """
    duration = end_time - start_time

    logger.info("=" * 80)
    logger.info("Inference Statistics")
    logger.info("=" * 80)
    logger.info(f"Image: {image_path}")
    logger.info(f"Duration: {duration:.2f}s")
    logger.info(f"Output length: {len(result.get('text', ''))} characters")

    if torch.cuda.is_available():
        memory_gb = torch.cuda.memory_allocated(0) / 1024**3
        logger.info(f"GPU Memory: {memory_gb:.2f} GB")

    logger.info("=" * 80)


# =============================================================================
# Error Handling
# =============================================================================

class InferenceError(Exception):
    """Base exception for inference errors."""
    pass


class ModelLoadError(InferenceError):
    """Exception raised when model fails to load."""
    pass


class ImageProcessingError(InferenceError):
    """Exception raised during image processing."""
    pass


class PredictionError(InferenceError):
    """Exception raised during prediction."""
    pass
