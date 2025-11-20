"""Data collator for vision-language models.

This collator handles batching of multimodal data (images + text) for
training vision-language models like LLaVA, Qwen-VL, and MiniCPM-V.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import torch
from PIL import Image
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VisionLanguageCollator:
    """Data collator for vision-language training.

    Handles:
    - Image loading and preprocessing
    - Text tokenization
    - Batching with padding
    - Mask and bbox handling (optional)

    Attributes:
        tokenizer: HuggingFace tokenizer
        processor: Vision-language processor (LLaVA/Qwen-VL/etc.)
        image_dir: Directory containing images
        max_length: Maximum sequence length
        padding: Padding strategy ('longest' or 'max_length')
        image_size: Target image size (width, height)
        return_masks: Whether to return segmentation masks
    """

    tokenizer: Any
    processor: Optional[Any] = None
    image_dir: Optional[Path] = None
    max_length: int = 512
    padding: str = "longest"
    image_size: tuple = (336, 336)
    return_masks: bool = False

    def __post_init__(self):
        """Initialize collator."""
        if self.image_dir:
            self.image_dir = Path(self.image_dir)

        # Check if processor exists, if not use tokenizer only
        self.use_processor = self.processor is not None

        if not self.use_processor:
            logger.warning(
                "No processor provided. Using tokenizer-only mode. "
                "Images will be loaded but not processed by a vision encoder."
            )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples.

        Args:
            features: List of dataset samples, each containing:
                - id: Sample identifier
                - image: Image path or PIL Image
                - conversations: List of conversation turns
                - masks: Optional RLE masks
                - bounding_boxes: Optional bboxes

        Returns:
            Batched tensors ready for model input
        """
        # Extract components
        images = []
        texts = []
        labels = []

        for feature in features:
            # Load image
            image = self._load_image(feature)
            if image is not None:
                images.append(image)

            # Process conversations to text
            text = self._format_conversation(feature.get("conversations", []))
            texts.append(text)

            # Create labels (same as input_ids for causal LM)
            labels.append(text)

        # Process with vision-language processor or tokenizer
        if self.use_processor:
            batch = self._process_with_processor(images, texts, labels)
        else:
            batch = self._process_with_tokenizer(texts, labels)
            # Add placeholder for images if needed
            if images:
                batch["pixel_values"] = self._stack_images(images)

        # Add masks if requested
        if self.return_masks:
            masks = [feature.get("masks", []) for feature in features]
            if any(masks):
                batch["masks"] = masks

        return batch

    def _load_image(self, feature: Dict[str, Any]) -> Optional[Image.Image]:
        """Load image from feature.

        Args:
            feature: Sample feature dict

        Returns:
            PIL Image or None
        """
        image_field = feature.get("image")

        if image_field is None:
            return None

        # If already PIL Image
        if isinstance(image_field, Image.Image):
            return image_field

        # If path string
        if isinstance(image_field, (str, Path)):
            image_path = Path(image_field)

            # Try absolute path first
            if image_path.exists():
                try:
                    return Image.open(image_path).convert("RGB")
                except Exception as e:
                    logger.error(f"Failed to load image {image_path}: {e}")
                    return None

            # Try relative to image_dir
            if self.image_dir:
                full_path = self.image_dir / image_path
                if full_path.exists():
                    try:
                        return Image.open(full_path).convert("RGB")
                    except Exception as e:
                        logger.error(f"Failed to load image {full_path}: {e}")
                        return None

            logger.warning(f"Image not found: {image_field}")
            return None

        logger.warning(f"Unknown image field type: {type(image_field)}")
        return None

    def _format_conversation(self, conversations: List[Dict[str, str]]) -> str:
        """Format conversation turns into a single text string.

        Args:
            conversations: List of conversation turns

        Returns:
            Formatted text string
        """
        if not conversations:
            return ""

        text_parts = []

        for turn in conversations:
            role = turn.get("from", "human")
            value = turn.get("value", "")

            # Remove <image> token for text processing
            # (it's handled separately by the processor)
            value = value.replace("<image>\n", "").replace("<image>", "")

            if role == "human":
                text_parts.append(f"USER: {value}")
            elif role == "gpt":
                text_parts.append(f"ASSISTANT: {value}")
            else:
                text_parts.append(value)

        return "\n".join(text_parts)

    def _process_with_processor(
        self,
        images: List[Image.Image],
        texts: List[str],
        labels: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Process batch using vision-language processor.

        Args:
            images: List of PIL Images
            texts: List of input texts
            labels: List of label texts

        Returns:
            Processed batch dict
        """
        try:
            # Process with multimodal processor
            batch = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=self.padding,
                truncation=True,
                max_length=self.max_length
            )

            # Create labels (same as input_ids for causal LM)
            batch["labels"] = batch["input_ids"].clone()

            return batch

        except Exception as e:
            logger.error(f"Processor failed: {e}. Falling back to tokenizer-only mode.")
            return self._process_with_tokenizer(texts, labels)

    def _process_with_tokenizer(
        self,
        texts: List[str],
        labels: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Process batch using tokenizer only (no image processing).

        Args:
            texts: List of input texts
            labels: List of label texts

        Returns:
            Processed batch dict
        """
        # Tokenize texts
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=self.padding,
            truncation=True,
            max_length=self.max_length
        )

        # Create labels
        label_batch = self.tokenizer(
            labels,
            return_tensors="pt",
            padding=self.padding,
            truncation=True,
            max_length=self.max_length
        )

        batch["labels"] = label_batch["input_ids"]

        # Replace padding token id in labels with -100 (ignore in loss)
        if self.tokenizer.pad_token_id is not None:
            batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100

        return batch

    def _stack_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Stack images into a tensor.

        Args:
            images: List of PIL Images

        Returns:
            Tensor of shape [batch, channels, height, width]
        """
        import torchvision.transforms as transforms

        # Define transform
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        # Transform and stack
        tensor_images = [transform(img) for img in images]
        return torch.stack(tensor_images)


# Convenience function
def create_vision_language_collator(
    tokenizer: Any,
    processor: Optional[Any] = None,
    image_dir: Optional[Path] = None,
    max_length: int = 512,
    image_size: tuple = (336, 336),
    return_masks: bool = False
) -> VisionLanguageCollator:
    """Create a vision-language data collator.

    Args:
        tokenizer: HuggingFace tokenizer
        processor: Vision-language processor (optional)
        image_dir: Directory containing images
        max_length: Maximum sequence length
        image_size: Target image size
        return_masks: Whether to return masks

    Returns:
        VisionLanguageCollator instance

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> collator = create_vision_language_collator(
        ...     tokenizer=tokenizer,
        ...     image_dir="./data/images",
        ...     max_length=512
        ... )
    """
    return VisionLanguageCollator(
        tokenizer=tokenizer,
        processor=processor,
        image_dir=image_dir,
        max_length=max_length,
        image_size=image_size,
        return_masks=return_masks
    )
