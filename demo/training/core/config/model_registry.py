"""Model registry for supported vision-language models."""

from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum


class ModelType(str, Enum):
    """Supported model types."""
    LLAVA = "llava"
    QWEN_VL = "qwen-vl"
    INSTRUCTBLIP = "instructblip"


@dataclass
class ModelInfo:
    """Information about a supported model.

    Attributes:
        name: HuggingFace model identifier
        type: Model type
        size_gb: Approximate model size in GB
        min_vram_gb: Minimum VRAM required (full precision)
        lora_vram_gb: VRAM required for LoRA training (FP16/BF16)
        qlora_vram_gb: VRAM required for QLoRA training (4-bit quantization)
        max_sequence_length: Maximum sequence length
        vision_encoder: Vision encoder name
        language_model: Language model name
        default_image_size: Default image input size
        supports_video: Whether model supports video input
        description: Model description
    """
    name: str
    type: ModelType
    size_gb: float
    min_vram_gb: int
    lora_vram_gb: int
    qlora_vram_gb: int
    max_sequence_length: int
    vision_encoder: str
    language_model: str
    default_image_size: int = 336
    supports_video: bool = False
    description: str = ""


class ModelRegistry:
    """Registry of supported vision-language models."""

    # Registry of known models
    MODELS: Dict[str, ModelInfo] = {
        "liuhaotian/llava-v1.5-7b": ModelInfo(
            name="liuhaotian/llava-v1.5-7b",
            type=ModelType.LLAVA,
            size_gb=13.5,
            min_vram_gb=40,
            lora_vram_gb=24,
            qlora_vram_gb=7,  # 4-bit quantized: ~3.5GB model + 2-3GB optimizer/activations
            max_sequence_length=2048,
            vision_encoder="CLIP ViT-L/14",
            language_model="Vicuna-7B",
            default_image_size=336,
            supports_video=False,
            description="LLaVA 1.5 with 7B parameters - good balance of performance and efficiency"
        ),
        "liuhaotian/llava-v1.5-13b": ModelInfo(
            name="liuhaotian/llava-v1.5-13b",
            type=ModelType.LLAVA,
            size_gb=26.0,
            min_vram_gb=80,
            lora_vram_gb=40,
            qlora_vram_gb=12,  # 4-bit quantized: ~6.5GB model + 4-5GB optimizer/activations
            max_sequence_length=2048,
            vision_encoder="CLIP ViT-L/14",
            language_model="Vicuna-13B",
            default_image_size=336,
            supports_video=False,
            description="LLaVA 1.5 with 13B parameters - higher quality, more VRAM needed"
        ),
        "Qwen/Qwen-VL-Chat": ModelInfo(
            name="Qwen/Qwen-VL-Chat",
            type=ModelType.QWEN_VL,
            size_gb=9.6,
            min_vram_gb=32,
            lora_vram_gb=20,
            qlora_vram_gb=6,  # 4-bit quantized: ~2.5GB model + 2-3GB optimizer/activations
            max_sequence_length=2048,
            vision_encoder="ViT-bigG",
            language_model="Qwen-7B",
            default_image_size=448,
            supports_video=True,
            description="Qwen-VL with multi-image and video support"
        ),
        "Salesforce/instructblip-vicuna-7b": ModelInfo(
            name="Salesforce/instructblip-vicuna-7b",
            type=ModelType.INSTRUCTBLIP,
            size_gb=14.0,
            min_vram_gb=40,
            lora_vram_gb=24,
            qlora_vram_gb=7,  # 4-bit quantized: ~3.5GB model + 2-3GB optimizer/activations
            max_sequence_length=2048,
            vision_encoder="EVA-CLIP",
            language_model="Vicuna-7B",
            default_image_size=224,
            supports_video=False,
            description="InstructBLIP with instruction-aware visual features"
        ),
    }

    @classmethod
    def get_model(cls, name: str) -> Optional[ModelInfo]:
        """Get model info by name.

        Args:
            name: Model name or HuggingFace identifier

        Returns:
            ModelInfo if found, None otherwise
        """
        return cls.MODELS.get(name)

    @classmethod
    def get_models_by_type(cls, model_type: ModelType) -> List[ModelInfo]:
        """Get all models of a specific type.

        Args:
            model_type: Type of models to retrieve

        Returns:
            List of ModelInfo for that type
        """
        return [
            info for info in cls.MODELS.values()
            if info.type == model_type
        ]

    @classmethod
    def list_all_models(cls) -> List[ModelInfo]:
        """Get list of all registered models.

        Returns:
            List of all ModelInfo
        """
        return list(cls.MODELS.values())

    @classmethod
    def is_supported(cls, name: str) -> bool:
        """Check if a model is supported.

        Args:
            name: Model name

        Returns:
            True if model is in registry
        """
        return name in cls.MODELS

    @classmethod
    def register_model(cls, model_info: ModelInfo) -> None:
        """Register a new model.

        Args:
            model_info: Model information to register
        """
        cls.MODELS[model_info.name] = model_info

    @classmethod
    def estimate_vram_requirements(
        cls,
        model_name: str,
        use_lora: bool = True,
        use_qlora: bool = False,
        batch_size: int = 4,
        gradient_accumulation: int = 1
    ) -> Dict[str, float]:
        """Estimate VRAM requirements for training.

        Args:
            model_name: Name of the model
            use_lora: Whether using LoRA (vs full fine-tuning)
            use_qlora: Whether using QLoRA (4-bit quantization)
            batch_size: Training batch size per device
            gradient_accumulation: Gradient accumulation steps

        Returns:
            Dictionary with VRAM estimates
        """
        model = cls.get_model(model_name)
        if not model:
            return {"error": "Model not found in registry"}

        if use_qlora:
            # QLoRA mode: 4-bit quantized model
            # Model is loaded in 4-bit (~25% of original size)
            quantized_model_size = model.size_gb * 0.25

            # LoRA adapters (very small, ~1-2% of model size)
            adapter_size = model.size_gb * 0.015

            # Optimizer states (only for LoRA params)
            optimizer_vram = adapter_size * 2.0

            # Gradients (only for LoRA params)
            gradient_vram = adapter_size

            # Activations (scales with batch size, but reduced with gradient checkpointing)
            activation_per_sample = 0.5  # GB per sample (reduced with checkpointing)
            activation_vram = activation_per_sample * batch_size

            total_vram = quantized_model_size + adapter_size + optimizer_vram + gradient_vram + activation_vram

            return {
                "method": "QLoRA (4-bit)",
                "model_vram_gb": quantized_model_size,
                "adapter_vram_gb": adapter_size,
                "optimizer_vram_gb": optimizer_vram,
                "gradient_vram_gb": gradient_vram,
                "activation_vram_gb": activation_vram,
                "total_vram_gb": total_vram,
                "effective_batch_size": batch_size * gradient_accumulation,
                "recommended_gpu": "RTX 4090 24GB" if total_vram > 16 else "RTX 3090 24GB" if total_vram > 12 else "RTX 4060 Ti 16GB" if total_vram > 8 else "RTX 3060 12GB / RTX 4060 8GB"
            }

        elif use_lora:
            # Regular LoRA mode: FP16/BF16
            base_vram = model.lora_vram_gb

            # Optimizer states (Adam: 2x parameters for momentum + variance)
            optimizer_vram = model.size_gb * 0.5

            # Gradients
            gradient_vram = model.size_gb * 0.2

            # Activations (scales with batch size)
            activation_per_sample = 2.0  # GB per sample
            activation_vram = activation_per_sample * batch_size

            total_vram = base_vram + optimizer_vram + gradient_vram + activation_vram

            return {
                "method": "LoRA (FP16/BF16)",
                "model_vram_gb": base_vram,
                "optimizer_vram_gb": optimizer_vram,
                "gradient_vram_gb": gradient_vram,
                "activation_vram_gb": activation_vram,
                "total_vram_gb": total_vram,
                "effective_batch_size": batch_size * gradient_accumulation,
                "recommended_gpu": "A100 80GB" if total_vram > 40 else "A100 40GB" if total_vram > 24 else "RTX 4090 24GB"
            }

        else:
            # Full fine-tuning mode
            base_vram = model.min_vram_gb
            optimizer_vram = model.size_gb * 2.0
            gradient_vram = model.size_gb * 1.0
            activation_per_sample = 2.0
            activation_vram = activation_per_sample * batch_size

            total_vram = base_vram + optimizer_vram + gradient_vram + activation_vram

            return {
                "method": "Full Fine-tuning",
                "model_vram_gb": base_vram,
                "optimizer_vram_gb": optimizer_vram,
                "gradient_vram_gb": gradient_vram,
                "activation_vram_gb": activation_vram,
                "total_vram_gb": total_vram,
                "effective_batch_size": batch_size * gradient_accumulation,
                "recommended_gpu": "A100 80GB" if total_vram > 40 else "A100 40GB"
            }
