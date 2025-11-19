"""Pre-configured training presets for common scenarios."""

from .training_config import (
    TrainingConfig,
    ModelConfig,
    DataConfig,
    TrainingHyperparameters,
    TrainingMethod,
    LoRAConfig,
    QuantizationConfig,
    HardwareConfig,
    MixedPrecision
)


class ConfigPresets:
    """Pre-configured training setups for common use cases."""

    @staticmethod
    def quick_test(data_path: str, model_name: str = "liuhaotian/llava-v1.5-7b") -> TrainingConfig:
        """Quick test configuration for debugging and validation.

        Minimal epochs, small batch size, frequent logging.

        Args:
            data_path: Path to training data
            model_name: Model to use

        Returns:
            TrainingConfig for quick testing
        """
        return TrainingConfig(
            model=ModelConfig(name=model_name, type="llava"),
            data=DataConfig(train_path=data_path, max_length=512),
            training=TrainingHyperparameters(
                method=TrainingMethod.LORA,
                learning_rate=1e-4,
                batch_size=2,
                gradient_accumulation_steps=2,
                num_epochs=1,
                warmup_ratio=0.1,
                lora=LoRAConfig(rank=8, alpha=16, dropout=0.05)
            ),
            hardware=HardwareConfig(
                mixed_precision=MixedPrecision.BF16,
                gradient_checkpointing=True
            ),
            experiment_name="quick_test"
        )

    @staticmethod
    def development(data_path: str, val_path: str = None, model_name: str = "liuhaotian/llava-v1.5-7b") -> TrainingConfig:
        """Development configuration for iterative experimentation.

        Moderate settings, good for iterating on model architecture.

        Args:
            data_path: Path to training data
            val_path: Path to validation data
            model_name: Model to use

        Returns:
            TrainingConfig for development
        """
        return TrainingConfig(
            model=ModelConfig(name=model_name, type="llava"),
            data=DataConfig(
                train_path=data_path,
                val_path=val_path,
                max_length=1024
            ),
            training=TrainingHyperparameters(
                method=TrainingMethod.LORA,
                learning_rate=2e-5,
                batch_size=4,
                gradient_accumulation_steps=4,
                num_epochs=3,
                warmup_ratio=0.03,
                lora=LoRAConfig(rank=32, alpha=64, dropout=0.05)
            ),
            hardware=HardwareConfig(
                mixed_precision=MixedPrecision.BF16,
                gradient_checkpointing=True,
                num_workers=4
            ),
            experiment_name="development"
        )

    @staticmethod
    def production(data_path: str, val_path: str, model_name: str = "liuhaotian/llava-v1.5-7b") -> TrainingConfig:
        """Production configuration for best quality results.

        Higher rank LoRA, more epochs, careful hyperparameters.

        Args:
            data_path: Path to training data
            val_path: Path to validation data (required)
            model_name: Model to use

        Returns:
            TrainingConfig for production training
        """
        return TrainingConfig(
            model=ModelConfig(name=model_name, type="llava"),
            data=DataConfig(
                train_path=data_path,
                val_path=val_path,
                max_length=2048
            ),
            training=TrainingHyperparameters(
                method=TrainingMethod.LORA,
                learning_rate=2e-5,
                batch_size=4,
                gradient_accumulation_steps=8,
                num_epochs=5,
                warmup_ratio=0.03,
                weight_decay=0.01,
                lora=LoRAConfig(rank=64, alpha=128, dropout=0.05)
            ),
            hardware=HardwareConfig(
                mixed_precision=MixedPrecision.BF16,
                gradient_checkpointing=True,
                num_workers=8
            ),
            experiment_name="production"
        )

    @staticmethod
    def memory_efficient(data_path: str, model_name: str = "liuhaotian/llava-v1.5-7b") -> TrainingConfig:
        """Memory-efficient configuration for limited VRAM.

        Small batch size, low rank LoRA, aggressive gradient checkpointing.

        Args:
            data_path: Path to training data
            model_name: Model to use

        Returns:
            TrainingConfig optimized for low VRAM
        """
        return TrainingConfig(
            model=ModelConfig(name=model_name, type="llava"),
            data=DataConfig(train_path=data_path, max_length=512),
            training=TrainingHyperparameters(
                method=TrainingMethod.LORA,
                learning_rate=2e-5,
                batch_size=1,
                gradient_accumulation_steps=16,
                num_epochs=3,
                warmup_ratio=0.03,
                lora=LoRAConfig(rank=16, alpha=32, dropout=0.05)
            ),
            hardware=HardwareConfig(
                mixed_precision=MixedPrecision.FP16,
                gradient_checkpointing=True,
                num_workers=2
            ),
            experiment_name="memory_efficient"
        )

    @staticmethod
    def high_quality(data_path: str, val_path: str, model_name: str = "liuhaotian/llava-v1.5-13b") -> TrainingConfig:
        """High-quality configuration with larger model.

        Uses 13B model, higher rank LoRA, more training.

        Args:
            data_path: Path to training data
            val_path: Path to validation data
            model_name: Larger model to use (default: 13B)

        Returns:
            TrainingConfig for highest quality
        """
        return TrainingConfig(
            model=ModelConfig(name=model_name, type="llava"),
            data=DataConfig(
                train_path=data_path,
                val_path=val_path,
                max_length=2048
            ),
            training=TrainingHyperparameters(
                method=TrainingMethod.LORA,
                learning_rate=1e-5,
                batch_size=2,
                gradient_accumulation_steps=16,
                num_epochs=5,
                warmup_ratio=0.05,
                weight_decay=0.01,
                lora=LoRAConfig(rank=128, alpha=256, dropout=0.05)
            ),
            hardware=HardwareConfig(
                mixed_precision=MixedPrecision.BF16,
                gradient_checkpointing=True,
                num_workers=8
            ),
            experiment_name="high_quality"
        )

    @staticmethod
    def ultra_low_memory(data_path: str, val_path: str = None, model_name: str = "liuhaotian/llava-v1.5-7b") -> TrainingConfig:
        """Ultra low memory configuration for 8GB GPUs using QLoRA.

        Uses 4-bit quantization (QLoRA), minimal batch size, aggressive settings.
        Optimized for consumer GPUs like RTX 3060 12GB, RTX 4060 8GB.

        Estimated VRAM usage: ~5-7GB (fits on 8GB GPUs!)

        Args:
            data_path: Path to training data
            val_path: Optional path to validation data
            model_name: Model to use (default: LLaVA-v1.5-7b, stable at ~7GB QLoRA)

        Returns:
            TrainingConfig optimized for 8GB VRAM
        """
        return TrainingConfig(
            model=ModelConfig(name=model_name, type="llava"),
            data=DataConfig(
                train_path=data_path,
                val_path=val_path,
                max_length=512,  # Shorter sequences to save memory
                image_size=336   # Smaller images
            ),
            training=TrainingHyperparameters(
                method=TrainingMethod.QLORA,  # 4-bit quantization!
                learning_rate=2e-4,  # Slightly higher LR for QLoRA
                batch_size=1,  # Minimal batch size
                gradient_accumulation_steps=32,  # Effective batch size = 32
                num_epochs=3,
                warmup_ratio=0.03,
                lora=LoRAConfig(
                    rank=8,  # Very low rank for memory savings
                    alpha=16,
                    dropout=0.05
                ),
                quantization=QuantizationConfig(
                    load_in_4bit=True,
                    load_in_8bit=False,
                    bnb_4bit_compute_dtype="bfloat16",
                    bnb_4bit_quant_type="nf4",  # NF4 quantization (better than FP4)
                    bnb_4bit_use_double_quant=True  # Extra memory savings
                )
            ),
            hardware=HardwareConfig(
                mixed_precision=MixedPrecision.BF16,
                gradient_checkpointing=True,  # Essential for memory savings
                num_workers=2  # Reduce CPU overhead
            ),
            experiment_name="ultra_low_memory_8gb"
        )

    @classmethod
    def list_presets(cls) -> dict:
        """List all available presets with descriptions.

        Returns:
            Dictionary of preset names to descriptions
        """
        return {
            "quick_test": "Fast testing setup (1 epoch, rank 8, batch 2)",
            "development": "Balanced for experimentation (3 epochs, rank 32, batch 4)",
            "production": "High quality for deployment (5 epochs, rank 64, batch 4)",
            "memory_efficient": "Optimized for low VRAM (rank 16, batch 1)",
            "high_quality": "Best results with 13B model (5 epochs, rank 128)",
            "ultra_low_memory": "🔥 8GB GPU support with QLoRA (LLaVA-7B, 4-bit, rank 8, ~7GB VRAM)"
        }
