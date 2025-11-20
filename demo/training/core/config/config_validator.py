"""Configuration validator for training configs.

Validates training configurations for correctness, compatibility,
and performance. Provides warnings and recommendations.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .training_config import TrainingConfig, TrainingMethod
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation message severity levels."""
    ERROR = "error"      # Configuration is invalid
    WARNING = "warning"  # Suboptimal configuration
    INFO = "info"        # Informational message


@dataclass
class ValidationMessage:
    """Validation message.

    Attributes:
        level: Severity level
        category: Message category
        message: Human-readable message
        suggestion: Optional suggestion to fix issue
    """
    level: ValidationLevel
    category: str
    message: str
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        """Format as string."""
        msg = f"[{self.level.value.upper()}] {self.category}: {self.message}"
        if self.suggestion:
            msg += f"\n  💡 Suggestion: {self.suggestion}"
        return msg


@dataclass
class ValidationResult:
    """Configuration validation result.

    Attributes:
        valid: Whether configuration is valid
        messages: List of validation messages
        errors: List of error messages
        warnings: List of warning messages
        infos: List of info messages
    """
    valid: bool
    messages: List[ValidationMessage]

    @property
    def errors(self) -> List[ValidationMessage]:
        """Get error messages."""
        return [m for m in self.messages if m.level == ValidationLevel.ERROR]

    @property
    def warnings(self) -> List[ValidationMessage]:
        """Get warning messages."""
        return [m for m in self.messages if m.level == ValidationLevel.WARNING]

    @property
    def infos(self) -> List[ValidationMessage]:
        """Get info messages."""
        return [m for m in self.messages if m.level == ValidationLevel.INFO]

    def print_summary(self) -> None:
        """Print validation summary."""
        print("=" * 80)
        print("Configuration Validation Summary")
        print("=" * 80)
        print()

        if self.valid:
            print("✅ Configuration is valid!")
        else:
            print("❌ Configuration has errors!")

        print()
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        print(f"Infos: {len(self.infos)}")
        print()

        # Print errors
        if self.errors:
            print("ERRORS:")
            print("-" * 80)
            for msg in self.errors:
                print(msg)
                print()

        # Print warnings
        if self.warnings:
            print("WARNINGS:")
            print("-" * 80)
            for msg in self.warnings:
                print(msg)
                print()

        # Print infos
        if self.infos:
            print("INFO:")
            print("-" * 80)
            for msg in self.infos:
                print(msg)
                print()


class ConfigValidator:
    """Validator for training configurations.

    Performs comprehensive validation of training configurations,
    including hardware compatibility, hyperparameter ranges,
    and performance checks.
    """

    def __init__(self, check_hardware: bool = True):
        """Initialize validator.

        Args:
            check_hardware: Whether to check hardware compatibility
        """
        self.check_hardware = check_hardware

    def validate(self, config: TrainingConfig) -> ValidationResult:
        """Validate a training configuration.

        Args:
            config: Training configuration to validate

        Returns:
            ValidationResult with errors, warnings, and suggestions
        """
        messages = []

        # Run all validation checks
        messages.extend(self._validate_model(config))
        messages.extend(self._validate_hyperparameters(config))
        messages.extend(self._validate_data(config))
        messages.extend(self._validate_hardware(config))
        messages.extend(self._validate_compatibility(config))
        messages.extend(self._validate_performance(config))

        # Configuration is valid if no errors
        valid = not any(m.level == ValidationLevel.ERROR for m in messages)

        return ValidationResult(valid=valid, messages=messages)

    def _validate_model(self, config: TrainingConfig) -> List[ValidationMessage]:
        """Validate model configuration.

        Args:
            config: Training configuration

        Returns:
            List of validation messages
        """
        messages = []

        # Check if model is in registry
        model = ModelRegistry.get_model(config.model.name)
        if model is None:
            messages.append(ValidationMessage(
                level=ValidationLevel.WARNING,
                category="Model",
                message=f"Model '{config.model.name}' not in registry. VRAM estimation unavailable.",
                suggestion="Use a registered model or add it to ModelRegistry for automatic VRAM checks."
            ))

        return messages

    def _validate_hyperparameters(self, config: TrainingConfig) -> List[ValidationMessage]:
        """Validate hyperparameters.

        Args:
            config: Training configuration

        Returns:
            List of validation messages
        """
        messages = []
        training = config.training

        # Learning rate checks
        if training.learning_rate > 1e-3:
            messages.append(ValidationMessage(
                level=ValidationLevel.WARNING,
                category="Hyperparameters",
                message=f"Learning rate {training.learning_rate} is very high (> 1e-3)",
                suggestion="Consider reducing to 2e-5 to 2e-4 range for stable training."
            ))

        if training.learning_rate < 1e-6:
            messages.append(ValidationMessage(
                level=ValidationLevel.WARNING,
                category="Hyperparameters",
                message=f"Learning rate {training.learning_rate} is very low (< 1e-6)",
                suggestion="Training may be very slow. Consider increasing to 1e-5 to 2e-4 range."
            ))

        # Batch size checks
        effective_bs = config.get_effective_batch_size()
        if effective_bs < 8:
            messages.append(ValidationMessage(
                level=ValidationLevel.INFO,
                category="Hyperparameters",
                message=f"Effective batch size {effective_bs} is small (< 8)",
                suggestion="Small batch sizes can cause instability. Consider increasing gradient_accumulation_steps."
            ))

        if effective_bs > 128:
            messages.append(ValidationMessage(
                level=ValidationLevel.WARNING,
                category="Hyperparameters",
                message=f"Effective batch size {effective_bs} is very large (> 128)",
                suggestion="Large batch sizes may require learning rate adjustment (linear scaling)."
            ))

        # LoRA configuration checks
        if training.method in [TrainingMethod.LORA, TrainingMethod.QLORA]:
            if training.lora is None:
                messages.append(ValidationMessage(
                    level=ValidationLevel.ERROR,
                    category="LoRA",
                    message="LoRA config missing for LoRA/QLoRA training method",
                    suggestion="Add LoRA configuration to training.lora"
                ))
            else:
                lora = training.lora

                # Rank checks
                if lora.rank < 4:
                    messages.append(ValidationMessage(
                        level=ValidationLevel.WARNING,
                        category="LoRA",
                        message=f"LoRA rank {lora.rank} is very low (< 4)",
                        suggestion="Low rank may limit model capacity. Consider rank >= 8."
                    ))

                if lora.rank > 128:
                    messages.append(ValidationMessage(
                        level=ValidationLevel.WARNING,
                        category="LoRA",
                        message=f"LoRA rank {lora.rank} is very high (> 128)",
                        suggestion="High rank increases memory and compute. Consider rank 16-64."
                    ))

                # Alpha/rank ratio check
                if lora.alpha < lora.rank:
                    messages.append(ValidationMessage(
                        level=ValidationLevel.WARNING,
                        category="LoRA",
                        message=f"LoRA alpha ({lora.alpha}) < rank ({lora.rank})",
                        suggestion="Typically alpha should be >= rank. Try alpha = rank * 2."
                    ))

        # QLoRA-specific checks
        if training.method == TrainingMethod.QLORA:
            if training.quantization is None:
                messages.append(ValidationMessage(
                    level=ValidationLevel.ERROR,
                    category="QLoRA",
                    message="Quantization config missing for QLoRA training",
                    suggestion="Add quantization configuration to training.quantization"
                ))

        return messages

    def _validate_data(self, config: TrainingConfig) -> List[ValidationMessage]:
        """Validate data configuration.

        Args:
            config: Training configuration

        Returns:
            List of validation messages
        """
        messages = []
        data = config.data

        # Check data paths exist
        from pathlib import Path

        train_path = Path(data.train_path)
        if not train_path.exists():
            messages.append(ValidationMessage(
                level=ValidationLevel.ERROR,
                category="Data",
                message=f"Training data not found: {data.train_path}",
                suggestion="Ensure training data path is correct."
            ))

        if data.val_path:
            val_path = Path(data.val_path)
            if not val_path.exists():
                messages.append(ValidationMessage(
                    level=ValidationLevel.WARNING,
                    category="Data",
                    message=f"Validation data not found: {data.val_path}",
                    suggestion="Validation data helps monitor training. Create validation split."
                ))

        # Sequence length checks
        if data.max_length > 2048:
            messages.append(ValidationMessage(
                level=ValidationLevel.WARNING,
                category="Data",
                message=f"Max sequence length {data.max_length} is very long (> 2048)",
                suggestion="Long sequences use more memory. Consider reducing to 512-1024."
            ))

        return messages

    def _validate_hardware(self, config: TrainingConfig) -> List[ValidationMessage]:
        """Validate hardware configuration and compatibility.

        Args:
            config: Training configuration

        Returns:
            List of validation messages
        """
        messages = []

        if not self.check_hardware:
            return messages

        # Check CUDA availability
        try:
            import torch
            if config.hardware.device == "cuda" and not torch.cuda.is_available():
                messages.append(ValidationMessage(
                    level=ValidationLevel.ERROR,
                    category="Hardware",
                    message="CUDA requested but not available",
                    suggestion="Install CUDA-enabled PyTorch or set device='cpu'"
                ))

            # Check VRAM if CUDA available
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

                # Estimate VRAM requirements
                model = ModelRegistry.get_model(config.model.name)
                if model:
                    use_qlora = config.training.method == TrainingMethod.QLORA
                    use_lora = config.training.method == TrainingMethod.LORA

                    vram_estimate = ModelRegistry.estimate_vram_requirements(
                        config.model.name,
                        use_lora=use_lora,
                        use_qlora=use_qlora,
                        batch_size=config.training.batch_size,
                        gradient_accumulation=config.training.gradient_accumulation_steps
                    )

                    estimated_vram = vram_estimate.get("total_vram_gb", 0)

                    if estimated_vram > gpu_memory_gb:
                        messages.append(ValidationMessage(
                            level=ValidationLevel.ERROR,
                            category="Hardware",
                            message=f"Estimated VRAM ({estimated_vram:.1f}GB) exceeds available GPU memory ({gpu_memory_gb:.1f}GB)",
                            suggestion=f"Try reducing batch_size, max_length, or use QLoRA. Recommended GPU: {vram_estimate.get('recommended_gpu', 'N/A')}"
                        ))
                    elif estimated_vram > gpu_memory_gb * 0.9:
                        messages.append(ValidationMessage(
                            level=ValidationLevel.WARNING,
                            category="Hardware",
                            message=f"Estimated VRAM ({estimated_vram:.1f}GB) is close to GPU limit ({gpu_memory_gb:.1f}GB)",
                            suggestion="Consider reducing batch_size or max_length for safety margin."
                        ))
                    else:
                        headroom = gpu_memory_gb - estimated_vram
                        messages.append(ValidationMessage(
                            level=ValidationLevel.INFO,
                            category="Hardware",
                            message=f"VRAM check passed! Estimated: {estimated_vram:.1f}GB, Available: {gpu_memory_gb:.1f}GB, Headroom: {headroom:.1f}GB",
                            suggestion=None
                        ))

        except ImportError:
            messages.append(ValidationMessage(
                level=ValidationLevel.WARNING,
                category="Hardware",
                message="PyTorch not available, skipping hardware checks",
                suggestion=None
            ))

        return messages

    def _validate_compatibility(self, config: TrainingConfig) -> List[ValidationMessage]:
        """Validate configuration compatibility.

        Args:
            config: Training configuration

        Returns:
            List of validation messages
        """
        messages = []

        # Check QLoRA + BF16 compatibility
        if config.training.method == TrainingMethod.QLORA:
            if config.hardware.mixed_precision.value not in ["bf16", "fp16"]:
                messages.append(ValidationMessage(
                    level=ValidationLevel.WARNING,
                    category="Compatibility",
                    message="QLoRA works best with mixed precision (BF16 or FP16)",
                    suggestion="Set hardware.mixed_precision to 'bf16' or 'fp16'"
                ))

            # Check gradient checkpointing
            if not config.hardware.gradient_checkpointing:
                messages.append(ValidationMessage(
                    level=ValidationLevel.WARNING,
                    category="Compatibility",
                    message="QLoRA without gradient checkpointing uses more memory",
                    suggestion="Enable hardware.gradient_checkpointing for memory savings"
                ))

        return messages

    def _validate_performance(self, config: TrainingConfig) -> List[ValidationMessage]:
        """Validate performance-related settings.

        Args:
            config: Training configuration

        Returns:
            List of validation messages
        """
        messages = []

        # Check warmup ratio
        if config.training.warmup_ratio > 0.1:
            messages.append(ValidationMessage(
                level=ValidationLevel.WARNING,
                category="Performance",
                message=f"Warmup ratio {config.training.warmup_ratio} is high (> 0.1)",
                suggestion="Long warmup may slow training. Typical range: 0.03-0.1"
            ))

        # Check number of workers
        if config.hardware.num_workers > 8:
            messages.append(ValidationMessage(
                level=ValidationLevel.WARNING,
                category="Performance",
                message=f"num_workers {config.hardware.num_workers} is very high (> 8)",
                suggestion="Too many workers can cause overhead. Try 2-8 workers."
            ))

        return messages
