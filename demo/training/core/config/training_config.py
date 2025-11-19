"""Training configuration schema using Pydantic."""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class TrainingMethod(str, Enum):
    """Supported training methods."""
    LORA = "lora"
    QLORA = "qlora"
    FULL = "full"


class MixedPrecision(str, Enum):
    """Mixed precision training options."""
    NO = "no"
    FP16 = "fp16"
    BF16 = "bf16"


class ModelConfig(BaseModel):
    """Model configuration.

    Attributes:
        name: HuggingFace model name or path
        type: Model type (llava, qwen-vl, instructblip)
        cache_dir: Optional cache directory for model downloads
    """
    name: str = Field(..., description="HuggingFace model name (e.g., 'liuhaotian/llava-v1.5-7b')")
    type: str = Field(..., description="Model type (llava, qwen-vl, instructblip)")
    cache_dir: Optional[str] = Field(None, description="Cache directory for model downloads")


class LoRAConfig(BaseModel):
    """LoRA-specific configuration.

    Attributes:
        rank: LoRA rank (typically 8-64)
        alpha: LoRA alpha (typically rank * 2)
        dropout: Dropout probability
        target_modules: Modules to apply LoRA to
        bias: Bias training strategy
    """
    rank: int = Field(64, ge=1, le=256, description="LoRA rank")
    alpha: int = Field(16, ge=1, description="LoRA alpha")
    dropout: float = Field(0.05, ge=0.0, le=1.0, description="LoRA dropout")
    target_modules: Optional[list[str]] = Field(
        None,
        description="Target modules (None = auto-detect)"
    )
    bias: str = Field("none", description="Bias strategy: none, all, lora_only")

    @field_validator('alpha')
    @classmethod
    def validate_alpha(cls, v, info):
        """Recommend alpha = rank * 2."""
        rank = info.data.get('rank', 64)
        if v < rank:
            raise ValueError(f"Alpha ({v}) should typically be >= rank ({rank})")
        return v


class DataConfig(BaseModel):
    """Data configuration.

    Attributes:
        train_path: Path to training data
        val_path: Optional validation data path
        max_length: Maximum sequence length
        image_size: Image size for preprocessing
    """
    train_path: str = Field(..., description="Path to training dataset")
    val_path: Optional[str] = Field(None, description="Path to validation dataset")
    max_length: int = Field(512, ge=128, le=4096, description="Maximum sequence length")
    image_size: int = Field(336, ge=224, le=1024, description="Image size")


class TrainingHyperparameters(BaseModel):
    """Training hyperparameters.

    Attributes:
        method: Training method (lora, qlora, full)
        learning_rate: Learning rate
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        num_epochs: Number of training epochs
        warmup_ratio: Warmup ratio
        weight_decay: Weight decay
        max_grad_norm: Maximum gradient norm for clipping
    """
    method: TrainingMethod = Field(TrainingMethod.LORA, description="Training method")
    learning_rate: float = Field(2e-5, gt=0, le=1e-3, description="Learning rate")
    batch_size: int = Field(4, ge=1, le=128, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(4, ge=1, description="Gradient accumulation steps")
    num_epochs: int = Field(3, ge=1, le=100, description="Number of epochs")
    warmup_ratio: float = Field(0.03, ge=0.0, le=1.0, description="Warmup ratio")
    weight_decay: float = Field(0.0, ge=0.0, le=1.0, description="Weight decay")
    max_grad_norm: float = Field(1.0, gt=0, description="Max gradient norm")

    lora: Optional[LoRAConfig] = Field(None, description="LoRA configuration")

    @field_validator('lora')
    @classmethod
    def validate_lora_config(cls, v, info):
        """Ensure LoRA config exists for LoRA/QLoRA methods."""
        method = info.data.get('method')
        if method in [TrainingMethod.LORA, TrainingMethod.QLORA] and v is None:
            # Create default LoRA config
            return LoRAConfig()
        return v


class HardwareConfig(BaseModel):
    """Hardware configuration.

    Attributes:
        device: Device to use (cuda, cpu)
        mixed_precision: Mixed precision mode
        gradient_checkpointing: Enable gradient checkpointing
        num_workers: Number of dataloader workers
    """
    device: str = Field("cuda", description="Device: cuda or cpu")
    mixed_precision: MixedPrecision = Field(MixedPrecision.BF16, description="Mixed precision")
    gradient_checkpointing: bool = Field(True, description="Enable gradient checkpointing")
    num_workers: int = Field(4, ge=0, description="Dataloader workers")


class CheckpointConfig(BaseModel):
    """Checkpoint configuration.

    Attributes:
        save_steps: Save checkpoint every N steps
        save_total_limit: Maximum number of checkpoints to keep
        output_dir: Directory to save checkpoints
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    save_steps: int = Field(100, ge=1, description="Save every N steps")
    save_total_limit: int = Field(3, ge=1, description="Max checkpoints to keep")
    output_dir: str = Field("./output", description="Output directory")
    resume_from_checkpoint: Optional[str] = Field(None, description="Resume from checkpoint")


class LoggingConfig(BaseModel):
    """Logging configuration.

    Attributes:
        log_steps: Log metrics every N steps
        tensorboard_dir: Tensorboard log directory
        report_to: Reporting destinations
    """
    log_steps: int = Field(10, ge=1, description="Log every N steps")
    tensorboard_dir: str = Field("./runs", description="Tensorboard directory")
    report_to: list[str] = Field(["tensorboard"], description="Report destinations")


class TrainingConfig(BaseModel):
    """Complete training configuration.

    This is the main configuration object that combines all training settings.
    """
    model: ModelConfig
    data: DataConfig
    training: TrainingHyperparameters
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    experiment_name: Optional[str] = Field(None, description="Experiment name")
    seed: int = Field(42, ge=0, description="Random seed")

    def model_dump_json(self, **kwargs) -> str:
        """Serialize to JSON string."""
        return super().model_dump_json(indent=2, **kwargs)

    @classmethod
    def from_json_file(cls, filepath: str) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            import json
            data = json.load(f)
        return cls(**data)

    def save_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.model_dump_json())

    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        return self.training.batch_size * self.training.gradient_accumulation_steps
