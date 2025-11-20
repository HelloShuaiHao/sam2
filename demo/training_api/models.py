"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from enum import Enum


# ==================== Data Preparation Models ====================

class ConvertRequest(BaseModel):
    """Request for converting SAM2 export to training format."""

    sam2_zip_path: str = Field(..., description="Path to SAM2 export ZIP file")
    output_dir: str = Field(..., description="Output directory for converted data")
    target_format: Literal["huggingface", "llava"] = Field(
        default="llava", description="Target format for conversion"
    )

    @field_validator("sam2_zip_path", "output_dir")
    def check_path_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")
        return v


class ConvertResponse(BaseModel):
    """Response after converting data."""

    success: bool
    output_dir: str
    num_samples: int
    message: str
    warnings: List[str] = []


class ValidateRequest(BaseModel):
    """Request for validating dataset quality."""

    data_path: str = Field(..., description="Path to converted dataset")
    format_type: Literal["huggingface", "llava"] = "llava"


class ValidationReport(BaseModel):
    """Validation report."""

    passed: bool
    num_errors: int
    num_warnings: int
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    recommendations: List[str]


class SplitRequest(BaseModel):
    """Request for splitting dataset."""

    data_path: str = Field(..., description="Path to dataset to split")
    output_dir: str = Field(..., description="Output directory for splits")
    strategy: Literal["stratified", "temporal", "random"] = "stratified"
    train_ratio: float = Field(0.7, ge=0.1, le=0.9)
    val_ratio: float = Field(0.2, ge=0.05, le=0.5)
    test_ratio: float = Field(0.1, ge=0.05, le=0.5)
    random_seed: int = 42

    @field_validator("train_ratio", "val_ratio", "test_ratio")
    def check_ratios_sum_to_one(cls, v, info):
        # This will be checked after all fields are set
        return v


class SplitResponse(BaseModel):
    """Response after splitting dataset."""

    success: bool
    train_path: str
    val_path: str
    test_path: str
    train_samples: int
    val_samples: int
    test_samples: int


# ==================== Training Models ====================

class JobStatus(str, Enum):
    """Training job status."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingConfig(BaseModel):
    """Training configuration."""

    # Model configuration
    model_name: str = Field(..., description="HuggingFace model name or path")
    use_lora: bool = Field(True, description="Use LoRA fine-tuning")
    lora_rank: int = Field(8, ge=1, le=256)
    lora_alpha: int = Field(16, ge=1, le=512)
    lora_dropout: float = Field(0.05, ge=0.0, le=0.5)
    use_qlora: bool = Field(False, description="Use 4-bit quantization (QLoRA)")

    # Training hyperparameters
    num_epochs: int = Field(3, ge=1, le=100)
    batch_size: int = Field(4, ge=1, le=128)
    gradient_accumulation_steps: int = Field(4, ge=1, le=64)
    learning_rate: float = Field(2e-4, gt=0.0)
    warmup_ratio: float = Field(0.03, ge=0.0, le=0.5)
    max_grad_norm: float = Field(1.0, gt=0.0)

    # Data configuration
    train_data_path: str
    val_data_path: Optional[str] = None
    max_length: int = Field(2048, ge=128, le=8192)

    # Hardware configuration
    fp16: bool = True
    bf16: bool = False
    device: str = "cuda"

    # Output configuration
    output_dir: str = Field(..., description="Directory to save checkpoints")
    save_steps: int = Field(100, ge=1)
    eval_steps: int = Field(100, ge=1)
    logging_steps: int = Field(10, ge=1)
    save_total_limit: int = Field(3, ge=1, le=10)


class StartTrainingRequest(BaseModel):
    """Request to start training job."""

    config: TrainingConfig
    experiment_name: Optional[str] = None
    tags: List[str] = []


class StartTrainingResponse(BaseModel):
    """Response after starting training."""

    success: bool
    job_id: str
    message: str
    estimated_duration_minutes: Optional[float] = None


class JobProgress(BaseModel):
    """Training job progress."""

    job_id: str
    status: JobStatus
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    progress_percentage: Optional[float] = None
    eta_seconds: Optional[float] = None

    # Metrics
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    learning_rate: Optional[float] = None

    # Timestamps
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Error info
    error_message: Optional[str] = None


class CancelJobResponse(BaseModel):
    """Response after cancelling job."""

    success: bool
    job_id: str
    message: str


# ==================== Experiment Models ====================

class ExperimentSummary(BaseModel):
    """Summary of an experiment."""

    experiment_id: str
    experiment_name: Optional[str]
    job_id: str
    status: JobStatus
    model_name: str

    # Metrics
    best_eval_loss: Optional[float] = None
    best_eval_accuracy: Optional[float] = None
    final_train_loss: Optional[float] = None

    # Configuration summary
    num_epochs: int
    batch_size: int
    learning_rate: float
    use_lora: bool
    use_qlora: bool

    # Metadata
    created_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    tags: List[str] = []


class ExperimentDetail(BaseModel):
    """Detailed experiment information."""

    # All summary fields
    experiment_id: str
    experiment_name: Optional[str]
    job_id: str
    status: JobStatus
    model_name: str

    # Full configuration
    config: TrainingConfig

    # All metrics history
    metrics_history: Dict[str, List[float]] = {}

    # Checkpoints
    checkpoints: List[Dict[str, Any]] = []
    best_checkpoint_path: Optional[str] = None

    # System info
    gpu_type: Optional[str] = None
    peak_memory_mb: Optional[float] = None

    # Metadata
    created_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    tags: List[str] = []


class CompareExperimentsRequest(BaseModel):
    """Request to compare multiple experiments."""

    experiment_ids: List[str] = Field(..., min_length=2, max_length=10)
    metrics: List[str] = ["eval_loss", "train_loss"]


class CompareExperimentsResponse(BaseModel):
    """Response with experiment comparison."""

    experiments: List[ExperimentSummary]
    comparison_table: Dict[str, Dict[str, Any]]  # metric -> {exp_id -> value}
    best_experiment_id: Optional[str] = None


# ==================== Export Models ====================

class ExportFormat(str, Enum):
    """Model export format."""

    HUGGINGFACE = "huggingface"  # Full model
    LORA_ADAPTER = "lora_adapter"  # Only LoRA weights
    ONNX = "onnx"  # ONNX format (future)
    TFLITE = "tflite"  # TensorFlow Lite (future)


class ExportRequest(BaseModel):
    """Request to export trained model."""

    job_id: str = Field(..., description="Training job ID")
    export_format: ExportFormat = ExportFormat.HUGGINGFACE
    output_dir: Optional[str] = None
    generate_model_card: bool = True
    merge_adapters: bool = Field(
        False, description="Merge LoRA adapters into base model (if using LoRA)"
    )


class ExportResponse(BaseModel):
    """Response after exporting model."""

    success: bool
    export_path: str
    export_format: ExportFormat
    file_size_mb: float
    message: str
    model_card_path: Optional[str] = None


class DownloadInfo(BaseModel):
    """Information for downloading exported model."""

    download_url: str
    filename: str
    file_size_mb: float
    expires_at: Optional[datetime] = None
