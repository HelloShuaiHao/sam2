"""
Training endpoints for starting, monitoring, and managing training jobs.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Optional
import sys
from pathlib import Path
import uuid
import time
import json
from datetime import datetime
import threading

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from training_api.models import (
    StartTrainingRequest,
    StartTrainingResponse,
    JobProgress,
    JobStatus,
    CancelJobResponse,
)

# Import core modules
from training.core.trainers.lora_trainer import LoRATrainer
from training.core.config.training_config import TrainingConfig as CoreTrainingConfig
from training.core.config.model_registry import ModelRegistry

router = APIRouter()

# In-memory job storage (in production, use Redis/database)
_active_jobs: Dict[str, Dict] = {}
_job_threads: Dict[str, threading.Thread] = {}


def estimate_training_time(config: StartTrainingRequest) -> float:
    """Estimate training duration in minutes."""
    # Simple heuristic based on model size, epochs, and batch size
    model_sizes = {
        "llava": 7000,  # 7B parameters
        "qwen": 9600,  # 9.6B parameters
        "instructblip": 13000,  # 13B parameters
    }

    # Guess model size from name
    model_size_mb = 7000  # default
    for key, size in model_sizes.items():
        if key in config.config.model_name.lower():
            model_size_mb = size
            break

    # Rough estimate: ~1 minute per epoch per 1B parameters with batch_size=4
    effective_batch = config.config.batch_size * config.config.gradient_accumulation_steps
    time_per_epoch = (model_size_mb / 1000) * (4 / effective_batch)

    if config.config.use_qlora:
        time_per_epoch *= 0.7  # QLoRA is faster due to quantization

    total_minutes = time_per_epoch * config.config.num_epochs
    return total_minutes


def run_training_job(job_id: str, config: StartTrainingRequest):
    """Background function to run training job."""
    try:
        # Update job status
        _active_jobs[job_id]["status"] = JobStatus.RUNNING
        _active_jobs[job_id]["started_at"] = datetime.now()

        # Import core config modules
        from training.core.config.training_config import (
            ModelConfig,
            DataConfig,
            TrainingHyperparameters,
            LoRAConfig as CoreLoRAConfig,
            QuantizationConfig,
            HardwareConfig,
            CheckpointConfig,
            LoggingConfig,
            TrainingMethod,
            MixedPrecision
        )

        # Convert API config to core config
        # Determine training method
        training_method = TrainingMethod.QLORA if config.config.use_qlora else (
            TrainingMethod.LORA if config.config.use_lora else TrainingMethod.FULL
        )

        # Create LoRA config if using LoRA/QLoRA
        lora_config = None
        if training_method in [TrainingMethod.LORA, TrainingMethod.QLORA]:
            lora_config = CoreLoRAConfig(
                rank=config.config.lora_rank,
                alpha=config.config.lora_alpha,
                dropout=config.config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )

        # Create quantization config if using QLoRA
        quantization_config = None
        if training_method == TrainingMethod.QLORA:
            quantization_config = QuantizationConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                bnb_4bit_compute_dtype="bfloat16" if config.config.bf16 else "float16",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

        # Determine mixed precision
        mixed_precision = MixedPrecision.BF16 if config.config.bf16 else (
            MixedPrecision.FP16 if config.config.fp16 else MixedPrecision.NO
        )

        core_config = CoreTrainingConfig(
            model=ModelConfig(
                name=config.config.model_name,
                type="llava",  # TODO: detect from model name
                cache_dir=None
            ),
            data=DataConfig(
                train_path=config.config.train_data_path,
                val_path=config.config.val_data_path,
                max_length=config.config.max_length,
                image_size=336  # Default image size
            ),
            training=TrainingHyperparameters(
                method=training_method,
                learning_rate=config.config.learning_rate,
                batch_size=config.config.batch_size,
                gradient_accumulation_steps=config.config.gradient_accumulation_steps,
                num_epochs=config.config.num_epochs,
                warmup_ratio=config.config.warmup_ratio,
                max_grad_norm=config.config.max_grad_norm,
                lora=lora_config,
                quantization=quantization_config
            ),
            hardware=HardwareConfig(
                device="cuda",
                mixed_precision=mixed_precision,
                gradient_checkpointing=True,
                num_workers=0
            ),
            checkpointing=CheckpointConfig(
                save_steps=config.config.save_steps,
                save_total_limit=config.config.save_total_limit,
                output_dir=config.config.output_dir
            ),
            logging=LoggingConfig(
                log_steps=config.config.logging_steps,
                tensorboard_dir=f"{config.config.output_dir}/runs",
                report_to=["tensorboard"]
            ),
            experiment_name=config.experiment_name
        )

        # Create trainer and setup model
        print(f"[Job {job_id}] Creating trainer and loading model...")
        trainer = LoRATrainer(core_config)

        print(f"[Job {job_id}] Loading model with QLoRA settings...")
        trainer.setup()

        print(f"[Job {job_id}] Model loaded successfully!")

        # Load datasets - create simple dataset from JSONL files
        print(f"[Job {job_id}] Loading datasets...")
        from torch.utils.data import Dataset
        import json

        class SimpleVLMDataset(Dataset):
            """Simple dataset that loads JSONL data for vision-language models."""
            def __init__(self, jsonl_path: str):
                self.data = []
                with open(jsonl_path, 'r') as f:
                    for line in f:
                        self.data.append(json.loads(line.strip()))

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                # Return raw data item - the Trainer's data collator will handle tokenization
                return self.data[idx]

        train_dataset = SimpleVLMDataset(config.config.train_data_path)
        eval_dataset = SimpleVLMDataset(config.config.val_data_path) if config.config.val_data_path else None

        print(f"[Job {job_id}] Starting actual training...")
        print(f"[Job {job_id}] Train samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"[Job {job_id}] Eval samples: {len(eval_dataset)}")

        # Run actual training
        result = trainer.train(train_dataset, eval_dataset)

        # Training completed successfully
        _active_jobs[job_id]["status"] = JobStatus.COMPLETED
        _active_jobs[job_id]["completed_at"] = datetime.now()
        _active_jobs[job_id]["progress_percentage"] = 100.0

        # Store final metrics
        if "metrics" in result:
            metrics = result["metrics"]
            _active_jobs[job_id]["train_loss"] = metrics.get("train_loss")
            _active_jobs[job_id]["eval_loss"] = metrics.get("eval_loss")

        print(f"[Job {job_id}] Training completed successfully!")
        print(f"[Job {job_id}] Output saved to: {result.get('output_dir')}")

    except Exception as e:
        # Training failed
        import traceback
        error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"[Job {job_id}] Training failed: {error_msg}")

        _active_jobs[job_id]["status"] = JobStatus.FAILED
        _active_jobs[job_id]["error_message"] = error_msg
        _active_jobs[job_id]["completed_at"] = datetime.now()


@router.post("/start", response_model=StartTrainingResponse)
async def start_training(request: StartTrainingRequest, background_tasks: BackgroundTasks):
    """
    Start a new training job.

    Args:
        request: Training configuration and metadata
        background_tasks: FastAPI background tasks

    Returns:
        Job ID and estimated duration
    """
    try:
        # Validate paths
        train_path = Path(request.config.train_data_path)
        if not train_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Training data not found: {train_path}"
            )

        if request.config.val_data_path:
            val_path = Path(request.config.val_data_path)
            if not val_path.exists():
                raise HTTPException(
                    status_code=404, detail=f"Validation data not found: {val_path}"
                )

        # Create output directory
        output_dir = Path(request.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Estimate duration
        estimated_duration = estimate_training_time(request)

        # Create job record
        _active_jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "config": request.config.dict(),
            "experiment_name": request.experiment_name,
            "tags": request.tags,
            "created_at": datetime.now(),
            "started_at": None,
            "updated_at": datetime.now(),
            "completed_at": None,
            "current_epoch": None,
            "total_epochs": request.config.num_epochs,
            "current_step": None,
            "total_steps": None,
            "progress_percentage": 0.0,
            "train_loss": None,
            "eval_loss": None,
            "learning_rate": None,
            "error_message": None,
        }

        # Start training in background thread
        thread = threading.Thread(target=run_training_job, args=(job_id, request))
        thread.daemon = True
        thread.start()
        _job_threads[job_id] = thread

        return StartTrainingResponse(
            success=True,
            job_id=job_id,
            message=f"Training job started successfully",
            estimated_duration_minutes=estimated_duration,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@router.get("/{job_id}/status", response_model=JobProgress)
async def get_job_status(job_id: str):
    """
    Get training job status and progress.

    Args:
        job_id: Training job ID

    Returns:
        Current job progress and metrics
    """
    if job_id not in _active_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job_data = _active_jobs[job_id]

    # Calculate ETA
    eta_seconds = None
    if job_data["status"] == JobStatus.RUNNING and job_data["progress_percentage"]:
        elapsed = (datetime.now() - job_data["started_at"]).total_seconds()
        progress_fraction = job_data["progress_percentage"] / 100.0
        if progress_fraction > 0:
            total_estimated = elapsed / progress_fraction
            eta_seconds = total_estimated - elapsed

    return JobProgress(
        job_id=job_id,
        status=job_data["status"],
        current_epoch=job_data.get("current_epoch"),
        total_epochs=job_data.get("total_epochs"),
        current_step=job_data.get("current_step"),
        total_steps=job_data.get("total_steps"),
        progress_percentage=job_data.get("progress_percentage"),
        eta_seconds=eta_seconds,
        train_loss=job_data.get("train_loss"),
        eval_loss=job_data.get("eval_loss"),
        learning_rate=job_data.get("learning_rate"),
        started_at=job_data.get("started_at"),
        updated_at=job_data.get("updated_at"),
        completed_at=job_data.get("completed_at"),
        error_message=job_data.get("error_message"),
    )


@router.post("/{job_id}/cancel", response_model=CancelJobResponse)
async def cancel_training(job_id: str):
    """
    Cancel a running training job.

    Args:
        job_id: Training job ID

    Returns:
        Cancellation result
    """
    if job_id not in _active_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job_data = _active_jobs[job_id]

    if job_data["status"] not in [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING]:
        return CancelJobResponse(
            success=False,
            job_id=job_id,
            message=f"Cannot cancel job in status: {job_data['status']}",
        )

    # Set status to cancelled
    _active_jobs[job_id]["status"] = JobStatus.CANCELLED
    _active_jobs[job_id]["completed_at"] = datetime.now()

    return CancelJobResponse(
        success=True, job_id=job_id, message="Training job cancelled successfully"
    )


@router.get("/jobs", response_model=list[JobProgress])
async def list_jobs(status: Optional[JobStatus] = None, limit: int = 100):
    """
    List all training jobs, optionally filtered by status.

    Args:
        status: Filter by job status
        limit: Maximum number of jobs to return

    Returns:
        List of job progress objects
    """
    jobs = []

    for job_id, job_data in _active_jobs.items():
        # Filter by status if specified
        if status and job_data["status"] != status:
            continue

        # Calculate ETA
        eta_seconds = None
        if job_data["status"] == JobStatus.RUNNING and job_data["progress_percentage"]:
            elapsed = (datetime.now() - job_data["started_at"]).total_seconds()
            progress_fraction = job_data["progress_percentage"] / 100.0
            if progress_fraction > 0:
                total_estimated = elapsed / progress_fraction
                eta_seconds = total_estimated - elapsed

        jobs.append(
            JobProgress(
                job_id=job_id,
                status=job_data["status"],
                current_epoch=job_data.get("current_epoch"),
                total_epochs=job_data.get("total_epochs"),
                current_step=job_data.get("current_step"),
                total_steps=job_data.get("total_steps"),
                progress_percentage=job_data.get("progress_percentage"),
                eta_seconds=eta_seconds,
                train_loss=job_data.get("train_loss"),
                eval_loss=job_data.get("eval_loss"),
                learning_rate=job_data.get("learning_rate"),
                started_at=job_data.get("started_at"),
                updated_at=job_data.get("updated_at"),
                completed_at=job_data.get("completed_at"),
                error_message=job_data.get("error_message"),
            )
        )

        if len(jobs) >= limit:
            break

    # Sort by creation time (most recent first)
    jobs.sort(key=lambda x: x.started_at or datetime.min, reverse=True)

    return jobs
