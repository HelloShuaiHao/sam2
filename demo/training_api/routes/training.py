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


# =============================================================================
# Path Mapping Utility
# =============================================================================

def map_docker_path(path_str: str) -> str:
    """Map Docker container paths to local filesystem paths.

    This allows the API to work seamlessly whether running in Docker or locally.

    Args:
        path_str: Path string (may be Docker container path or local path)

    Returns:
        Mapped local filesystem path

    Examples:
        /app/output -> /path/to/project/demo/training/output
        /data/exports -> /path/to/project/demo/data/exports
        /local/path -> /local/path (unchanged)
    """
    # Get project root (4 levels up from this file)
    project_root = Path(__file__).parent.parent.parent.parent

    # Map Docker paths to local paths
    if path_str.startswith("/app/"):
        # /app/output -> demo/training/output
        relative_path = path_str.replace("/app/", "demo/training/")
        return str(project_root / relative_path)
    elif path_str.startswith("/data/"):
        # /data/exports -> demo/data/exports
        relative_path = path_str.replace("/data/", "demo/data/")
        return str(project_root / relative_path)
    else:
        # Already a local path
        return path_str


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


def _check_cancellation(job_id: str) -> bool:
    """Check if job has been cancelled.

    Returns:
        True if cancelled, False otherwise
    """
    if job_id not in _active_jobs:
        return True
    return _active_jobs[job_id]["status"] == JobStatus.CANCELLED


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

        # Check cancellation before expensive operations
        if _check_cancellation(job_id):
            print(f"[Job {job_id}] Job cancelled before starting")
            return

        # Create trainer and setup model
        print(f"[Job {job_id}] Creating trainer and loading model...")
        trainer = LoRATrainer(core_config)

        # Check cancellation before model download/load
        if _check_cancellation(job_id):
            print(f"[Job {job_id}] Job cancelled before model load")
            return

        print(f"[Job {job_id}] Loading model with QLoRA settings...")
        print(f"[Job {job_id}] Note: Model download may take 5-30 minutes on first run")
        print(f"[Job {job_id}] You can cancel anytime - downloads will resume on retry")

        # Setup model with cancellation checks
        # Note: HuggingFace downloads have built-in retry, but we can't interrupt them
        # The model will still download, but we'll stop immediately after
        trainer.setup()

        # Check cancellation after model load
        if _check_cancellation(job_id):
            print(f"[Job {job_id}] Job cancelled after model load")
            return

        print(f"[Job {job_id}] Model loaded successfully!")

        # Load datasets - create simple dataset from JSONL files
        print(f"[Job {job_id}] Loading datasets...")
        from torch.utils.data import Dataset
        import json
        from PIL import Image

        class SimpleVLMDataset(Dataset):
            """Simple dataset that loads JSONL data for vision-language models."""
            def __init__(self, jsonl_path: str, base_path: str = None):
                self.data = []
                self.base_path = Path(base_path) if base_path else Path(jsonl_path).parent
                with open(jsonl_path, 'r') as f:
                    for line in f:
                        self.data.append(json.loads(line.strip()))

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]

                # Load image if path is provided
                if 'image' in item:
                    image_path = self.base_path / item['image']
                    if image_path.exists():
                        item['image'] = Image.open(image_path).convert('RGB')

                # Return the item - will be processed by custom collator
                return item

        # Custom data collator for vision-language models
        from dataclasses import dataclass
        from typing import List, Dict, Any
        import torch
        from transformers import DataCollatorWithPadding

        @dataclass
        class VLMDataCollator:
            """Custom data collator for vision-language model training."""
            processor: Any  # AutoProcessor or tokenizer

            def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
                """Process a batch of vision-language samples.

                Args:
                    features: List of samples from dataset, each containing:
                        - image: PIL Image or image path
                        - conversations: List of conversation turns

                Returns:
                    Batch dictionary ready for model training
                """
                batch = {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                }

                # Check if we have images in the batch
                has_images = any('image' in f for f in features)
                if has_images:
                    batch["pixel_values"] = []

                for feature in features:
                    # Format conversation text
                    conversations = feature.get('conversations', [])
                    text = self._format_conversations(conversations)

                    # Tokenize text
                    if has_images and hasattr(self.processor, 'tokenizer'):
                        # Use processor's tokenizer for vision-language models
                        encoding = self.processor.tokenizer(
                            text,
                            truncation=True,
                            max_length=2048,
                            return_tensors="pt"
                        )
                    else:
                        # Use tokenizer directly
                        encoding = self.processor(
                            text,
                            truncation=True,
                            max_length=2048,
                            return_tensors="pt"
                        )

                    batch["input_ids"].append(encoding["input_ids"][0])
                    batch["attention_mask"].append(encoding["attention_mask"][0])

                    # Labels are the same as input_ids for causal LM
                    batch["labels"].append(encoding["input_ids"][0].clone())

                    # Process image if present
                    if has_images and 'image' in feature:
                        if hasattr(self.processor, 'image_processor'):
                            image_inputs = self.processor.image_processor(
                                feature['image'],
                                return_tensors="pt"
                            )
                            batch["pixel_values"].append(image_inputs["pixel_values"][0])
                        else:
                            # Fallback: create dummy pixel values if image processor not available
                            batch["pixel_values"].append(torch.zeros(3, 336, 336))

                # Pad sequences to same length
                max_len = max(len(ids) for ids in batch["input_ids"])
                pad_token_id = self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else self.processor.pad_token_id

                for i in range(len(batch["input_ids"])):
                    padding_length = max_len - len(batch["input_ids"][i])
                    if padding_length > 0:
                        batch["input_ids"][i] = torch.cat([
                            batch["input_ids"][i],
                            torch.full((padding_length,), pad_token_id, dtype=torch.long)
                        ])
                        batch["attention_mask"][i] = torch.cat([
                            batch["attention_mask"][i],
                            torch.zeros(padding_length, dtype=torch.long)
                        ])
                        batch["labels"][i] = torch.cat([
                            batch["labels"][i],
                            torch.full((padding_length,), -100, dtype=torch.long)  # -100 is ignored in loss
                        ])

                # Stack into tensors
                batch["input_ids"] = torch.stack(batch["input_ids"])
                batch["attention_mask"] = torch.stack(batch["attention_mask"])
                batch["labels"] = torch.stack(batch["labels"])

                if has_images and len(batch["pixel_values"]) > 0:
                    batch["pixel_values"] = torch.stack(batch["pixel_values"])

                return batch

            def _format_conversations(self, conversations: List[Dict]) -> str:
                """Format conversation list into training text.

                Args:
                    conversations: List of {from: str, value: str} dicts

                Returns:
                    Formatted conversation text
                """
                formatted_text = ""
                for conv in conversations:
                    role = conv.get('from', 'human')
                    value = conv.get('value', '')

                    if role == 'human' or role == 'user':
                        formatted_text += f"USER: {value}\n"
                    elif role == 'gpt' or role == 'assistant':
                        formatted_text += f"ASSISTANT: {value}\n"
                    else:
                        formatted_text += f"{value}\n"

                return formatted_text.strip()

        train_dataset = SimpleVLMDataset(config.config.train_data_path)
        eval_dataset = SimpleVLMDataset(config.config.val_data_path) if config.config.val_data_path else None

        # Create custom data collator
        data_collator = VLMDataCollator(processor=trainer.tokenizer)

        # Check cancellation before training
        if _check_cancellation(job_id):
            print(f"[Job {job_id}] Job cancelled before training")
            return

        print(f"[Job {job_id}] Starting actual training...")
        print(f"[Job {job_id}] Train samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"[Job {job_id}] Eval samples: {len(eval_dataset)}")

        # Run actual training
        # Note: Once training starts, it's harder to cancel mid-epoch
        # The HuggingFace Trainer will handle KeyboardInterrupt for clean shutdown
        result = trainer.train(train_dataset, eval_dataset, data_collator=data_collator)

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
        # Check if job was cancelled (not a real failure)
        if _check_cancellation(job_id):
            print(f"[Job {job_id}] Job was cancelled (exception during cancellation: {str(e)})")
            # Status already set to CANCELLED by cancel endpoint
            if _active_jobs[job_id]["completed_at"] is None:
                _active_jobs[job_id]["completed_at"] = datetime.now()
        else:
            # Training failed due to real error
            import traceback
            error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"[Job {job_id}] Training failed: {error_msg}")

            _active_jobs[job_id]["status"] = JobStatus.FAILED
            _active_jobs[job_id]["error_message"] = error_msg
            _active_jobs[job_id]["completed_at"] = datetime.now()
    finally:
        # Cleanup: Remove from active threads
        if job_id in _job_threads:
            print(f"[Job {job_id}] Cleaning up thread")
            del _job_threads[job_id]

        # Force GPU memory cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"[Job {job_id}] GPU cache cleared")
        except Exception as cleanup_error:
            print(f"[Job {job_id}] Cleanup warning: {cleanup_error}")


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
        # Map Docker paths to local paths
        request.config.train_data_path = map_docker_path(request.config.train_data_path)
        if request.config.val_data_path:
            request.config.val_data_path = map_docker_path(request.config.val_data_path)
        request.config.output_dir = map_docker_path(request.config.output_dir)

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
