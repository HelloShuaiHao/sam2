"""
Experiment tracking and comparison endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models import (
    ExperimentSummary,
    ExperimentDetail,
    CompareExperimentsRequest,
    CompareExperimentsResponse,
    JobStatus,
    TrainingConfig,
)

# Import from training router
from . import training

router = APIRouter()

# In-memory experiment storage (in production, use database)
_experiments: Dict[str, Dict] = {}


def job_to_experiment(job_id: str, job_data: Dict) -> ExperimentDetail:
    """Convert job data to experiment detail."""
    # Calculate duration
    duration_seconds = None
    if job_data.get("completed_at") and job_data.get("started_at"):
        duration_seconds = (
            job_data["completed_at"] - job_data["started_at"]
        ).total_seconds()

    # Load metrics history from checkpoint dir (mock for now)
    metrics_history = {
        "train_loss": [],
        "eval_loss": [],
        "learning_rate": [],
    }

    # Get config
    config_dict = job_data.get("config", {})

    return ExperimentDetail(
        experiment_id=job_id,
        experiment_name=job_data.get("experiment_name"),
        job_id=job_id,
        status=job_data["status"],
        model_name=config_dict.get("model_name", "unknown"),
        config=TrainingConfig(**config_dict),
        metrics_history=metrics_history,
        checkpoints=[],
        best_checkpoint_path=None,
        gpu_type=None,
        peak_memory_mb=None,
        created_at=job_data["created_at"],
        completed_at=job_data.get("completed_at"),
        duration_seconds=duration_seconds,
        tags=job_data.get("tags", []),
    )


@router.get("/", response_model=List[ExperimentSummary])
async def list_experiments(
    status: Optional[JobStatus] = None,
    tag: Optional[str] = None,
    limit: int = 100,
    sort_by: str = "created_at",
):
    """
    List all experiments with optional filtering.

    Args:
        status: Filter by experiment status
        tag: Filter by tag
        limit: Maximum number of experiments to return
        sort_by: Sort field (created_at, duration_seconds, best_eval_loss)

    Returns:
        List of experiment summaries
    """
    experiments = []

    # Get all jobs from training router
    for job_id, job_data in training._active_jobs.items():
        # Apply filters
        if status and job_data["status"] != status:
            continue

        if tag and tag not in job_data.get("tags", []):
            continue

        # Calculate duration
        duration_seconds = None
        if job_data.get("completed_at") and job_data.get("started_at"):
            duration_seconds = (
                job_data["completed_at"] - job_data["started_at"]
            ).total_seconds()

        # Get config
        config_dict = job_data.get("config", {})

        experiments.append(
            ExperimentSummary(
                experiment_id=job_id,
                experiment_name=job_data.get("experiment_name"),
                job_id=job_id,
                status=job_data["status"],
                model_name=config_dict.get("model_name", "unknown"),
                best_eval_loss=job_data.get("eval_loss"),
                best_eval_accuracy=None,  # Not tracked yet
                final_train_loss=job_data.get("train_loss"),
                num_epochs=config_dict.get("num_epochs", 0),
                batch_size=config_dict.get("batch_size", 0),
                learning_rate=config_dict.get("learning_rate", 0.0),
                use_lora=config_dict.get("use_lora", False),
                use_qlora=config_dict.get("use_qlora", False),
                created_at=job_data["created_at"],
                completed_at=job_data.get("completed_at"),
                duration_seconds=duration_seconds,
                tags=job_data.get("tags", []),
            )
        )

        if len(experiments) >= limit:
            break

    # Sort experiments
    if sort_by == "created_at":
        experiments.sort(key=lambda x: x.created_at, reverse=True)
    elif sort_by == "duration_seconds":
        experiments.sort(
            key=lambda x: x.duration_seconds or float("inf"), reverse=True
        )
    elif sort_by == "best_eval_loss":
        experiments.sort(
            key=lambda x: x.best_eval_loss or float("inf"), reverse=False
        )

    return experiments


@router.get("/{experiment_id}", response_model=ExperimentDetail)
async def get_experiment(experiment_id: str):
    """
    Get detailed information about a specific experiment.

    Args:
        experiment_id: Experiment ID (same as job ID)

    Returns:
        Detailed experiment information
    """
    if experiment_id not in training._active_jobs:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")

    job_data = training._active_jobs[experiment_id]
    return job_to_experiment(experiment_id, job_data)


@router.post("/compare", response_model=CompareExperimentsResponse)
async def compare_experiments(request: CompareExperimentsRequest):
    """
    Compare multiple experiments side-by-side.

    Args:
        request: List of experiment IDs and metrics to compare

    Returns:
        Comparison data with best experiment highlighted
    """
    # Validate all experiment IDs exist
    experiments = []
    for exp_id in request.experiment_ids:
        if exp_id not in training._active_jobs:
            raise HTTPException(status_code=404, detail=f"Experiment not found: {exp_id}")

        job_data = training._active_jobs[exp_id]

        # Calculate duration
        duration_seconds = None
        if job_data.get("completed_at") and job_data.get("started_at"):
            duration_seconds = (
                job_data["completed_at"] - job_data["started_at"]
            ).total_seconds()

        # Get config
        config_dict = job_data.get("config", {})

        experiments.append(
            ExperimentSummary(
                experiment_id=exp_id,
                experiment_name=job_data.get("experiment_name"),
                job_id=exp_id,
                status=job_data["status"],
                model_name=config_dict.get("model_name", "unknown"),
                best_eval_loss=job_data.get("eval_loss"),
                best_eval_accuracy=None,
                final_train_loss=job_data.get("train_loss"),
                num_epochs=config_dict.get("num_epochs", 0),
                batch_size=config_dict.get("batch_size", 0),
                learning_rate=config_dict.get("learning_rate", 0.0),
                use_lora=config_dict.get("use_lora", False),
                use_qlora=config_dict.get("use_qlora", False),
                created_at=job_data["created_at"],
                completed_at=job_data.get("completed_at"),
                duration_seconds=duration_seconds,
                tags=job_data.get("tags", []),
            )
        )

    # Build comparison table
    comparison_table: Dict[str, Dict[str, Any]] = {}

    for metric in request.metrics:
        comparison_table[metric] = {}

        for exp in experiments:
            if metric == "eval_loss":
                comparison_table[metric][exp.experiment_id] = exp.best_eval_loss
            elif metric == "train_loss":
                comparison_table[metric][exp.experiment_id] = exp.final_train_loss
            elif metric == "duration":
                comparison_table[metric][exp.experiment_id] = exp.duration_seconds
            else:
                # Unknown metric
                comparison_table[metric][exp.experiment_id] = None

    # Find best experiment (lowest eval loss)
    best_experiment_id = None
    best_eval_loss = float("inf")

    for exp in experiments:
        if exp.best_eval_loss and exp.best_eval_loss < best_eval_loss:
            best_eval_loss = exp.best_eval_loss
            best_experiment_id = exp.experiment_id

    return CompareExperimentsResponse(
        experiments=experiments,
        comparison_table=comparison_table,
        best_experiment_id=best_experiment_id,
    )


@router.delete("/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """
    Delete an experiment and its associated data.

    Args:
        experiment_id: Experiment ID to delete

    Returns:
        Deletion confirmation
    """
    if experiment_id not in training._active_jobs:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")

    job_data = training._active_jobs[experiment_id]

    # Don't allow deletion of running jobs
    if job_data["status"] == JobStatus.RUNNING:
        raise HTTPException(
            status_code=400, detail="Cannot delete running experiment. Cancel it first."
        )

    # Delete checkpoint directory if it exists
    output_dir = Path(job_data["config"]["output_dir"])
    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)

    # Remove from active jobs
    del training._active_jobs[experiment_id]

    return {"success": True, "message": f"Experiment {experiment_id} deleted"}


@router.get("/{experiment_id}/metrics")
async def get_experiment_metrics(experiment_id: str):
    """
    Get detailed metrics history for an experiment.

    Args:
        experiment_id: Experiment ID

    Returns:
        Metrics history with timestamps
    """
    if experiment_id not in training._active_jobs:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")

    job_data = training._active_jobs[experiment_id]

    # In a real implementation, load from checkpoint/tensorboard logs
    # For now, return mock data
    metrics = {
        "train_loss": [0.8, 0.6, 0.5, 0.45, 0.42],
        "eval_loss": [0.9, 0.7, 0.6, 0.55, 0.52],
        "learning_rate": [2e-4, 1.8e-4, 1.5e-4, 1.2e-4, 1e-4],
        "steps": [0, 100, 200, 300, 400],
    }

    return {
        "experiment_id": experiment_id,
        "metrics": metrics,
        "current_epoch": job_data.get("current_epoch"),
        "total_epochs": job_data.get("total_epochs"),
    }
