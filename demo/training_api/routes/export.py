"""
Model export and download endpoints.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import shutil
import uuid

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from training_api.models import (
    ExportRequest,
    ExportResponse,
    ExportFormat,
    DownloadInfo,
    JobStatus,
)

# Import from training router
from . import training

# Import core export modules
from training.core.export.hf_exporter import HuggingFaceExporter
from training.core.export.lora_exporter import LoRAExporter
from training.core.export.model_card_generator import ModelCardGenerator

router = APIRouter()

# In-memory export storage (in production, use S3/cloud storage)
_exports: dict[str, dict] = {}


def get_directory_size_mb(path: Path) -> float:
    """Calculate directory size in MB."""
    total_size = 0
    for item in path.rglob("*"):
        if item.is_file():
            total_size += item.stat().st_size
    return total_size / (1024 * 1024)  # Convert to MB


@router.post("/{job_id}", response_model=ExportResponse)
async def export_model(job_id: str, request: ExportRequest):
    """
    Export trained model to specified format.

    Args:
        job_id: Training job ID
        request: Export configuration

    Returns:
        Export result with path and metadata
    """
    try:
        # Validate job exists
        if job_id not in training._active_jobs:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        job_data = training._active_jobs[job_id]

        # Check job is completed
        if job_data["status"] != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot export model from job with status: {job_data['status']}",
            )

        # Get checkpoint directory
        checkpoint_dir = Path(job_data["config"]["output_dir"])
        if not checkpoint_dir.exists():
            raise HTTPException(
                status_code=404, detail=f"Checkpoint directory not found: {checkpoint_dir}"
            )

        # Determine output directory
        if request.output_dir:
            export_dir = Path(request.output_dir)
        else:
            export_dir = checkpoint_dir.parent / f"export_{job_id[:8]}"

        export_dir.mkdir(parents=True, exist_ok=True)

        # Export based on format
        model_card_path = None

        if request.export_format == ExportFormat.HUGGINGFACE:
            # Export full model in HuggingFace format
            exporter = HuggingFaceExporter()
            export_path = export_dir / "huggingface_model"
            export_path.mkdir(exist_ok=True)

            exporter.export(
                checkpoint_path=str(checkpoint_dir),
                output_path=str(export_path),
                merge_lora=request.merge_adapters,
            )

        elif request.export_format == ExportFormat.LORA_ADAPTER:
            # Export only LoRA adapters
            if not job_data["config"].get("use_lora"):
                raise HTTPException(
                    status_code=400,
                    detail="Cannot export LoRA adapters from non-LoRA training",
                )

            exporter = LoRAExporter()
            export_path = export_dir / "lora_adapters"
            export_path.mkdir(exist_ok=True)

            exporter.export(
                checkpoint_path=str(checkpoint_dir), output_path=str(export_path)
            )

        elif request.export_format in [ExportFormat.ONNX, ExportFormat.TFLITE]:
            # Future formats - not implemented yet
            raise HTTPException(
                status_code=501, detail=f"Export format {request.export_format} not implemented yet"
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unknown export format: {request.export_format}")

        # Generate model card
        if request.generate_model_card:
            card_generator = ModelCardGenerator()
            card_path = export_dir / "MODEL_CARD.md"

            # Prepare training info
            training_info = {
                "model_name": job_data["config"]["model_name"],
                "training_config": job_data["config"],
                "final_metrics": {
                    "train_loss": job_data.get("train_loss"),
                    "eval_loss": job_data.get("eval_loss"),
                },
                "training_duration": None,
                "num_parameters": None,  # Would need to calculate
            }

            if job_data.get("started_at") and job_data.get("completed_at"):
                duration = (job_data["completed_at"] - job_data["started_at"]).total_seconds()
                training_info["training_duration"] = duration

            card_content = card_generator.generate(
                training_info=training_info, output_path=str(card_path)
            )

            model_card_path = str(card_path)

        # Calculate export size
        file_size_mb = get_directory_size_mb(export_path)

        # Store export info
        export_id = str(uuid.uuid4())
        _exports[export_id] = {
            "export_id": export_id,
            "job_id": job_id,
            "export_path": str(export_path),
            "export_format": request.export_format,
            "file_size_mb": file_size_mb,
            "created_at": datetime.now(),
        }

        return ExportResponse(
            success=True,
            export_path=str(export_path),
            export_format=request.export_format,
            file_size_mb=file_size_mb,
            message=f"Model exported successfully to {export_path}",
            model_card_path=model_card_path,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/{job_id}/download", response_model=DownloadInfo)
async def get_download_info(job_id: str):
    """
    Get download information for exported model.

    Args:
        job_id: Training job ID

    Returns:
        Download URL and metadata
    """
    # Find export by job_id
    export_data = None
    for export_id, data in _exports.items():
        if data["job_id"] == job_id:
            export_data = data
            break

    if not export_data:
        raise HTTPException(status_code=404, detail=f"No export found for job: {job_id}")

    # Generate download URL (in production, use signed URLs)
    download_url = f"/api/export/download_file/{export_data['export_id']}"

    # Set expiration (7 days from now)
    expires_at = datetime.now() + timedelta(days=7)

    return DownloadInfo(
        download_url=download_url,
        filename=f"model_{job_id[:8]}.zip",
        file_size_mb=export_data["file_size_mb"],
        expires_at=expires_at,
    )


@router.get("/download_file/{export_id}")
async def download_export(export_id: str):
    """
    Download exported model as ZIP file.

    Args:
        export_id: Export ID

    Returns:
        ZIP file containing exported model
    """
    if export_id not in _exports:
        raise HTTPException(status_code=404, detail=f"Export not found: {export_id}")

    export_data = _exports[export_id]
    export_path = Path(export_data["export_path"])

    if not export_path.exists():
        raise HTTPException(status_code=404, detail=f"Export files not found: {export_path}")

    # Create ZIP archive
    zip_path = export_path.parent / f"{export_path.name}.zip"

    if not zip_path.exists():
        # Create ZIP
        shutil.make_archive(
            str(export_path), "zip", export_path.parent, export_path.name
        )

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"model_{export_data['job_id'][:8]}.zip",
    )


@router.get("/list")
async def list_exports():
    """
    List all available exports.

    Returns:
        List of export metadata
    """
    exports_list = []

    for export_id, data in _exports.items():
        exports_list.append(
            {
                "export_id": export_id,
                "job_id": data["job_id"],
                "export_format": data["export_format"],
                "file_size_mb": data["file_size_mb"],
                "created_at": data["created_at"],
                "download_url": f"/api/export/download_file/{export_id}",
            }
        )

    # Sort by creation time (most recent first)
    exports_list.sort(key=lambda x: x["created_at"], reverse=True)

    return {"exports": exports_list, "total": len(exports_list)}


@router.delete("/{export_id}")
async def delete_export(export_id: str):
    """
    Delete an exported model.

    Args:
        export_id: Export ID

    Returns:
        Deletion confirmation
    """
    if export_id not in _exports:
        raise HTTPException(status_code=404, detail=f"Export not found: {export_id}")

    export_data = _exports[export_id]
    export_path = Path(export_data["export_path"])

    # Delete files
    if export_path.exists():
        shutil.rmtree(export_path)

    # Delete ZIP if exists
    zip_path = export_path.parent / f"{export_path.name}.zip"
    if zip_path.exists():
        zip_path.unlink()

    # Remove from registry
    del _exports[export_id]

    return {"success": True, "message": f"Export {export_id} deleted"}
