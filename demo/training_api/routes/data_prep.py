"""
Data preparation endpoints for SAM2 annotation conversion, validation, and splitting.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
import sys
from pathlib import Path
import json
import shutil
import uuid

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from training_api.models import (
    ConvertRequest,
    ConvertResponse,
    ValidateRequest,
    ValidationReport,
    SplitRequest,
    SplitResponse,
)

# Import core modules
from training.core.data_converter.sam2_parser import SAM2Parser
from training.core.data_converter.llava_converter import LLaVAConverter
from training.core.data_converter.huggingface_converter import HuggingFaceConverter
from training.core.validation.validator import Validator
from training.core.validation.report_generator import ReportGenerator
from training.core.data_splitter.split_config import SplitConfig, SplitStrategy
from training.core.data_splitter.stratified_splitter import StratifiedSplitter
from training.core.data_splitter.temporal_splitter import TemporalSplitter
from training.core.data_splitter.random_splitter import RandomSplitter

router = APIRouter()

# Upload directory for SAM2 exports
UPLOAD_DIR = Path("/app/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_sam2_export(file: UploadFile = File(...)):
    """
    Upload SAM2 export ZIP file.

    Args:
        file: ZIP file containing SAM2 annotations

    Returns:
        Upload result with file path
    """
    try:
        # Validate file type
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="Only ZIP files are supported")

        # Generate unique filename
        file_id = str(uuid.uuid4())[:8]
        filename = f"{file_id}_{file.filename}"
        file_path = UPLOAD_DIR / filename

        # Save uploaded file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        return {
            "success": True,
            "file_path": str(file_path),
            "filename": filename,
            "file_size_mb": round(file_size_mb, 2),
            "message": f"File uploaded successfully: {filename}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/convert", response_model=ConvertResponse)
async def convert_sam2_data(request: ConvertRequest):
    """
    Convert SAM2 annotation export to training format.

    Args:
        request: Conversion request with source path and target format

    Returns:
        Conversion result with statistics
    """
    try:
        # Validate input path
        sam2_path = Path(request.sam2_zip_path)
        if not sam2_path.exists():
            raise HTTPException(status_code=404, detail=f"SAM2 export not found: {sam2_path}")

        # Create output directory
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse SAM2 export
        parser = SAM2Parser()
        sam2_data = parser.parse(str(sam2_path))

        # Convert to target format
        warnings = []
        if request.target_format == "llava":
            converter = LLaVAConverter()
            output_path = output_dir / "llava_format.jsonl"
            converter.convert(sam2_data, str(output_path))
            num_samples = len(sam2_data.get("annotations", []))

        elif request.target_format == "huggingface":
            converter = HuggingFaceConverter()
            output_path = output_dir / "huggingface_dataset"
            dataset = converter.convert(sam2_data, str(output_path))
            num_samples = len(dataset)

        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported format: {request.target_format}"
            )

        return ConvertResponse(
            success=True,
            output_dir=str(output_dir),
            num_samples=num_samples,
            message=f"Successfully converted {num_samples} samples to {request.target_format} format",
            warnings=warnings,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")


@router.post("/validate", response_model=ValidationReport)
async def validate_dataset(request: ValidateRequest):
    """
    Validate dataset quality and generate report.

    Args:
        request: Validation request with dataset path

    Returns:
        Validation report with errors, warnings, and recommendations
    """
    try:
        # Validate input path
        data_path = Path(request.data_path)
        if not data_path.exists():
            raise HTTPException(status_code=404, detail=f"Dataset not found: {data_path}")

        # Load dataset
        if request.format_type == "llava":
            # Load JSONL
            samples = []
            with open(data_path, "r") as f:
                for line in f:
                    samples.append(json.loads(line))
        else:
            # Load HuggingFace dataset
            from datasets import load_from_disk

            dataset = load_from_disk(str(data_path))
            samples = list(dataset)

        # Run validation
        validator = Validator()
        validation_results = validator.validate(samples)

        # Generate report
        report_gen = ReportGenerator()
        report = report_gen.generate(validation_results)

        return ValidationReport(
            passed=report["passed"],
            num_errors=report["num_errors"],
            num_warnings=report["num_warnings"],
            errors=report["errors"],
            warnings=report["warnings"],
            statistics=report["statistics"],
            recommendations=report["recommendations"],
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.post("/split", response_model=SplitResponse)
async def split_dataset(request: SplitRequest):
    """
    Split dataset into train/val/test sets.

    Args:
        request: Split request with strategy and ratios

    Returns:
        Split result with paths and sample counts
    """
    try:
        # Validate input path
        data_path = Path(request.data_path)
        if not data_path.exists():
            raise HTTPException(status_code=404, detail=f"Dataset not found: {data_path}")

        # Validate ratios sum to 1.0
        total_ratio = request.train_ratio + request.val_ratio + request.test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise HTTPException(
                status_code=400,
                detail=f"Ratios must sum to 1.0, got {total_ratio}",
            )

        # Create output directory
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create split configuration
        split_config = SplitConfig(
            train_ratio=request.train_ratio,
            val_ratio=request.val_ratio,
            test_ratio=request.test_ratio,
            random_seed=request.random_seed,
        )

        # Load dataset
        with open(data_path, "r") as f:
            samples = [json.loads(line) for line in f]

        # Choose splitter based on strategy
        if request.strategy == "stratified":
            splitter = StratifiedSplitter(split_config)
        elif request.strategy == "temporal":
            splitter = TemporalSplitter(split_config)
        elif request.strategy == "random":
            splitter = RandomSplitter(split_config)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown strategy: {request.strategy}"
            )

        # Perform split
        train_data, val_data, test_data = splitter.split(samples)

        # Save splits
        train_path = output_dir / "train.jsonl"
        val_path = output_dir / "val.jsonl"
        test_path = output_dir / "test.jsonl"

        with open(train_path, "w") as f:
            for sample in train_data:
                f.write(json.dumps(sample) + "\n")

        with open(val_path, "w") as f:
            for sample in val_data:
                f.write(json.dumps(sample) + "\n")

        with open(test_path, "w") as f:
            for sample in test_data:
                f.write(json.dumps(sample) + "\n")

        return SplitResponse(
            success=True,
            train_path=str(train_path),
            val_path=str(val_path),
            test_path=str(test_path),
            train_samples=len(train_data),
            val_samples=len(val_data),
            test_samples=len(test_data),
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Split failed: {str(e)}")


@router.get("/download_validation_report/{report_id}")
async def download_validation_report(report_id: str):
    """
    Download validation report as Markdown file.

    Args:
        report_id: Report identifier

    Returns:
        Markdown file for download
    """
    # In a real implementation, you would store reports in a database/storage
    # For now, this is a placeholder
    report_path = Path(f"/tmp/validation_report_{report_id}.md")

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(
        path=report_path,
        media_type="text/markdown",
        filename=f"validation_report_{report_id}.md",
    )
