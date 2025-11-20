"""FastAPI inference server for deployed models.

Production-ready REST API for serving fine-tuned vision-language models.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import io

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn

from inference_example import ModelInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH = Path("/app/model")  # Default path in Docker container
UPLOAD_DIR = Path("/app/uploads")
OUTPUT_DIR = Path("/app/outputs")

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Initialize FastAPI App
# =============================================================================

app = FastAPI(
    title="IDoctor Model Inference API",
    description="REST API for medical image segmentation using fine-tuned vision-language models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Global Model Instance
# =============================================================================

model_inference: Optional[ModelInference] = None


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model_inference

    logger.info("=" * 80)
    logger.info("Starting IDoctor Inference API...")
    logger.info("=" * 80)

    try:
        # Check if model exists
        if not MODEL_PATH.exists():
            logger.warning(f"Model path does not exist: {MODEL_PATH}")
            logger.warning("API will start but inference will fail until model is loaded")
            return

        logger.info(f"Loading model from {MODEL_PATH}...")
        model_inference = ModelInference(
            model_path=MODEL_PATH,
            use_4bit=True,  # Use 4-bit for memory efficiency
            device="auto"
        )

        logger.info("✅ Model loaded successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error("API will start but inference endpoints will not work")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down IDoctor Inference API...")


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "IDoctor Model Inference API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_inference is not None,
        "model_path": str(MODEL_PATH)
    }


@app.get("/model/info")
async def model_info():
    """Get model information."""
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_path": str(model_inference.model_path),
        "use_4bit": model_inference.use_4bit,
        "device": model_inference.device,
        "status": "ready"
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="Medical image file (JPG, PNG, etc.)"),
    prompt: str = Form("Describe and segment all visible structures in this medical image.",
                       description="Text prompt for the model"),
    max_length: int = Form(512, description="Maximum generation length"),
    temperature: float = Form(0.7, description="Sampling temperature (0.0 = greedy, 1.0 = random)")
):
    """Run inference on a medical image.

    Args:
        image: Uploaded image file
        prompt: Text prompt describing the task
        max_length: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        JSON with prediction results

    Example:
        curl -X POST "http://localhost:8000/predict" \\
          -F "image=@medical_scan.jpg" \\
          -F "prompt=Identify and segment the tumor" \\
          -F "max_length=256"
    """
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read and validate image
        logger.info(f"Received image: {image.filename}")
        image_bytes = await image.read()

        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()  # Verify it's a valid image
            img = Image.open(io.BytesIO(image_bytes))  # Re-open after verify
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

        # Save temporarily
        temp_path = UPLOAD_DIR / f"temp_{image.filename}"
        img.save(temp_path)
        logger.info(f"Saved to: {temp_path}")

        # Run inference
        logger.info(f"Running inference with prompt: {prompt}")
        result = model_inference.predict(
            image_path=temp_path,
            prompt=prompt,
            max_length=max_length,
            temperature=temperature
        )

        # Clean up temporary file
        temp_path.unlink()

        logger.info("✓ Inference complete")

        return JSONResponse(content={
            "success": True,
            "result": result,
            "metadata": {
                "filename": image.filename,
                "prompt": prompt,
                "max_length": max_length,
                "temperature": temperature
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(
    images: list[UploadFile] = File(..., description="Multiple image files"),
    prompts: Optional[str] = Form(None, description="Comma-separated prompts (one per image)"),
    max_length: int = Form(512),
    temperature: float = Form(0.7)
):
    """Run batch inference on multiple images.

    Args:
        images: List of uploaded image files
        prompts: Comma-separated prompts (optional, uses default if not provided)
        max_length: Maximum generation length
        temperature: Sampling temperature

    Returns:
        JSON with list of prediction results
    """
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(images) == 0:
        raise HTTPException(status_code=400, detail="No images provided")

    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")

    try:
        # Parse prompts
        if prompts:
            prompt_list = [p.strip() for p in prompts.split(",")]
            if len(prompt_list) != len(images):
                raise HTTPException(
                    status_code=400,
                    detail=f"Number of prompts ({len(prompt_list)}) must match number of images ({len(images)})"
                )
        else:
            prompt_list = ["Segment all visible structures."] * len(images)

        # Save images temporarily
        temp_paths = []
        for img_file in images:
            image_bytes = await img_file.read()
            img = Image.open(io.BytesIO(image_bytes))
            temp_path = UPLOAD_DIR / f"batch_{img_file.filename}"
            img.save(temp_path)
            temp_paths.append(temp_path)

        # Run batch inference
        logger.info(f"Running batch inference on {len(images)} images...")
        results = model_inference.batch_predict(
            image_paths=temp_paths,
            prompts=prompt_list,
            max_length=max_length,
            temperature=temperature
        )

        # Clean up
        for path in temp_paths:
            path.unlink()

        logger.info("✓ Batch inference complete")

        return JSONResponse(content={
            "success": True,
            "count": len(results),
            "results": results
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")


@app.post("/model/reload")
async def reload_model(
    model_path: str = Form(None, description="Optional new model path")
):
    """Reload the model (useful for model updates).

    Args:
        model_path: Optional path to new model (uses default if not provided)

    Returns:
        JSON with reload status
    """
    global model_inference

    try:
        path = Path(model_path) if model_path else MODEL_PATH

        logger.info(f"Reloading model from {path}...")
        model_inference = ModelInference(
            model_path=path,
            use_4bit=True,
            device="auto"
        )

        logger.info("✓ Model reloaded successfully")

        return JSONResponse(content={
            "success": True,
            "message": "Model reloaded successfully",
            "model_path": str(path)
        })

    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IDoctor Inference API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind to")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model directory")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")

    args = parser.parse_args()

    # Update model path if provided
    if args.model_path:
        MODEL_PATH = Path(args.model_path)

    # Run server
    uvicorn.run(
        "inference_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
