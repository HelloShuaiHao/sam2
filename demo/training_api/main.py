"""
FastAPI application for LLM fine-tuning pipeline.

This API provides endpoints for:
- Data preparation (convert, validate, split)
- Training job management (start, monitor, cancel)
- Experiment tracking and comparison
- Model export and download
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from training_api.routes import data_prep, training, experiments, export as export_routes

# Create FastAPI app
app = FastAPI(
    title="LLM Fine-tuning API",
    description="API for end-to-end LLM fine-tuning pipeline with SAM2 annotations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "status": "healthy",
        "service": "LLM Fine-tuning API",
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": Path(__file__).stat().st_mtime,
    }


# Include routers
app.include_router(data_prep.router, prefix="/api/data", tags=["Data Preparation"])
app.include_router(training.router, prefix="/api/train", tags=["Training"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["Experiments"])
app.include_router(export_routes.router, prefix="/api/export", tags=["Export"])


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "type": type(exc).__name__,
        },
    )


def main():
    """Run the FastAPI application."""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info",
    )


if __name__ == "__main__":
    main()
