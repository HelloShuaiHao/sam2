# Multi-stage Dockerfile for IDoctor Fine-tuned Model Deployment
# Optimized for production inference with GPU support

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Set environment variables
ENV PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

# Stage 2: Dependencies
FROM base as dependencies

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Transformers, PEFT, and related packages
RUN pip install --no-cache-dir \
    transformers>=4.36.0 \
    peft>=0.7.0 \
    accelerate>=0.25.0 \
    bitsandbytes>=0.41.0 \
    sentencepiece>=0.1.99 \
    pillow>=10.0.0 \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    python-multipart>=0.0.6

# Stage 3: Application
FROM dependencies as application

# Set working directory
WORKDIR /app

# Copy model files (to be replaced with actual model path during build)
# Usage: docker build --build-arg MODEL_PATH=./my_model .
ARG MODEL_PATH=./model
COPY ${MODEL_PATH} /app/model

# Copy inference code
COPY inference_api.py /app/
COPY inference_utils.py /app/

# Create necessary directories
RUN mkdir -p /app/uploads /app/outputs

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run inference API
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000"]


# =============================================================================
# Build Instructions:
# =============================================================================
#
# 1. Basic build:
#    docker build -t idoctor-model:latest .
#
# 2. Build with specific model:
#    docker build --build-arg MODEL_PATH=./exported_model -t idoctor-model:v1 .
#
# 3. Build for CPU only (smaller image):
#    docker build -f Dockerfile.cpu -t idoctor-model:cpu .
#
# =============================================================================
# Run Instructions:
# =============================================================================
#
# 1. Run with GPU:
#    docker run --gpus all -p 8000:8000 idoctor-model:latest
#
# 2. Run with specific GPU:
#    docker run --gpus '"device=0"' -p 8000:8000 idoctor-model:latest
#
# 3. Run with environment variables:
#    docker run --gpus all -p 8000:8000 \
#      -e MODEL_PATH=/app/model \
#      -e MAX_LENGTH=512 \
#      idoctor-model:latest
#
# 4. Run with volume mount (for dynamic model loading):
#    docker run --gpus all -p 8000:8000 \
#      -v /path/to/model:/app/model \
#      idoctor-model:latest
#
# =============================================================================
# API Usage:
# =============================================================================
#
# Once running, access the API at:
# - Interactive docs: http://localhost:8000/docs
# - Health check: http://localhost:8000/health
# - Inference: POST http://localhost:8000/predict
#
# Example curl:
#   curl -X POST "http://localhost:8000/predict" \
#     -F "image=@medical_image.jpg" \
#     -F "prompt=Segment all visible structures"
#
# =============================================================================
