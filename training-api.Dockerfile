# Training API Dockerfile
# For LLM fine-tuning pipeline

# Use lightweight base image
FROM ubuntu:22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the existing conda environment from host
# Context is parent dir now, so path is anaconda3/envs/...
COPY anaconda3/envs/py39-torch201-cuda118 /opt/conda/envs/torch-env

# Activate environment by default
ENV PATH=/opt/conda/envs/torch-env/bin:$PATH
ENV CONDA_DEFAULT_ENV=torch-env

# Copy training API code (add sam2/ prefix since context is parent dir)
COPY sam2/demo/training_api/ /app/training_api/
COPY sam2/demo/training/ /app/training/

# Install Python dependencies
COPY sam2/demo/training_api/requirements.txt /app/

# PyTorch already in conda env, just install other packages from Aliyun
# Fast because skipping the huge PyTorch download!
# Use full path to pip from conda env
RUN /opt/conda/envs/torch-env/bin/pip install --no-cache-dir --default-timeout=1000 \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    pydantic==2.5.0 \
    email-validator==2.1.0 \
    python-jose[cryptography]==3.3.0 \
    passlib[bcrypt]==1.7.4 \
    celery==5.3.4 \
    redis==5.0.1 \
    transformers>=4.35.0 \
    datasets>=2.14.0 \
    peft>=0.6.0 \
    sentencepiece>=0.1.99 \
    protobuf>=3.20.0 \
    tensorboard>=2.15.0 \
    wandb>=0.16.0 \
    numpy>=1.24.0 \
    pillow>=10.0.0 \
    hf-transfer>=0.1.4

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application
CMD ["/opt/conda/envs/torch-env/bin/uvicorn", "training_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
