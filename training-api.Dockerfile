# Training API Dockerfile
# For LLM fine-tuning pipeline

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Copy training API code
COPY demo/training_api/ /app/training_api/
COPY demo/training/ /app/training/

# Install Python dependencies
COPY demo/training_api/requirements.txt /app/

# Step 1: Install PyTorch with CUDA 11.8 from Tsinghua mirror
# Tsinghua mirror has good speeds for PyTorch in China
RUN pip install --no-cache-dir --default-timeout=1000 \
    torch torchvision torchaudio \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# Step 2: Install CUDA-dependent packages from Tsinghua
# bitsandbytes and accelerate need to be compatible with PyTorch CUDA
RUN pip install --no-cache-dir --default-timeout=1000 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    bitsandbytes>=0.41.0 \
    accelerate>=0.24.0

# Step 3: Install other dependencies from Aliyun (faster for general packages)
RUN pip install --no-cache-dir --default-timeout=1000 \
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
CMD ["uvicorn", "training_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
