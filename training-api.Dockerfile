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
# Use Aliyun mirror for faster downloads in China and increase timeout
# PyTorch alone is 900MB, needs sufficient timeout for slow networks
RUN pip install --no-cache-dir \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --default-timeout=1000 \
    -r requirements.txt

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application
CMD ["uvicorn", "training_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
