#!/bin/bash

# Start Training API using local conda environment
# This avoids slow Docker builds with PyTorch downloads

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Training API with local conda environment...${NC}"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set PYTHONPATH to include current directory and demo directory
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/demo:${PYTHONPATH}"

# Environment variables from docker-compose.yaml
export SERVER_ENVIRONMENT=DEV

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0

# Enable detailed error logging
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1

# HuggingFace settings
export HF_HOME="${SCRIPT_DIR}/cache/huggingface"
export TRANSFORMERS_CACHE="${SCRIPT_DIR}/cache/transformers"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export HUGGINGFACE_HUB_CACHE="${SCRIPT_DIR}/cache/huggingface"

# Increase download timeout
export REQUESTS_TIMEOUT=300
export HTTP_TIMEOUT=300

# Training API settings
export MAX_WORKERS=1
export LOG_LEVEL=info

# Create cache directories if they don't exist
mkdir -p "${SCRIPT_DIR}/cache/huggingface"
mkdir -p "${SCRIPT_DIR}/cache/transformers"
mkdir -p "${SCRIPT_DIR}/demo/training/output"
mkdir -p "${SCRIPT_DIR}/checkpoints"

echo -e "${YELLOW}Environment configured:${NC}"
echo "  PYTHONPATH: ${PYTHONPATH}"
echo "  HF_HOME: ${HF_HOME}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo ""

# Check if uvicorn is installed
python3 -c "import uvicorn" 2>/dev/null || {
    echo -e "${YELLOW}Warning: uvicorn not found. Install with: pip install uvicorn[standard]${NC}"
    exit 1
}

# Check if required Python packages are installed
python3 -c "import fastapi" 2>/dev/null || {
    echo -e "${YELLOW}Warning: fastapi not found. Install requirements with:${NC}"
    echo "  pip install -r demo/training_api/requirements.txt"
    exit 1
}

echo -e "${GREEN}Starting uvicorn server on port 7264...${NC}"
echo -e "${YELLOW}Access at: http://localhost:7264${NC}"
echo -e "${YELLOW}API docs at: http://localhost:7264/docs${NC}"
echo ""

# Change to demo directory and start uvicorn
cd "${SCRIPT_DIR}/demo"

# Start uvicorn using python -m to avoid shebang path issues
# (port 7264 to match docker-compose external port)
# Use --host 0.0.0.0 to allow external access
# Use --reload for auto-reload on code changes
python3 -m uvicorn training_api.main:app \
    --host 0.0.0.0 \
    --port 7264 \
    --workers 1 \
    --reload \
    --log-level info
