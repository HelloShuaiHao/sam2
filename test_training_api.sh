#!/bin/bash

# Test script for training API
# This script tests the training API by calling it directly with curl

set -e

API_URL="http://localhost:7264"

echo "=========================================="
echo "Testing SAM2 Training API"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check API health
echo -e "\n${YELLOW}Step 1: Checking API health...${NC}"
health_response=$(curl -s "${API_URL}/health" || echo "FAILED")
if [[ "$health_response" == *"healthy"* ]]; then
    echo -e "${GREEN}✓ API is healthy${NC}"
else
    echo -e "${RED}✗ API is not responding. Please start it first with ./start_training_api.sh${NC}"
    exit 1
fi

# Step 2: Start a small training job
echo -e "\n${YELLOW}Step 2: Starting training job...${NC}"

# Training configuration
TRAIN_CONFIG=$(cat <<EOF
{
  "config": {
    "model_name": "liuhaotian/llava-v1.5-7b",
    "train_data_path": "/media/bygpu/c61f8350-02db-4a47-88ca-3121e00c63cc/sam2/demo/training/output/splits/train.jsonl",
    "val_data_path": "/media/bygpu/c61f8350-02db-4a47-88ca-3121e00c63cc/sam2/demo/training/output/splits/val.jsonl",
    "output_dir": "/media/bygpu/c61f8350-02db-4a47-88ca-3121e00c63cc/sam2/demo/training/output/checkpoints/test_run",
    "num_epochs": 1,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "max_length": 2048,
    "use_qlora": true,
    "use_lora": false,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bf16": false,
    "fp16": true,
    "warmup_ratio": 0.03,
    "max_grad_norm": 1.0,
    "logging_steps": 1,
    "save_steps": 100,
    "save_total_limit": 2
  },
  "experiment_name": "api_test_$(date +%s)",
  "tags": ["test", "qlora", "api-test"]
}
EOF
)

echo "Sending training request..."
train_response=$(curl -s -X POST "${API_URL}/api/train/start" \
    -H "Content-Type: application/json" \
    -d "$TRAIN_CONFIG")

echo "Response: $train_response"

# Extract job ID
JOB_ID=$(echo "$train_response" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$JOB_ID" ]; then
    echo -e "${RED}✗ Failed to start training job${NC}"
    echo "Response: $train_response"
    exit 1
fi

echo -e "${GREEN}✓ Training job started: ${JOB_ID}${NC}"

# Step 3: Monitor training progress
echo -e "\n${YELLOW}Step 3: Monitoring training progress...${NC}"
echo "Checking status every 5 seconds..."
echo "Press Ctrl+C to stop monitoring (training will continue)"

COUNTER=0
while true; do
    sleep 5
    COUNTER=$((COUNTER + 1))

    status_response=$(curl -s "${API_URL}/api/train/${JOB_ID}/status")

    # Extract status fields
    status=$(echo "$status_response" | grep -o '"status":"[^"]*"' | head -1 | cut -d'"' -f4)
    progress=$(echo "$status_response" | grep -o '"progress_percentage":[0-9.]*' | cut -d':' -f2)
    error_msg=$(echo "$status_response" | grep -o '"error_message":"[^"]*"' | cut -d'"' -f4)

    echo -e "\n[Check #${COUNTER}] Status: ${status}, Progress: ${progress}%"

    if [ "$status" = "completed" ]; then
        echo -e "${GREEN}✓ Training completed successfully!${NC}"
        break
    elif [ "$status" = "failed" ]; then
        echo -e "${RED}✗ Training failed!${NC}"
        if [ -n "$error_msg" ]; then
            echo -e "${RED}Error: ${error_msg}${NC}"
        fi
        exit 1
    elif [ "$status" = "cancelled" ]; then
        echo -e "${YELLOW}○ Training was cancelled${NC}"
        exit 1
    fi

    # Show partial response for debugging
    echo "Full status: $status_response" | head -c 200
    echo "..."
done

echo -e "\n${GREEN}=========================================="
echo "Training test completed successfully!"
echo "==========================================${NC}"
