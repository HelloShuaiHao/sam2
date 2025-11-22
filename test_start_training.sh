#!/bin/bash

# 测试 Training API start endpoint 的脚本

echo "=== Testing Training API Start Endpoint ==="
echo ""

# 测试数据路径
TRAIN_DATA="/media/bygpu/c61f8350-02db-4a47-88ca-3121e00c63cc/sam2/demo/training/output/training_data/splits/train.jsonl"
VAL_DATA="/media/bygpu/c61f8350-02db-4a47-88ca-3121e00c63cc/sam2/demo/training/output/training_data/splits/val.jsonl"
OUTPUT_DIR="/media/bygpu/c61f8350-02db-4a47-88ca-3121e00c63cc/sam2/demo/training/output/test_training"

# 检查文件是否存在
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found: $TRAIN_DATA"
    echo "Please run split endpoint first"
    exit 1
fi

echo "Found training data: $TRAIN_DATA"
echo ""
echo "Testing start training endpoint..."
echo ""

# 调用 start training API
curl -v -X POST http://localhost:7264/api/train/start \
  -H "Content-Type: application/json" \
  -d "{
    \"experiment_name\": \"test_training\",
    \"tags\": [\"test\"],
    \"config\": {
      \"model_name\": \"llava-hf/llava-1.5-7b-hf\",
      \"train_data_path\": \"$TRAIN_DATA\",
      \"val_data_path\": \"$VAL_DATA\",
      \"output_dir\": \"$OUTPUT_DIR\",
      \"num_epochs\": 1,
      \"batch_size\": 1,
      \"learning_rate\": 0.0002,
      \"use_qlora\": true,
      \"lora_r\": 16,
      \"lora_alpha\": 32,
      \"gradient_accumulation_steps\": 4,
      \"save_strategy\": \"epoch\",
      \"logging_steps\": 10
    }
  }" 2>&1 | grep -A 50 "< HTTP"

echo ""
echo ""
echo "=== Test complete ==="
