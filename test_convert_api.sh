#!/bin/bash

# 测试 Training API convert endpoint 的脚本

echo "=== Testing Training API Convert Endpoint ==="
echo ""

# 测试数据
TEST_ZIP="/media/bygpu/c61f8350-02db-4a47-88ca-3121e00c63cc/sam2/demo/uploads/*.zip"

# 查找最近的上传文件
if ls $TEST_ZIP 1> /dev/null 2>&1; then
    LATEST_ZIP=$(ls -t $TEST_ZIP | head -1)
    echo "Found uploaded file: $LATEST_ZIP"
else
    echo "No ZIP files found in demo/uploads/"
    echo "Please upload a file first using the web interface"
    exit 1
fi

echo ""
echo "Testing convert endpoint..."
echo ""

# 调用 convert API
curl -v -X POST http://localhost:7264/api/data/convert \
  -H "Content-Type: application/json" \
  -d "{
    \"sam2_zip_path\": \"$LATEST_ZIP\",
    \"output_dir\": \"/media/bygpu/c61f8350-02db-4a47-88ca-3121e00c63cc/sam2/demo/training/output/training_data\",
    \"target_format\": \"llava\"
  }" 2>&1 | tee /tmp/convert_test_output.txt

echo ""
echo ""
echo "=== Full output saved to /tmp/convert_test_output.txt ==="
