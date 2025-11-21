#!/bin/bash

################################################################################
# 修复 HuggingFace 下载超时问题
################################################################################

set -e

echo "🔧 修复 HuggingFace 模型下载超时"
echo ""

cd "$(dirname "$0")"

echo "方案选择:"
echo "  1) 快速修复 - 安装依赖并重启 (推荐)"
echo "  2) 使用国内模型源 - ModelScope"
echo "  3) 手动下载模型"
echo ""
read -p "选择方案 [1/2/3]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 方案 1: 安装 hf_transfer 加速下载"
        echo ""

        # 安装 sentencepiece (如果还没装)
        echo "1. 安装必要依赖..."
        docker compose exec training-api pip install sentencepiece protobuf hf-transfer

        echo ""
        echo "2. 重启服务..."
        docker compose restart training-api

        echo ""
        echo "3. 等待服务就绪..."
        sleep 5

        echo ""
        echo "✅ 完成! 现在可以:"
        echo "   1. 重新点击 Start Training"
        echo "   2. 观察日志: docker compose logs -f training-api"
        ;;

    2)
        echo ""
        echo "🇨🇳 方案 2: 配置 ModelScope (国内镜像)"
        echo ""

        cat > /tmp/modelscope_fix.py << 'EOF'
# 在容器中安装 ModelScope
import subprocess
import sys

print("安装 ModelScope...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])

print("✅ ModelScope 安装完成")
print("")
print("⚠️  注意: 你需要修改前端,将模型名称改为:")
print("   AI-ModelScope/llava-1.5-7b-hf")
EOF

        docker compose cp /tmp/modelscope_fix.py training-api:/tmp/
        docker compose exec training-api python /tmp/modelscope_fix.py

        echo ""
        echo "✅ ModelScope 已安装"
        echo ""
        echo "下一步:"
        echo "  1. 修改前端模型选择,使用 ModelScope 模型 ID"
        echo "  2. 或者手动下载模型(方案 3)"
        ;;

    3)
        echo ""
        echo "📥 方案 3: 手动下载模型指南"
        echo ""

        cat << 'INSTRUCTIONS'
如果你有代理或VPN,可以手动下载模型:

步骤 1: 在本地下载模型
--------------------------
# 使用 huggingface-cli (需要代理)
pip install huggingface-hub
huggingface-cli download liuhaotian/llava-v1.5-7b \
  --local-dir ./llava-v1.5-7b \
  --local-dir-use-symlinks False

步骤 2: 上传到服务器
--------------------
# 压缩模型
tar -czf llava-v1.5-7b.tar.gz llava-v1.5-7b/

# 上传到服务器
scp llava-v1.5-7b.tar.gz user@server:/path/to/sam2/models/

步骤 3: 在服务器解压
--------------------
ssh user@server
cd /path/to/sam2
mkdir -p models
tar -xzf models/llava-v1.5-7b.tar.gz -C models/

步骤 4: 修改 docker-compose.yaml
--------------------------------
添加模型挂载:

services:
  training-api:
    volumes:
      - ./models:/app/models:ro

步骤 5: 重启服务
----------------
docker compose restart training-api

步骤 6: 修改前端
----------------
将模型名称改为: /app/models/llava-v1.5-7b

INSTRUCTIONS
        ;;

    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 如果还是超时,请检查:"
echo ""
echo "1. 网络连接"
echo "   docker compose exec training-api ping -c 3 hf-mirror.com"
echo ""
echo "2. DNS 配置"
echo "   docker compose exec training-api cat /etc/resolv.conf"
echo ""
echo "3. 查看详细日志"
echo "   docker compose logs -f training-api"
echo ""
echo "4. 尝试使用代理"
echo "   编辑 docker-compose.yaml,添加:"
echo "   environment:"
echo "     - HTTP_PROXY=http://你的代理:端口"
echo "     - HTTPS_PROXY=http://你的代理:端口"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
