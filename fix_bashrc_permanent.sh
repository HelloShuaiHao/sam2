#!/bin/bash

################################################################################
# 永久修复 .bashrc 配置
################################################################################

set -e

NEW_PATH="/media/bygpu/c61f8350-02db-4a47-88ca-3121e00c63cc/anaconda3"

echo "🔧 永久修复 .bashrc 配置"
echo ""

# 1. 在 .bashrc 最前面添加新路径(优先级最高)
echo "[1/2] 添加新路径到 .bashrc 开头..."

# 先删除可能存在的旧配置
sed -i '/# Anaconda3 SSD Path/d' ~/.bashrc
sed -i "s|export PATH=\"$NEW_PATH/bin:\$PATH\"|# removed|g" ~/.bashrc
sed -i '/# removed/d' ~/.bashrc

# 在文件开头添加(在所有其他配置之前)
cat > /tmp/conda_path.txt << EOF
# Anaconda3 SSD Path - Added by migration script
# This MUST be at the top to override conda init
export PATH="$NEW_PATH/bin:\$PATH"

EOF

# 合并到 .bashrc 开头
cat /tmp/conda_path.txt ~/.bashrc > /tmp/bashrc.new
mv /tmp/bashrc.new ~/.bashrc
rm /tmp/conda_path.txt

echo "✓ 已添加"

# 2. 验证
echo ""
echo "[2/2] 验证配置..."
if head -5 ~/.bashrc | grep -q "$NEW_PATH"; then
    echo "✓ 新路径已添加到 .bashrc 开头"
else
    echo "❌ 配置失败"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 配置已永久修复!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📝 现在所有新的 terminal tab 都会自动使用新路径"
echo ""
echo "验证:"
echo "  1. 打开新的 terminal tab"
echo "  2. 执行: which conda"
echo "  3. 应该显示: $NEW_PATH/bin/conda"
echo ""
echo "当前 tab 需要重新加载:"
echo "  source ~/.bashrc"
echo ""
