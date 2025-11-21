#!/bin/bash

################################################################################
# 修复 .bashrc 语法错误
################################################################################

set -e

echo "🔧 修复 .bashrc 语法错误"
echo ""

# 1. 备份当前的 .bashrc
echo "[1/4] 备份当前 .bashrc..."
cp ~/.bashrc ~/.bashrc.broken.$(date +%Y%m%d_%H%M%S)
echo "✓ 已备份到 ~/.bashrc.broken.*"

# 2. 检查是否有可用的备份
echo ""
echo "[2/4] 查找可用的备份..."
ls -lt ~/.bashrc.backup* 2>/dev/null | head -5

BACKUP_FILE=$(ls -t ~/.bashrc.backup* 2>/dev/null | head -1)

if [ -z "$BACKUP_FILE" ]; then
    echo "❌ 未找到备份文件"
    echo ""
    echo "手动修复方法:"
    echo "  1. 编辑文件: nano ~/.bashrc"
    echo "  2. 查看第 148 行附近的 if/fi 配对"
    echo "  3. 确保每个 if 都有对应的 fi"
    exit 1
fi

echo "✓ 找到最新备份: $BACKUP_FILE"

# 3. 恢复备份
echo ""
echo "[3/4] 恢复备份..."
read -p "是否恢复此备份? [y/N]: " confirm

if [ "$confirm" != "y" ]; then
    echo "取消恢复"
    exit 0
fi

cp "$BACKUP_FILE" ~/.bashrc
echo "✓ 已恢复备份"

# 4. 添加正确的配置
echo ""
echo "[4/4] 添加正确的 Anaconda 配置..."

NEW_PATH="/media/bygpu/c61f8350-02db-4a47-88ca-3121e00c63cc/anaconda3"

# 删除所有旧的添加
sed -i '/# Anaconda3 SSD Path/d' ~/.bashrc

# 在文件末尾添加(避免破坏原有结构)
cat >> ~/.bashrc << EOF

# Anaconda3 SSD Path - Migrated
export PATH="$NEW_PATH/bin:\$PATH"
EOF

echo "✓ 配置已添加"

# 5. 验证语法
echo ""
echo "验证 .bashrc 语法..."
if bash -n ~/.bashrc 2>&1; then
    echo "✓ 语法正确"
else
    echo "❌ 仍有语法错误"
    echo ""
    echo "建议手动检查: nano ~/.bashrc"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ .bashrc 已修复!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "现在执行:"
echo "  source ~/.bashrc"
echo "  which conda"
echo ""
