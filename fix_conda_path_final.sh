#!/bin/bash

################################################################################
# 修复 Conda 路径配置 - 最终版本
# 强制替换所有旧路径为新路径
################################################################################

set -e

NEW_PATH="/media/bygpu/c61f8350-02db-4a47-88ca-3121e00c63cc/anaconda3"
OLD_PATH="/home/bygpu/anaconda3"

echo "🔧 修复 Conda 路径配置 (最终版)"
echo ""
echo "新路径: $NEW_PATH"
echo "旧路径: $OLD_PATH"
echo ""

# 1. 备份
echo "[1/4] 备份配置文件..."
cp ~/.bashrc ~/.bashrc.backup.final.$(date +%Y%m%d_%H%M%S)
echo "✓ 已备份"

# 2. 替换所有旧路径
echo ""
echo "[2/4] 替换 .bashrc 中的所有旧路径..."
sed -i "s|$OLD_PATH|$NEW_PATH|g" ~/.bashrc
echo "✓ 路径已替换"

# 3. 验证替换结果
echo ""
echo "[3/4] 验证替换..."
if grep -q "$OLD_PATH" ~/.bashrc; then
    echo "⚠️  警告: 仍然发现旧路径,可能需要手动检查"
    grep -n "$OLD_PATH" ~/.bashrc
else
    echo "✓ 所有旧路径已替换"
fi

# 4. 重新加载并测试
echo ""
echo "[4/4] 测试新配置..."
source ~/.bashrc

# 使用绝对路径测试
NEW_CONDA="$NEW_PATH/bin/conda"
if [ -x "$NEW_CONDA" ]; then
    echo "✓ 新 conda 可执行"
    $NEW_CONDA --version
else
    echo "❌ 错误: 新路径的 conda 不可执行"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 配置修复完成!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📝 现在执行以下命令:"
echo ""
echo "1. 重新加载配置:"
echo "   source ~/.bashrc"
echo ""
echo "2. 清除命令缓存:"
echo "   hash -r"
echo ""
echo "3. 验证路径:"
echo "   which conda"
echo "   # 应该显示: $NEW_PATH/bin/conda"
echo ""
echo "4. 测试功能:"
echo "   conda --version"
echo "   conda env list"
echo ""
echo "5. 如果一切正常,删除旧目录:"
echo "   rm -rf $OLD_PATH"
echo "   df -h /  # 查看释放的空间"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "备注: 配置文件已备份到 ~/.bashrc.backup.final.*"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
