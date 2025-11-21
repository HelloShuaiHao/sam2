#!/bin/bash

################################################################################
# 修复 Conda 路径配置
# 在服务器上执行此脚本
################################################################################

set -e

NEW_PATH="/media/bygpu/c61f8350-02db-4a47-88ca-3121e00c63cc/anaconda3"
OLD_PATH="/home/bygpu/anaconda3"

echo "🔧 修复 Conda 路径配置"
echo ""
echo "新路径: $NEW_PATH"
echo "旧路径: $OLD_PATH"
echo ""

# 1. 备份当前配置
echo "[1/5] 备份配置文件..."
cp ~/.bashrc ~/.bashrc.backup.fix.$(date +%Y%m%d_%H%M%S)

# 2. 完全删除旧的 conda 初始化代码
echo "[2/5] 清理旧的 conda 配置..."
sed -i '/# >>> conda initialize >>>/,/# <<< conda initialize <<</d' ~/.bashrc

# 3. 手动添加新的 conda 路径到 PATH
echo "[3/5] 添加新的 PATH..."
cat >> ~/.bashrc << 'EOF'

# Conda Path (migrated to SSD)
export PATH="/media/bygpu/c61f8350-02db-4a47-88ca-3121e00c63cc/anaconda3/bin:$PATH"
EOF

# 4. 运行新的 conda init
echo "[4/5] 初始化新的 conda..."
$NEW_PATH/bin/conda init bash

# 5. 验证
echo "[5/5] 验证配置..."
echo ""

# 显示 bashrc 中的 conda 配置
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "~/.bashrc 中的 Conda 配置:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
grep -A 5 "conda initialize" ~/.bashrc || echo "未找到 conda initialize 块"
echo ""
grep "anaconda3" ~/.bashrc || echo "未找到 anaconda3 路径"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "✅ 配置已更新"
echo ""
echo "📝 下一步:"
echo "   1. 重新加载配置:"
echo "      source ~/.bashrc"
echo ""
echo "   2. 验证路径:"
echo "      which conda"
echo "      which python"
echo ""
echo "   3. 如果还是显示旧路径,执行:"
echo "      export PATH=\"$NEW_PATH/bin:\$PATH\""
echo "      hash -r"
echo ""
