#!/bin/bash

################################################################################
# 验证迁移并删除旧目录
################################################################################

set -e

NEW_PATH="/media/bygpu/c61f8350-02db-4a47-88ca-3121e00c63cc/anaconda3"
OLD_PATH="/home/bygpu/anaconda3"

echo "🔍 验证 Anaconda 迁移"
echo ""

# 清除所有缓存
hash -r
unset -f conda 2>/dev/null || true

# 强制使用新路径
export PATH="$NEW_PATH/bin:$PATH"

echo "[1/4] 测试新路径的 conda..."
if $NEW_PATH/bin/conda --version; then
    echo "✓ 新路径的 conda 可用"
else
    echo "❌ 新路径的 conda 不可用"
    exit 1
fi

echo ""
echo "[2/4] 测试新路径的 python..."
if $NEW_PATH/bin/python --version; then
    echo "✓ 新路径的 python 可用"
else
    echo "❌ 新路径的 python 不可用"
    exit 1
fi

echo ""
echo "[3/4] 列出 conda 环境..."
$NEW_PATH/bin/conda env list

echo ""
echo "[4/4] 测试 Python 导入..."
$NEW_PATH/bin/python -c "import sys; print('Python prefix:', sys.prefix)"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 所有测试通过!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查旧目录大小
if [ -d "$OLD_PATH" ]; then
    OLD_SIZE=$(du -sh "$OLD_PATH" | cut -f1)
    echo "📊 旧目录信息:"
    echo "   路径: $OLD_PATH"
    echo "   大小: $OLD_SIZE"
    echo ""

    read -p "❓ 是否删除旧目录? [y/N]: " confirm

    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        echo ""
        echo "🗑️  正在删除旧目录..."
        rm -rf "$OLD_PATH"
        echo "✓ 已删除: $OLD_PATH"
        echo ""

        echo "📊 系统盘空间:"
        df -h / | grep -E "Filesystem|/$"
        echo ""

        echo "🎉 释放了 $OLD_SIZE 空间!"
    else
        echo ""
        echo "⏸️  保留旧目录"
        echo "   你可以稍后手动删除: rm -rf $OLD_PATH"
    fi
else
    echo "ℹ️  旧目录已不存在"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 配置 Shell (重要!)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "由于 'which conda' 仍显示旧路径,这是 shell 缓存问题。"
echo ""
echo "解决方法 (选一个):"
echo ""
echo "方法 1: 退出并重新登录 SSH (推荐)"
echo "   exit"
echo "   ssh bygpu@服务器"
echo "   which conda  # 应该显示新路径"
echo ""
echo "方法 2: 在当前 shell 中强制使用新路径"
echo "   添加到 ~/.bashrc 最后一行:"
echo "   export PATH=\"$NEW_PATH/bin:\$PATH\""
echo "   hash -r"
echo ""
echo "方法 3: 创建别名 (临时)"
echo "   alias conda='$NEW_PATH/bin/conda'"
echo "   alias python='$NEW_PATH/bin/python'"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
