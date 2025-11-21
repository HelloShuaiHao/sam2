#!/bin/bash

################################################################################
# Anaconda 无痛迁移脚本
# 功能: 将 Anaconda 从系统盘迁移到其他磁盘,不影响功能
################################################################################

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  Anaconda 迁移工具${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# 默认路径
OLD_PATH="/home/bygpu/anaconda3"
NEW_PATH=""

# 步骤 1: 检查当前安装
echo -e "${YELLOW}步骤 1/6: 检查当前 Anaconda 安装${NC}"
echo ""

if [ ! -d "$OLD_PATH" ]; then
    echo -e "${RED}❌ 错误: $OLD_PATH 不存在${NC}"
    read -p "请输入 Anaconda 实际安装路径: " OLD_PATH

    if [ ! -d "$OLD_PATH" ]; then
        echo -e "${RED}❌ 路径仍然无效,退出${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ 找到 Anaconda: $OLD_PATH${NC}"

# 检查大小
ANACONDA_SIZE=$(du -sh "$OLD_PATH" 2>/dev/null | cut -f1)
echo -e "${GREEN}✓ 当前大小: $ANACONDA_SIZE${NC}"
echo ""

# 步骤 2: 选择目标位置
echo -e "${YELLOW}步骤 2/6: 选择目标 SSD 磁盘${NC}"
echo ""

echo "可用磁盘:"
df -h | grep -E "Filesystem|/dev/sd|/dev/nvme|/mnt|/media"
echo ""

read -p "请输入目标路径 (例如 /mnt/ssd/anaconda3): " NEW_PATH

if [ -z "$NEW_PATH" ]; then
    echo -e "${RED}❌ 未输入路径,退出${NC}"
    exit 1
fi

# 检查目标是否已存在
if [ -d "$NEW_PATH" ]; then
    echo -e "${RED}❌ 错误: $NEW_PATH 已存在${NC}"
    read -p "是否删除并继续? [y/N]: " confirm
    if [ "$confirm" != "y" ]; then
        echo "取消迁移"
        exit 1
    fi
    rm -rf "$NEW_PATH"
fi

# 检查目标磁盘空间
PARENT_DIR=$(dirname "$NEW_PATH")
if [ ! -d "$PARENT_DIR" ]; then
    echo -e "${YELLOW}⚠️  父目录不存在,将创建: $PARENT_DIR${NC}"
    read -p "继续? [y/N]: " confirm
    if [ "$confirm" != "y" ]; then
        echo "取消迁移"
        exit 1
    fi
    mkdir -p "$PARENT_DIR"
fi

AVAILABLE_SPACE=$(df -BG "$PARENT_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
echo -e "${GREEN}✓ 目标磁盘可用空间: ${AVAILABLE_SPACE}GB${NC}"
echo ""

# 步骤 3: 确认迁移计划
echo -e "${YELLOW}步骤 3/6: 确认迁移计划${NC}"
echo ""
echo -e "  源路径: ${BLUE}$OLD_PATH${NC}"
echo -e "  目标路径: ${BLUE}$NEW_PATH${NC}"
echo -e "  数据大小: ${BLUE}$ANACONDA_SIZE${NC}"
echo ""
echo -e "${YELLOW}⚠️  警告:${NC}"
echo "  1. 迁移期间请勿使用 conda/python"
echo "  2. 建议先关闭所有使用 Anaconda 的程序"
echo "  3. 整个过程可能需要 5-30 分钟"
echo ""
read -p "确认开始迁移? [y/N]: " confirm

if [ "$confirm" != "y" ]; then
    echo "取消迁移"
    exit 1
fi

# 步骤 4: 复制 Anaconda
echo ""
echo -e "${YELLOW}步骤 4/6: 复制 Anaconda 到新位置${NC}"
echo -e "${YELLOW}这可能需要 10-30 分钟,请耐心等待...${NC}"
echo ""

# 使用 rsync 保留权限和符号链接
if command -v rsync &> /dev/null; then
    echo "使用 rsync 复制(保留权限)..."
    rsync -av --info=progress2 "$OLD_PATH/" "$NEW_PATH/"
else
    echo "使用 cp 复制..."
    cp -r "$OLD_PATH" "$NEW_PATH"
fi

echo ""
echo -e "${GREEN}✓ 复制完成${NC}"
echo ""

# 步骤 5: 更新配置
echo -e "${YELLOW}步骤 5/6: 更新环境变量和配置${NC}"
echo ""

# 检测 shell
if [ -n "$BASH_VERSION" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
else
    SHELL_RC="$HOME/.bashrc"
fi

echo "检测到 shell 配置文件: $SHELL_RC"

# 备份配置文件
cp "$SHELL_RC" "${SHELL_RC}.backup.$(date +%Y%m%d_%H%M%S)"
echo -e "${GREEN}✓ 已备份配置文件${NC}"

# 更新 PATH
echo ""
echo "正在更新 PATH..."

# 删除旧的 conda 初始化代码
sed -i.bak "/# >>> conda initialize >>>/,/# <<< conda initialize <<</d" "$SHELL_RC"

# 运行新的 conda init
"$NEW_PATH/bin/conda" init bash 2>/dev/null || true
if [ -f "$HOME/.zshrc" ]; then
    "$NEW_PATH/bin/conda" init zsh 2>/dev/null || true
fi

echo -e "${GREEN}✓ 更新完成${NC}"
echo ""

# 步骤 6: 验证
echo -e "${YELLOW}步骤 6/6: 验证新安装${NC}"
echo ""

# 测试 conda
"$NEW_PATH/bin/conda" --version
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Conda 工作正常${NC}"
else
    echo -e "${RED}❌ Conda 验证失败${NC}"
    exit 1
fi

# 测试 python
"$NEW_PATH/bin/python" --version
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python 工作正常${NC}"
else
    echo -e "${RED}❌ Python 验证失败${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✅ 迁移成功完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}下一步操作:${NC}"
echo ""
echo "1. ${GREEN}重新加载配置${NC}"
echo "   source $SHELL_RC"
echo ""
echo "2. ${GREEN}验证安装${NC}"
echo "   which conda"
echo "   which python"
echo "   conda env list"
echo ""
echo "3. ${YELLOW}测试功能正常后,删除旧目录${NC}"
echo "   # 先测试几天,确认一切正常"
echo "   # 然后执行:"
echo "   sudo rm -rf $OLD_PATH"
echo ""
echo "4. ${GREEN}清理系统盘空间${NC}"
echo "   df -h  # 查看释放的空间"
echo ""
echo -e "${BLUE}备注:${NC}"
echo "  - 旧配置文件备份在: ${SHELL_RC}.backup.*"
echo "  - 如有问题,可以恢复备份"
echo "  - 建议先测试 1-2 天再删除旧目录"
echo ""

# 显示空间对比
echo -e "${YELLOW}空间对比:${NC}"
echo "系统盘:"
df -h / | grep -v Filesystem
echo ""
echo "SSD 盘:"
df -h "$NEW_PATH" | grep -v Filesystem
echo ""
