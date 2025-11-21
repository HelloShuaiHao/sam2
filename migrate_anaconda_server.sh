#!/bin/bash

################################################################################
# Anaconda 服务器迁移脚本
# 使用方法:
#   1. 上传此脚本到服务器
#   2. chmod +x migrate_anaconda_server.sh
#   3. ./migrate_anaconda_server.sh
################################################################################

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Anaconda 无痛迁移工具${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 默认路径
OLD_PATH="/home/bygpu/anaconda3"

# 步骤 1: 检查当前安装
echo -e "${YELLOW}[1/7] 检查当前 Anaconda 安装${NC}"
echo ""

if [ ! -d "$OLD_PATH" ]; then
    echo -e "${RED}❌ 默认路径不存在: $OLD_PATH${NC}"
    read -p "请输入实际安装路径: " OLD_PATH
    if [ ! -d "$OLD_PATH" ]; then
        echo -e "${RED}❌ 路径无效,退出${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ 找到 Anaconda: $OLD_PATH${NC}"

# 检查大小
echo "正在计算大小..."
ANACONDA_SIZE=$(du -sh "$OLD_PATH" | cut -f1)
echo -e "${GREEN}✓ 当前大小: $ANACONDA_SIZE${NC}"

# 检查文件数量
FILE_COUNT=$(find "$OLD_PATH" -type f | wc -l)
echo -e "${GREEN}✓ 文件数量: $FILE_COUNT${NC}"
echo ""

# 步骤 2: 显示磁盘信息
echo -e "${YELLOW}[2/7] 查看可用磁盘${NC}"
echo ""

echo "当前磁盘使用情况:"
df -h
echo ""

# 步骤 3: 选择目标位置
echo -e "${YELLOW}[3/7] 选择目标位置${NC}"
echo ""

echo "常见 SSD 挂载点:"
echo "  - /mnt/ssd"
echo "  - /data"
echo "  - /home/ssd"
echo ""

read -p "请输入目标完整路径 (例如 /mnt/ssd/anaconda3): " NEW_PATH

if [ -z "$NEW_PATH" ]; then
    echo -e "${RED}❌ 未输入路径,退出${NC}"
    exit 1
fi

# 检查父目录
PARENT_DIR=$(dirname "$NEW_PATH")
if [ ! -d "$PARENT_DIR" ]; then
    echo -e "${RED}❌ 父目录不存在: $PARENT_DIR${NC}"
    echo "请先创建或选择正确的路径"
    exit 1
fi

# 检查目标是否存在
if [ -d "$NEW_PATH" ]; then
    echo -e "${YELLOW}⚠️  目标已存在: $NEW_PATH${NC}"
    read -p "是否删除并继续? [y/N]: " confirm
    if [ "$confirm" != "y" ]; then
        exit 1
    fi
    rm -rf "$NEW_PATH"
fi

# 检查空间
AVAILABLE_SPACE=$(df -BG "$PARENT_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
echo -e "${GREEN}✓ 目标磁盘可用空间: ${AVAILABLE_SPACE}GB${NC}"
echo ""

# 步骤 4: 确认
echo -e "${YELLOW}[4/7] 确认迁移计划${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  源路径:     ${BLUE}$OLD_PATH${NC}"
echo -e "  目标路径:   ${BLUE}$NEW_PATH${NC}"
echo -e "  数据大小:   ${BLUE}$ANACONDA_SIZE${NC}"
echo -e "  文件数量:   ${BLUE}$FILE_COUNT${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${YELLOW}⚠️  注意事项:${NC}"
echo "  1. 迁移期间请勿使用 conda/python"
echo "  2. 建议在 screen/tmux 中运行"
echo "  3. 预计时间: 10-30 分钟"
echo ""
read -p "确认开始? [y/N]: " confirm

if [ "$confirm" != "y" ]; then
    echo "已取消"
    exit 0
fi

# 步骤 5: 复制数据
echo ""
echo -e "${YELLOW}[5/7] 复制 Anaconda 到新位置${NC}"
echo ""

START_TIME=$(date +%s)

if command -v rsync &> /dev/null; then
    echo "使用 rsync 复制(保留权限和符号链接)..."
    rsync -av --info=progress2 "$OLD_PATH/" "$NEW_PATH/"
else
    echo "使用 cp 复制..."
    cp -rp "$OLD_PATH" "$NEW_PATH"
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}✓ 复制完成 (耗时: ${ELAPSED}秒)${NC}"
echo ""

# 验证复制
NEW_SIZE=$(du -sh "$NEW_PATH" | cut -f1)
echo -e "${GREEN}✓ 新位置大小: $NEW_SIZE${NC}"
echo ""

# 步骤 6: 更新配置
echo -e "${YELLOW}[6/7] 更新环境变量${NC}"
echo ""

# 检测用户的 shell 配置文件
SHELL_RC=""
if [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_RC="$HOME/.bash_profile"
elif [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
else
    echo -e "${RED}❌ 未找到 shell 配置文件${NC}"
    exit 1
fi

echo "配置文件: $SHELL_RC"

# 备份
BACKUP_FILE="${SHELL_RC}.backup.$(date +%Y%m%d_%H%M%S)"
cp "$SHELL_RC" "$BACKUP_FILE"
echo -e "${GREEN}✓ 已备份: $BACKUP_FILE${NC}"

# 删除旧的 conda 初始化
echo "移除旧的 conda 配置..."
sed -i.bak '/# >>> conda initialize >>>/,/# <<< conda initialize <<</d' "$SHELL_RC"

# 运行新的 conda init
echo "初始化新的 conda..."
"$NEW_PATH/bin/conda" init bash 2>/dev/null || true
if [ -f "$HOME/.zshrc" ]; then
    "$NEW_PATH/bin/conda" init zsh 2>/dev/null || true
fi

echo -e "${GREEN}✓ 配置更新完成${NC}"
echo ""

# 步骤 7: 验证
echo -e "${YELLOW}[7/7] 验证新安装${NC}"
echo ""

# 测试 conda
if "$NEW_PATH/bin/conda" --version; then
    echo -e "${GREEN}✓ Conda 可用${NC}"
else
    echo -e "${RED}❌ Conda 验证失败${NC}"
    exit 1
fi

# 测试 python
if "$NEW_PATH/bin/python" --version; then
    echo -e "${GREEN}✓ Python 可用${NC}"
else
    echo -e "${RED}❌ Python 验证失败${NC}"
    exit 1
fi

# 列出环境
echo ""
echo "Conda 环境列表:"
"$NEW_PATH/bin/conda" env list

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✅ 迁移成功完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}下一步操作:${NC}"
echo ""
echo -e "${YELLOW}1. 重新加载配置 (必须执行)${NC}"
echo "   source $SHELL_RC"
echo ""
echo -e "${YELLOW}2. 验证安装${NC}"
echo "   which conda    # 应该显示: $NEW_PATH/bin/conda"
echo "   which python   # 应该显示: $NEW_PATH/bin/python"
echo "   conda --version"
echo ""
echo -e "${YELLOW}3. 测试功能${NC}"
echo "   conda env list"
echo "   python -c 'import sys; print(sys.prefix)'"
echo ""
echo -e "${YELLOW}4. 删除旧目录 (建议先测试几天)${NC}"
echo "   # 确认一切正常后执行:"
echo "   sudo rm -rf $OLD_PATH"
echo ""
echo -e "${YELLOW}5. 查看释放的空间${NC}"
echo "   df -h /"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}备注:${NC}"
echo "  • 配置备份: $BACKUP_FILE"
echo "  • 如有问题可恢复: cp $BACKUP_FILE $SHELL_RC"
echo "  • 建议测试 1-2 天后再删除旧目录"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
