# Training API 启动指南

## 概述

Training API 现在支持两种启动方式：
1. **本地启动（推荐）** - 使用本地 conda 环境，快速启动，无需下载 PyTorch
2. **Docker 启动** - 完整的容器化部署，适合生产环境

## 方式一：本地启动（推荐）⚡

### 优势
- ✅ 启动速度快（无需下载 PyTorch）
- ✅ 使用本地已安装的 GPU PyTorch (2.3.1+cu121)
- ✅ 支持热重载（代码修改自动生效）
- ✅ 调试方便

### 启动命令

```bash
# 启动 Training API（端口 7264）
./start_training_api.sh

# 或者在后台启动
nohup ./start_training_api.sh > training_api.log 2>&1 &
```

### 访问地址
- 主页：http://localhost:7264
- 健康检查：http://localhost:7264/health
- API 文档：http://localhost:7264/docs
- ReDoc：http://localhost:7264/redoc

### 环境配置
启动脚本自动配置以下环境变量：
- GPU: `CUDA_VISIBLE_DEVICES=0`
- PyTorch: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- HuggingFace: `HF_ENDPOINT=https://hf-mirror.com`
- 监听地址: `0.0.0.0:7264`（支持外部访问）

### 停止服务

```bash
# 查找进程
ps aux | grep "uvicorn training_api"

# 停止服务
pkill -f "uvicorn training_api"

# 或者使用 Ctrl+C（如果在前台运行）
```

## 方式二：Docker 启动

### 仅启动 Frontend + Backend

```bash
# 默认只启动 frontend 和 backend（不包括 training-api）
docker compose up -d

# 或者强制重新构建
docker compose up -d --build
```

### 启动完整服务（包括 Training API）

```bash
# 使用 profile 启动所有服务（会下载 PyTorch，很慢）
docker compose --profile training up -d

# 或者强制重新构建
docker compose --profile training up -d --build
```

## 混合部署（推荐）

**最佳实践：Docker 运行 Frontend + Backend，本地运行 Training API**

```bash
# 1. 本地启动 Training API
./start_training_api.sh

# 2. Docker 启动 Frontend + Backend
docker compose up -d

# 3. 验证所有服务
curl http://localhost:7262  # Frontend
curl http://localhost:7263  # Backend
curl http://localhost:7264/health  # Training API
```

## 端口映射

| 服务 | 容器端口 | 主机端口 | 访问地址 |
|------|---------|---------|---------|
| Frontend | 80 | 7262 | http://localhost:7262 |
| Backend | 5000 | 7263 | http://localhost:7263 |
| Training API | 8000 | 7264 | http://localhost:7264 |

## GPU 配置验证

### 检查本地 PyTorch GPU 支持

```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### 检查 Training API GPU 状态

```bash
# 查看 GPU 使用情况
nvidia-smi

# 查看 Training API 进程环境变量
tr '\0' '\n' < /proc/$(pgrep -f "uvicorn training_api" | head -1)/environ | grep CUDA
```

### 验证网络监听

```bash
# 确认 Training API 监听在 0.0.0.0:7264
ss -tlnp | grep 7264
# 输出应包含: 0.0.0.0:7264
```

## 故障排查

### Training API 无法启动

1. 检查端口是否被占用
```bash
lsof -i:7264
# 或
ss -tlnp | grep 7264
```

2. 检查 Python 环境
```bash
which python3
python3 -m pip list | grep -E "uvicorn|fastapi|torch"
```

3. 查看日志
```bash
# 如果在后台运行
tail -f training_api.log

# 或查看进程输出
ps aux | grep "uvicorn training_api"
```

### 依赖缺失

```bash
# 安装 Training API 依赖
pip install -r demo/training_api/requirements.txt

# 或者只安装核心依赖
pip install fastapi uvicorn[standard] torch transformers
```

### 模块导入错误

```bash
# 确保 demo/training 是 Python 包
ls -la demo/training/__init__.py

# 如果不存在，创建
touch demo/training/__init__.py
```

## 日志和监控

### 启动日志

本地启动时会显示详细的启动信息：
```
Starting Training API with local conda environment...
Environment configured:
  PYTHONPATH: /path/to/sam2:/path/to/sam2/demo
  HF_HOME: /path/to/sam2/cache/huggingface
  CUDA_VISIBLE_DEVICES: 0

Starting uvicorn server on port 7264...
Access at: http://localhost:7264
API docs at: http://localhost:7264/docs

INFO:     Uvicorn running on http://0.0.0.0:7264
INFO:     Application startup complete.
```

### 实时监控

```bash
# 监控 GPU 使用
watch -n 1 nvidia-smi

# 监控 API 请求
tail -f training_api.log

# 查看系统资源
htop
```

## 开发建议

### 代码热重载

本地启动默认启用 `--reload`，修改代码后自动重启：
- ✅ 修改 `demo/training_api/*.py` - 自动重载
- ✅ 修改 `demo/training/core/*.py` - 自动重载
- ⚠️ 修改环境变量 - 需要手动重启

### 性能优化

```bash
# 增加 worker 数量（多核 CPU）
# 编辑 start_training_api.sh，修改：
--workers 4  # 建议为 CPU 核心数

# 禁用热重载（生产环境）
# 移除 --reload 参数
```

## 生产部署

生产环境建议使用 Docker 方式部署，或者使用 systemd 管理本地服务：

### 创建 systemd 服务

```bash
sudo tee /etc/systemd/system/training-api.service > /dev/null <<EOF
[Unit]
Description=SAM2 Training API
After=network.target

[Service]
Type=simple
User=bygpu
WorkingDirectory=/media/bygpu/c61f8350-02db-4a47-88ca-3121e00c63cc/sam2
ExecStart=/media/bygpu/c61f8350-02db-4a47-88ca-3121e00c63cc/sam2/start_training_api.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 启动服务
sudo systemctl daemon-reload
sudo systemctl enable training-api
sudo systemctl start training-api

# 查看状态
sudo systemctl status training-api
```

## 总结

| 场景 | 推荐方式 | 命令 |
|------|---------|------|
| 开发调试 | 本地启动 | `./start_training_api.sh` |
| 快速测试 | 本地启动 | `./start_training_api.sh` |
| 生产部署 | Docker | `docker compose --profile training up -d` |
| 混合部署 | 本地 Training API + Docker 其他服务 | 见"混合部署"章节 |
