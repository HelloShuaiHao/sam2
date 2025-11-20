# Docker Deployment Guide

## 🐳 服务架构

你的SAM2项目现在包含3个Docker服务：

```
┌─────────────────────────────────────────────────────────────┐
│                        服务器架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │   Frontend     │  │    Backend     │  │ Training API │ │
│  │  (React/Vite)  │  │   (GraphQL)    │  │  (FastAPI)   │ │
│  │   Port: 7262   │  │  Port: 7263    │  │  Port: 7264  │ │
│  │                │  │                │  │              │ │
│  │  - SAM2 UI     │  │  - SAM2 推理   │  │  - 数据准备  │ │
│  │  - 视频标注    │  │  - 视频处理    │  │  - 模型训练  │ │
│  │  - 训练UI ✨   │  │  - GraphQL API │  │  - 实验跟踪  │ │
│  └────────────────┘  └────────────────┘  └──────────────┘ │
│         │                    │                   │         │
│         │                    ├───────────────────┤         │
│         │                    │                   │         │
│         │              ┌─────▼─────┐      ┌──────▼──────┐ │
│         │              │   GPU 0   │      │   GPU 0    │ │
│         │              │  (推理)   │      │  (训练)    │ │
│         │              └───────────┘      └────────────┘ │
│         │                                                  │
│         └──────────────────────────────────────────────── │
│                        所有服务共享                         │
└─────────────────────────────────────────────────────────────┘
```

## 📋 服务说明

### 1. Frontend (`sam2/frontend`)
- **端口**: 7262
- **功能**:
  - SAM2视频标注界面
  - **新增**：LLM训练工作流UI
  - **新增**：实验管理仪表板
- **访问**: `http://ai.bygpu.com:7262`

### 2. Backend (`sam2/backend`)
- **端口**: 7263
- **功能**:
  - SAM2推理服务
  - GraphQL API
  - 视频处理
- **GPU**: CUDA_VISIBLE_DEVICES=0
- **访问**: `http://ai.bygpu.com:7263`

### 3. Training API (`sam2/training-api`) ✨ 新增
- **端口**: 7264
- **功能**:
  - 数据准备（SAM2→训练格式）
  - LLM模型训练（LoRA/QLoRA）
  - 实验跟踪和对比
  - 模型导出和下载
- **GPU**: CUDA_VISIBLE_DEVICES=0 (与backend共享)
- **访问**:
  - API: `http://ai.bygpu.com:7264`
  - Docs: `http://ai.bygpu.com:7264/docs`

## 🚀 部署步骤

### 1. 准备工作

```bash
# 确保在项目根目录
cd ~/Desktop/sam2

# 创建必要的目录
mkdir -p checkpoints
mkdir -p demo/training/output
```

### 2. 构建镜像

```bash
# 构建所有服务
docker compose build

# 或者只构建training-api
docker compose build training-api
```

### 3. 启动服务

```bash
# 启动所有服务
docker compose up -d

# 查看日志
docker compose logs -f training-api

# 查看所有服务状态
docker compose ps
```

### 4. 验证服务

```bash
# 检查training API健康状态
curl http://localhost:7264/health

# 访问API文档
curl http://localhost:7264/docs

# 测试数据准备端点
curl -X POST http://localhost:7264/api/data/convert \
  -H "Content-Type: application/json" \
  -d '{
    "sam2_zip_path": "/data/export.zip",
    "output_dir": "/app/output",
    "target_format": "llava"
  }'
```

## 📦 卷挂载说明

### Frontend
- 无卷挂载（静态文件在镜像内）

### Backend
- `./demo/data/:/data/:rw` - SAM2数据目录（读写）

### Training API ✨
- `./demo/data/:/data/:ro` - SAM2导出数据（只读）
- `./demo/training/output/:/app/output/:rw` - 训练输出（读写）
- `./checkpoints/:/app/checkpoints/:rw` - 模型检查点（读写）

## ⚙️ 环境变量

### Frontend
```env
VITE_API_URL=http://ai.bygpu.com:7264  # Training API地址
```

### Training API
```env
# PyTorch设置
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
CUDA_VISIBLE_DEVICES=0

# HuggingFace缓存
HF_HOME=/app/cache/huggingface
TRANSFORMERS_CACHE=/app/cache/transformers

# API设置
MAX_WORKERS=1
LOG_LEVEL=info
```

## 🔧 GPU配置

### 当前配置（共享单GPU）
- Backend和Training API都使用`CUDA_VISIBLE_DEVICES=0`
- 通过Docker资源限制防止OOM：
  ```yaml
  limits:
    memory: 32G  # Training API最大内存
  ```

### 如果有多GPU（推荐配置）
```yaml
backend:
  environment:
    - CUDA_VISIBLE_DEVICES=0  # GPU 0 用于推理

training-api:
  environment:
    - CUDA_VISIBLE_DEVICES=1  # GPU 1 用于训练
```

## 🎯 访问地址

| 服务 | 端口 | 地址 | 说明 |
|------|------|------|------|
| Frontend | 7262 | http://ai.bygpu.com:7262 | Web界面 |
| Backend API | 7263 | http://ai.bygpu.com:7263 | GraphQL API |
| Training API | 7264 | http://ai.bygpu.com:7264 | REST API |
| API Docs | 7264 | http://ai.bygpu.com:7264/docs | Swagger文档 |
| API ReDoc | 7264 | http://ai.bygpu.com:7264/redoc | ReDoc文档 |

## 📊 资源使用

### 内存估算
- Frontend: ~500 MB
- Backend (推理): ~8-12 GB (含GPU)
- Training API:
  - 空闲: ~2 GB
  - QLoRA训练: ~8-10 GB GPU + 4 GB RAM
  - LoRA训练: ~12-16 GB GPU + 8 GB RAM

### 磁盘空间
- 镜像: ~15 GB (总计)
- 训练数据: 取决于数据集大小
- 模型检查点: ~50 MB - 30 GB (取决于模型和格式)

## 🔄 常用命令

```bash
# 重启training API
docker compose restart training-api

# 查看training API日志
docker compose logs -f training-api

# 进入training API容器
docker compose exec training-api bash

# 停止所有服务
docker compose down

# 停止并删除卷
docker compose down -v

# 重新构建并启动
docker compose up -d --build training-api

# 清理未使用的镜像
docker system prune -a
```

## 🐛 故障排查

### Training API无法启动

1. **检查GPU可用性**:
```bash
docker compose exec training-api nvidia-smi
```

2. **检查日志**:
```bash
docker compose logs training-api
```

3. **检查端口占用**:
```bash
netstat -tulpn | grep 7264
```

### GPU内存不足

1. **减少batch size**:
   - 在训练配置中使用`batch_size=1`
   - 启用`use_qlora=true`

2. **调整CUDA内存配置**:
```yaml
environment:
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256  # 降低分配大小
```

3. **限制容器内存**:
```yaml
deploy:
  resources:
    limits:
      memory: 16G  # 降低限制
```

### 前端无法连接API

1. **检查环境变量**:
```bash
docker compose exec frontend env | grep VITE_API_URL
```

2. **验证API可访问**:
```bash
curl http://ai.bygpu.com:7264/health
```

3. **检查CORS配置** (在training_api/main.py):
```python
allow_origins=["*"]  # 开发环境允许所有来源
```

## 📝 更新部署

### 更新Training API代码

```bash
# 1. 拉取最新代码
git pull

# 2. 重新构建镜像
docker compose build training-api

# 3. 重启服务
docker compose up -d training-api
```

### 更新Frontend

```bash
# 1. 重新构建
docker compose build frontend

# 2. 重启
docker compose up -d frontend
```

## 🔒 生产环境建议

1. **启用HTTPS** (使用Nginx反向代理)
2. **添加认证** (JWT tokens)
3. **限制CORS** (指定允许的域名)
4. **使用Redis** (代替内存存储任务状态)
5. **使用PostgreSQL** (存储实验数据)
6. **启用日志轮转**
7. **配置监控** (Prometheus + Grafana)
8. **定期备份检查点**

## 📖 相关文档

- Training API文档: `demo/training_api/README.md`
- Frontend UI文档: `demo/frontend/src/training/README.md`
- Tasks进度: `openspec/changes/enable-llm-finetuning-pipeline/tasks.md`
