# 🚀 SAM2 Training API - 快速部署指南

## 📦 新增内容总结

### Docker服务架构

你的SAM2项目现在有**3个Docker容器**：

```
┌─────────────────────────────────────────────────┐
│  Frontend (7262)                                │
│  ├─ SAM2 视频标注界面                           │
│  └─ LLM训练工作流UI ✨新增                      │
├─────────────────────────────────────────────────┤
│  Backend (7263)                                 │
│  ├─ SAM2推理服务                                │
│  └─ GraphQL API                                 │
├─────────────────────────────────────────────────┤
│  Training API (7264) ✨新增                     │
│  ├─ 数据准备 (SAM2 → LLaVA格式)               │
│  ├─ 模型训练 (LoRA/QLoRA for 8GB GPU)         │
│  ├─ 实验跟踪                                    │
│  └─ 模型导出                                    │
└─────────────────────────────────────────────────┘
```

### 新增文件

```
sam2/
├── docker-compose.yaml          # ✅ 已更新 (新增training-api服务)
├── training-api.Dockerfile      # ✅ 新建 (Training API镜像)
├── deploy.sh                    # ✅ 新建 (一键部署脚本)
├── DOCKER_DEPLOYMENT.md         # ✅ 新建 (详细部署文档)
├── .dockerignore                # ✅ 新建
│
├── demo/training_api/           # ✅ 新建 (FastAPI后端)
│   ├── main.py                  # FastAPI应用
│   ├── models.py                # Pydantic数据模型
│   ├── requirements.txt         # Python依赖
│   ├── README.md                # API文档
│   └── routes/                  # 18个API端点
│       ├── data_prep.py         # 数据准备
│       ├── training.py          # 训练管理
│       ├── experiments.py       # 实验跟踪
│       └── export.py            # 模型导出
│
└── demo/frontend/src/           # ✅ 已更新 (React UI)
    ├── lib/
    │   ├── utils.ts             # 工具函数
    │   └── api-client.ts        # API客户端
    ├── components/ui/           # Shadcn/ui组件
    │   ├── button.tsx
    │   ├── card.tsx
    │   ├── progress.tsx
    │   └── badge.tsx
    └── training/                # 训练UI模块
        ├── TrainingWorkflow.tsx      # 主工作流
        ├── DataPreparationStep.tsx   # 数据准备UI
        ├── TrainingConfigStep.tsx    # 训练配置UI
        ├── TrainingMonitorStep.tsx   # 训练监控UI
        ├── ExportStep.tsx            # 模型导出UI
        ├── ExperimentDashboard.tsx   # 实验仪表板
        └── README.md                 # UI文档
```

## 🎯 一键部署

### 方式1: 使用自动化脚本 (推荐)

```bash
cd ~/Desktop/sam2

# 运行部署脚本
./deploy.sh

# 选择选项 1: Full deployment
```

脚本会自动：
- ✅ 检查Docker和GPU
- ✅ 创建必要目录
- ✅ 构建镜像
- ✅ 启动所有服务
- ✅ 健康检查
- ✅ 显示访问地址

### 方式2: 手动部署

```bash
cd ~/Desktop/sam2

# 1. 创建目录
mkdir -p checkpoints
mkdir -p demo/training/output

# 2. 构建并启动
docker compose up -d --build

# 3. 查看日志
docker compose logs -f training-api

# 4. 检查状态
docker compose ps
```

## 📍 访问地址

部署成功后，访问以下地址：

| 服务 | 地址 | 说明 |
|------|------|------|
| **Frontend** | http://ai.bygpu.com:7262 | Web界面 + 训练UI |
| **Backend** | http://ai.bygpu.com:7263 | SAM2推理API |
| **Training API** | http://ai.bygpu.com:7264 | 训练管理API |
| **API文档** | http://ai.bygpu.com:7264/docs | Swagger文档 |

## 🎨 UI功能演示

### 1. 训练工作流
访问: `http://ai.bygpu.com:7262/training`

**4步向导**:
1. **数据准备** - 上传SAM2导出，转换格式，验证质量
2. **训练配置** - 选择模型(LLaVA-7B QLoRA推荐)，设置超参数
3. **训练监控** - 实时查看进度、损失曲线、ETA
4. **模型导出** - 下载LoRA适配器或完整模型

### 2. 实验仪表板
访问: `http://ai.bygpu.com:7262/experiments`

**功能**:
- 查看所有训练实验
- 对比多个实验的指标
- 删除旧实验
- 按状态/损失排序

## 🧪 测试API

```bash
# 1. 健康检查
curl http://localhost:7264/health

# 2. 转换SAM2数据
curl -X POST http://localhost:7264/api/data/convert \
  -H "Content-Type: application/json" \
  -d '{
    "sam2_zip_path": "/data/your_export.zip",
    "output_dir": "/app/output",
    "target_format": "llava"
  }'

# 3. 启动训练
curl -X POST http://localhost:7264/api/train/start \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "model_name": "liuhaotian/llava-v1.5-7b",
      "use_qlora": true,
      "num_epochs": 3,
      "batch_size": 1,
      "train_data_path": "/app/output/splits/train.jsonl",
      "val_data_path": "/app/output/splits/val.jsonl",
      "output_dir": "/app/checkpoints"
    },
    "experiment_name": "test-training",
    "tags": ["qlora", "8gb"]
  }'

# 4. 查看训练状态
curl http://localhost:7264/api/train/{job_id}/status
```

## 🔍 查看日志

```bash
# 查看training API日志
docker compose logs -f training-api

# 查看所有服务日志
docker compose logs -f

# 只看错误日志
docker compose logs training-api | grep ERROR
```

## 🐛 常见问题

### Q: Training API启动失败？
```bash
# 检查日志
docker compose logs training-api

# 常见原因：
# 1. GPU驱动问题 - 检查: nvidia-smi
# 2. 端口被占用 - 检查: netstat -tulpn | grep 7264
# 3. 内存不足 - 检查: free -h
```

### Q: 前端无法连接Training API？
```bash
# 1. 检查环境变量
docker compose exec frontend env | grep VITE_API_URL

# 2. 验证API可访问
curl http://ai.bygpu.com:7264/health

# 3. 重启frontend
docker compose restart frontend
```

### Q: GPU内存不足？
**解决方案**:
1. 使用QLoRA配置 (推荐8GB GPU)
2. 降低batch_size到1
3. 减少max_length到1024
4. 调整Docker内存限制

```yaml
# docker-compose.yaml
training-api:
  deploy:
    resources:
      limits:
        memory: 16G  # 降低限制
```

### Q: 如何分配多GPU？
如果你有多个GPU，修改`docker-compose.yaml`:

```yaml
backend:
  environment:
    - CUDA_VISIBLE_DEVICES=0  # GPU 0用于推理

training-api:
  environment:
    - CUDA_VISIBLE_DEVICES=1  # GPU 1用于训练
```

## 🔄 更新和维护

### 更新代码

```bash
cd ~/Desktop/sam2
git pull

# 重新构建training API
docker compose build training-api

# 重启服务
docker compose up -d training-api
```

### 清理资源

```bash
# 停止所有服务
docker compose down

# 清理旧的检查点（释放磁盘空间）
rm -rf checkpoints/old-*
rm -rf demo/training/output/old-*

# 清理Docker缓存
docker system prune -a
```

## 📊 资源监控

### 实时GPU监控

```bash
# 方式1: 在host机器
watch -n 1 nvidia-smi

# 方式2: 在容器内
docker compose exec training-api nvidia-smi
```

### 磁盘使用

```bash
# 检查各目录大小
du -sh checkpoints/
du -sh demo/training/output/
```

## 📚 详细文档

- **部署详情**: `DOCKER_DEPLOYMENT.md`
- **API文档**: `demo/training_api/README.md`
- **UI文档**: `demo/frontend/src/training/README.md`
- **任务进度**: `openspec/changes/enable-llm-finetuning-pipeline/tasks.md`

## 🎉 开始训练

一切就绪后：

1. **准备数据**:
   - 在SAM2界面标注视频
   - 导出标注数据

2. **启动训练**:
   - 访问 `http://ai.bygpu.com:7262/training`
   - 按照4步向导操作
   - 选择8GB GPU优化的QLoRA配置

3. **监控训练**:
   - 实时查看进度和损失
   - 预计训练时间：3 epochs约1-2小时（取决于数据量）

4. **导出模型**:
   - 训练完成后下载LoRA适配器
   - 文件大小约10-50 MB

5. **使用模型**:
   ```python
   from peft import PeftModel
   from transformers import AutoModel

   base_model = AutoModel.from_pretrained("liuhaotian/llava-v1.5-7b")
   model = PeftModel.from_pretrained(base_model, "./lora_adapters")
   ```

## 🆘 获取帮助

遇到问题？

1. 查看日志: `docker compose logs training-api`
2. 查看文档: `DOCKER_DEPLOYMENT.md`
3. 检查API: `http://ai.bygpu.com:7264/docs`
4. 运行脚本: `./deploy.sh` 选择选项6查看状态

---

**祝训练顺利！** 🚀
