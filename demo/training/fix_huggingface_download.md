# 🔧 解决 HuggingFace 模型下载问题

## 问题描述

```
ConnectionResetError(104, 'Connection reset by peer')
thrown while requesting HEAD https://huggingface.co/liuhaotian/llava-v1.5-7b/...
```

这是网络连接 HuggingFace 时失败,通常发生在:
1. 中国大陆访问 HuggingFace (被墙)
2. 网络不稳定
3. Docker 容器网络配置问题

---

## ✅ 解决方案

### 方案 1: 使用 HuggingFace 镜像 (推荐,最快)

使用国内镜像站加速下载:

**步骤 1: 修改环境变量**

编辑 `demo/docker-compose.yml`,在 `training-api` 服务中添加:

```yaml
services:
  training-api:
    environment:
      # 使用 HuggingFace 镜像
      HF_ENDPOINT: https://hf-mirror.com
      # 或者使用 modelscope
      # HF_ENDPOINT: https://www.modelscope.cn
```

**步骤 2: 重启服务**

```bash
cd demo
docker-compose restart training-api
```

**步骤 3: 重新开始训练**

点击 "Start Training" 按钮重试。

---

### 方案 2: 提前下载模型 (离线使用)

如果你有代理或者可以访问 HuggingFace,可以提前下载模型。

**步骤 1: 在宿主机下载模型**

```bash
# 安装 huggingface-cli
pip install huggingface-hub

# 登录 (如果模型需要权限)
huggingface-cli login

# 下载模型到本地
huggingface-cli download \
  liuhaotian/llava-v1.5-7b \
  --local-dir /Users/mbp/Desktop/Work/Life/IDoctor/sam2/models/llava-v1.5-7b \
  --local-dir-use-symlinks False
```

**步骤 2: 挂载到 Docker 容器**

编辑 `demo/docker-compose.yml`:

```yaml
services:
  training-api:
    volumes:
      - ../models:/app/models  # 挂载模型目录
      - ./training:/app/training
      # ... 其他挂载
    environment:
      # 使用本地模型
      HF_HOME: /app/models
      TRANSFORMERS_CACHE: /app/models
```

**步骤 3: 修改前端配置**

在前端选择模型时,使用本地路径:

- 原来: `liuhaotian/llava-v1.5-7b`
- 改为: `/app/models/llava-v1.5-7b`

---

### 方案 3: 使用代理

如果你有 VPN/代理:

**步骤 1: 配置 Docker 代理**

编辑 `demo/docker-compose.yml`:

```yaml
services:
  training-api:
    environment:
      HTTP_PROXY: http://host.docker.internal:7890
      HTTPS_PROXY: http://host.docker.internal:7890
      NO_PROXY: localhost,127.0.0.1
```

**步骤 2: 重启服务**

```bash
docker-compose restart training-api
```

---

### 方案 4: 使用 ModelScope (国内替代方案)

使用魔搭社区的模型:

**步骤 1: 修改代码使用 ModelScope**

编辑 `demo/training/core/trainers/lora_trainer.py`:

在文件开头添加:

```python
# 使用 ModelScope
import os
os.environ['MODELSCOPE_CACHE'] = '/app/models'

# 导入 modelscope
try:
    from modelscope import snapshot_download
    USE_MODELSCOPE = True
except ImportError:
    USE_MODELSCOPE = False
```

在 `setup()` 方法中,模型加载部分修改为:

```python
if USE_MODELSCOPE and not self.config.model.name.startswith('/'):
    # 使用 ModelScope 下载
    model_id = "AI-ModelScope/llava-v1.5-7b"  # ModelScope 上的模型 ID
    cache_dir = snapshot_download(model_id)
    self.config.model.name = cache_dir
```

**步骤 2: 安装依赖**

在 `demo/training_api/requirements.txt` 添加:

```
modelscope
```

**步骤 3: 重新构建镜像**

```bash
docker-compose build training-api
docker-compose up -d
```

---

## 🚀 快速修复 (推荐)

**最简单的方法: 使用 HuggingFace 镜像**

1. 编辑 `demo/docker-compose.yml`:

```yaml
services:
  training-api:
    environment:
      - HF_ENDPOINT=https://hf-mirror.com
```

2. 重启:

```bash
docker-compose restart training-api
```

3. 重新点击 "Start Training"

---

## 📊 验证是否成功

重新开始训练后,观察日志:

**成功的标志:**
```
[Job xxx] Creating trainer and loading model...
[Job xxx] Loading model with QLoRA settings...
Loading checkpoint shards: 100%|██████████| 3/3 [00:05<00:00,  1.67s/it]
[Job xxx] Model loaded successfully!
[Job xxx] Loading datasets...
[Job xxx] Starting actual training...
```

**如果还是失败,检查:**

```bash
# 查看完整错误日志
docker-compose logs -f training-api

# 测试容器内网络
docker-compose exec training-api ping hf-mirror.com
docker-compose exec training-api curl -I https://hf-mirror.com

# 检查 DNS
docker-compose exec training-api cat /etc/resolv.conf
```

---

## 💡 其他提示

### 模型很大 (13GB)

LLaVA-v1.5-7b 模型约 13GB,下载需要:
- 良好的网络: ~10-30 分钟
- 慢速网络: 1-2 小时
- 中国大陆无镜像: 可能失败或极慢

**建议:**
1. 第一次训练使用方案 1 (镜像)
2. 下载完成后会缓存,后续训练秒开
3. 或使用方案 2 预下载

### 检查磁盘空间

```bash
# 检查 Docker 卷空间
docker system df

# 需要至少 20GB 空余空间 (模型 13GB + 训练数据 + checkpoints)
df -h
```

### HuggingFace Token (某些模型需要)

如果模型需要授权:

```bash
# 在容器内设置
docker-compose exec training-api bash
export HF_TOKEN="hf_xxxxxxxxxxxxx"
```

或在 docker-compose.yml 中:

```yaml
services:
  training-api:
    environment:
      - HF_TOKEN=hf_xxxxxxxxxxxxx
```
