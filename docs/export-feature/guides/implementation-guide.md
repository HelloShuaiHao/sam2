# SAM2 视频注释导出功能 - 实施指南

## 🎯 功能概述

为 SAM2 Demo 添加了视频注释导出功能，支持：
- ✅ **帧率可配置的导出**：用户可选择 0.5-30 FPS 进行导出
- ✅ **JSON 注释格式**：带有 RLE 编码的掩码、边界框和元数据
- ✅ **进度跟踪**：实时显示导出进度
- ✅ **后台处理**：导出作业在后台异步处理
- ✅ **自动下载**：完成后自动触发 ZIP 文件下载

## 📁 新增文件结构

### 后端 (Python)
```
demo/backend/server/
├── data/
│   ├── data_types.py          # ✅ 已更新 - 新增导出类型
│   ├── schema.py              # ✅ 已更新 - 新增 mutation/query
│   └── export_service.py      # ✅ 新建 - 导出服务核心逻辑
└── utils/                     # ✅ 新建目录
    ├── __init__.py
    ├── frame_sampler.py       # 基于时间的帧采样
    ├── rle_encoder.py         # RLE 掩码编码/解码
    └── annotation_serializer.py  # JSON 序列化器
```

### 前端 (TypeScript/React)
```
demo/frontend/src/common/components/export/  # ✅ 新建目录
├── FrameRateSelector.tsx      # 帧率选择组件
├── ExportConfigModal.tsx      # 配置模态框
├── ExportProgress.tsx         # 进度指示器
├── ExportButton.tsx           # 导出按钮（集成组件）
└── useExport.ts               # 自定义 Hook（状态管理）
```

## 🚀 启动和使用

### 启动 SAM2 Demo

使用 Docker Compose 启动（推荐方式）：

```bash
# 1. 构建并启动服务
docker compose up --build

# 或者后台运行
docker compose up -d --build

# 2. 访问应用
# 前端: http://localhost:7262
# 后端: http://localhost:7263/graphql
```

**服务说明**：
- `frontend`: React 应用 (端口 7262)
- `backend`: Flask + GraphQL API (端口 7263)
- 后端使用 GPU 加速 (需要 NVIDIA GPU)

### 停止服务

```bash
# 停止服务
docker compose down

# 停止并删除 volumes
docker compose down -v
```

### 查看日志

```bash
# 查看所有日志
docker compose logs -f

# 只查看后端日志
docker compose logs -f backend

# 只查看前端日志
docker compose logs -f frontend

# 查看导出相关日志
docker compose logs -f backend | grep -i export
```

## 🔌 集成步骤

### 步骤 1: 在 DemoVideoEditor 中添加导出按钮

打开 `demo/frontend/src/common/components/video/editor/DemoVideoEditor.tsx`，添加导出按钮：

```tsx
import ExportButton from '@/common/components/export/ExportButton';
import {trackletObjectsAtom, sessionAtom} from '@/demo/atoms';
import {useAtomValue} from 'jotai';

// 在组件内部
export default function DemoVideoEditor({video: inputVideo}: Props) {
  const session = useAtomValue(sessionAtom);
  const trackletObjects = useAtomValue(trackletObjectsAtom);

  // 计算视频元数据（用于导出）
  const videoMetadata = {
    duration: video?.metadata?.duration || 0,
    fps: 30, // 从视频元数据中获取
    totalFrames: video?.metadata?.totalFrames || 0,
    width: video?.metadata?.width || 1920,
    height: video?.metadata?.height || 1080,
  };

  return (
    <div {...stylex.props(styles.container)}>
      <VideoEditor
        // ... 现有 props ...
      >
        {/* 在工具栏或适当位置添加导出按钮 */}
        <ExportButton
          sessionId={session?.id || null}
          videoMetadata={videoMetadata}
          hasTrackedObjects={trackletObjects.length > 0}
        />

        {/* ... 其他子组件 ... */}
      </VideoEditor>
    </div>
  );
}
```

### 步骤 2: 重新构建 Docker 镜像

添加新文件后，需要重新构建镜像：

```bash
# 重新构建并启动
docker compose up --build

# 如果需要强制重建（清除缓存）
docker compose build --no-cache
docker compose up
```

### 步骤 3: 验证功能

1. 访问 http://localhost:7262
2. 上传或选择视频
3. 添加对象标注
4. 点击"Export"按钮
5. 配置导出参数
6. 等待导出完成
7. 下载 ZIP 文件

## 📊 Docker Compose 配置

### 环境变量

在 `docker-compose.yaml` 中已配置的关键环境变量：

```yaml
environment:
  - API_URL=http://localhost:7263          # 后端 API 地址
  - DEFAULT_VIDEO_PATH=gallery/05_default_juggle.mp4
  - VIDEO_ENCODE_FPS=24                    # 视频编码 FPS
  - VIDEO_ENCODE_MAX_WIDTH=1280
  - VIDEO_ENCODE_MAX_HEIGHT=720
```

### 数据持久化

导出文件保存在挂载的 volume 中：

```yaml
volumes:
  - ./demo/data/:/data/:rw
```

导出文件路径：`./demo/data/exports/{job_id}.zip`

### GPU 配置

需要 NVIDIA GPU 和 nvidia-docker runtime：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

**如果没有 GPU**，需要修改配置：
1. 在 `docker-compose.yaml` 中移除 `deploy` 部分
2. 设置环境变量：`SAM2_DEMO_FORCE_CPU_DEVICE=1`

## 🔧 配置选项

### 自定义端口

修改 `docker-compose.yaml` 中的端口映射：

```yaml
services:
  frontend:
    ports:
      - "8080:80"  # 修改为 8080

  backend:
    ports:
      - "8081:5000"  # 修改为 8081
    environment:
      - API_URL=http://localhost:8081  # 同步更新
```

### 调整 Worker 数量

```yaml
backend:
  environment:
    - GUNICORN_WORKERS=2    # 增加 workers
    - GUNICORN_THREADS=4    # 增加 threads
```

### 修改视频编码设置

```yaml
backend:
  environment:
    - VIDEO_ENCODE_FPS=30          # 提高 FPS
    - VIDEO_ENCODE_MAX_WIDTH=1920  # 提高分辨率
    - VIDEO_ENCODE_MAX_HEIGHT=1080
    - VIDEO_ENCODE_CRF=18          # 降低 CRF = 更高质量
```

## 🐛 故障排除

### 问题 1: 容器启动失败

**症状**: `docker compose up` 报错

**检查步骤**:
```bash
# 查看容器状态
docker compose ps

# 查看详细日志
docker compose logs backend
docker compose logs frontend

# 检查端口占用
lsof -i :7262
lsof -i :7263
```

### 问题 2: GPU 不可用

**症状**: 后端日志显示 "CUDA not available"

**解决方案**:
```bash
# 1. 检查 nvidia-docker 安装
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 2. 如果失败，使用 CPU 模式
# 在 docker-compose.yaml 中添加：
environment:
  - SAM2_DEMO_FORCE_CPU_DEVICE=1

# 并移除 deploy.resources 部分
```

### 问题 3: 导出文件无法访问

**症状**: 下载 404 或权限错误

**解决方案**:
```bash
# 检查 volume 挂载
docker compose exec backend ls -la /data/exports/

# 检查权限
docker compose exec backend chmod -R 777 /data/exports/

# 或者在 docker-compose.yaml 中添加：
user: "${UID}:${GID}"
```

### 问题 4: 前端无法连接后端

**症状**: 导出失败，网络错误

**解决方案**:
```bash
# 检查后端是否正常
curl http://localhost:7263/healthy

# 检查 GraphQL 端点
curl -X POST http://localhost:7263/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ __schema { types { name } } }"}'

# 检查前端配置
# 确保前端的 API 端点配置正确
```

### 问题 5: 导出进度卡住

**症状**: 导出一直显示 "Processing"

**调试步骤**:
```bash
# 1. 查看后端日志
docker compose logs -f backend | grep -i "export\|error"

# 2. 进入容器检查
docker compose exec backend bash
cd /data/exports
ls -lh

# 3. 手动测试 GraphQL
curl -X POST http://localhost:7263/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query { exportJobStatus(jobId: \"YOUR_JOB_ID\") { status } }"
  }'
```

## 📂 导出文件位置

### 在容器内
```
/data/exports/{job_id}.zip
```

### 在宿主机
```
./demo/data/exports/{job_id}.zip
```

### 访问导出文件

```bash
# 列出所有导出
ls -lh demo/data/exports/

# 解压查看
unzip demo/data/exports/{job_id}.zip -d /tmp/export
cat /tmp/export/annotations.json | jq '.'
```

## 🔄 开发模式

### 前端开发（热重载）

如果需要前端热重载，使用本地开发模式：

```bash
# 1. 只启动后端
docker compose up backend

# 2. 本地运行前端
cd demo/frontend
yarn install
yarn dev --port 7262
```

### 后端开发（代码更改）

```bash
# 1. 修改代码后重新构建
docker compose build backend

# 2. 重启后端服务
docker compose up backend
```

或者使用 volume 挂载实现热重载：

```yaml
# 在 docker-compose.yaml 中添加
backend:
  volumes:
    - ./demo/backend/server:/app/server:ro
```

## 📊 性能监控

### 查看资源使用

```bash
# 查看容器资源使用
docker stats

# 查看 GPU 使用
docker compose exec backend nvidia-smi

# 查看导出目录大小
du -sh demo/data/exports/
```

### 清理旧导出文件

```bash
# 删除 24 小时前的导出
find demo/data/exports/ -name "*.zip" -mtime +1 -delete

# 或在容器内设置定时任务
docker compose exec backend bash -c "
  find /data/exports -name '*.zip' -mtime +1 -delete
"
```

## 📝 相关文档

- **API 参考**: `docs/export-feature/api/graphql-api.md`
- **测试指南**: `docs/export-feature/testing/testing-guide.md`
- **架构设计**: `docs/export-feature/guides/architecture.md`

## 🔗 有用的命令

```bash
# 完整的重启流程
docker compose down
docker compose build --no-cache
docker compose up -d
docker compose logs -f

# 检查服务健康状态
curl http://localhost:7263/healthy

# 备份导出数据
tar -czf exports_backup.tar.gz demo/data/exports/

# 清理 Docker 资源
docker compose down --volumes --remove-orphans
docker system prune -a
```
