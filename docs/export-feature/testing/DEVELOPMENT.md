# SAM2 开发模式快速启动指南

## 🚀 开发模式 (热重载，无需重新构建)

使用 `docker-compose.dev.yaml` 配置，修改代码后自动刷新，无需重新 build。

### 首次启动

```bash
# 1. 首次构建镜像（只需要一次）
docker compose -f docker-compose.dev.yaml build

# 2. 启动开发服务
docker compose -f docker-compose.dev.yaml up
```

### 日常开发

```bash
# 直接启动（修改代码会自动刷新）
docker compose -f docker-compose.dev.yaml up

# 或后台运行
docker compose -f docker-compose.dev.yaml up -d

# 查看日志
docker compose -f docker-compose.dev.yaml logs -f
```

### 停止服务

```bash
docker compose -f docker-compose.dev.yaml down
```

## ⚡ 热重载说明

### 前端 (React/TypeScript)
- **修改位置**: `demo/frontend/src/` 下的任何文件
- **生效时间**: 保存后 1-2 秒自动刷新浏览器
- **无需重启**: Vite 开发服务器自动检测变化

### 后端 (Python/Flask)
- **修改位置**: `demo/backend/server/` 下的任何 `.py` 文件
- **生效时间**: 保存后 2-3 秒自动重启服务
- **无需重启**: Gunicorn `--reload` 模式自动检测变化

## 📝 修改示例

### 修改前端组件
```bash
# 编辑文件
vim demo/frontend/src/common/components/export/ExportButton.tsx

# 保存后，浏览器自动刷新 ✅
```

### 修改后端 API
```bash
# 编辑文件
vim demo/backend/server/data/export_service.py

# 保存后，Gunicorn 自动重启 ✅
```

## 🔍 故障排除

### 前端没有热重载
```bash
# 检查前端日志
docker compose -f docker-compose.dev.yaml logs frontend-dev

# 重启前端服务
docker compose -f docker-compose.dev.yaml restart frontend-dev
```

### 后端修改不生效
```bash
# 检查后端日志
docker compose -f docker-compose.dev.yaml logs backend-dev

# 确认 GUNICORN_RELOAD 已启用
docker compose -f docker-compose.dev.yaml exec backend-dev env | grep RELOAD

# 手动重启后端
docker compose -f docker-compose.dev.yaml restart backend-dev
```

### 完全重新开始
```bash
# 停止并删除容器
docker compose -f docker-compose.dev.yaml down

# 重新构建（如果修改了 Dockerfile 或 package.json）
docker compose -f docker-compose.dev.yaml build

# 重新启动
docker compose -f docker-compose.dev.yaml up
```

## 📊 对比：生产模式 vs 开发模式

| 特性 | 生产模式 | 开发模式 |
|------|----------|----------|
| 启动命令 | `docker compose up` | `docker compose -f docker-compose.dev.yaml up` |
| 代码修改 | 需要重新 build | 自动热重载 ✅ |
| 构建时间 | 5-10 分钟 | 首次 5-10 分钟，之后秒级 |
| 前端 | nginx 静态文件 | Vite dev server |
| 后端 | Gunicorn 生产模式 | Gunicorn --reload |
| 适用场景 | 部署到服务器 | 本地开发调试 |

## 💡 提示

1. **开发模式启动后，修改代码无需任何操作，保存即生效**
2. 如果修改了 `package.json` 或 `requirements.txt`，需要重新 build
3. 开发模式仅用于本地开发，不要用于生产环境
4. 第一次启动需要 build，之后就很快了

## 🎯 推荐工作流

```bash
# 早上启动
docker compose -f docker-compose.dev.yaml up -d

# 开发一整天，修改代码，保存，自动刷新 ✅

# 下班关闭
docker compose -f docker-compose.dev.yaml down
```

没错，就是这么简单！🎉
