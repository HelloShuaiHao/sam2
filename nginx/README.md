# SAM2 Training Platform - Nginx Deployment Guide

## 📁 Nginx配置文件位置

新创建的配置文件：`nginx/sam2.conf`

## 🎯 配置方案

### 统一访问架构

```
http://ai.bygpu.com
├── /                          → Frontend (7262)
│   ├── /training              → 训练工作流UI
│   └── /experiments           → 实验仪表板
│
├── /api/sam2                  → Backend (7263)
│   ├── /api/sam2/graphql      → GraphQL API
│   └── /api/sam2/...          → SAM2推理服务
│
└── /api/training              → Training API (7264)
    ├── /api/training/data     → 数据准备端点
    ├── /api/training/train    → 训练管理端点
    ├── /api/training/experiments → 实验跟踪端点
    ├── /api/training/export   → 模型导出端点
    ├── /api/training/docs     → Swagger文档
    └── /api/training/redoc    → ReDoc文档
```

### URL映射表

| 原始端口访问 | Nginx代理后访问 | 说明 |
|--------------|----------------|------|
| http://ai.bygpu.com:7262 | http://ai.bygpu.com/ | 前端界面 |
| http://ai.bygpu.com:7263 | http://ai.bygpu.com/api/sam2 | SAM2 API |
| http://ai.bygpu.com:7264 | http://ai.bygpu.com/api/training | Training API |
| http://ai.bygpu.com:7264/docs | http://ai.bygpu.com/api/training/docs | API文档 |

## 🚀 部署步骤

### 方式1: 使用项目内的配置文件

```bash
# 1. 创建软链接到Nginx配置目录
sudo ln -s /home/bygpu/Desktop/sam2/nginx/sam2.conf \
    /etc/nginx/sites-available/sam2.conf

# 2. 启用配置
sudo ln -s /etc/nginx/sites-available/sam2.conf \
    /etc/nginx/sites-enabled/sam2.conf

# 3. 测试配置
sudo nginx -t

# 4. 重新加载Nginx
sudo systemctl reload nginx
```

### 方式2: 直接复制配置文件

```bash
# 1. 复制配置文件
sudo cp nginx/sam2.conf /etc/nginx/sites-available/sam2.conf

# 2. 启用配置
sudo ln -s /etc/nginx/sites-available/sam2.conf \
    /etc/nginx/sites-enabled/sam2.conf

# 3. 测试配置
sudo nginx -t

# 4. 重新加载Nginx
sudo systemctl reload nginx
```

### 方式3: 集成到现有Nginx配置

如果你已经有Nginx配置文件，可以将`nginx/sam2.conf`的内容添加到现有配置中。

## 🔧 配置特性

### 1. 反向代理
- 所有服务通过统一域名访问
- 自动路由到对应的Docker容器

### 2. WebSocket支持
- 支持实时推理（SAM2 Backend）
- 支持实时训练监控（Training API）

### 3. 大文件上传
- `client_max_body_size 2G` - 支持大型SAM2导出文件
- 流式上传，不缓冲

### 4. 性能优化
- Gzip压缩
- 静态文件缓存（1年）
- Keep-alive连接池

### 5. 安全配置
- CORS配置
- XSS防护
- 点击劫持防护
- 隐藏服务器版本

## 🔒 HTTPS配置（生产环境）

### 1. 获取SSL证书

```bash
# 使用Let's Encrypt免费证书
sudo apt-get install certbot python3-certbot-nginx

# 自动配置SSL
sudo certbot --nginx -d ai.bygpu.com

# 证书会自动续期
```

### 2. 手动配置SSL

如果已有证书，取消注释配置文件中的SSL部分：

```nginx
listen 443 ssl http2;
ssl_certificate /etc/nginx/ssl/ai.bygpu.com.crt;
ssl_certificate_key /etc/nginx/ssl/ai.bygpu.com.key;
```

## 📊 监控和日志

### 查看访问日志

```bash
# 实时查看所有访问
sudo tail -f /var/log/nginx/sam2_access.log

# 查看错误日志
sudo tail -f /var/log/nginx/sam2_error.log

# 过滤Training API的请求
sudo tail -f /var/log/nginx/sam2_access.log | grep "/api/training"
```

### 日志格式

```
访问日志: /var/log/nginx/sam2_access.log
错误日志: /var/log/nginx/sam2_error.log
```

## 🧪 测试配置

### 1. 测试基本连通性

```bash
# 测试主页
curl http://ai.bygpu.com/

# 测试SAM2 API
curl http://ai.bygpu.com/api/sam2/

# 测试Training API
curl http://ai.bygpu.com/api/training/health

# 测试API文档
curl http://ai.bygpu.com/api/training/docs
```

### 2. 测试文件上传

```bash
# 测试大文件上传
curl -X POST http://ai.bygpu.com/api/training/data/convert \
  -H "Content-Type: application/json" \
  -d '{
    "sam2_zip_path": "/data/test.zip",
    "output_dir": "/app/output",
    "target_format": "llava"
  }'
```

### 3. 测试CORS

```bash
# 测试OPTIONS请求
curl -X OPTIONS http://ai.bygpu.com/api/training/health \
  -H "Origin: http://ai.bygpu.com" \
  -v
```

## 🔄 更新配置

### 修改配置后重新加载

```bash
# 1. 测试配置语法
sudo nginx -t

# 2. 如果测试通过，重新加载
sudo systemctl reload nginx

# 3. 如果需要完全重启
sudo systemctl restart nginx
```

### 回滚配置

```bash
# 禁用新配置
sudo rm /etc/nginx/sites-enabled/sam2.conf

# 重新加载
sudo systemctl reload nginx
```

## 🐛 故障排查

### Nginx无法启动

```bash
# 检查配置语法
sudo nginx -t

# 查看详细错误
sudo systemctl status nginx
sudo journalctl -u nginx -n 50
```

### 502 Bad Gateway

原因：后端服务未启动

```bash
# 检查Docker容器状态
docker compose ps

# 重启后端服务
docker compose restart frontend backend training-api
```

### 504 Gateway Timeout

原因：超时设置过短

```nginx
# 增加超时时间
proxy_connect_timeout 600s;
proxy_send_timeout 600s;
proxy_read_timeout 600s;
```

### CORS错误

检查配置中的CORS头是否正确：

```nginx
add_header Access-Control-Allow-Origin * always;
add_header Access-Control-Allow-Methods "GET, POST, OPTIONS" always;
```

## 📝 Frontend环境变量更新

更新`docker-compose.yaml`中的frontend环境变量：

```yaml
frontend:
  environment:
    # 使用Nginx代理的路径（无需端口号）
    - VITE_API_URL=http://ai.bygpu.com/api/training
    # 或使用相对路径
    - VITE_API_URL=/api/training
```

然后重新构建frontend：

```bash
docker compose build frontend
docker compose up -d frontend
```

## 🔗 访问地址（Nginx代理后）

| 功能 | URL | 说明 |
|------|-----|------|
| **主页** | http://ai.bygpu.com | SAM2界面 |
| **训练UI** | http://ai.bygpu.com/training | 训练工作流 |
| **实验仪表板** | http://ai.bygpu.com/experiments | 实验管理 |
| **SAM2 API** | http://ai.bygpu.com/api/sam2 | GraphQL |
| **Training API** | http://ai.bygpu.com/api/training | REST API |
| **API文档** | http://ai.bygpu.com/api/training/docs | Swagger |
| **健康检查** | http://ai.bygpu.com/health | Nginx健康 |

## 🎯 完整部署流程

```bash
# 1. 部署Nginx配置
sudo ln -s /home/bygpu/Desktop/sam2/nginx/sam2.conf \
    /etc/nginx/sites-available/sam2.conf
sudo ln -s /etc/nginx/sites-available/sam2.conf \
    /etc/nginx/sites-enabled/sam2.conf

# 2. 测试配置
sudo nginx -t

# 3. 重新加载Nginx
sudo systemctl reload nginx

# 4. 更新frontend环境变量
cd ~/Desktop/sam2
# 编辑 docker-compose.yaml，更新VITE_API_URL

# 5. 重新构建并启动服务
docker compose up -d --build

# 6. 验证服务
curl http://ai.bygpu.com/health
curl http://ai.bygpu.com/api/training/health

# 7. 访问UI
# 浏览器打开: http://ai.bygpu.com
```

## 🆘 需要帮助？

遇到问题时的检查顺序：

1. **检查Nginx状态**: `sudo systemctl status nginx`
2. **检查Nginx日志**: `sudo tail -f /var/log/nginx/sam2_error.log`
3. **检查Docker容器**: `docker compose ps`
4. **检查端口监听**: `sudo netstat -tulpn | grep -E "7262|7263|7264"`
5. **测试后端直连**: `curl http://localhost:7264/health`

---

**配置文件已准备就绪！** 🚀

运行 `sudo nginx -t && sudo systemctl reload nginx` 启用新配置。
