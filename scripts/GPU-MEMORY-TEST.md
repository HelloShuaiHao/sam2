# 显存泄漏修复 - 测试指南

## 修复内容

1. **添加详细的显存监控日志**
   - GPU 内存使用情况（清理前后对比）
   - Images tensor 的大小、类型和设备信息
   - 每个清理步骤的执行情况

2. **改进显存释放逻辑**
   - 先将 tensor 移到 CPU 释放 GPU 显存
   - 正确的清理顺序：reset_state → 清理 images → 清理其他结构
   - 递归清理所有嵌套的 tensors

3. **配置日志输出**
   - 所有日志输出到 stdout（Docker 可以捕获）
   - 日志级别设置为 INFO
   - 格式化输出，便于查看

## 测试步骤

### 1. 重新构建并启动服务

```bash
cd /Users/mbp/Desktop/Work/Life/IDoctor/sam2

# 停止并重新构建
docker compose down
docker compose build
docker compose up
```

### 2. 在新终端监控 GPU 显存

```bash
# 终端 1: 监控 GPU 显存变化
watch -n 1 nvidia-smi

# 或者只看显存占用
watch -n 1 "nvidia-smi | grep MiB"
```

### 3. 在另一个终端监控日志

```bash
# 终端 2: 查看详细日志
cd /Users/mbp/Desktop/Work/Life/IDoctor/sam2
./scripts/monitor-gpu-logs.sh

# 或者直接使用 docker compose logs
docker compose logs -f | grep -E "GPU memory|Clearing|Deleted|session"
```

### 4. 执行测试操作

1. **打开浏览器**，访问 SAM2 页面
2. **选择第一个视频**（例如 dog.mp4）
3. **等待加载完成**，观察 GPU 显存和日志
4. **切换到第二个视频**（例如 coffee.mp4）
5. **关键**：观察日志输出

### 5. 预期日志输出

切换视频时，你应该看到类似这样的日志：

```
2025-11-18 14:43:16 - inference.predictor - INFO - GPU memory BEFORE cleanup - allocated: 4500 MiB, reserved: 5000 MiB
2025-11-18 14:43:16 - inference.predictor - INFO - Calling predictor.reset_state()
2025-11-18 14:43:16 - sam2.sam2_video_predictor - INFO - Resetting inference state...
2025-11-18 14:43:16 - sam2.sam2_video_predictor - INFO - Clearing 10 cached features
2025-11-18 14:43:16 - inference.predictor - INFO - Clearing images tensor/loader, type: <class 'torch.Tensor'>
2025-11-18 14:43:16 - inference.predictor - INFO - Images tensor shape: torch.Size([120, 3, 1024, 1024]), device: cuda:0, size: 1440 MiB
2025-11-18 14:43:16 - inference.predictor - INFO - Deleted images tensor
2025-11-18 14:43:16 - inference.predictor - INFO - Clearing inference_state data structures
2025-11-18 14:43:16 - inference.predictor - INFO - Running garbage collection and clearing CUDA cache
2025-11-18 14:43:17 - inference.predictor - INFO - GPU memory AFTER cleanup - allocated: 1200 MiB, reserved: 2000 MiB
2025-11-18 14:43:17 - inference.predictor - INFO - removed session xxx and cleared GPU memory
```

### 6. 验证结果

✅ **成功标志**：
- 日志显示 `GPU memory BEFORE cleanup` 和 `AFTER cleanup`
- `AFTER cleanup` 的 `allocated` 内存明显减少（应该减少约 1440 MiB 或更多）
- nvidia-smi 显示 GPU 显存从 9000M 降回到 5000M 左右
- 切换多个视频后，显存不再持续增长

❌ **失败标志**：
- 没有看到 GPU memory 相关日志
- `AFTER cleanup` 的内存没有明显减少
- nvidia-smi 显示显存持续增长

## 如果没有看到日志

如果你没有看到 "GPU memory" 相关的日志，说明：

1. **closeSession 没有被调用** - 检查前端是否正确调用了 closeSession
2. **Docker 镜像没有更新** - 确认 `docker compose build` 成功
3. **日志被过滤** - 尝试直接查看所有日志：
   ```bash
   docker compose logs -f
   ```

## 常见问题

### Q1: 日志中只有 HTTP 请求，没有 GPU 日志？
A: 确保你已经重新构建：`docker compose down && docker compose build && docker compose up`

### Q2: 显存还是没有释放？
A: 发送以下信息：
```bash
# 1. 查看所有日志
docker compose logs > /tmp/sam2-logs.txt

# 2. 查看 GPU 相关日志
docker compose logs 2>&1 | grep -C 10 "GPU memory"

# 3. 查看是否有错误
docker compose logs 2>&1 | grep -i error
```

### Q3: 如何确认新代码已生效？
A: 启动时应该看到：
```
2025-11-18 XX:XX:XX - server.app - INFO - === SAM2 Backend Starting ===
```

## 下一步

测试完成后，请分享：
1. 完整的日志输出（特别是包含 "GPU memory" 的部分）
2. nvidia-smi 的显存变化截图
3. 是否成功释放显存

祝测试顺利！
