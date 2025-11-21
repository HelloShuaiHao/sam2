# 🔧 训练取消功能修复

## 问题描述

**原始问题:**
- 点击 "Cancel Training" 后,后台线程继续运行
- HuggingFace 模型下载重试不会停止
- 资源泄漏:GPU 内存、线程、网络连接持续占用

## ✅ 修复内容

### 1. 添加取消检查点

在训练流程的关键位置添加了取消检查:

```python
# routes/training.py

def _check_cancellation(job_id: str) -> bool:
    """检查任务是否被取消"""
    if job_id not in _active_jobs:
        return True
    return _active_jobs[job_id]["status"] == JobStatus.CANCELLED

# 在关键操作前检查
if _check_cancellation(job_id):
    print(f"[Job {job_id}] Job cancelled")
    return
```

**检查点位置:**
1. ✅ 开始配置前
2. ✅ 模型加载前
3. ✅ 模型加载后
4. ✅ 数据加载后
5. ✅ 训练开始前

### 2. 资源清理机制

添加了 `finally` 块确保资源始终被释放:

```python
finally:
    # 清理线程引用
    if job_id in _job_threads:
        del _job_threads[job_id]

    # 清理 GPU 内存
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 3. 区分取消和失败

修复了异常处理,区分主动取消和真实错误:

```python
except Exception as e:
    if _check_cancellation(job_id):
        # 主动取消,不是错误
        print(f"[Job {job_id}] Job was cancelled")
    else:
        # 真实错误
        _active_jobs[job_id]["status"] = JobStatus.FAILED
        _active_jobs[job_id]["error_message"] = str(e)
```

---

## ⚠️ 已知限制

### HuggingFace 下载无法立即中断

**问题:**
当模型正在下载时(显示 "Retrying" 消息),**无法立即停止下载**。

**原因:**
- HuggingFace `transformers` 库的下载器不支持外部中断
- 下载使用的是 `requests` 库,没有暴露取消接口
- 线程间通信无法强制终止 I/O 操作

**实际行为:**
```
用户点击 Cancel
    ↓
后端设置 status = CANCELLED
    ↓
后台线程检查状态(每个检查点)
    ↓
如果在下载中 → 下载继续直到当前文件完成
如果在其他阶段 → 立即退出
    ↓
下载完成后 → 线程立即退出
    ↓
资源清理
```

**时间线:**
- ✅ **配置阶段取消**: 立即生效(< 1 秒)
- ⚠️ **下载阶段取消**: 当前分片下载完成后生效(可能 10-60 秒)
- ✅ **模型加载后取消**: 立即生效(< 1 秒)
- ⚠️ **训练中取消**: 当前 step 完成后生效(10-30 秒)

---

## 🎯 最佳实践

### 用户端

1. **首次下载时**:
   - 如果取消,等待 1-2 分钟确认线程退出
   - 查看日志确认清理完成
   - 模型文件会部分下载到缓存,下次继续

2. **训练过程中**:
   - 取消后等待当前 step 完成
   - 观察 GPU 内存释放(`nvidia-smi`)
   - checkpoint 可能已保存,可以恢复训练

### 开发端

**监控取消状态:**

```bash
# 查看日志
docker compose logs -f training-api | grep "cancelled"

# 检查线程清理
docker compose logs -f training-api | grep "Cleaning up"

# 监控 GPU 内存
watch -n 1 nvidia-smi
```

**强制清理(紧急情况):**

```bash
# 重启服务(会终止所有训练)
docker compose restart training-api

# 清理 GPU 内存
docker compose exec training-api python -c "import torch; torch.cuda.empty_cache()"
```

---

## 🔬 技术细节

### 为什么不能强制中断下载?

**尝试过的方案:**

1. ❌ **发送 SIGINT 信号**
   - 会终止整个容器,不可接受

2. ❌ **使用 threading.Event**
   - 无法中断阻塞的 I/O 操作(socket.recv)

3. ❌ **使用 timeout**
   - HuggingFace 已经有重试机制,添加 timeout 会导致重试

4. ⚠️ **轮询检查**
   - 只能在操作间隙检查,无法中断正在进行的下载

**当前方案(最优):**

```python
# 在每个操作前后检查
操作前检查 → 执行操作 → 操作后检查 → 下一步
     ↑                              ↑
  可以取消                       可以取消
```

### 下载恢复机制

HuggingFace 下载器有内置的断点续传:

```python
# 第一次下载(被取消)
model_part_1.bin  ← 已下载,缓存
model_part_2.bin  ← 下载中,部分缓存
model_part_3.bin  ← 未开始

# 第二次下载(重试)
model_part_1.bin  ← 跳过(从缓存读取)
model_part_2.bin  ← 继续下载
model_part_3.bin  ← 开始下载
```

所以多次取消/重试不会浪费带宽。

---

## 📊 测试验证

### 测试场景 1: 配置阶段取消

```
1. 点击 Start Training
2. 立即点击 Cancel Training
3. 预期:立即取消,无资源占用
```

**验证:**
```bash
docker compose logs training-api | tail -20
# 应该看到:
# [Job xxx] Job cancelled before starting
# [Job xxx] Cleaning up thread
```

### 测试场景 2: 下载阶段取消

```
1. 点击 Start Training
2. 等待看到 "Loading model" 消息
3. 点击 Cancel Training
4. 预期:当前下载完成后退出(最多 1-2 分钟)
```

**验证:**
```bash
# 观察日志
docker compose logs -f training-api

# 应该看到:
# [Job xxx] Loading model with QLoRA settings...
# 用户点击取消
# Downloading shards: 33%|███ ...  (继续一会儿)
# [Job xxx] Job cancelled after model load
# [Job xxx] Cleaning up thread
# [Job xxx] GPU cache cleared
```

### 测试场景 3: 训练阶段取消

```
1. 开始训练(使用已下载的模型)
2. 等待训练开始
3. 点击 Cancel Training
4. 预期:当前 step 完成后退出
```

---

## 🚀 未来改进方向

### 短期(可实现)

1. **添加取消进度提示**
   ```typescript
   // 前端显示
   "Cancelling... Waiting for current operation to complete"
   ```

2. **自动清理僵尸进程**
   ```python
   # 定期检查超时的 CANCELLED 任务
   async def cleanup_stale_jobs():
       for job_id, job in _active_jobs.items():
           if job["status"] == "cancelled":
               age = datetime.now() - job["completed_at"]
               if age > timedelta(minutes=5):
                   # 强制清理
   ```

3. **添加强制终止选项**
   ```python
   @router.post("/{job_id}/force-kill")
   async def force_kill_job(job_id: str):
       # 使用 multiprocessing.Process.terminate()
   ```

### 长期(需要上游支持)

1. **使用 multiprocessing 替代 threading**
   - 可以强制 kill 进程
   - 完全隔离资源

2. **自定义 HuggingFace 下载器**
   - 添加取消回调
   - 支持中断当前下载

3. **使用异步下载**
   - 使用 `aiohttp` 替代 `requests`
   - 可以取消异步任务

---

## 📝 总结

### ✅ 已修复

- ✅ 取消后线程会退出(虽然可能延迟)
- ✅ GPU 内存自动清理
- ✅ 线程引用正确删除
- ✅ 区分取消和失败状态
- ✅ 添加详细日志

### ⚠️ 已知限制

- ⚠️ 模型下载中取消有延迟(最多 1-2 分钟)
- ⚠️ 训练中取消需等待当前 step

### 💡 建议

1. **首次训练**:让模型下载完成,不要在下载时取消
2. **测试训练**:使用小数据集,可以快速取消
3. **生产训练**:使用预下载的模型,避免下载阶段

---

## 🐛 问题反馈

如果遇到取消不生效:

1. 查看日志确认线程是否清理
2. 检查 GPU 内存是否释放
3. 如果超过 5 分钟仍未退出,重启服务:
   ```bash
   docker compose restart training-api
   ```
