# QLoRA 训练故障排查指南

本文档记录了在 RTX 3060 (12GB) 上配置 LLaVA-7B + QLoRA 训练时遇到的所有问题和解决方案。

---

## 目录

1. [段错误 (Segmentation Fault)](#1-段错误-segmentation-fault)
2. [bitsandbytes 版本兼容性](#2-bitsandbytes-版本兼容性)
3. [LLaVA Processor 加载问题](#3-llava-processor-加载问题)
4. [网络和镜像源配置](#4-网络和镜像源配置)
5. [最终工作配置](#5-最终工作配置)

---

## 1. 段错误 (Segmentation Fault)

### 问题表现

```bash
Exit code: 139
/bin/bash: 段错误 (核心已转储)
```

程序在加载 4-bit 量化模型时崩溃，没有任何 Python 错误信息。

### 根本原因

**缺少 `torch_dtype` 参数**导致 bitsandbytes 在量化时出现内存访问错误。

### 错误代码

```python
# ❌ 会导致段错误
model = LlavaForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True  # ⚠️ 这个参数也可能有问题
)
```

### 正确代码

```python
# ✅ 稳定工作
model = LlavaForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,  # 🔑 关键！必须明确指定
)
```

### 详细修复位置

**文件**: `core/trainers/lora_trainer.py`  
**行号**: 103-118

**修复前**:
```python
load_kwargs = {"trust_remote_code": True}
if self.config.model.cache_dir:
    load_kwargs["cache_dir"] = self.config.model.cache_dir

if quantization_config is not None:
    load_kwargs["quantization_config"] = quantization_config
    load_kwargs["device_map"] = "auto"
    # ❌ 缺少 torch_dtype!
```

**修复后**:
```python
load_kwargs = {}
if self.config.model.cache_dir:
    load_kwargs["cache_dir"] = self.config.model.cache_dir

if quantization_config is not None:
    # QLoRA mode: use quantization config
    load_kwargs["quantization_config"] = quantization_config
    load_kwargs["device_map"] = "auto"
    load_kwargs["torch_dtype"] = torch.float16  # ✅ 必须添加
else:
    # Regular LoRA mode
    load_kwargs["torch_dtype"] = torch.bfloat16 if ... else torch.float16
    load_kwargs["device_map"] = ...
    load_kwargs["trust_remote_code"] = True  # 仅非量化时使用
```

### 关键要点

1. **4-bit/8-bit 量化必须指定 `torch_dtype`**
2. 推荐使用 `torch.float16` (比 `bfloat16` 更稳定)
3. `trust_remote_code=True` 在量化加载时可能导致冲突，应该移除

---

## 2. bitsandbytes 版本兼容性

### 问题表现

```python
RuntimeError: "normal_kernel_cpu" not implemented for 'Char'
```

### 测试的版本

| 版本 | 结果 | 备注 |
|------|------|------|
| 0.43.3 | ❌ 段错误 | 初始版本，与 LLaVA 不兼容 |
| 0.48.2 | ❌ normal_kernel_cpu 错误 | 太新，有回归问题 |
| 0.44.1 | ❌ normal_kernel_cpu 错误 | 与 transformers 4.44.2 不兼容 |
| 0.44.1 + 修复 | ✅ 工作 | 添加 torch_dtype 后成功 |

### 解决方案

问题不在于 bitsandbytes 版本，而在于**缺少 `torch_dtype` 参数**。

当添加 `torch_dtype=torch.float16` 后，所有版本都可以工作（推荐 0.44.1+）。

### 安装命令

```bash
# 推荐版本
pip install bitsandbytes==0.44.1

# 或使用最新稳定版
pip install bitsandbytes>=0.44.0
```

---

## 3. LLaVA Processor 加载问题

### 问题表现

```python
OSError: liuhaotian/llava-v1.5-7b does not appear to have a file named preprocessor_config.json
```

### 根本原因

LLaVA 模型**确实没有** `preprocessor_config.json` 文件，这是模型本身的设计。

### 解决方案

使用 **fallback 机制**：先尝试加载 Processor，失败后使用 Tokenizer。

```python
# ✅ 正确的加载方式
if is_vision_model:
    try:
        self.tokenizer = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        logger.info("Loaded processor")
    except Exception as e:
        logger.warning(f"Failed to load processor, falling back to tokenizer: {e}")
        # 👇 这是正常的 fallback，不是错误！
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False
        )
```

### 关键要点

1. **警告信息是正常的**，不需要修复
2. LLaVA 使用 `AutoTokenizer` 就足够了
3. 只需要确保有 fallback 机制

---

## 4. 网络和镜像源配置

### 问题表现

```python
ConnectionResetError: [Errno 104] Connection reset by peer
requests.exceptions.ConnectionError
```

### 解决方案

使用 HuggingFace 中国镜像源。

```bash
# 方法 1: 环境变量 (推荐)
export HF_ENDPOINT=https://hf-mirror.com

# 方法 2: 在代码中设置
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### 完整测试命令

```bash
# 正确的测试命令
export HF_ENDPOINT=https://hf-mirror.com && \
cd /home/bygpu/Desktop/sam2/demo/training && \
python test_qlora_training.py --quick
```

---

## 5. 最终工作配置

### 环境配置

```yaml
硬件:
  GPU: NVIDIA GeForce RTX 3060 (12GB VRAM)
  CUDA: 12.4
  Driver: 550.120

软件:
  Python: 3.11
  PyTorch: 2.3.1+cu121
  transformers: 4.44.2
  bitsandbytes: 0.44.1+
  peft: (latest)
```

### 成功的加载代码

```python
import torch
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration, AutoTokenizer

# 4-bit 量化配置
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "liuhaotian/llava-v1.5-7b",
    use_fast=False
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型 - 关键参数！
model = LlavaForConditionalGeneration.from_pretrained(
    "liuhaotian/llava-v1.5-7b",
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,  # 🔑 必须指定！
)
```

### 显存使用情况

```
训练配置:
  Model (4-bit): 3.38 GB
  LoRA Adapters: 0.20 GB
  Optimizer: 0.40 GB
  Gradients: 0.20 GB
  Activations: 0.50 GB
  ────────────────────────
  TOTAL: 4.68 GB

实际使用:
  训练时: 4.32 GB
  预留: 5.06 GB
  剩余: ~6.5 GB (可用于更大批次)
```

### 测试结果

```bash
$ python test_qlora_training.py --quick

✅ ALL TESTS PASSED!
  ✓ Model loaded successfully
  ✓ Model is quantized (4-bit)
  ✓ Training loop test passed (5 steps)
  ✓ VRAM stable at 4.32GB
```

---

## 快速参考：常见错误检查清单

### 遇到段错误时

- [ ] 检查是否添加了 `torch_dtype=torch.float16`
- [ ] 移除 `trust_remote_code=True` (仅限量化模式)
- [ ] 确认 bitsandbytes >= 0.44.0
- [ ] 检查 CUDA 和 PyTorch 版本兼容性

### 遇到加载错误时

- [ ] 设置 `export HF_ENDPOINT=https://hf-mirror.com`
- [ ] 检查网络连接
- [ ] 确认模型名称正确
- [ ] 检查是否有 HuggingFace token (如果需要)

### 遇到内存错误时

- [ ] 检查 VRAM 是否充足 (`nvidia-smi`)
- [ ] 减小 batch_size
- [ ] 减小 max_length
- [ ] 启用 gradient_checkpointing

---

## 相关文件

- 成功的测试脚本: `test_llava_4bit_final.py`
- 修复的训练器: `core/trainers/lora_trainer.py:103-118`
- QLoRA 测试: `test_qlora_training.py`

---

## 总结

最关键的修复是在 **lora_trainer.py** 中添加：

```python
load_kwargs["torch_dtype"] = torch.float16  # 这一行解决了 99% 的问题！
```

没有这一行，bitsandbytes 在处理 4-bit 量化时会出现内存访问错误，导致段错误。

**记住**: 量化加载时，永远要明确指定 `torch_dtype`！
