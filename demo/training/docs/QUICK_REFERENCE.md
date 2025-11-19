# QLoRA 快速参考卡片

## 🚀 成功加载 LLaVA + QLoRA 的最小代码

```python
import torch
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration, AutoTokenizer

# 配置
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# 加载
model = LlavaForConditionalGeneration.from_pretrained(
    "liuhaotian/llava-v1.5-7b",
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,  # ⚠️ 必须！不加会段错误
)
```

## ⚡ 关键要点

1. **永远添加 `torch_dtype=torch.float16`** - 这是避免段错误的关键
2. **量化模式不要用 `trust_remote_code=True`**
3. **设置镜像源**: `export HF_ENDPOINT=https://hf-mirror.com`
4. **LLaVA processor 警告是正常的** - 会自动 fallback 到 tokenizer

## 🔧 常见问题速查

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| 段错误 (139) | 缺少 torch_dtype | 添加 `torch_dtype=torch.float16` |
| normal_kernel_cpu | 同上 | 同上 |
| preprocessor_config.json | LLaVA 没这个文件 | 正常，忽略警告 |
| Connection reset | 网络问题 | `export HF_ENDPOINT=https://hf-mirror.com` |

## 📊 显存使用参考 (RTX 3060 12GB)

```
LLaVA-7B + QLoRA:
  训练中: 4.32 GB
  剩余: ~7 GB
  
建议配置:
  batch_size: 1
  gradient_accumulation: 32
  max_length: 512
```

## 🧪 测试命令

```bash
# 设置镜像源 + 运行测试
export HF_ENDPOINT=https://hf-mirror.com && \
cd /home/bygpu/Desktop/sam2/demo/training && \
python test_qlora_training.py --quick
```

## 📁 相关文件

- 完整文档: `docs/TROUBLESHOOTING.md`
- 成功示例: `test_llava_4bit_final.py`
- 修复位置: `core/trainers/lora_trainer.py:113`
