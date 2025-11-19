# Memory Optimization Guide for 8GB GPUs

This guide helps you train vision-language models on consumer GPUs with limited VRAM (8GB).

## 🎯 Quick Start for 8GB GPUs

**Recommended model**: `Qwen/Qwen-VL-Chat` (only 6GB with QLoRA!)

```python
from core.config import ConfigPresets

# One-line configuration for 8GB GPUs
config = ConfigPresets.ultra_low_memory(
    data_path="./output/splits/train.jsonl",
    val_path="./output/splits/val.jsonl"
)

# Estimated VRAM: ~5-7GB ✅
```

---

## 📊 VRAM Breakdown (QLoRA vs LoRA)

### Regular LoRA (FP16/BF16) - ❌ Won't fit in 8GB
```
Model:           24.0 GB
Optimizer:        6.8 GB
Gradients:        2.7 GB
Activations:      2.0 GB
────────────────────────
TOTAL:           35.5 GB  ❌ Needs RTX 4090 24GB
```

### QLoRA (4-bit) - ✅ Fits in 8GB!
```
Model (4-bit):    3.4 GB  (75% reduction!)
LoRA adapters:    0.20 GB
Optimizer:        0.04 GB
Gradients:        0.02 GB
Activations:      0.5 GB
────────────────────────
TOTAL:            4.2 GB  ✅ Works on RTX 3060 12GB / RTX 4060 8GB
```

**Memory savings: ~88% reduction!**

---

## 🔧 Optimization Strategies

### 1. **Use QLoRA Instead of LoRA**

QLoRA uses 4-bit quantization to reduce model size by 75%.

```python
from core.config import TrainingMethod, QuantizationConfig

training=TrainingHyperparameters(
    method=TrainingMethod.QLORA,  # ✅ Use QLoRA
    quantization=QuantizationConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NF4 is better than FP4
        bnb_4bit_use_double_quant=True  # Extra savings
    )
)
```

**Savings**: ~75% model memory

---

### 2. **Reduce LoRA Rank**

Lower rank = fewer trainable parameters = less memory.

```python
lora=LoRAConfig(
    rank=8,   # ✅ Start with 8 for 8GB GPUs (can try 16 or 32 if you have headroom)
    alpha=16  # Keep alpha = rank * 2
)
```

| Rank | Trainable Params | Memory  | Quality |
|------|-----------------|---------|---------|
| 8    | ~4M             | Lowest  | Good    |
| 16   | ~8M             | Low     | Better  |
| 32   | ~16M            | Medium  | Great   |
| 64   | ~32M            | High    | Best    |

**Recommendation for 8GB**: rank=8 or rank=16

---

### 3. **Enable Gradient Checkpointing**

Trade computation for memory by recomputing activations during backward pass.

```python
hardware=HardwareConfig(
    gradient_checkpointing=True  # ✅ Essential for 8GB!
)
```

**Savings**: ~40% activation memory
**Cost**: ~20% slower training

---

### 4. **Reduce Batch Size + Use Gradient Accumulation**

```python
training=TrainingHyperparameters(
    batch_size=1,                      # ✅ Minimal batch size
    gradient_accumulation_steps=32,    # Effective batch = 32
)
```

This gives you the same gradient updates as batch_size=32, but uses 32x less memory.

**Effective batch size** = `batch_size × gradient_accumulation_steps`

---

### 5. **Reduce Sequence Length**

Shorter sequences use less memory.

```python
data=DataConfig(
    max_length=512,   # ✅ Good for 8GB (can try 768 if stable)
    image_size=336    # ✅ Smaller images save memory
)
```

| Max Length | Memory | Use Case |
|------------|--------|----------|
| 256        | Lowest | Very short conversations |
| 512        | Low    | ✅ Recommended for 8GB |
| 1024       | Medium | Standard |
| 2048       | High   | Long contexts |

---

### 6. **Choose the Right Model**

Not all models are equal in memory usage:

| Model | Size | QLoRA VRAM | Fits 8GB? |
|-------|------|------------|-----------|
| **Qwen/Qwen-VL-Chat** | 9.6GB | **6GB** | ✅ **BEST** |
| LLaVA 1.5 7B | 13.5GB | 7GB | ✅ Good |
| InstructBLIP 7B | 14.0GB | 7GB | ✅ OK |
| LLaVA 1.5 13B | 26.0GB | 12GB | ❌ No |

**Recommendation**: Start with `Qwen/Qwen-VL-Chat` for maximum headroom.

---

### 7. **Reduce Image Resolution**

Vision-language models process images through vision encoders.

```python
data=DataConfig(
    image_size=336  # ✅ Good balance (224 for more savings, 448 for quality)
)
```

| Image Size | Memory | Quality |
|------------|--------|---------|
| 224        | Low    | OK      |
| 336        | Medium | ✅ Recommended |
| 448        | High   | Best    |

---

### 8. **Mixed Precision Training**

Use BF16 or FP16 for reduced precision.

```python
hardware=HardwareConfig(
    mixed_precision=MixedPrecision.BF16  # ✅ Use BF16 if available
)
```

**BF16** (BFloat16): Better numerical stability, recommended for modern GPUs
**FP16**: Slightly faster on older GPUs

---

## 🚀 Complete 8GB Configuration Example

```python
from core.config import *

config = TrainingConfig(
    model=ModelConfig(
        name="Qwen/Qwen-VL-Chat",  # Smallest QLoRA footprint
        type="qwen-vl"
    ),
    data=DataConfig(
        train_path="./output/splits/train.jsonl",
        val_path="./output/splits/val.jsonl",
        max_length=512,      # ✅ Reduced sequence length
        image_size=336       # ✅ Balanced resolution
    ),
    training=TrainingHyperparameters(
        method=TrainingMethod.QLORA,  # ✅ 4-bit quantization
        learning_rate=2e-4,
        batch_size=1,                 # ✅ Minimal batch
        gradient_accumulation_steps=32,  # ✅ Effective batch = 32
        num_epochs=3,
        lora=LoRAConfig(
            rank=8,              # ✅ Low rank
            alpha=16,
            dropout=0.05
        ),
        quantization=QuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True  # ✅ Extra savings
        )
    ),
    hardware=HardwareConfig(
        mixed_precision=MixedPrecision.BF16,
        gradient_checkpointing=True,  # ✅ Essential!
        num_workers=2
    ),
    experiment_name="8gb_optimized"
)
```

**Estimated VRAM**: ~5-6GB
**Headroom**: ~2-3GB for safety

---

## 📈 Monitoring Memory Usage

### Before Training
```bash
# Check available GPU memory
nvidia-smi
```

### During Training
```bash
# Monitor in real-time (every 1 second)
nvidia-smi -l 1

# Or use watch
watch -n 1 nvidia-smi
```

### Python Memory Profiling
```python
import torch

# Check current GPU memory
allocated = torch.cuda.memory_allocated() / 1024**3  # GB
reserved = torch.cuda.memory_reserved() / 1024**3    # GB
print(f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# Clear cache if needed
torch.cuda.empty_cache()
```

---

## ⚠️ Troubleshooting OOM (Out of Memory)

### If you get OOM errors:

1. **Reduce batch size** (but keep gradient_accumulation high)
   ```python
   batch_size=1
   gradient_accumulation_steps=64  # Increase this
   ```

2. **Reduce sequence length**
   ```python
   max_length=256  # or even 128 for testing
   ```

3. **Reduce image size**
   ```python
   image_size=224  # Smaller images
   ```

4. **Lower LoRA rank**
   ```python
   lora=LoRAConfig(rank=4, alpha=8)  # Extreme low memory
   ```

5. **Switch to smaller model**
   ```python
   model_name="Qwen/Qwen-VL-Chat"  # Smallest footprint
   ```

6. **Close other GPU applications**
   ```bash
   # Check what's using GPU
   nvidia-smi

   # Kill processes if needed
   kill -9 <PID>
   ```

---

## 🎓 Trade-offs Summary

| Optimization | Memory Saved | Speed Impact | Quality Impact |
|--------------|--------------|--------------|----------------|
| QLoRA (4-bit) | 🟢🟢🟢🟢 75% | 🔴 -30% slower | 🟡 Minimal loss |
| Gradient checkpointing | 🟢🟢🟢 40% | 🔴 -20% slower | ✅ None |
| Lower LoRA rank (64→8) | 🟢🟢 ~15% | ✅ Faster | 🔴 Some loss |
| Reduce batch size | 🟢 Variable | 🔴 Slower convergence | ✅ None |
| Reduce max_length | 🟢 Variable | ✅ Faster | 🔴 Context loss |
| Reduce image_size | 🟢 Variable | ✅ Faster | 🔴 Detail loss |

---

## 🏆 Best Practices for 8GB GPUs

### ✅ DO:
- Use QLoRA (4-bit quantization)
- Enable gradient checkpointing
- Start with `batch_size=1` and `gradient_accumulation_steps=32`
- Monitor VRAM usage with `nvidia-smi -l 1`
- Close all other GPU applications
- Use `Qwen/Qwen-VL-Chat` for smallest footprint
- Start with `rank=8`, increase if you have headroom

### ❌ DON'T:
- Don't use regular LoRA (requires 24GB+)
- Don't disable gradient checkpointing
- Don't use batch_size > 1 initially
- Don't use max_length > 512 initially
- Don't train without monitoring VRAM
- Don't run other GPU tasks simultaneously

---

## 📊 Expected Performance

### Training Speed (relative to A100):
- **A100 80GB** (regular LoRA): 1.0x (baseline)
- **RTX 4090 24GB** (regular LoRA): 0.7x
- **RTX 3060 12GB** (QLoRA): 0.3-0.4x
- **RTX 4060 8GB** (QLoRA): 0.25-0.35x

**Note**: QLoRA is ~30-40% slower, but enables training on consumer GPUs!

### Quality Impact:
- QLoRA produces **95-99%** of the quality of regular LoRA
- Most papers report minimal quality difference
- Perfect for fine-tuning on specific tasks

---

## 🔗 Useful Resources

- **QLoRA Paper**: https://arxiv.org/abs/2305.14314
- **PEFT Library**: https://github.com/huggingface/peft
- **bitsandbytes**: https://github.com/TimDettmers/bitsandbytes

---

## 💡 Pro Tips

1. **Start small, scale up**:
   - Test with 10-100 samples first
   - Verify VRAM usage is stable
   - Then train on full dataset

2. **Save checkpoints frequently**:
   - Training can be interrupted on consumer GPUs
   - Save every 50-100 steps

3. **Use gradient accumulation wisely**:
   - Higher values = more stable gradients
   - But slower feedback loop
   - Recommended: 16-32 for 8GB GPUs

4. **Monitor temperature**:
   - Consumer GPUs can thermal throttle
   - Use `nvidia-smi` to check temperature
   - Keep under 80°C for stability

5. **Consider overnight training**:
   - QLoRA is slower, plan accordingly
   - 1000 steps might take 3-6 hours on 8GB GPU

---

**Last Updated**: Phase 2 Complete (QLoRA Implementation)

For questions or issues, see `example_qlora_8gb.py` for a working demo.
