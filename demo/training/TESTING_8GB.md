# Testing QLoRA on 8GB GPUs

This guide helps you verify that QLoRA training works on your 8GB GPU.

## 📋 Prerequisites

### Hardware
- NVIDIA GPU with 8GB+ VRAM
  - ✅ RTX 3060 12GB
  - ✅ RTX 4060 8GB
  - ✅ RTX 3060 Ti 8GB
  - ✅ RTX 2060 Super 8GB

### Software
```bash
# Check CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU
nvidia-smi
```

### HuggingFace Account
You need access to the models. Log in:
```bash
huggingface-cli login
```

---

## 🚀 Quick Test (5 minutes)

This test verifies QLoRA works WITHOUT real data:

```bash
cd demo/training

# Quick test (10 samples, 5 steps)
python test_qlora_training.py --quick
```

### Expected Output

```
================================================================================
QLoRA Training Test for 8GB GPUs
================================================================================

GPU Information
================================================================================
GPU: NVIDIA GeForce RTX 3060
Total VRAM: 12.0 GB
Currently allocated: 0.00 GB
Currently reserved: 0.00 GB
Available: 12.00 GB

[1/3] Creating dummy dataset with 10 samples...
  ✓ Created ./test_data/train.jsonl (8 samples)
  ✓ Created ./test_data/val.jsonl (2 samples)

[2/3] Testing QLoRA configuration...
  ✓ Configuration loaded
    Model: Qwen/Qwen-VL-Chat
    Method: qlora
    LoRA rank: 8
    Batch size: 1
    Gradient accumulation: 32

  VRAM Estimate:
    Model (4-bit): 2.40 GB
    Adapters: 0.14 GB
    Optimizer: 0.03 GB
    Gradients: 0.01 GB
    Activations: 0.50 GB
    ─────────────────────
    TOTAL: 3.08 GB

  ✓ Estimated VRAM fits! Headroom: 8.9GB

[3/3] Testing model loading with QLoRA...
  (This will download the model if not cached - may take 5-10 minutes)

  [Before setup] VRAM: 0.00GB allocated, 0.00GB reserved

  Loading model in 4-bit precision...

  [After model load] VRAM: 3.2GB allocated, 3.5GB reserved

  ✓ Model loaded successfully!
    Tokenizer: QWenTokenizer
    Model: PeftModel
    ✓ Model is quantized (4-bit)

[BONUS] Testing training loop (5 steps)...
  Note: Using dummy forward passes (real data loading not implemented yet)

  [Before training] VRAM: 3.2GB allocated, 3.5GB reserved

    Step 1/5... VRAM: 3.2GB
    Step 2/5... VRAM: 3.2GB
    Step 3/5... VRAM: 3.2GB
    Step 4/5... VRAM: 3.2GB
    Step 5/5... VRAM: 3.2GB

  [After training] VRAM: 3.2GB allocated, 3.5GB reserved

  ✓ Training loop test passed!

================================================================================
✅ ALL TESTS PASSED!
================================================================================

🎉 QLoRA is working on your GPU!

Next steps:
  1. Prepare real SAM2 data: python example_data_preparation.py
  2. Create full training script with data loading
  3. Run actual training on your dataset

  [Final] VRAM: 3.2GB allocated, 3.5GB reserved
```

---

## 📊 Monitoring VRAM

### Terminal 1: Run test
```bash
python test_qlora_training.py --quick
```

### Terminal 2: Monitor GPU
```bash
# Update every 1 second
watch -n 1 nvidia-smi

# Or continuous monitoring
nvidia-smi -l 1
```

### Expected VRAM Usage

| Stage | VRAM (Qwen-VL) | VRAM (LLaVA-7B) |
|-------|----------------|-----------------|
| Before loading | 0 GB | 0 GB |
| After model load | 3-4 GB | 4-5 GB |
| During training | 3-4 GB | 4-5 GB |
| Peak usage | 4-5 GB | 5-6 GB |

**Safe for 8GB GPU**: ✅ Yes (3-5GB headroom)

---

## 🔧 Troubleshooting

### Error: "CUDA out of memory"

**Solution 1**: Switch to smaller model
```python
# In test_qlora_training.py, modify config:
model_name = "Qwen/Qwen-VL-Chat"  # Smallest (6GB QLoRA)
```

**Solution 2**: Reduce memory usage
Edit the ultra_low_memory preset:
- `max_length=256` (instead of 512)
- `image_size=224` (instead of 336)
- `rank=4` (instead of 8)

**Solution 3**: Close other GPU applications
```bash
# Check what's using GPU
nvidia-smi

# Kill processes
kill -9 <PID>
```

---

### Error: "Failed to load model"

**Possible causes:**

1. **No HuggingFace login**
   ```bash
   huggingface-cli login
   ```

2. **No internet connection**
   - Model needs to download (~10GB first time)
   - Check network

3. **Insufficient disk space**
   - Models cached in `~/.cache/huggingface/`
   - Need ~15GB free space

---

### Error: "CUDA not available"

**Check PyTorch installation:**
```bash
python -c "import torch; print(torch.version.cuda)"
```

**Reinstall with CUDA support:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📈 Full Test (longer, more realistic)

```bash
# Test with 50 samples, 10 steps
python test_qlora_training.py --samples 50 --steps 10
```

---

## ✅ What This Test Validates

- [x] CUDA is working
- [x] GPU has enough VRAM
- [x] QLoRA configuration loads
- [x] Model downloads successfully
- [x] 4-bit quantization works
- [x] PEFT/LoRA applies correctly
- [x] Training loop executes
- [x] No OOM errors
- [x] VRAM usage matches estimates

## ❌ What This Test Does NOT Validate

- [ ] Real data loading (no DataLoader yet)
- [ ] Actual gradient updates
- [ ] Loss computation
- [ ] Validation loop
- [ ] Checkpoint saving
- [ ] Multi-epoch training

---

## 🎯 Next Steps After Test Passes

### 1. Prepare Real Data
```bash
python example_data_preparation.py your_sam2_export.zip output/
```

### 2. Create Full Training Script
We need to implement:
- Custom DataLoader for vision-language data
- Image preprocessing
- Text tokenization
- Loss computation
- Checkpoint saving

### 3. Run Real Training
```bash
python train_qlora.py \
  --data output/splits/train.jsonl \
  --val output/splits/val.jsonl \
  --epochs 3 \
  --output models/my_model
```

---

## 💡 Tips

### Before Each Training Run:

1. **Close other applications**
   - Web browsers with GPU acceleration
   - Other ML/DL processes
   - Video editing software

2. **Monitor temperature**
   ```bash
   watch -n 1 nvidia-smi
   # Keep GPU < 80°C
   ```

3. **Free up disk space**
   - Models + checkpoints can use 20-30GB
   - Clean `~/.cache/huggingface/` if needed

4. **Use `tmux` or `screen`**
   - Training might take hours
   - Don't let terminal disconnect

---

## 📊 Performance Expectations

### Training Speed (8GB GPU vs Cloud)

| GPU | VRAM | Steps/sec | Relative Speed |
|-----|------|-----------|----------------|
| A100 80GB | 80GB | 2.0 | 1.0x (baseline) |
| RTX 4090 | 24GB | 1.4 | 0.7x |
| RTX 3060 12GB | 12GB | 0.6 | 0.3x |
| RTX 4060 8GB | 8GB | 0.5 | 0.25x |

**QLoRA is ~30-40% slower** than regular LoRA, but enables training on consumer GPUs!

### Training Time Estimates (8GB GPU)

| Dataset Size | Steps | Time (RTX 3060) | Time (RTX 4060) |
|--------------|-------|-----------------|-----------------|
| 100 samples | 100 | ~30 min | ~40 min |
| 500 samples | 500 | ~2.5 hours | ~3.5 hours |
| 1000 samples | 1000 | ~5 hours | ~7 hours |

**Tip**: Start overnight for large datasets!

---

## 🐛 Common Issues

### Issue: Model downloads slowly
**Solution**: Use mirror or download manually
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Issue: GPU memory keeps increasing
**Solution**: Add garbage collection
```python
import gc
torch.cuda.empty_cache()
gc.collect()
```

### Issue: Training crashes after N steps
**Solution**: Lower memory usage
- Reduce `max_length`
- Reduce `image_size`
- Use `rank=4` instead of `rank=8`

---

## 📞 Getting Help

If tests fail:

1. **Check the error message** - most issues are self-explanatory
2. **Read MEMORY_OPTIMIZATION.md** - optimization tips
3. **Run with `--quick`** - isolate the issue
4. **Share GPU info**: `nvidia-smi`, VRAM usage, error logs

---

## ✨ Success Criteria

You're ready for real training when:

- ✅ `test_qlora_training.py --quick` passes
- ✅ VRAM usage stable (no continuous growth)
- ✅ Peak VRAM < 80% of total (safety margin)
- ✅ GPU temperature < 80°C
- ✅ No OOM errors

**If all above pass**: 🎉 Your 8GB GPU is ready for QLoRA training!

---

Last Updated: Phase 2 Complete (QLoRA Testing)
