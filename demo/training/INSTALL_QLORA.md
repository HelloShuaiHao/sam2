# QLoRA Installation Guide for 8GB GPUs

Quick installation guide to get QLoRA testing working.

## Step 1: Check Your CUDA Version

```bash
nvidia-smi
```

Look for "CUDA Version" in the output. Common versions:
- CUDA 11.8
- CUDA 12.1
- CUDA 12.4

## Step 2: Install PyTorch with CUDA Support

⚠️ **IMPORTANT**: Install PyTorch FIRST with correct CUDA version!

### For CUDA 11.8 (Most Common)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### For CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### For CUDA 12.4
```bash
pip install torch torchvision torchaudio
```

### Verify PyTorch CUDA
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

**Expected output:**
```
PyTorch: 2.x.x+cu118
CUDA available: True
CUDA version: 11.8
```

## Step 3: Install QLoRA Dependencies

### Option A: Minimal (Just for testing QLoRA)
```bash
cd /Users/mbp/Desktop/Work/Life/IDoctor/sam2/demo/training
pip install -r requirements_qlora_minimal.txt
```

### Option B: Full (All features)
```bash
pip install -r requirements.txt
```

**Recommended**: Start with Option A, upgrade to Option B later if needed.

## Step 4: Verify Installation

```bash
python -c "
import torch
import transformers
import peft
import bitsandbytes
print('✅ All packages installed!')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

**Expected output:**
```
✅ All packages installed!
PyTorch: 2.x.x
Transformers: 4.36.x
PEFT: 0.7.x
CUDA: True
```

## Step 5: Test QLoRA

```bash
# Quick test (5 minutes)
python test_qlora_training.py --quick
```

---

## Troubleshooting

### Error: "CUDA not available"

**Fix PyTorch CUDA installation:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Error: "No module named 'bitsandbytes'"

**Install bitsandbytes:**
```bash
pip install bitsandbytes>=0.41.0
```

### Error: "bitsandbytes CUDA setup failed"

This usually means PyTorch CUDA is not installed correctly.

**Solution:**
```bash
# 1. Check CUDA
nvidia-smi

# 2. Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 3. Reinstall bitsandbytes
pip uninstall bitsandbytes
pip install bitsandbytes
```

### Error: "Illegal instruction (core dumped)"

bitsandbytes might not support your CPU.

**Solution**: Use newer version
```bash
pip install bitsandbytes>=0.43.0
```

---

## Conda Environment (Recommended)

If you want a clean environment:

```bash
# Create new environment
conda create -n qlora python=3.10
conda activate qlora

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install QLoRA dependencies
pip install -r requirements_qlora_minimal.txt

# Test
python test_qlora_training.py --quick
```

---

## Quick Install (Copy-Paste)

```bash
# 1. Check CUDA version
nvidia-smi

# 2. Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install QLoRA dependencies
cd /Users/mbp/Desktop/Work/Life/IDoctor/sam2/demo/training
pip install -r requirements_qlora_minimal.txt

# 4. Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 5. Test QLoRA
python test_qlora_training.py --quick
```

---

## Package Versions

| Package | Minimum | Tested |
|---------|---------|--------|
| Python | 3.9 | 3.10 |
| PyTorch | 2.0.0 | 2.1.0 |
| transformers | 4.36.0 | 4.36.0 |
| peft | 0.7.0 | 0.7.1 |
| bitsandbytes | 0.41.0 | 0.41.3 |
| accelerate | 0.25.0 | 0.25.0 |

---

## Disk Space Requirements

- PyTorch + dependencies: ~5GB
- Model cache (first download): ~10-15GB
- Total: ~20GB free space recommended

---

## HuggingFace Login

Some models require authentication:

```bash
pip install huggingface-hub
huggingface-cli login
```

Enter your HuggingFace token when prompted.

---

Last Updated: QLoRA Installation Guide
