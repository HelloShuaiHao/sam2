#!/usr/bin/env python3
"""Test LLaVA without quantization to check basic loading."""

import sys
import torch
from transformers import LlavaForConditionalGeneration, AutoTokenizer

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

model_name = "liuhaotian/llava-v1.5-7b"

print("\n=== Testing LLaVA WITHOUT quantization (FP16) ===")
print("Warning: This may use >10GB VRAM")

try:
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✓ Tokenizer loaded")

    print("\n2. Loading model in FP16...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        low_cpu_mem_usage=True
    )
    print(f"   ✓ Model loaded successfully")

    # Check VRAM usage
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1024**3
        vram_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\n3. VRAM usage:")
        print(f"   Allocated: {vram_used:.2f} GB")
        print(f"   Reserved: {vram_reserved:.2f} GB")

    print("\n✓ SUCCESS! LLaVA loaded without quantization")

except Exception as e:
    print(f"\n✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
