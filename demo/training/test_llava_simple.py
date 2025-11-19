#!/usr/bin/env python3
"""Test LLaVA without processor - just for training setup."""

import sys
import torch
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration, AutoTokenizer

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

model_name = "liuhaotian/llava-v1.5-7b"

print("\n=== Testing LLaVA with 8-bit quantization (no processor) ===")

try:
    # 8-bit quantization config
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✓ Tokenizer loaded")

    print("\n2. Loading model with 8-bit quantization...")
    print("   (This may take 1-2 minutes for first load...)")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print(f"   ✓ Model loaded successfully")

    # Check VRAM usage
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1024**3
        vram_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\n3. VRAM usage:")
        print(f"   Allocated: {vram_used:.2f} GB")
        print(f"   Reserved: {vram_reserved:.2f} GB")

    print("\n✓ SUCCESS! LLaVA loaded with 8-bit quantization")
    print("  Ready for LoRA training!")

except Exception as e:
    print(f"\n✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
