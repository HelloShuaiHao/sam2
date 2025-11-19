#!/usr/bin/env python3
"""Test LLaVA with 4-bit quantization (proper loading)."""

import sys
import torch
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration, AutoTokenizer

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

model_name = "liuhaotian/llava-v1.5-7b"

print("\n=== Testing LLaVA with 4-bit quantization ===")

try:
    # 4-bit quantization config (uses less VRAM than 8-bit)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✓ Tokenizer loaded")

    print("\n2. Loading model with 4-bit quantization...")
    print("   (This will take 2-3 minutes...)")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print(f"   ✓ Model loaded successfully!")

    # Check VRAM usage
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1024**3
        vram_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\n3. VRAM usage:")
        print(f"   Allocated: {vram_used:.2f} GB")
        print(f"   Reserved: {vram_reserved:.2f} GB")

    print("\n✓✓✓ SUCCESS! LLaVA-7B loaded with 4-bit quantization ✓✓✓")
    print(f"  Model ready for LoRA training on 8GB GPU!")

except Exception as e:
    print(f"\n✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
