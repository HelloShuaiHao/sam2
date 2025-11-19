#!/usr/bin/env python3
"""Test LLaVA model loading with and without quantization."""

import sys
import torch

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

model_name = "liuhaotian/llava-v1.5-7b"

# Test 1: Load without quantization (FP16)
print("\n=== Test 1: Loading LLaVA without quantization (FP16) ===")
try:
    from transformers import LlavaForConditionalGeneration

    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print(f"✓ Model loaded successfully (FP16)")
    print(f"  Device: {model.device}")
    print(f"  Dtype: {model.dtype}")
    del model
    torch.cuda.empty_cache()
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Load with 4-bit quantization
print("\n=== Test 2: Loading LLaVA with 4-bit quantization ===")
try:
    from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print(f"✓ Model loaded successfully (4-bit)")
    print(f"  Device: {model.device}")
    print(f"  Dtype: {model.dtype}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
