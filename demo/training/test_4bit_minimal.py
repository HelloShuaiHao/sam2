#!/usr/bin/env python3
"""Minimal test for 4-bit quantization to diagnose segfault."""

import sys
import torch
from transformers import BitsAndBytesConfig

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    import bitsandbytes as bnb
    print(f"bitsandbytes version: {bnb.__version__}")
except Exception as e:
    print(f"Failed to import bitsandbytes: {e}")
    sys.exit(1)

print("\n=== Testing 4-bit quantization config ===")
try:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print("✓ BitsAndBytesConfig created successfully")
except Exception as e:
    print(f"✗ Failed to create BitsAndBytesConfig: {e}")
    sys.exit(1)

print("\n=== Testing model loading with 4-bit ===")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Use a tiny model for testing
    model_name = "gpt2"  # Small model for testing
    print(f"Loading {model_name} with 4-bit quantization...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"✓ Model loaded successfully")
    print(f"  Model device: {model.device}")
    print(f"  Model dtype: {model.dtype}")

except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
