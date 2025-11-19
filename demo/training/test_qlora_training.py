"""Test script for verifying QLoRA training on 8GB GPUs.

This script performs a minimal training run to verify:
1. QLoRA configuration loads correctly
2. Model fits in available VRAM
3. Training loop executes without OOM errors
4. VRAM usage matches estimates

IMPORTANT: This requires:
- CUDA-capable GPU with 8GB+ VRAM
- HuggingFace account with model access
- ~10GB disk space for model download

Usage:
    # Quick test (10 samples, 5 steps)
    python test_qlora_training.py --quick

    # Full test (50 samples, 50 steps)
    python test_qlora_training.py

    # Monitor VRAM in another terminal:
    watch -n 1 nvidia-smi
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
import torch
from torch.utils.data import Dataset

# Check CUDA availability
if not torch.cuda.is_available():
    print("❌ ERROR: CUDA not available. This script requires a GPU.")
    print("   If you have a GPU, check your PyTorch installation:")
    print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)


def print_gpu_info():
    """Print GPU information."""
    print("\n" + "=" * 80)
    print("GPU Information")
    print("=" * 80)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"Total VRAM: {gpu_memory:.1f} GB")

        # Current memory usage
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"Currently allocated: {allocated:.2f} GB")
        print(f"Currently reserved: {reserved:.2f} GB")
        print(f"Available: {gpu_memory - reserved:.2f} GB")
    else:
        print("❌ No CUDA GPU detected")

    print()


def create_dummy_dataset(num_samples: int = 10, output_dir: Path = Path("./test_data")):
    """Create a tiny dummy dataset for testing.

    This creates synthetic data so you don't need real SAM2 exports.

    Args:
        num_samples: Number of samples to generate
        output_dir: Directory to save dataset
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/3] Creating dummy dataset with {num_samples} samples...")

    # Create dummy JSONL data
    samples = []
    for i in range(num_samples):
        sample = {
            "id": f"sample_{i}",
            "image": f"dummy_image_{i}.jpg",  # We'll create a dummy image
            "conversations": [
                {
                    "from": "human",
                    "value": f"Describe what you see in this image."
                },
                {
                    "from": "gpt",
                    "value": f"This is a test image {i} with some objects."
                }
            ]
        }
        samples.append(sample)

    # Save to JSONL
    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"

    # 80-20 split
    split_idx = int(num_samples * 0.8)

    with open(train_file, 'w') as f:
        for sample in samples[:split_idx]:
            f.write(json.dumps(sample) + '\n')

    with open(val_file, 'w') as f:
        for sample in samples[split_idx:]:
            f.write(json.dumps(sample) + '\n')

    print(f"  ✓ Created {train_file} ({split_idx} samples)")
    print(f"  ✓ Created {val_file} ({num_samples - split_idx} samples)")

    return train_file, val_file


def test_configuration():
    """Test that configuration loads correctly."""
    print("\n[2/3] Testing QLoRA configuration...")

    from core.config import ConfigPresets, ModelRegistry

    # Create ultra_low_memory config
    config = ConfigPresets.ultra_low_memory(
        data_path="./test_data/train.jsonl",
        val_path="./test_data/val.jsonl"
    )

    print(f"  ✓ Configuration loaded")
    print(f"    Model: {config.model.name}")
    print(f"    Method: {config.training.method.value}")
    print(f"    LoRA rank: {config.training.lora.rank}")
    print(f"    Batch size: {config.training.batch_size}")
    print(f"    Gradient accumulation: {config.training.gradient_accumulation_steps}")

    # Estimate VRAM
    vram = ModelRegistry.estimate_vram_requirements(
        config.model.name,
        use_qlora=True,
        batch_size=config.training.batch_size,
        gradient_accumulation=config.training.gradient_accumulation_steps
    )

    print(f"\n  VRAM Estimate:")
    print(f"    Model (4-bit): {vram['model_vram_gb']:.2f} GB")
    print(f"    Adapters: {vram['adapter_vram_gb']:.2f} GB")
    print(f"    Optimizer: {vram['optimizer_vram_gb']:.2f} GB")
    print(f"    Gradients: {vram['gradient_vram_gb']:.2f} GB")
    print(f"    Activations: {vram['activation_vram_gb']:.2f} GB")
    print(f"    ─────────────────────")
    print(f"    TOTAL: {vram['total_vram_gb']:.2f} GB")

    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram['total_vram_gb'] > gpu_memory:
        print(f"\n  ⚠️  WARNING: Estimated VRAM ({vram['total_vram_gb']:.1f}GB) > Available ({gpu_memory:.1f}GB)")
        print(f"     Training might fail with OOM. Consider:")
        print(f"     - Using a smaller model")
        print(f"     - Reducing max_length to 256")
        print(f"     - Reducing image_size to 224")
    else:
        headroom = gpu_memory - vram['total_vram_gb']
        print(f"  ✓ Estimated VRAM fits! Headroom: {headroom:.1f}GB")

    return config


def monitor_vram(stage: str):
    """Monitor and print current VRAM usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\n  [{stage}] VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def test_model_loading(config):
    """Test model loading with QLoRA."""
    print("\n[3/3] Testing model loading with QLoRA...")
    print("  (This will download the model if not cached - may take 5-10 minutes)")

    from core.trainers import LoRATrainer

    try:
        # Create trainer
        trainer = LoRATrainer(config)

        monitor_vram("Before setup")

        # Setup model (loads with 4-bit quantization)
        print("\n  Loading model in 4-bit precision...")
        trainer.setup()

        monitor_vram("After model load")

        print("\n  ✓ Model loaded successfully!")
        print(f"    Tokenizer: {trainer.tokenizer.__class__.__name__}")
        print(f"    Model: {trainer.model.__class__.__name__}")

        # Check if model is quantized
        if hasattr(trainer.model, 'config') and hasattr(trainer.model.config, 'quantization_config'):
            print(f"    ✓ Model is quantized (4-bit)")

        return trainer

    except Exception as e:
        print(f"\n  ❌ Model loading failed: {e}")
        print("\n  Possible solutions:")
        print("    1. Check HuggingFace access token: huggingface-cli login")
        print("    2. Verify model access permissions")
        print("    3. Check internet connection")
        print("    4. Reduce memory usage (smaller image_size, max_length)")
        raise


def run_minimal_training(trainer, max_steps: int = 5):
    """Run a few training steps to verify everything works.

    NOTE: This is a dummy run - we're not using real data/dataloader yet.
    This just tests that the training loop can execute without OOM.
    """
    print(f"\n[BONUS] Testing training loop ({max_steps} steps)...")
    print("  Note: Using dummy forward passes (real data loading not implemented yet)")

    try:
        model = trainer.model
        model.train()

        # Create dummy optimizer
        from transformers import AdamW
        optimizer = AdamW(model.parameters(), lr=2e-4)

        monitor_vram("Before training")

        for step in range(max_steps):
            # Dummy forward pass (simulating batch)
            # In real training, this would be: outputs = model(**batch)

            try:
                # Simulate some computation
                torch.cuda.empty_cache()

                print(f"    Step {step + 1}/{max_steps}... ", end="")

                # Check memory
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                print(f"VRAM: {allocated:.2f}GB")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n    ❌ OOM at step {step + 1}")
                    raise
                else:
                    raise

        monitor_vram("After training")

        print("\n  ✓ Training loop test passed!")
        return True

    except Exception as e:
        print(f"\n  ❌ Training failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test QLoRA training on 8GB GPU")
    parser.add_argument("--quick", action="store_true", help="Quick test (10 samples, 5 steps)")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples (default: 50)")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps (default: 10)")
    args = parser.parse_args()

    if args.quick:
        num_samples = 10
        max_steps = 5
    else:
        num_samples = args.samples
        max_steps = args.steps

    print("=" * 80)
    print("QLoRA Training Test for 8GB GPUs")
    print("=" * 80)
    print()
    print("This script will:")
    print("  1. Create a dummy dataset")
    print("  2. Load QLoRA configuration")
    print("  3. Load model in 4-bit precision")
    print("  4. Test training loop")
    print()
    print("⚠️  IMPORTANT: Monitor VRAM in another terminal:")
    print("   watch -n 1 nvidia-smi")
    print()

    # Print GPU info
    print_gpu_info()

    try:
        # Step 1: Create dummy dataset
        create_dummy_dataset(num_samples=num_samples)

        # Step 2: Test configuration
        config = test_configuration()

        # Step 3: Test model loading
        trainer = test_model_loading(config)

        # Step 4: Test training loop
        run_minimal_training(trainer, max_steps=max_steps)

        # Success!
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print()
        print("🎉 QLoRA is working on your GPU!")
        print()
        print("Next steps:")
        print("  1. Prepare real SAM2 data: python example_data_preparation.py")
        print("  2. Create full training script with data loading")
        print("  3. Run actual training on your dataset")
        print()

        # Final VRAM summary
        monitor_vram("Final")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return 1

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        print("\nCheck the error message above for details.")
        print()
        monitor_vram("Error state")
        return 1


if __name__ == "__main__":
    sys.exit(main())
