"""Example script demonstrating QLoRA training for 8GB GPUs.

This script shows how to:
1. Use the ultra_low_memory preset for 8GB GPUs
2. Estimate VRAM requirements with QLoRA
3. Compare QLoRA vs regular LoRA memory usage
4. Configure 4-bit quantization settings

IMPORTANT: This is a configuration demo. Actual training requires:
- A CUDA-capable GPU with 8GB+ VRAM
- HuggingFace account with access to the model
- Prepared training data (use example_data_preparation.py first)

Usage:
    python example_qlora_8gb.py
"""

from pathlib import Path
from core.config import (
    ConfigPresets,
    ModelRegistry,
    TrainingConfig,
    ModelConfig,
    DataConfig,
    TrainingHyperparameters,
    TrainingMethod,
    LoRAConfig,
    QuantizationConfig,
    HardwareConfig,
    MixedPrecision
)


def main():
    print("=" * 80)
    print("QLoRA Training for 8GB GPUs - Configuration Demo")
    print("=" * 80)
    print()
    print("⚡ This demo shows how to train LLMs on 8GB consumer GPUs using QLoRA!")
    print()

    # =========================================================================
    # Part 1: VRAM Comparison - LoRA vs QLoRA
    # =========================================================================
    print("[1/4] VRAM Requirements Comparison")
    print("=" * 80)
    print()

    model_name = "liuhaotian/llava-v1.5-7b"
    batch_size = 1

    # Regular LoRA (FP16/BF16)
    lora_estimate = ModelRegistry.estimate_vram_requirements(
        model_name,
        use_lora=True,
        use_qlora=False,
        batch_size=batch_size,
        gradient_accumulation=32
    )

    # QLoRA (4-bit quantized)
    qlora_estimate = ModelRegistry.estimate_vram_requirements(
        model_name,
        use_lora=False,
        use_qlora=True,
        batch_size=batch_size,
        gradient_accumulation=32
    )

    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size} (with gradient accumulation = 32)")
    print()

    print(f"❌ Regular LoRA (FP16/BF16):")
    print(f"   Model VRAM:      {lora_estimate['model_vram_gb']:.1f} GB")
    print(f"   Optimizer:       {lora_estimate['optimizer_vram_gb']:.1f} GB")
    print(f"   Gradients:       {lora_estimate['gradient_vram_gb']:.1f} GB")
    print(f"   Activations:     {lora_estimate['activation_vram_gb']:.1f} GB")
    print(f"   ─────────────────────────────")
    print(f"   💥 TOTAL:        {lora_estimate['total_vram_gb']:.1f} GB")
    print(f"   GPU Needed:      {lora_estimate['recommended_gpu']}")
    print()

    print(f"✅ QLoRA (4-bit quantized):")
    print(f"   Model (4-bit):   {qlora_estimate['model_vram_gb']:.1f} GB")
    print(f"   LoRA adapters:   {qlora_estimate['adapter_vram_gb']:.2f} GB")
    print(f"   Optimizer:       {qlora_estimate['optimizer_vram_gb']:.2f} GB")
    print(f"   Gradients:       {qlora_estimate['gradient_vram_gb']:.2f} GB")
    print(f"   Activations:     {qlora_estimate['activation_vram_gb']:.1f} GB")
    print(f"   ─────────────────────────────")
    print(f"   🎉 TOTAL:        {qlora_estimate['total_vram_gb']:.1f} GB")
    print(f"   GPU Needed:      {qlora_estimate['recommended_gpu']}")
    print()

    memory_savings = lora_estimate['total_vram_gb'] - qlora_estimate['total_vram_gb']
    savings_percent = (memory_savings / lora_estimate['total_vram_gb']) * 100

    print(f"💾 Memory savings: {memory_savings:.1f} GB ({savings_percent:.0f}% reduction!)")
    print()

    # =========================================================================
    # Part 2: Ultra Low Memory Preset
    # =========================================================================
    print("[2/4] Using ultra_low_memory Preset (8GB optimized)")
    print("=" * 80)
    print()

    # Create config using the preset
    config = ConfigPresets.ultra_low_memory(
        data_path="./output/splits/train.jsonl",
        val_path="./output/splits/val.jsonl"
    )

    print(f"Preset: ultra_low_memory")
    print(f"  Model: {config.model.name}")
    print(f"  Training method: {config.training.method.value}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.get_effective_batch_size()}")
    print(f"  LoRA rank: {config.training.lora.rank}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print()

    print("Quantization settings:")
    quant = config.training.quantization
    print(f"  ✓ 4-bit quantization: {quant.load_in_4bit}")
    print(f"  ✓ Quantization type: {quant.bnb_4bit_quant_type}")
    print(f"  ✓ Compute dtype: {quant.bnb_4bit_compute_dtype}")
    print(f"  ✓ Double quantization: {quant.bnb_4bit_use_double_quant}")
    print(f"  ✓ Gradient checkpointing: {config.hardware.gradient_checkpointing}")
    print()

    # =========================================================================
    # Part 3: Custom QLoRA Configuration
    # =========================================================================
    print("[3/4] Creating Custom QLoRA Configuration")
    print("=" * 80)
    print()

    # Create a custom QLoRA config
    custom_config = TrainingConfig(
        model=ModelConfig(
            name="liuhaotian/llava-v1.5-7b",
            type="llava"
        ),
        data=DataConfig(
            train_path="./output/splits/train.jsonl",
            val_path="./output/splits/val.jsonl",
            max_length=512,
            image_size=336
        ),
        training=TrainingHyperparameters(
            method=TrainingMethod.QLORA,  # QLoRA mode
            learning_rate=2e-4,
            batch_size=1,
            gradient_accumulation_steps=32,
            num_epochs=3,
            warmup_ratio=0.03,
            lora=LoRAConfig(
                rank=16,  # Can adjust based on your needs
                alpha=32,
                dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            ),
            quantization=QuantizationConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="bfloat16",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        ),
        hardware=HardwareConfig(
            device="auto",
            mixed_precision=MixedPrecision.BF16,
            gradient_checkpointing=True,
            num_workers=2
        ),
        experiment_name="custom_qlora_8gb"
    )

    print("Custom QLoRA configuration created:")
    print(f"  Model: {custom_config.model.name}")
    print(f"  LoRA rank: {custom_config.training.lora.rank}")
    print(f"  Effective batch size: {custom_config.get_effective_batch_size()}")
    print()

    # Save configuration
    config_path = Path("output/qlora_8gb_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    custom_config.save_json(str(config_path))
    print(f"✓ Configuration saved to: {config_path}")
    print()

    # =========================================================================
    # Part 4: All Models QLoRA Comparison
    # =========================================================================
    print("[4/4] QLoRA VRAM Requirements for All Models")
    print("=" * 80)
    print()

    all_models = ModelRegistry.list_all_models()

    print(f"{'Model':<40} {'Size':<10} {'QLoRA VRAM':<12} {'Fits 8GB?'}")
    print("─" * 80)

    for model in all_models:
        fits_8gb = "✅ YES" if model.qlora_vram_gb <= 8 else "❌ NO"
        print(f"{model.name:<40} {model.size_gb:>6.1f} GB  {model.qlora_vram_gb:>6.0f} GB      {fits_8gb}")

    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 80)
    print("QLoRA Demo Complete!")
    print("=" * 80)
    print()
    print("🎯 Key Takeaways:")
    print()
    print("  1. QLoRA reduces VRAM by ~70% compared to regular LoRA")
    print("  2. You CAN train 7B models on 8GB GPUs (RTX 3060, RTX 4060)")
    print("  3. Use ultra_low_memory preset for automatic 8GB optimization")
    print("  4. Trade-off: Slower training (~30-40%), but enables training on consumer GPUs")
    print()
    print("🚀 Next Steps:")
    print()
    print("  1. Prepare your data with example_data_preparation.py")
    print("  2. Test this config: python -c 'from core.config import ConfigPresets; ConfigPresets.ultra_low_memory(...)'")
    print("  3. Start training with a small dataset to verify VRAM usage")
    print("  4. Monitor VRAM with: nvidia-smi -l 1")
    print()
    print("⚠️  Memory Tips for 8GB GPUs:")
    print()
    print("  • Close other applications before training")
    print("  • Use max_length=512 or lower for sequences")
    print("  • Use image_size=336 or lower for images")
    print("  • Keep batch_size=1 with high gradient_accumulation")
    print("  • Enable gradient_checkpointing (essential!)")
    print()
    print("Files Created:")
    print(f"  - {config_path}")
    print()


if __name__ == "__main__":
    main()
