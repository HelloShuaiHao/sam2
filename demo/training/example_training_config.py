"""Example script demonstrating Phase 2: Training Configuration and Setup.

This script shows how to:
1. Create training configurations using presets
2. Set up the model registry
3. Estimate VRAM requirements
4. Configure LoRA training (without actually training - requires GPU)

Usage:
    python example_training_config.py
"""

from pathlib import Path
from core.config import (
    TrainingConfig,
    ModelConfig,
    DataConfig,
    TrainingHyperparameters,
    TrainingMethod,
    LoRAConfig,
    ModelRegistry,
    ConfigPresets
)

def main():
    print("=" * 80)
    print("SAM2 LLM Fine-tuning Pipeline - Phase 2 Demo")
    print("Training Configuration & Model Registry")
    print("=" * 80)
    print()

    # =========================================================================
    # Part 1: Model Registry
    # =========================================================================
    print("[1/4] Exploring Model Registry...")
    print()

    # List all supported models
    all_models = ModelRegistry.list_all_models()
    print(f"Found {len(all_models)} supported models:")
    for model in all_models:
        print(f"  - {model.name}")
        print(f"    Type: {model.type.value}, Size: {model.size_gb:.1f}GB")
        print(f"    LoRA VRAM: {model.lora_vram_gb}GB, Full FT VRAM: {model.min_vram_gb}GB")
        print(f"    {model.description}")
        print()

    # Get specific model info
    model_name = "liuhaotian/llava-v1.5-7b"
    model_info = ModelRegistry.get_model(model_name)
    if model_info:
        print(f"Selected model: {model_info.name}")
        print(f"  Vision encoder: {model_info.vision_encoder}")
        print(f"  Language model: {model_info.language_model}")
        print(f"  Max sequence length: {model_info.max_sequence_length}")
        print()

    # Estimate VRAM requirements
    print("VRAM Estimation for different configurations:")
    for batch_size in [2, 4, 8]:
        estimate = ModelRegistry.estimate_vram_requirements(
            model_name,
            use_lora=True,
            batch_size=batch_size
        )
        print(f"  Batch size {batch_size}:")
        print(f"    Total VRAM: {estimate['total_vram_gb']:.1f}GB")
        print(f"    Recommended GPU: {estimate['recommended_gpu']}")
    print()

    # =========================================================================
    # Part 2: Configuration Presets
    # =========================================================================
    print("[2/4] Exploring Configuration Presets...")
    print()

    presets = ConfigPresets.list_presets()
    print(f"Available presets: {len(presets)}")
    for name, description in presets.items():
        print(f"  - {name}: {description}")
    print()

    # Create configurations from presets
    data_path = "./output/splits/train.jsonl"
    val_path = "./output/splits/val.jsonl"

    configs = {
        "Quick Test": ConfigPresets.quick_test(data_path),
        "Development": ConfigPresets.development(data_path, val_path),
        "Production": ConfigPresets.production(data_path, val_path),
        "Memory Efficient": ConfigPresets.memory_efficient(data_path)
    }

    for name, config in configs.items():
        print(f"{name} Configuration:")
        print(f"  Model: {config.model.name}")
        print(f"  Batch size: {config.training.batch_size}")
        print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
        print(f"  Effective batch size: {config.get_effective_batch_size()}")
        print(f"  LoRA rank: {config.training.lora.rank}")
        print(f"  Learning rate: {config.training.learning_rate}")
        print(f"  Epochs: {config.training.num_epochs}")
        print()

    # =========================================================================
    # Part 3: Custom Configuration
    # =========================================================================
    print("[3/4] Creating Custom Configuration...")
    print()

    # Create a custom configuration
    custom_config = TrainingConfig(
        model=ModelConfig(
            name="liuhaotian/llava-v1.5-7b",
            type="llava"
        ),
        data=DataConfig(
            train_path=data_path,
            val_path=val_path,
            max_length=1024,
            image_size=336
        ),
        training=TrainingHyperparameters(
            method=TrainingMethod.LORA,
            learning_rate=2e-5,
            batch_size=4,
            gradient_accumulation_steps=4,
            num_epochs=3,
            warmup_ratio=0.03,
            lora=LoRAConfig(
                rank=64,
                alpha=128,
                dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
        ),
        experiment_name="custom_sam2_finetuning",
        seed=42
    )

    print("Custom configuration created:")
    print(f"  Experiment: {custom_config.experiment_name}")
    print(f"  Model: {custom_config.model.name}")
    print(f"  Effective batch size: {custom_config.get_effective_batch_size()}")
    print()

    # =========================================================================
    # Part 4: Save and Load Configuration
    # =========================================================================
    print("[4/4] Saving and Loading Configuration...")
    print()

    # Save configuration to JSON
    config_path = Path("output/example_training_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    custom_config.save_json(str(config_path))
    print(f"✓ Configuration saved to: {config_path}")

    # Load configuration back
    loaded_config = TrainingConfig.from_json_file(str(config_path))
    print(f"✓ Configuration loaded successfully")
    print(f"  Verified: {loaded_config.experiment_name == custom_config.experiment_name}")
    print()

    # Display JSON preview
    print("Configuration JSON preview:")
    print(custom_config.model_dump_json()[:500] + "...")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 80)
    print("Phase 2 Configuration Demo Complete!")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Model registry with 4 vision-language models")
    print("  ✓ VRAM estimation for different configurations")
    print("  ✓ 5 pre-configured training presets")
    print("  ✓ Custom configuration creation")
    print("  ✓ Configuration serialization (JSON)")
    print()
    print("Next Steps:")
    print("  1. Choose a configuration preset or create custom config")
    print("  2. Ensure you have a GPU with sufficient VRAM")
    print("  3. Use the configuration to start LoRA training")
    print("  4. Note: Actual training requires HuggingFace account & model access")
    print()
    print("Files Created:")
    print(f"  - {config_path}")
    print()


if __name__ == "__main__":
    main()
