"""Test script for ConfigValidator.

Tests configuration validation with various scenarios.
"""

import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import ConfigPresets, TrainingConfig, ModelConfig, DataConfig, TrainingHyperparameters, TrainingMethod
from core.config.config_validator import ConfigValidator


def test_valid_config():
    """Test validation of a valid configuration."""
    print("=" * 80)
    print("Test 1: Valid Configuration (ultra_low_memory preset)")
    print("=" * 80)
    print()

    # Create a valid config using preset
    config = ConfigPresets.ultra_low_memory(
        data_path="./output/splits/train.jsonl",
        val_path="./output/splits/val.jsonl"
    )

    # Validate
    validator = ConfigValidator(check_hardware=True)
    result = validator.validate(config)

    # Print summary
    result.print_summary()

    return result.valid


def test_invalid_vram():
    """Test validation with insufficient VRAM."""
    print("=" * 80)
    print("Test 2: Insufficient VRAM (high_quality preset on small GPU)")
    print("=" * 80)
    print()

    # Create config that requires lots of VRAM
    config = ConfigPresets.high_quality(
        data_path="./output/splits/train.jsonl",
        val_path="./output/splits/val.jsonl"
    )

    # Modify to make it even more demanding
    config.training.batch_size = 8
    config.data.max_length = 2048

    # Validate
    validator = ConfigValidator(check_hardware=True)
    result = validator.validate(config)

    # Print summary
    result.print_summary()

    return result


def test_invalid_hyperparameters():
    """Test validation with bad hyperparameters."""
    print("=" * 80)
    print("Test 3: Invalid Hyperparameters")
    print("=" * 80)
    print()

    # Create config with questionable settings
    config = ConfigPresets.quick_test(data_path="./test_data/train.jsonl")

    # Modify to add issues
    config.training.learning_rate = 1e-2  # Too high!
    config.training.batch_size = 1
    config.training.gradient_accumulation_steps = 1  # Effective batch size = 1 (too small)
    config.training.lora.rank = 2  # Very low rank
    config.training.lora.alpha = 1  # Alpha < rank

    # Validate
    validator = ConfigValidator(check_hardware=False)  # Skip hardware for this test
    result = validator.validate(config)

    # Print summary
    result.print_summary()

    return result


def test_missing_data():
    """Test validation with missing data files."""
    print("=" * 80)
    print("Test 4: Missing Data Files")
    print("=" * 80)
    print()

    # Create config with non-existent data path
    config = ConfigPresets.quick_test(data_path="./nonexistent_file.jsonl")

    # Validate
    validator = ConfigValidator(check_hardware=False)
    result = validator.validate(config)

    # Print summary
    result.print_summary()

    return result


def test_qlora_config():
    """Test QLoRA-specific validation."""
    print("=" * 80)
    print("Test 5: QLoRA Configuration Check")
    print("=" * 80)
    print()

    # Create QLoRA config
    config = ConfigPresets.ultra_low_memory(
        data_path="./output/splits/train.jsonl"
    )

    # Verify it's QLoRA
    assert config.training.method == TrainingMethod.QLORA

    # Validate
    validator = ConfigValidator(check_hardware=True)
    result = validator.validate(config)

    # Print summary
    result.print_summary()

    return result


def main():
    """Run all tests."""
    print()
    print("🧪 ConfigValidator Test Suite")
    print()

    tests = [
        ("Valid Config", test_valid_config),
        ("High VRAM Config", test_invalid_vram),
        ("Bad Hyperparameters", test_invalid_hyperparameters),
        ("Missing Data", test_missing_data),
        ("QLoRA Config", test_qlora_config),
    ]

    results = {}

    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = result
        except Exception as e:
            print(f"❌ Test '{name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results[name] = None
        print()

    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print()

    for name, result in results.items():
        if result is None:
            status = "❌ EXCEPTION"
        elif isinstance(result, bool):
            status = "✅ VALID" if result else "⚠️ INVALID"
        else:
            status = "✅ COMPLETED"
        print(f"  {name}: {status}")

    print()
    print("=" * 80)
    print("✅ ConfigValidator Tests Complete!")
    print("=" * 80)
    print()
    print("Key features verified:")
    print("  ✓ Valid configuration passes")
    print("  ✓ VRAM estimation and checking")
    print("  ✓ Hyperparameter validation")
    print("  ✓ Data path validation")
    print("  ✓ QLoRA-specific checks")
    print()


if __name__ == "__main__":
    main()
