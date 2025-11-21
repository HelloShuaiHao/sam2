"""
Verify data flow from preparation to training.

This script checks that:
1. Split API returns correct train/val paths
2. Training config receives those paths
3. Training job uses the correct data files

Usage:
    python verify_data_flow.py <split_result_json>
"""

import json
import sys
from pathlib import Path


def verify_split_result(split_result_path: str):
    """Verify split result contains required paths."""
    print("=" * 80)
    print("Step 1: Verify Split Result")
    print("=" * 80)

    with open(split_result_path, 'r') as f:
        split_result = json.load(f)

    print(f"✓ Split result loaded from: {split_result_path}")
    print(f"\nSplit Result:")
    print(json.dumps(split_result, indent=2))

    required_fields = ['train_path', 'val_path', 'test_path', 'train_samples', 'val_samples', 'test_samples']
    for field in required_fields:
        if field not in split_result:
            print(f"❌ Missing field: {field}")
            return None
        print(f"✓ {field}: {split_result[field]}")

    return split_result


def verify_training_config(split_result: dict, training_config_path: str):
    """Verify training config uses correct paths from split result."""
    print("\n" + "=" * 80)
    print("Step 2: Verify Training Config")
    print("=" * 80)

    with open(training_config_path, 'r') as f:
        training_config = json.load(f)

    print(f"✓ Training config loaded from: {training_config_path}")
    print(f"\nTraining Config:")
    print(json.dumps(training_config, indent=2))

    # Check if training config uses the split paths
    expected_train_path = split_result['train_path']
    expected_val_path = split_result['val_path']

    actual_train_path = training_config.get('train_data_path')
    actual_val_path = training_config.get('val_data_path')

    print(f"\n📊 Path Verification:")
    print(f"Expected train_path: {expected_train_path}")
    print(f"Actual train_path:   {actual_train_path}")

    if actual_train_path == expected_train_path:
        print("✅ Train path matches!")
    else:
        print("❌ Train path mismatch!")
        return False

    print(f"\nExpected val_path: {expected_val_path}")
    print(f"Actual val_path:   {actual_val_path}")

    if actual_val_path == expected_val_path:
        print("✅ Val path matches!")
    else:
        print("❌ Val path mismatch!")
        return False

    return True


def verify_data_files(training_config: dict):
    """Verify that the data files actually exist."""
    print("\n" + "=" * 80)
    print("Step 3: Verify Data Files Exist")
    print("=" * 80)

    train_path = Path(training_config['train_data_path'])
    val_path = Path(training_config['val_data_path'])

    print(f"\n📁 Checking files:")

    if train_path.exists():
        # Count lines
        with open(train_path, 'r') as f:
            train_samples = sum(1 for _ in f)
        print(f"✅ Train file exists: {train_path}")
        print(f"   Samples: {train_samples}")
    else:
        print(f"❌ Train file missing: {train_path}")
        return False

    if val_path.exists():
        with open(val_path, 'r') as f:
            val_samples = sum(1 for _ in f)
        print(f"✅ Val file exists: {val_path}")
        print(f"   Samples: {val_samples}")
    else:
        print(f"❌ Val file missing: {val_path}")
        return False

    # Verify file format (first sample)
    print(f"\n📄 Checking data format:")
    with open(train_path, 'r') as f:
        first_line = f.readline()
        sample = json.loads(first_line)
        print(f"✓ First training sample:")
        print(json.dumps(sample, indent=2)[:500] + "...")

    return True


def main():
    print("\n🔍 SAM2 Training Data Flow Verification\n")

    # For interactive use without arguments
    if len(sys.argv) < 2:
        print("Usage: python verify_data_flow.py <split_result_json> [training_config_json]")
        print("\nAlternatively, provide the paths when prompted:")

        split_result_path = input("\nEnter split result JSON path: ").strip()
        training_config_path = input("Enter training config JSON path (optional): ").strip()
    else:
        split_result_path = sys.argv[1]
        training_config_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Step 1: Verify split result
    split_result = verify_split_result(split_result_path)
    if not split_result:
        print("\n❌ Split result verification failed!")
        return 1

    # Step 2: Verify training config (if provided)
    if training_config_path:
        if not verify_training_config(split_result, training_config_path):
            print("\n❌ Training config verification failed!")
            return 1

        # Step 3: Verify data files exist
        with open(training_config_path, 'r') as f:
            training_config = json.load(f)

        if not verify_data_files(training_config):
            print("\n❌ Data file verification failed!")
            return 1
    else:
        # If no training config, just verify the split files exist
        print("\n" + "=" * 80)
        print("Step 2: Verify Split Files Exist")
        print("=" * 80)

        for key in ['train_path', 'val_path', 'test_path']:
            path = Path(split_result[key])
            if path.exists():
                with open(path, 'r') as f:
                    samples = sum(1 for _ in f)
                print(f"✅ {key}: {path}")
                print(f"   Samples: {samples}")
            else:
                print(f"❌ {key} missing: {path}")
                return 1

    # Success!
    print("\n" + "=" * 80)
    print("✅ ALL CHECKS PASSED!")
    print("=" * 80)
    print("\n✓ Split result is valid")
    if training_config_path:
        print("✓ Training config uses correct data paths")
        print("✓ Data files exist and are readable")
    else:
        print("✓ Split data files exist and are readable")

    print("\n🎉 Data flow is correctly configured!")
    print("   Training will use the prepared data from the split step.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
