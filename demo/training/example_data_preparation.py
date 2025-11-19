"""Example script demonstrating Phase 1: Data Preparation pipeline.

This script shows how to:
1. Parse a SAM2 export
2. Convert to training formats (HuggingFace, LLaVA)
3. Validate dataset quality
4. Split dataset for training

Usage:
    python example_data_preparation.py path/to/sam2_export.zip output_dir
"""

import sys
import json
from pathlib import Path
from core.data_converter import SAM2Parser, HuggingFaceConverter, LLaVAConverter
from core.validation import (
    Validator,
    BasicChecks,
    BalanceAnalyzer,
    QualityMetrics,
    ReportGenerator
)
from core.data_splitter import SplitConfig, SplitStrategy, StratifiedSplitter


def main():
    if len(sys.argv) < 3:
        print("Usage: python example_data_preparation.py <sam2_export.zip> <output_dir>")
        sys.exit(1)

    sam2_export_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SAM2 LLM Fine-tuning Pipeline - Phase 1 Demo")
    print("=" * 80)
    print()

    # Step 1: Parse SAM2 Export
    print("[1/5] Parsing SAM2 export...")
    parser = SAM2Parser()

    try:
        data = parser.parse_zip(sam2_export_path)
        print(f"✓ Parsed successfully")
        print(f"  - Video: {parser.get_video_metadata().get('filename', 'unknown')}")
        print(f"  - Frames: {len(parser.get_frames())}")
        print(f"  - Classes: {len(parser.get_class_distribution())}")
        print()
    except Exception as e:
        print(f"✗ Failed to parse: {e}")
        sys.exit(1)

    # Step 2: Convert to HuggingFace format
    print("[2/5] Converting to HuggingFace format...")
    hf_converter = HuggingFaceConverter()
    hf_output = output_dir / "huggingface"

    try:
        stats = hf_converter.convert(sam2_export_path, hf_output)
        print(f"✓ Converted to HuggingFace format")
        print(f"  - Samples: {stats['total_samples']}")
        print(f"  - Classes: {stats['total_classes']}")
        print(f"  - Output: {hf_output}")
        print()
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        print()

    # Step 3: Convert to LLaVA format
    print("[3/5] Converting to LLaVA format...")
    llava_converter = LLaVAConverter()
    llava_output = output_dir / "llava"

    try:
        stats = llava_converter.convert(sam2_export_path, llava_output)
        print(f"✓ Converted to LLaVA format")
        print(f"  - Samples: {stats['total_samples']}")
        print(f"  - Output: {llava_output}")
        print()
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        print()

    # Step 4: Validate dataset quality
    print("[4/5] Validating dataset quality...")
    validator = Validator()

    # Add all validation rules
    for check in BasicChecks.get_all_checks():
        validator.add_rule(check)
    validator.add_rule(BalanceAnalyzer(imbalance_threshold=10.0))
    validator.add_rule(QualityMetrics(min_object_area_percent=0.1))

    # Prepare data for validation
    validation_data = {
        "frames": parser.get_frames(),
        "metadata": data
    }

    try:
        results = validator.validate(validation_data)
        print(f"✓ Validation complete: {results['status'].upper()}")
        print(f"  - Passed: {results['summary']['passed']}")
        print(f"  - Warnings: {results['summary']['warnings']}")
        print(f"  - Errors: {results['summary']['errors']}")

        # Generate report
        report = ReportGenerator.generate_report(results, dataset_name="Example Dataset")
        report_path = output_dir / "validation_report.md"
        ReportGenerator.save_report(report, str(report_path))
        print(f"  - Report: {report_path}")
        print()
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        print()

    # Step 5: Split dataset
    print("[5/5] Splitting dataset...")

    # Load HuggingFace dataset for splitting
    try:
        dataset_file = hf_output / "dataset.jsonl"
        if dataset_file.exists():
            samples = []
            with open(dataset_file, 'r') as f:
                for line in f:
                    samples.append(json.loads(line))

            # Configure split
            config = SplitConfig(
                strategy=SplitStrategy.STRATIFIED,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                seed=42
            )

            # Perform split
            splitter = StratifiedSplitter(config)
            train, val, test = splitter.split(samples)

            # Save splits
            splits_dir = output_dir / "splits"
            splits_dir.mkdir(exist_ok=True)

            for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
                split_file = splits_dir / f"{split_name}.jsonl"
                with open(split_file, 'w') as f:
                    for sample in split_data:
                        f.write(json.dumps(sample) + '\n')

            print(f"✓ Dataset split complete")
            print(f"  - Train: {len(train)} samples ({len(train)/len(samples)*100:.1f}%)")
            print(f"  - Val: {len(val)} samples ({len(val)/len(samples)*100:.1f}%)")
            print(f"  - Test: {len(test)} samples ({len(test)/len(samples)*100:.1f}%)")
            print(f"  - Output: {splits_dir}")

            # Validate split
            validation = splitter.validate_split(train, val, test, tolerance=0.05)
            if validation['is_valid']:
                print(f"  ✓ Split maintains class balance (max deviation: {validation['max_deviation_percent']}%)")
            else:
                print(f"  ⚠ Split has imbalance (max deviation: {validation['max_deviation_percent']}%)")

            # Save split config
            config.to_json(str(splits_dir / "split_config.json"))
            print()

        else:
            print(f"✗ Dataset file not found: {dataset_file}")
            print()

    except Exception as e:
        print(f"✗ Splitting failed: {e}")
        print()

    # Summary
    print("=" * 80)
    print("Phase 1 Demo Complete!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Review validation report: {}/validation_report.md".format(output_dir))
    print("  2. Use the splits for training: {}/splits/".format(output_dir))
    print("  3. Proceed to Phase 2: Training Orchestration (coming soon)")
    print()


if __name__ == "__main__":
    main()
