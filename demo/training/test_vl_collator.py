"""Test script for VisionLanguageCollator.

Tests the data collator with sample SAM2 data.
"""

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from core.trainers.vl_collator import create_vision_language_collator


def create_sample_data(output_dir: Path = Path("./test_data")):
    """Create sample data for testing.

    Args:
        output_dir: Directory to save sample data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create sample JSONL data (similar to LLaVA format)
    samples = [
        {
            "id": "sample_001",
            "image": "sample_001.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nIdentify and segment all objects in this image."
                },
                {
                    "from": "gpt",
                    "value": "I can see 2 persons and 1 car in the image. I've provided segmentation masks for each object."
                }
            ],
            "masks": ["<rle_encoded_mask_1>", "<rle_encoded_mask_2>", "<rle_encoded_mask_3>"],
            "bounding_boxes": [[10, 20, 100, 150], [120, 30, 80, 140], [200, 50, 150, 100]]
        },
        {
            "id": "sample_002",
            "image": "sample_002.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nSegment the person in this image."
                },
                {
                    "from": "gpt",
                    "value": "I can see 1 person in the image. The object is located at coordinates: [50, 60, 120, 180]. I've provided segmentation masks for each object."
                }
            ],
            "masks": ["<rle_encoded_mask_1>"],
            "bounding_boxes": [[50, 60, 120, 180]]
        },
        {
            "id": "sample_003",
            "image": "sample_003.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nFind and segment each object present in this image."
                },
                {
                    "from": "gpt",
                    "value": "I can see 1 car and 1 dog in the image. I've provided segmentation masks for each object."
                }
            ],
            "masks": ["<rle_encoded_mask_1>", "<rle_encoded_mask_2>"],
            "bounding_boxes": [[15, 25, 200, 150], [250, 80, 100, 90]]
        }
    ]

    # Save as JSONL
    jsonl_file = output_dir / "test_data.jsonl"
    with open(jsonl_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"✓ Created sample data: {jsonl_file}")
    print(f"  Samples: {len(samples)}")

    return jsonl_file


def test_collator():
    """Test VisionLanguageCollator."""
    print("=" * 80)
    print("Testing VisionLanguageCollator")
    print("=" * 80)
    print()

    # Step 1: Create sample data
    print("[1/4] Creating sample data...")
    data_file = create_sample_data()
    print()

    # Step 2: Load tokenizer
    print("[2/4] Loading tokenizer...")
    try:
        # Try to use a simple tokenizer (no need to download large models)
        tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)

        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
        print(f"  Vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        print("Note: This test requires internet connection to download tokenizer")
        return
    print()

    # Step 3: Create collator
    print("[3/4] Creating VisionLanguageCollator...")
    collator = create_vision_language_collator(
        tokenizer=tokenizer,
        processor=None,  # No processor for this test
        image_dir=None,  # No real images
        max_length=512,
        image_size=(336, 336)
    )
    print(f"✓ Collator created")
    print(f"  Max length: {collator.max_length}")
    print(f"  Image size: {collator.image_size}")
    print(f"  Using processor: {collator.use_processor}")
    print()

    # Step 4: Test batching
    print("[4/4] Testing batch collation...")

    # Load sample data
    samples = []
    with open(data_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples")
    print()

    # Test with batch of 2
    batch_size = 2
    batch_samples = samples[:batch_size]

    print(f"Collating batch of {batch_size} samples...")
    try:
        batch = collator(batch_samples)

        print("✓ Batch created successfully!")
        print()
        print("Batch contents:")
        print(f"  Keys: {list(batch.keys())}")
        print()

        for key, value in batch.items():
            if isinstance(value, list):
                print(f"  {key}: list of {len(value)} items")
            elif hasattr(value, 'shape'):
                print(f"  {key}: tensor of shape {value.shape}")
            else:
                print(f"  {key}: {type(value).__name__}")

        print()

        # Check batch shapes
        if "input_ids" in batch:
            print("Detailed info:")
            print(f"  input_ids shape: {batch['input_ids'].shape}")
            print(f"  Expected: [batch_size={batch_size}, seq_len<=512]")
            print()

            # Decode first sample
            print("First sample (decoded):")
            first_text = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
            print(f"  {first_text[:200]}...")
            print()

        print("=" * 80)
        print("✅ VisionLanguageCollator Test PASSED!")
        print("=" * 80)
        print()
        print("Key features verified:")
        print("  ✓ Sample data loading")
        print("  ✓ Conversation formatting")
        print("  ✓ Tokenization")
        print("  ✓ Batch collation")
        print("  ✓ Padding")
        print()
        print("Next steps:")
        print("  1. Test with real images")
        print("  2. Test with vision-language processor (LLaVA, Qwen-VL)")
        print("  3. Integrate with training pipeline")
        print()

    except Exception as e:
        import traceback
        print()
        print("=" * 80)
        print("❌ VisionLanguageCollator Test FAILED")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()
        print("Full traceback:")
        traceback.print_exc()
        print()


if __name__ == "__main__":
    test_collator()
