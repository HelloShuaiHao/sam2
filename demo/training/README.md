# LLM Fine-tuning Pipeline for SAM2

End-to-end pipeline for fine-tuning vision-language models on SAM2 annotation exports.

## Features

### Phase 1: Data Preparation ✅ (COMPLETE & TESTED)

- **Format Conversion**
  - Convert SAM2 JSON exports to HuggingFace Datasets format
  - Convert to LLaVA instruction-tuning format (JSONL)
  - Flexible SAM2 export parser (handles nested directories, PNG/JPG frames)
  - Automated instruction-response pair generation with varied templates

- **Quality Validation**
  - Missing annotation detection
  - Duplicate frame checks
  - Mask RLE integrity validation
  - Bounding box validation
  - Class balance analysis (detects >10:1 imbalances)
  - Quality metrics calculation (object areas, annotation density)
  - Automated Markdown report generation

- **Dataset Splitting**
  - Stratified splitting (maintains class balance within 5%)
  - Temporal splitting (prevents data leakage for videos)
  - Random splitting with reproducible seeds
  - Split validation and statistics

### Phase 2: Training Configuration ✅ (COMPLETE)

- **Model Registry**
  - 4 pre-registered vision-language models (LLaVA 7B/13B, Qwen-VL, InstructBLIP)
  - Automatic VRAM estimation for different configurations
  - Model metadata (size, VRAM requirements, sequence lengths)

- **Configuration System**
  - Pydantic-based type-safe configuration
  - 5 pre-configured presets:
    - **Quick Test**: Fast iteration (1 epoch, rank 8)
    - **Development**: Balanced experimentation (3 epochs, rank 32)
    - **Production**: High quality (5 epochs, rank 64)
    - **Memory Efficient**: Low VRAM (rank 16, batch 1)
    - **High Quality**: Best results with 13B model (rank 128)
  - JSON serialization for reproducibility
  - Automatic validation of hyperparameters

- **LoRA Trainer (Core Implementation)**
  - PEFT integration for parameter-efficient fine-tuning
  - Automatic target module detection
  - Gradient checkpointing support
  - Mixed precision training (FP16/BF16)
  - HuggingFace Transformers integration

- **Checkpoint Management**
  - Automatic checkpoint pruning (keep best + N recent)
  - Best model tracking by metric
  - Checkpoint metadata persistence
  - Resume from checkpoint support

### Phase 3-6: Coming Soon

- Experiment tracking with Tensorboard
- Model export and deployment
- FastAPI REST API
- React frontend UI
- Celery job queue for distributed training

## Installation

```bash
cd demo/training
pip install -r requirements.txt
```

## Quick Start

### Phase 1: Data Preparation (Tested with Real SAM2 Exports!)

```python
from core.data_converter import SAM2Parser, HuggingFaceConverter, LLaVAConverter
from core.validation import Validator, BasicChecks, BalanceAnalyzer, QualityMetrics
from core.data_splitter import SplitConfig, SplitStrategy, StratifiedSplitter
from pathlib import Path

# 1. Parse SAM2 export
parser = SAM2Parser()
data = parser.parse_zip(Path("path/to/sam2_export.zip"))

# 2. Convert to HuggingFace format
hf_converter = HuggingFaceConverter()
stats = hf_converter.convert(
    input_path=Path("path/to/sam2_export.zip"),
    output_path=Path("output/hf_dataset")
)
print(f"Converted {stats['total_samples']} samples")

# 3. Validate dataset quality
validator = Validator()
for check in BasicChecks.get_all_checks():
    validator.add_rule(check)
validator.add_rule(BalanceAnalyzer(imbalance_threshold=10.0))
validator.add_rule(QualityMetrics(min_object_area_percent=0.1))

results = validator.validate({
    "frames": parser.get_frames(),
    "metadata": parser.metadata
})

# 4. Generate validation report
from core.validation import ReportGenerator
report = ReportGenerator.generate_report(results, dataset_name="My Dataset")
ReportGenerator.save_report(report, "validation_report.md")

# 5. Split dataset
config = SplitConfig(
    strategy=SplitStrategy.STRATIFIED,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
)

splitter = StratifiedSplitter(config)
train, val, test = splitter.split(dataset_samples)

# Validate split maintains class balance
validation = splitter.validate_split(train, val, test, tolerance=0.05)
print(f"Split is valid: {validation['is_valid']}")
```

**Or use the complete example script:**

```bash
cd demo/training
python example_data_preparation.py path/to/sam2_export.zip output/
```

### Phase 2: Training Configuration

```python
from core.config import ConfigPresets, ModelRegistry, TrainingConfig

# Option 1: Use a preset
config = ConfigPresets.production(
    data_path="./output/splits/train.jsonl",
    val_path="./output/splits/val.jsonl"
)

# Option 2: Customize configuration
from core.config import (
    TrainingConfig, ModelConfig, DataConfig,
    TrainingHyperparameters, TrainingMethod, LoRAConfig
)

config = TrainingConfig(
    model=ModelConfig(name="liuhaotian/llava-v1.5-7b", type="llava"),
    data=DataConfig(
        train_path="./output/splits/train.jsonl",
        val_path="./output/splits/val.jsonl",
        max_length=1024
    ),
    training=TrainingHyperparameters(
        method=TrainingMethod.LORA,
        learning_rate=2e-5,
        batch_size=4,
        gradient_accumulation_steps=4,
        num_epochs=3,
        lora=LoRAConfig(rank=64, alpha=128)
    ),
    experiment_name="sam2_segmentation"
)

# Estimate VRAM requirements
vram_est = ModelRegistry.estimate_vram_requirements(
    "liuhaotian/llava-v1.5-7b",
    use_lora=True,
    batch_size=4
)
print(f"Estimated VRAM: {vram_est['total_vram_gb']:.1f}GB")
print(f"Recommended GPU: {vram_est['recommended_gpu']}")

# Save configuration
config.save_json("training_config.json")
```

**Or explore the configuration system:**

```bash
cd demo/training
python example_training_config.py
```

## Project Structure

```
demo/training/
├── core/
│   ├── data_converter/      # ✅ Format conversion (SAM2 → HF/LLaVA)
│   │   ├── sam2_parser.py
│   │   ├── huggingface_converter.py
│   │   └── llava_converter.py
│   ├── validation/          # ✅ Quality validation
│   │   ├── validator.py
│   │   ├── basic_checks.py
│   │   ├── balance_analysis.py
│   │   ├── quality_metrics.py
│   │   └── report_generator.py
│   ├── data_splitter/       # ✅ Dataset splitting
│   │   ├── split_config.py
│   │   ├── stratified_splitter.py
│   │   ├── temporal_splitter.py
│   │   └── random_splitter.py
│   ├── config/              # ✅ Training configuration
│   │   ├── training_config.py
│   │   ├── model_registry.py
│   │   └── presets.py
│   ├── trainers/            # ✅ LoRA trainer
│   │   ├── base_trainer.py
│   │   └── lora_trainer.py
│   └── checkpoints/         # ✅ Checkpoint management
│       ├── checkpoint_manager.py
│       └── best_model_tracker.py
├── requirements.txt         # ✅ All dependencies
├── README.md                # ✅ Documentation
├── example_data_preparation.py   # ✅ Phase 1 demo
└── example_training_config.py    # ✅ Phase 2 demo
```

## Development Status

- [x] **Phase 1: Data Preparation** - ✅ COMPLETE & TESTED
  - [x] SAM2 parser (handles nested directories, PNG frames)
  - [x] HuggingFace converter
  - [x] LLaVA converter
  - [x] Quality validation framework (6 validation rules)
  - [x] Dataset splitting utilities (3 strategies)
  - [x] **Tested with real SAM2 exports** (40 frames, 3 classes, 120 objects)

- [x] **Phase 2: Training Configuration** - ✅ COMPLETE
  - [x] Training configuration schema (Pydantic)
  - [x] Model registry (4 vision-language models)
  - [x] Configuration presets (5 presets)
  - [x] LoRA trainer implementation
  - [x] Checkpoint management
  - [x] Best model tracking
  - [x] VRAM estimation

- [ ] **Phase 3: Experiment Tracking** - In Progress
  - [ ] Tensorboard integration
  - [ ] Custom metrics logging
  - [ ] Training visualization

- [ ] **Phase 4: Model Export**
- [ ] **Phase 5: API & UI**
- [ ] **Phase 6: Testing & Documentation**

**Overall Progress**: 19/58 tasks complete (33%)

## Testing

```bash
# Run tests (coming soon)
pytest tests/

# Check code quality
ruff check core/
mypy core/
```

## Contributing

This is part of the SAM2 demo project. See the main project README for contribution guidelines.

## License

Same as SAM2 project license.
