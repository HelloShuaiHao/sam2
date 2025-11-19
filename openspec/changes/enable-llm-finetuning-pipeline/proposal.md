# Proposal: Enable End-to-End LLM Fine-tuning Pipeline

## Metadata
- **Change ID**: `enable-llm-finetuning-pipeline`
- **Status**: Proposed
- **Author**: iDoctor Team
- **Created**: 2025-11-19
- **Related Changes**:
  - `add-frame-export-annotation` (depends on export functionality)

## Problem Statement

The SAM2 demo currently provides excellent video segmentation and annotation export capabilities, but lacks the final step needed for practical AI model training workflows:

**Key Problems:**
1. **No training data preparation**: Exported annotations are in JSON format but not in formats suitable for LLM fine-tuning (JSONL, HuggingFace datasets, etc.)
2. **Manual data processing required**: Users must manually convert exported data, split datasets, and prepare training configurations
3. **No integrated training pipeline**: No way to directly feed exported annotations into model training workflows
4. **Missing quality validation**: No automated checks for annotation quality, dataset balance, or format correctness before training
5. **No training orchestration**: Users must set up their own training infrastructure, hyperparameter management, and experiment tracking

**Impact:**
- Exported annotation data sits unused without significant additional engineering effort
- High barrier to entry for researchers and practitioners who want to train custom models
- Lost opportunity to create complete annotation-to-deployment workflow
- Manual processes lead to errors and inconsistency in training data preparation

## Business Value

Adding an end-to-end LLM fine-tuning pipeline will:

1. **Complete the annotation workflow**: Transform SAM2 demo from annotation tool to full AI development platform
2. **Lower barrier to entry**: Enable non-ML-experts to train custom vision models from annotated data
3. **Accelerate research**: Researchers can iterate faster from annotation to trained model
4. **Competitive differentiation**: Few tools offer integrated annotation + training in one platform
5. **Monetization opportunity**: Training pipeline can be metered/charged as premium feature
6. **Community growth**: Attract ML practitioners and researchers to the platform

## Proposed Solution

### Overview

Build an end-to-end LLM fine-tuning pipeline that:
1. Converts exported SAM2 annotations to LLM-compatible training formats
2. Performs dataset validation, cleaning, and splitting
3. Orchestrates model training with popular frameworks (LoRA, QLoRA, etc.)
4. Tracks experiments, metrics, and model artifacts
5. Exports trained models for deployment

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SAM2 Demo (Existing)                        │
│  ┌──────────┐    ┌──────────┐    ┌─────────────────────┐      │
│  │  Upload  │ -> │ Annotate │ -> │ Export Annotations  │       │
│  │  Video   │    │  Objects │    │     (JSON/ZIP)      │       │
│  └──────────┘    └──────────┘    └─────────────────────┘       │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 v
┌─────────────────────────────────────────────────────────────────┐
│               LLM Fine-tuning Pipeline (NEW)                    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. Data Preparation Service                             │  │
│  │  - Convert JSON -> JSONL/HuggingFace format             │  │
│  │  - Generate vision-language instruction pairs            │  │
│  │  - Quality validation and filtering                      │  │
│  │  - Train/val/test split (configurable ratios)           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          v                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2. Training Orchestration Service                       │  │
│  │  - Support multiple frameworks: LoRA, QLoRA, Full FT    │  │
│  │  - Configurable hyperparameters (learning rate, batch    │  │
│  │    size, epochs, etc.)                                   │  │
│  │  - Distributed training support (multi-GPU optional)     │  │
│  │  - Training job queue and resource management            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          v                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  3. Experiment Tracking Service                          │  │
│  │  - Log training metrics (loss, accuracy, etc.)           │  │
│  │  - Tensorboard integration                                │  │
│  │  - Model checkpoint management                            │  │
│  │  - Experiment comparison and visualization               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          v                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  4. Model Export & Deployment Service                    │  │
│  │  - Export in standard formats (HuggingFace, ONNX, etc.) │  │
│  │  - Model quantization and optimization                   │  │
│  │  - Deployment package generation                         │  │
│  │  - Version control for trained models                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

#### Phase 1: Data Preparation & Validation (Weeks 1-2)

**1.1 Format Conversion**
- Convert SAM2 JSON exports to instruction-tuning formats:
  ```jsonl
  {
    "image": "frame_0000.jpg",
    "conversations": [
      {"role": "user", "content": "Segment all persons in this frame"},
      {"role": "assistant", "content": "I found 2 persons: [mask_data]"}
    ],
    "masks": [...],
    "bounding_boxes": [...]
  }
  ```
- Support multiple output formats:
  - HuggingFace Datasets format
  - LLaVA instruction format
  - Custom JSONL with masks

**1.2 Quality Validation**
- Automated checks:
  - Minimum annotations per class
  - Mask quality (area, boundary smoothness)
  - Dataset balance analysis
  - Duplicate detection
  - Missing data detection
- Generate quality report with warnings/errors

**1.3 Dataset Splitting**
- Configurable train/val/test ratios (default: 80/10/10)
- Stratified splitting to maintain class balance
- Support for custom split strategies (temporal, random, etc.)

#### Phase 2: Training Orchestration (Weeks 3-4)

**2.1 Training Configuration UI**
- Model selection:
  - Base models: LLaVA, InstructBLIP, Qwen-VL, etc.
  - Model sizes: 7B, 13B, etc.
- Training method:
  - LoRA (Low-Rank Adaptation)
  - QLoRA (Quantized LoRA)
  - Full fine-tuning (for smaller models)
- Hyperparameters:
  - Learning rate, batch size, epochs
  - LoRA rank and alpha
  - Gradient accumulation steps

**2.2 Training Job Management**
- Job queue system (Celery or Ray)
- Resource allocation (GPU selection, memory limits)
- Progress monitoring (real-time loss curves)
- Pause/resume/cancel capabilities
- Email/webhook notifications on completion

**2.3 Distributed Training (Optional)**
- Multi-GPU support via DeepSpeed or FSDP
- Automatic device placement
- Mixed precision training (FP16/BF16)

#### Phase 3: Experiment Tracking (Week 5)

**3.1 Metrics Logging**
- Training metrics: loss, learning rate, gradient norm
- Validation metrics: accuracy, IoU, mAP
- Custom metrics for segmentation tasks
- Real-time visualization with Tensorboard

**3.2 Checkpoint Management**
- Automatic checkpoint saving (configurable frequency)
- Best model selection based on validation metrics
- Checkpoint pruning to save storage
- Easy rollback to previous checkpoints

**3.3 Experiment Comparison**
- Side-by-side comparison of multiple runs
- Hyperparameter sweep results
- Export experiment data for reporting

#### Phase 4: Model Export & Deployment (Week 6)

**4.1 Model Export Formats**
- HuggingFace format (most common)
- ONNX export for production inference
- Quantized models (INT8, INT4) for edge deployment
- Adapter-only export for LoRA models

**4.2 Deployment Artifacts**
- Model card generation (metadata, performance, usage)
- Inference example code
- Docker container with model + inference API
- Configuration files for deployment platforms

### User Workflow

1. **Annotate Video** (Existing)
   - Upload video to SAM2 demo
   - Annotate objects and track across frames
   - Export annotations (JSON + frames)

2. **Prepare Training Data** (NEW)
   - Upload exported ZIP to training pipeline
   - Select target format (HuggingFace, LLaVA, etc.)
   - Configure dataset split ratios
   - Review quality validation report
   - Confirm and generate training dataset

3. **Configure Training** (NEW)
   - Select base model (e.g., LLaVA-1.5-7B)
   - Choose training method (LoRA recommended)
   - Set hyperparameters or use defaults
   - Estimate training time and cost
   - Start training job

4. **Monitor Training** (NEW)
   - View real-time training progress
   - Check loss curves and validation metrics
   - Receive notification on completion
   - Review experiment summary

5. **Export Model** (NEW)
   - Select best checkpoint
   - Choose export format (HuggingFace, ONNX, etc.)
   - Download model artifacts
   - Use in production or further development

### Tech Stack

**New Components:**
- **Backend**: Python FastAPI (separate from SAM2 demo backend)
- **Training**: PyTorch + HuggingFace Transformers + PEFT
- **Job Queue**: Celery + Redis (or Ray for distributed)
- **Experiment Tracking**: Tensorboard + MLflow (optional)
- **Storage**: S3-compatible storage for models and datasets
- **Frontend**: React extension to SAM2 demo UI

**Integration with Existing:**
- Reads exported ZIP files from SAM2 demo export service
- Shares authentication and quota system with SAM2 demo
- Can be deployed as separate microservice or integrated into demo backend

## Scope

### In Scope (MVP - 6 weeks)

**Phase 1: Data Preparation**
- [x] JSON to JSONL converter for instruction tuning
- [x] HuggingFace dataset format export
- [x] Basic quality validation (missing data, duplicates)
- [x] Configurable train/val/test split
- [x] Quality report generation

**Phase 2: Training**
- [x] LoRA fine-tuning support for vision-language models
- [x] Training configuration UI
- [x] Job queue for training management
- [x] Basic hyperparameter tuning
- [x] Progress monitoring and logging

**Phase 3: Tracking**
- [x] Tensorboard integration for metrics
- [x] Checkpoint saving and management
- [x] Basic experiment comparison

**Phase 4: Export**
- [x] HuggingFace format model export
- [x] Model card generation
- [x] Simple deployment package (Docker + API)

### Out of Scope (Future Enhancements)

**Advanced Features:**
- [ ] QLoRA and full fine-tuning support (start with LoRA only)
- [ ] Multi-GPU distributed training (single GPU in MVP)
- [ ] MLflow integration (Tensorboard sufficient for MVP)
- [ ] Custom model architecture support (predefined models only)
- [ ] AutoML and hyperparameter optimization
- [ ] Active learning and data augmentation
- [ ] Model compression and pruning
- [ ] A/B testing and deployment monitoring
- [ ] Integration with cloud training platforms (SageMaker, VertexAI)

**Advanced Data Processing:**
- [ ] COCO format support (JSONL only in MVP)
- [ ] Video augmentation (spatial, temporal)
- [ ] Semi-supervised learning with unlabeled data
- [ ] Cross-validation beyond simple split

## Dependencies

### Existing Infrastructure
- **SAM2 Demo**: Annotation and export functionality (dependency)
- **Export Service**: Provides JSON annotations in ZIP format
- **Auth/Quota System**: For user management and usage tracking

### New Python Dependencies
```python
# Training
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0  # For LoRA
accelerate>=0.24.0
bitsandbytes>=0.41.0  # For quantization

# Data processing
datasets>=2.14.0
pillow>=10.0.0
opencv-python>=4.8.0

# Experiment tracking
tensorboard>=2.14.0
# mlflow>=2.8.0  # Optional for MVP

# Job queue
celery>=5.3.0
redis>=5.0.0
# ray>=2.8.0  # Alternative, for distributed

# API
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.4.0

# Storage
boto3>=1.28.0  # For S3 storage
```

### Infrastructure Requirements
- **GPU**: Minimum 1x NVIDIA GPU with 24GB VRAM (A100/RTX 4090/L40)
- **Storage**: 500GB for models, checkpoints, and datasets
- **Redis**: For job queue (can be containerized)
- **S3/MinIO**: For model and dataset storage

### External Services (Optional)
- **HuggingFace Hub**: For downloading base models
- **Weights & Biases**: For advanced experiment tracking (future)

## Risks and Mitigation

### Technical Risks

**1. GPU Resource Contention**
- *Risk*: Training jobs may conflict with SAM2 inference workloads
- *Mitigation*:
  - Separate GPU allocation for training vs inference
  - Implement priority-based job scheduling
  - Queue training jobs during low-traffic periods

**2. Training Time Variability**
- *Risk*: Training can take hours or days, leading to poor UX
- *Mitigation*:
  - Provide accurate time estimates before starting
  - Support checkpoint resuming for interrupted jobs
  - Implement email notifications for long-running jobs

**3. Model Compatibility**
- *Risk*: Base models may have incompatible architectures or dependencies
- *Mitigation*:
  - Start with well-tested models (LLaVA, Qwen-VL)
  - Provide clear compatibility matrix
  - Validate model before starting training

**4. Data Format Mismatches**
- *Risk*: Exported SAM2 data may not directly map to LLM formats
- *Mitigation*:
  - Design flexible conversion layer
  - Support multiple output formats
  - Provide data preview before training

**5. Storage Explosion**
- *Risk*: Model checkpoints and datasets can consume TBs of storage
- *Mitigation*:
  - Implement automatic checkpoint pruning
  - Compress old experiments
  - Set storage quotas per user

### Operational Risks

**1. Training Costs**
- *Risk*: GPU training is expensive, potential for runaway costs
- *Mitigation*:
  - Set time/cost limits per training job
  - Require cost estimation approval before start
  - Implement quota system for training hours

**2. Model Quality Issues**
- *Risk*: Trained models may have poor quality, wasting resources
- *Mitigation*:
  - Require minimum validation set size
  - Automated quality checks during training
  - Early stopping on degrading metrics

**3. Security Concerns**
- *Risk*: User-uploaded models/data may contain malicious code
- *Mitigation*:
  - Sandboxed training environment
  - Scan uploaded files for malware
  - Limit model execution capabilities

## Success Criteria

### Functional Requirements

**Data Preparation:**
- [ ] Converts SAM2 JSON exports to at least 2 LLM formats (HuggingFace, JSONL)
- [ ] Detects and reports data quality issues (missing annotations, class imbalance)
- [ ] Performs train/val/test split with configurable ratios
- [ ] Generates human-readable quality report

**Training:**
- [ ] Successfully fine-tunes LLaVA-1.5-7B with LoRA on sample dataset
- [ ] Supports at least 2 base models (LLaVA, Qwen-VL)
- [ ] Allows configuration of key hyperparameters (LR, batch size, epochs)
- [ ] Displays real-time training progress (loss curve, ETA)

**Experiment Tracking:**
- [ ] Logs all training metrics to Tensorboard
- [ ] Saves checkpoints at configurable intervals
- [ ] Allows comparison of 2+ experiments side-by-side

**Model Export:**
- [ ] Exports trained model in HuggingFace format
- [ ] Generates model card with metadata and performance stats
- [ ] Provides runnable inference example code

### Performance Targets

**Data Processing:**
- [ ] Converts 10,000 frame annotations in < 5 minutes
- [ ] Validates dataset quality in < 2 minutes for typical dataset

**Training:**
- [ ] Initiates training job within 30 seconds of submission
- [ ] Supports batch sizes up to 16 on 24GB GPU
- [ ] Achieves > 70% GPU utilization during training

**Experiment Management:**
- [ ] Stores up to 100 experiment runs per user
- [ ] Displays metrics with < 1 second lag on dashboard
- [ ] Checkpoint load/save completes in < 10 seconds

### Quality Gates

**Code Quality:**
- [ ] All Python code passes type checking (mypy)
- [ ] Unit test coverage > 80% for core modules
- [ ] Integration tests for full training pipeline

**Documentation:**
- [ ] API documentation for all endpoints
- [ ] User guide with examples for each workflow step
- [ ] Troubleshooting guide for common training issues

**Security:**
- [ ] Training jobs run in isolated containers
- [ ] User data and models are encrypted at rest
- [ ] No model execution outside sandboxed environment

## Timeline Estimate

**Week 1-2: Data Preparation Service**
- JSON to JSONL/HuggingFace converter
- Quality validation logic
- Dataset splitting utilities
- Quality report generation
- Unit tests and documentation

**Week 3-4: Training Orchestration**
- Training configuration API and UI
- LoRA fine-tuning integration with HuggingFace
- Job queue setup (Celery + Redis)
- Progress monitoring and logging
- Integration with SAM2 demo export

**Week 5: Experiment Tracking**
- Tensorboard integration
- Checkpoint management
- Experiment comparison dashboard
- Metrics visualization

**Week 6: Model Export & Polish**
- HuggingFace model export
- Model card generation
- Deployment package creation
- End-to-end testing
- Documentation and examples

**Total Duration**: 6 weeks (1 developer full-time)

**Parallelization Opportunities:**
- Frontend UI (weeks 3-5) can be developed in parallel with backend
- Documentation can be written incrementally throughout

## Open Questions

1. **Base Model Selection**
   - *Question*: Which vision-language models should we prioritize for MVP?
   - *Options*: LLaVA, InstructBLIP, Qwen-VL, MiniGPT-4
   - *Recommendation*: Start with LLaVA-1.5 (most popular, well-documented)

2. **Training Method Priority**
   - *Question*: Should MVP support only LoRA, or include QLoRA/full FT?
   - *Recommendation*: LoRA only in MVP - simpler, faster, less memory

3. **Job Queue Technology**
   - *Question*: Celery + Redis vs Ray for job orchestration?
   - *Recommendation*: Celery for MVP (simpler), migrate to Ray if distributed training needed

4. **Storage Backend**
   - *Question*: Local filesystem vs S3 for model/dataset storage?
   - *Recommendation*: Start with local, add S3 support in v2 for scalability

5. **Experiment Tracking**
   - *Question*: Tensorboard vs MLflow vs Weights & Biases?
   - *Recommendation*: Tensorboard for MVP (free, integrated with PyTorch), add MLflow in v2

6. **Integration vs Standalone**
   - *Question*: Should training service be integrated into SAM2 demo backend or separate microservice?
   - *Recommendation*: Separate microservice for better scaling and resource isolation

7. **GPU Allocation**
   - *Question*: How to share GPUs between inference (SAM2) and training?
   - *Recommendation*: Dedicate separate GPU for training, or implement priority-based scheduling

8. **Quota System**
   - *Question*: How to meter training usage - by time, GPU-hours, model size?
   - *Recommendation*: Track GPU-hours (easy to understand, aligns with cloud pricing)

## Alternatives Considered

### Alternative 1: Use External Training Platform (SageMaker, HuggingFace AutoTrain)
- **Pros**: No infrastructure management, proven scalability, automatic optimization
- **Cons**: High costs, less control, vendor lock-in, complex integration
- **Decision**: Rejected - user wants integrated solution, more control over costs

### Alternative 2: Support Only Cloud-Based Training
- **Pros**: Infinite scalability, no local GPU requirements
- **Cons**: Ongoing cloud costs, data transfer latency, privacy concerns
- **Decision**: Rejected - start with local training, add cloud option in v2

### Alternative 3: Pre-configured Training Templates Only
- **Pros**: Simpler UX, fewer configuration errors, faster setup
- **Cons**: Less flexibility, may not fit all use cases
- **Decision**: Hybrid approach - provide templates but allow customization

### Alternative 4: Focus on Traditional CV Models (YOLO, Mask R-CNN)
- **Pros**: Simpler training, faster inference, smaller models
- **Cons**: Less flexible than LLMs, limited to specific tasks
- **Decision**: Rejected - user specifically requested LLM fine-tuning for broader capabilities

## References

- **Related Proposals**:
  - `add-frame-export-annotation`: Provides input data for training pipeline
- **SAM2 Demo**:
  - Export Service: `demo/backend/server/data/export_service.py`
  - Annotation Format: See exported JSON structure
- **Training Frameworks**:
  - HuggingFace PEFT: https://github.com/huggingface/peft
  - LLaVA: https://github.com/haotian-liu/LLaVA
  - Qwen-VL: https://github.com/QwenLM/Qwen-VL
- **Experiment Tracking**:
  - Tensorboard: https://www.tensorflow.org/tensorboard
  - MLflow: https://mlflow.org/
- **Deployment**:
  - HuggingFace Hub: https://huggingface.co/docs/hub/
  - ONNX Export: https://huggingface.co/docs/transformers/serialization#onnx

## Appendix

### Example: SAM2 JSON to LLaVA Format Conversion

**Input (SAM2 Export JSON):**
```json
{
  "video": {
    "filename": "example.mp4",
    "width": 1920,
    "height": 1080,
    "source_fps": 30,
    "total_frames": 900
  },
  "annotations": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "objects": [
        {
          "object_id": 1,
          "label": "person",
          "mask_rle": "...",
          "bbox": [100, 200, 300, 400]
        }
      ]
    }
  ]
}
```

**Output (LLaVA Instruction Format):**
```jsonl
{
  "id": "example_frame_0",
  "image": "frame_0000.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nIdentify and segment all objects in this image."
    },
    {
      "from": "gpt",
      "value": "I can see 1 person in the image. The person is located at coordinates [100, 200, 300, 400] with a segmentation mask."
    }
  ],
  "masks": ["..."],
  "bounding_boxes": [[100, 200, 300, 400]]
}
```

### Example: Training Configuration

```yaml
# training_config.yaml
model:
  name: "liuhaotian/llava-v1.5-7b"
  type: "llava"

training:
  method: "lora"
  lora_rank: 64
  lora_alpha: 16
  learning_rate: 2e-5
  batch_size: 4
  gradient_accumulation_steps: 4
  num_epochs: 3
  warmup_ratio: 0.03

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  max_samples: null  # Use all data

hardware:
  device: "cuda"
  mixed_precision: "bf16"
  gradient_checkpointing: true

checkpointing:
  save_steps: 100
  save_total_limit: 3

logging:
  log_steps: 10
  tensorboard_dir: "./runs"
```

### Example: Quality Validation Report

```markdown
# Dataset Quality Report

## Summary
- Total Frames: 1,500
- Total Objects: 4,200
- Object Classes: person (2,800), car (900), dog (500)
- Train/Val/Test Split: 1,200 / 150 / 150 frames

## Quality Checks

### ✅ Passed
- All frames have at least one annotation
- No duplicate frames detected
- All masks are valid RLE format
- Bounding boxes within image boundaries

### ⚠️ Warnings
- Class imbalance: 'person' has 5.6x more samples than 'dog'
- 5 frames have unusually small object area (< 1% of image)

### ❌ Errors
- None

## Recommendations
1. Consider data augmentation for minority classes (car, dog)
2. Review frames with small objects for annotation quality
3. Current dataset is suitable for training, estimated time: 2.5 hours
```
