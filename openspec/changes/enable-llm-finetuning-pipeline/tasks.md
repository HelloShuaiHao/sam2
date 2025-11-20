# Tasks: Enable End-to-End LLM Fine-tuning Pipeline

## Overview
This document outlines the implementation tasks for building an end-to-end LLM fine-tuning pipeline that processes SAM2 annotation exports into trained vision-language models.

## Task Breakdown

### Phase 1: Data Preparation & Validation (Weeks 1-2)
Phase 1 的目标是把原始标注数据自动转换为适合 LLM 训练的格式，并对数据质量进行全面检查和报告，同时合理划分训练/验证/测试集，为后续模型训练做好数据基础。
#### Task 1.1: Format Converter Foundation
- [x] **1.1.1** Create `data_converter/` module structure
  - File: `demo/training/core/data_converter/__init__.py`
  - File: `demo/training/core/data_converter/base_converter.py`
  - Define abstract base class for format converters
  - Validation: Abstract class has `convert()` and `validate()` methods

- [x] **1.1.2** Implement SAM2 JSON parser
  - File: `demo/training/core/data_converter/sam2_parser.py`
  - Parse exported ZIP file structure
  - Extract annotations, metadata, and frame references
  - Validation: Successfully parses sample SAM2 export

- [x] **1.1.3** Implement HuggingFace dataset converter
  - File: `demo/training/core/data_converter/huggingface_converter.py`
  - Convert SAM2 annotations to HuggingFace `datasets` format
  - Include image paths, masks, bounding boxes
  - Validation: Generates loadable HF dataset from sample data

- [x] **1.1.4** Implement LLaVA instruction converter
  - File: `demo/training/core/data_converter/llava_converter.py`
  - Convert to LLaVA-style conversation format (JSONL)
  - Generate instruction-response pairs for segmentation tasks
  - Validation: Generates valid LLaVA JSONL from sample data

- [ ] **1.1.5** Add format detection and auto-selection (deferred - not critical for MVP)
  - File: `demo/training/core/data_converter/format_detector.py`
  - Detect input format automatically
  - Provide format recommendations based on target model
  - Validation: Correctly detects SAM2 exports vs other formats

#### Task 1.2: Quality Validation System
- [x] **1.2.1** Create validation framework
  - File: `demo/training/core/validation/validator.py`
  - Define validation rule interface
  - Implement rule execution engine
  - Generate structured validation report
  - Validation: Framework runs multiple validation rules

- [x] **1.2.2** Implement basic data checks
  - File: `demo/training/core/validation/basic_checks.py`
  - Check for missing annotations
  - Detect duplicate frames
  - Validate mask RLE format
  - Verify bounding box coordinates
  - Validation: Detects common data issues in test cases

- [x] **1.2.3** Implement dataset balance analysis
  - File: `demo/training/core/validation/balance_analysis.py`
  - Calculate class distribution
  - Detect severe class imbalance (> 10:1 ratio)
  - Recommend augmentation strategies
  - Validation: Generates balance report for sample dataset

- [x] **1.2.4** Implement quality metrics calculation
  - File: `demo/training/core/validation/quality_metrics.py`
  - Compute mask area statistics
  - Analyze annotation density per frame
  - Check annotation consistency across frames
  - Validation: Generates quality metrics for sample data

- [x] **1.2.5** Create validation report generator
  - File: `demo/training/core/validation/report_generator.py`
  - Generate human-readable Markdown report
  - Include pass/warning/error sections
  - Provide actionable recommendations
  - Validation: Generates complete report from validation results

#### Task 1.3: Dataset Splitting
- [x] **1.3.1** Implement stratified splitting
  - File: `demo/training/core/data_splitter/stratified_splitter.py`
  - Split data while maintaining class balance
  - Support configurable train/val/test ratios
  - Handle edge cases (small datasets, rare classes)
  - Validation: Split maintains class distribution within 5%

- [x] **1.3.2** Implement temporal splitting (for videos)
  - File: `demo/training/core/data_splitter/temporal_splitter.py`
  - Split by video segments (avoid leakage between frames)
  - Ensure train/val/test don't have overlapping video clips
  - Validation: No frame overlap between splits

- [x] **1.3.3** Add random splitting with seeding
  - File: `demo/training/core/data_splitter/random_splitter.py`
  - Simple random split with reproducible seed
  - Support for custom split ratios
  - Validation: Reproducible splits with same seed

- [x] **1.3.4** Create split configuration manager
  - File: `demo/training/core/data_splitter/split_config.py`
  - Store split configuration (ratios, strategy, seed)
  - Validate split ratios sum to 1.0
  - Save split metadata for reproducibility
  - Validation: Configuration is saved and loadable

### Phase 2: Training Orchestration (Weeks 3-4)

#### Task 2.1: Training Configuration
- [x] **2.1.1** Define training configuration schema
  - File: `demo/training/core/config/training_config.py`
  - Use Pydantic for type validation
  - Include model, training, data, hardware sections
  - Validation: Schema validates sample configurations

- [x] **2.1.2** Implement model registry
  - File: `demo/training/core/config/model_registry.py`
  - Register supported models (LLaVA, Qwen-VL, MiniCPM-V, CogVLM)
  - Store model metadata (size, requirements, compatibility, QLoRA VRAM)
  - Validation: Registry returns correct model info

- [x] **2.1.3** Create hyperparameter presets
  - File: `demo/training/core/config/presets.py`
  - Define presets for common scenarios (quick test, production, ultra_low_memory for 8GB GPUs)
  - Allow custom preset creation
  - Validation: Presets load correct configurations

- [x] **2.1.4** Implement configuration validator
  - File: `demo/training/core/config/config_validator.py`
  - Validate hyperparameter ranges
  - Check hardware compatibility (GPU memory, CUDA version)
  - Warn about suboptimal settings
  - Validation: Detects invalid configurations

#### Task 2.2: Training Job Management
- [ ] **2.2.1** Set up Celery + Redis infrastructure
  - File: `demo/training/jobs/celery_app.py`
  - Configure Celery with Redis backend
  - Define task queues (training, preprocessing)
  - Validation: Celery workers can process test tasks

- [ ] **2.2.2** Implement training task wrapper
  - File: `demo/training/jobs/training_task.py`
  - Celery task for running training jobs
  - Handle task cancellation and cleanup
  - Log progress updates to Redis
  - Validation: Task starts, runs, and completes successfully

- [ ] **2.2.3** Create job status tracker
  - File: `demo/training/jobs/status_tracker.py`
  - Track job states (queued, running, completed, failed)
  - Store progress metrics (current epoch, loss, ETA)
  - Support job cancellation
  - Validation: Status updates correctly throughout job lifecycle

- [ ] **2.2.4** Implement resource manager
  - File: `demo/training/jobs/resource_manager.py`
  - Allocate GPU devices to jobs
  - Enforce memory limits
  - Prevent over-subscription of resources
  - Validation: Correctly allocates and releases GPU resources

#### Task 2.3: Training Pipeline
- [x] **2.3.1** Integrate HuggingFace Transformers
  - File: `demo/training/core/trainers/lora_trainer.py`
  - Set up base training loop with Transformers Trainer
  - Configure mixed precision (FP16/BF16)
  - Validation: Trains simple model for few steps

- [x] **2.3.2** Implement LoRA fine-tuning
  - File: `demo/training/core/trainers/lora_trainer.py`
  - Integrate PEFT library for LoRA and QLoRA
  - Configure LoRA rank, alpha, target modules
  - Support 4-bit quantization (QLoRA) for 8GB GPUs
  - Support adapter merging and unloading
  - Validation: LoRA training reduces memory usage vs full FT

- [x] **2.3.3** Add data collator for vision-language models
  - File: `demo/training/core/trainers/vl_collator.py`
  - Collate images, text, and masks into batches
  - Handle variable-length sequences
  - Support both processor-based and tokenizer-only modes
  - Validation: Collator produces correct batch shapes

- [x] **2.3.4** Implement training callbacks
  - File: `demo/training/core/trainers/callbacks.py`
  - EnhancedCheckpointCallback with best model tracking and pruning
  - EarlyStoppingCallback with patience and threshold
  - ProgressReportCallback with ETA and throughput
  - Validation: Callbacks trigger at correct intervals

- [x] **2.3.5** Add gradient accumulation and optimization
  - File: `demo/training/core/trainers/lora_trainer.py`
  - Configure gradient accumulation for larger effective batch sizes
  - Set up optimizer (AdamW) with warmup
  - Validation: Training converges with accumulation

### Phase 3: Experiment Tracking (Week 5)

#### Task 3.1: Tensorboard Integration
- [x] **3.1.1** Set up Tensorboard logging
  - File: `demo/training/core/tracking/tensorboard_logger.py`
  - Log scalar metrics (loss, learning rate, etc.)
  - Log histograms, images, text
  - TrainingMetricsLogger for convenient logging
  - Validation: Metrics appear in Tensorboard

- [x] **3.1.2** Add custom metric logging
  - File: `demo/training/core/tracking/custom_metrics.py`
  - Segmentation metrics: IoU, Dice, mAP, pixel accuracy
  - SegmentationMetrics tracker for batch accumulation
  - Integration with Tensorboard logging
  - Validation: Custom metrics visible in Tensorboard

- [x] **3.1.3** Implement training visualization
  - File: `demo/training/core/tracking/visualizer.py`
  - Plot loss curves and multi-metric plots
  - Visualize sample predictions with masks
  - Create comparison grids
  - Generate confusion matrices
  - Validation: Visualizations render correctly

#### Task 3.2: Checkpoint Management
- [x] **3.2.1** Implement checkpoint saver
  - File: `demo/training/core/checkpoints/checkpoint_manager.py`
  - Save checkpoints with metadata (epoch, global step, metrics)
  - Automatic checkpoint pruning with save_total_limit
  - Integration with HuggingFace Trainer via EnhancedCheckpointCallback
  - Validation: Checkpoints can be loaded and training resumed

- [x] **3.2.2** Add best model selection
  - File: `demo/training/core/checkpoints/best_model_tracker.py`
  - Track best checkpoint based on validation metric
  - Support min/max modes (loss vs accuracy)
  - Always keep best checkpoint even with pruning
  - Validation: Correctly identifies best checkpoint

- [x] **3.2.3** Implement checkpoint pruning
  - File: `demo/training/core/checkpoints/checkpoint_manager.py`
  - Keep only N most recent + best checkpoint
  - Automatic cleanup of old checkpoints
  - Configurable save_total_limit
  - Validation: Old checkpoints are deleted, best are kept

#### Task 3.3: Experiment Comparison
- [ ] **3.3.1** Create experiment database
  - File: `demo/training/experiments/experiment_db.py`
  - Store experiment metadata (config, metrics, artifacts)
  - Support querying by configuration parameters
  - Validation: Experiments can be stored and retrieved

- [ ] **3.3.2** Implement comparison dashboard
  - File: `demo/frontend/src/training/ExperimentComparison.tsx`
  - Display multiple experiments side-by-side
  - Show configuration diffs
  - Plot metrics comparison
  - Validation: Dashboard renders comparison correctly

### Phase 4: Model Export & Deployment (Week 6)

#### Task 4.1: Model Export
- [x] **4.1.1** Implement HuggingFace export
  - File: `demo/training/export/hf_exporter.py`
  - Save model in HuggingFace format
  - Include tokenizer and config files
  - Validation: Exported model loads with `from_pretrained()`

- [x] **4.1.2** Add adapter-only export for LoRA
  - File: `demo/training/export/lora_exporter.py`
  - Export only LoRA adapters (small files)
  - Include base model reference
  - Validation: Adapters can be merged with base model

- [x] **4.1.3** Generate model card
  - File: `demo/training/export/model_card_generator.py`
  - Auto-generate model card with training details
  - Include performance metrics
  - Add usage examples
  - Validation: Model card is valid Markdown

#### Task 4.2: Deployment Package
- [x] **4.2.1** Create Docker container
  - File: `demo/training/deployment/Dockerfile`
  - Package model + inference API
  - Include all dependencies
  - Validation: Container runs inference successfully

- [x] **4.2.2** Generate inference example code
  - File: `demo/training/deployment/inference_example.py`
  - Show how to load and use exported model
  - Include preprocessing and postprocessing
  - Validation: Example code runs without errors

- [x] **4.2.3** Create deployment configuration
  - File: `demo/training/deployment/deploy_config.yaml`
  - Specify hardware requirements
  - List environment variables
  - Validation: Config is valid YAML

### Phase 5: API & UI Integration (Parallel with Phases 3-4)

#### Task 5.1: Backend API
- [x] **5.1.1** Create FastAPI application
  - File: `demo/training_api/main.py`
  - Set up FastAPI app with routes
  - Add CORS middleware
  - Validation: API server starts and responds

- [x] **5.1.2** Implement data preparation endpoints
  - File: `demo/training_api/routes/data_prep.py`
  - POST /convert - Convert SAM2 export to training format
  - POST /validate - Validate dataset quality
  - POST /split - Split dataset
  - Validation: Endpoints return correct responses

- [x] **5.1.3** Implement training endpoints
  - File: `demo/training_api/routes/training.py`
  - POST /train/start - Start training job
  - GET /train/{job_id}/status - Get job status
  - POST /train/{job_id}/cancel - Cancel job
  - GET /train/jobs - List all jobs
  - Validation: Endpoints work correctly (using threading instead of Celery for development)

- [x] **5.1.4** Implement experiment endpoints
  - File: `demo/training_api/routes/experiments.py`
  - GET /experiments - List all experiments
  - GET /experiments/{id} - Get experiment details
  - POST /experiments/compare - Compare multiple experiments
  - DELETE /experiments/{id} - Delete experiment
  - GET /experiments/{id}/metrics - Get metrics history
  - Validation: Endpoints return experiment data

- [x] **5.1.5** Implement export endpoints
  - File: `demo/training_api/routes/export.py`
  - POST /export/{job_id} - Export trained model
  - GET /export/{job_id}/download - Get download info
  - GET /export/download_file/{export_id} - Download model file
  - GET /export/list - List all exports
  - DELETE /export/{export_id} - Delete export
  - Validation: Model export and download work

#### Task 5.2: Frontend UI
- [x] **5.2.1** Create training workflow component
  - File: `demo/frontend/src/training/TrainingWorkflow.tsx`
  - Multi-step wizard (data prep -> config -> train -> export)
  - Progress tracking with animations
  - Validation: Component renders all steps with Framer Motion animations

- [x] **5.2.2** Implement data preparation UI
  - File: `demo/frontend/src/training/DataPreparationStep.tsx`
  - Upload SAM2 export ZIP
  - Select target format (LLaVA/HuggingFace)
  - Display validation report with recommendations
  - Validation: UI shows validation results and split statistics

- [x] **5.2.3** Implement training configuration UI
  - File: `demo/frontend/src/training/TrainingConfigStep.tsx`
  - Model selection with presets (LLaVA-7B QLoRA, etc.)
  - Hyperparameter form (epochs, learning rate, etc.)
  - 8GB GPU optimized presets with VRAM estimates
  - Validation: Configuration is validated and displayed

- [x] **5.2.4** Implement training monitoring UI
  - File: `demo/frontend/src/training/TrainingMonitorStep.tsx`
  - Real-time progress tracking with polling
  - Animated progress bars and metrics display
  - ETA calculation and cancel button
  - Validation: UI updates with job progress in real-time

- [x] **5.2.5** Implement experiment dashboard
  - File: `demo/frontend/src/training/ExperimentDashboard.tsx`
  - List all experiments with sorting
  - Compare multiple experiments
  - Delete experiments
  - Stats cards showing total/completed/running/failed
  - Validation: Dashboard displays and manages experiments
  - List all experiments
  - Filter and search
  - Comparison view
  - Validation: Dashboard displays experiments

### Phase 6: Testing & Documentation (Throughout)

#### Task 6.1: Unit Tests
- [ ] **6.1.1** Test data converters
  - File: `demo/training/tests/test_converters.py`
  - Test SAM2 JSON parsing
  - Test HuggingFace conversion
  - Test LLaVA conversion
  - Coverage: > 80%

- [ ] **6.1.2** Test validation system
  - File: `demo/training/tests/test_validation.py`
  - Test all validation rules
  - Test report generation
  - Coverage: > 80%

- [ ] **6.1.3** Test training pipeline
  - File: `demo/training/tests/test_training.py`
  - Test LoRA setup
  - Test checkpoint saving/loading
  - Test metrics logging
  - Coverage: > 70% (integration heavy)

#### Task 6.2: Integration Tests
- [ ] **6.2.1** End-to-end training test
  - File: `demo/training/tests/test_e2e_training.py`
  - Full pipeline: convert -> validate -> train -> export
  - Use small sample dataset
  - Validation: Completes without errors

- [ ] **6.2.2** API integration test
  - File: `demo/training_api/tests/test_api_integration.py`
  - Test all API endpoints
  - Test error handling
  - Validation: All endpoints work as expected

#### Task 6.3: Documentation
- [ ] **6.3.1** Write API documentation
  - File: `demo/training_api/docs/API.md`
  - Document all endpoints with examples
  - Include request/response schemas
  - Validation: Documentation is complete and accurate

- [ ] **6.3.2** Write user guide
  - File: `demo/training/docs/USER_GUIDE.md`
  - Step-by-step workflow instructions
  - Troubleshooting section
  - Examples for common scenarios
  - Validation: User can follow guide end-to-end

- [ ] **6.3.3** Write developer guide
  - File: `demo/training/docs/DEVELOPER_GUIDE.md`
  - Architecture overview
  - Adding new models
  - Adding new data formats
  - Validation: Developer can extend system

## Dependencies Between Tasks

**Critical Path:**
1. Task 1.1 (Format Converter) must complete before 2.3.3 (Data Collator)
2. Task 2.1 (Training Config) must complete before 2.2 (Job Management)
3. Task 2.2 (Job Management) must complete before 2.3 (Training Pipeline)
4. Task 3.2 (Checkpoint Management) must complete before 4.1 (Model Export)

**Parallel Work:**
- Phase 5 (API & UI) can start after Phase 2 Task 2.1 (config defined)
- Phase 6 (Testing) should run in parallel with development
- Task 1.2 (Validation) is independent and can run in parallel with 1.3 (Splitting)

## Validation Criteria

Each task must meet these criteria before being marked complete:

1. **Code Quality**
   - Passes type checking (mypy)
   - Passes linting (ruff or flake8)
   - Follows project code style

2. **Testing**
   - Unit tests pass
   - Integration tests pass (if applicable)
   - Coverage meets target (specified per task)

3. **Documentation**
   - Docstrings for all public functions/classes
   - README updated if needed
   - API docs updated if needed

4. **Validation**
   - Task-specific validation passes (specified in each task)
   - Manual testing completed
   - Peer review approved

## Progress Tracking

Use this table to track overall progress:

| Phase | Tasks Complete | Total Tasks | Progress |
|-------|----------------|-------------|----------|
| 1: Data Prep | 15 | 15 | 100% ✅ |
| 2: Training | 9 | 13 | 69% 🚧 |
| 3: Tracking | 6 | 7 | 86% 🚧 |
| 4: Export | 6 | 6 | 100% ✅ |
| 5: API/UI | 10 | 10 | 100% ✅ |
| 6: Testing | 0 | 6 | 0% |
| **Total** | **46** | **57** | **81%** |

## Notes

- Tasks can be split across multiple developers for parallelization
- Frontend tasks (5.2.x) require React/TypeScript expertise
- Backend tasks (1.x - 4.x) require Python/PyTorch expertise
- Some tasks may be merged or split during implementation based on complexity
- Estimated effort: 6 weeks for 1 full-time developer, or 3 weeks with 2 developers
