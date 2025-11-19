# Design: End-to-End LLM Fine-tuning Pipeline

## Architecture Overview

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Client Layer (Browser)                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │         SAM2 Demo UI (React + TypeScript)                        │  │
│  │  ├─ Video Annotation Interface (Existing)                        │  │
│  │  └─ Training Pipeline Interface (NEW)                            │  │
│  │     ├─ Data Preparation Wizard                                   │  │
│  │     ├─ Training Configuration Form                               │  │
│  │     ├─ Training Monitor Dashboard                                │  │
│  │     └─ Experiment Comparison View                                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ HTTPS
                                │ GraphQL (Annotation)
                                │ REST API (Training)
┌───────────────────────────────┴─────────────────────────────────────────┐
│                         Application Layer                               │
│  ┌─────────────────────────┐    ┌─────────────────────────────────┐   │
│  │  SAM2 Backend           │    │  Training API Service (NEW)     │   │
│  │  (Flask + GraphQL)      │    │  (FastAPI)                      │   │
│  │  - Video Segmentation   │    │  ┌──────────────────────────┐  │   │
│  │  - Annotation Export    │───▶│  │ Data Preparation API     │  │   │
│  │  - Session Management   │    │  │ /api/data/convert        │  │   │
│  └─────────────────────────┘    │  │ /api/data/validate       │  │   │
│                                  │  │ /api/data/split          │  │   │
│                                  │  └──────────────────────────┘  │   │
│                                  │  ┌──────────────────────────┐  │   │
│                                  │  │ Training Orchestration   │  │   │
│                                  │  │ /api/train/start         │  │   │
│                                  │  │ /api/train/{id}/status   │  │   │
│                                  │  │ /api/train/{id}/cancel   │  │   │
│                                  │  └──────────────────────────┘  │   │
│                                  │  ┌──────────────────────────┐  │   │
│                                  │  │ Experiment Tracking      │  │   │
│                                  │  │ /api/experiments         │  │   │
│                                  │  │ /api/experiments/compare │  │   │
│                                  │  └──────────────────────────┘  │   │
│                                  │  ┌──────────────────────────┐  │   │
│                                  │  │ Model Export API         │  │   │
│                                  │  │ /api/export/{id}         │  │   │
│                                  │  │ /api/export/{id}/download│  │   │
│                                  │  └──────────────────────────┘  │   │
│                                  └─────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────────────┐
│                         Processing Layer                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              Job Queue (Celery + Redis)                          │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │  │
│  │  │ Data Prep    │  │ Training     │  │ Export       │         │  │
│  │  │ Queue        │  │ Queue        │  │ Queue        │         │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                  │                                      │
│  ┌──────────────────────────────┼─────────────────────────────────┐   │
│  │         Celery Workers        │                                 │   │
│  │  ┌────────────────────────────┴───────────────────────────┐    │   │
│  │  │  Worker 1: Data Preparation                            │    │   │
│  │  │  - SAM2 JSON Parser                                    │    │   │
│  │  │  - Format Converters (HF, LLaVA)                       │    │   │
│  │  │  - Quality Validators                                  │    │   │
│  │  │  - Dataset Splitter                                    │    │   │
│  │  └────────────────────────────────────────────────────────┘    │   │
│  │  ┌────────────────────────────────────────────────────────┐    │   │
│  │  │  Worker 2-N: Training (GPU Required)                   │    │   │
│  │  │  - Model Loader (HuggingFace)                          │    │   │
│  │  │  - LoRA Trainer (PEFT)                                 │    │   │
│  │  │  - Checkpoint Manager                                  │    │   │
│  │  │  - Metrics Logger (Tensorboard)                        │    │   │
│  │  └────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────────────┐
│                         Data & Storage Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │  PostgreSQL     │  │  Redis          │  │  File Storage   │        │
│  │  - Experiments  │  │  - Job Status   │  │  (S3/MinIO)     │        │
│  │  - Configs      │  │  - Metrics      │  │  - Datasets     │        │
│  │  - Users        │  │  - Cache        │  │  - Checkpoints  │        │
│  └─────────────────┘  └─────────────────┘  │  - Models       │        │
│                                             │  - Exports      │        │
│                                             └─────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

## File Structure

### Complete Project Structure

```
sam2/
├── demo/
│   ├── frontend/                          # Existing + Extended
│   │   ├── src/
│   │   │   ├── ...                       # Existing SAM2 demo files
│   │   │   └── training/                 # NEW: Training Pipeline UI
│   │   │       ├── components/
│   │   │       │   ├── DataPreparation/
│   │   │       │   │   ├── UploadZip.tsx
│   │   │       │   │   ├── FormatSelector.tsx
│   │   │       │   │   ├── ValidationReport.tsx
│   │   │       │   │   └── SplitConfig.tsx
│   │   │       │   ├── TrainingConfig/
│   │   │       │   │   ├── ModelSelector.tsx
│   │   │       │   │   ├── MethodSelector.tsx       # LoRA/QLoRA/Full
│   │   │       │   │   ├── HyperparameterForm.tsx
│   │   │       │   │   └── PresetSelector.tsx
│   │   │       │   ├── TrainingMonitor/
│   │   │       │   │   ├── JobStatus.tsx
│   │   │       │   │   ├── LossCurve.tsx
│   │   │       │   │   ├── MetricsDisplay.tsx
│   │   │       │   │   └── ProgressBar.tsx
│   │   │       │   ├── ExperimentDashboard/
│   │   │       │   │   ├── ExperimentList.tsx
│   │   │       │   │   ├── ExperimentComparison.tsx
│   │   │       │   │   └── ExperimentDetail.tsx
│   │   │       │   └── ModelExport/
│   │   │       │       ├── ExportForm.tsx
│   │   │       │       ├── ModelCard.tsx
│   │   │       │       └── DownloadButton.tsx
│   │   │       ├── pages/
│   │   │       │   ├── TrainingWorkflow.tsx          # Main wizard
│   │   │       │   ├── DataPrepPage.tsx
│   │   │       │   ├── TrainingPage.tsx
│   │   │       │   ├── ExperimentsPage.tsx
│   │   │       │   └── ExportPage.tsx
│   │   │       ├── hooks/
│   │   │       │   ├── useDataPreparation.ts
│   │   │       │   ├── useTrainingJob.ts
│   │   │       │   ├── useExperiments.ts
│   │   │       │   └── useModelExport.ts
│   │   │       ├── api/
│   │   │       │   ├── trainingApi.ts                # API client
│   │   │       │   └── types.ts                      # TypeScript types
│   │   │       └── utils/
│   │   │           ├── formatters.ts
│   │   │           └── validators.ts
│   │   └── package.json
│   │
│   ├── backend/                           # Existing SAM2 backend (unchanged)
│   │   └── server/
│   │       ├── ...                        # Existing files
│   │       └── data/export_service.py     # Used by training pipeline
│   │
│   └── training/                          # NEW: Training Pipeline Backend
│       ├── README.md
│       ├── requirements.txt               # Python dependencies
│       ├── docker-compose.training.yaml   # Training services
│       │
│       ├── api/                           # FastAPI Application
│       │   ├── __init__.py
│       │   ├── main.py                    # FastAPI app entry point
│       │   ├── dependencies.py            # Dependency injection
│       │   ├── middleware.py              # Auth, CORS, logging
│       │   │
│       │   ├── routes/
│       │   │   ├── __init__.py
│       │   │   ├── data_preparation.py    # POST /api/data/*
│       │   │   ├── training.py            # POST /api/train/*
│       │   │   ├── experiments.py         # GET /api/experiments/*
│       │   │   └── export.py              # POST /api/export/*
│       │   │
│       │   ├── schemas/                   # Pydantic models
│       │   │   ├── __init__.py
│       │   │   ├── data_prep.py
│       │   │   ├── training.py
│       │   │   ├── experiment.py
│       │   │   └── export.py
│       │   │
│       │   └── models/                    # Database models (SQLAlchemy)
│       │       ├── __init__.py
│       │       ├── experiment.py
│       │       ├── training_job.py
│       │       └── dataset.py
│       │
│       ├── core/                          # Core business logic
│       │   ├── __init__.py
│       │   │
│       │   ├── data_converter/           # Data Preparation Module
│       │   │   ├── __init__.py
│       │   │   ├── base_converter.py     # Abstract base class
│       │   │   ├── sam2_parser.py        # Parse SAM2 JSON exports
│       │   │   ├── huggingface_converter.py
│       │   │   ├── llava_converter.py
│       │   │   └── format_detector.py
│       │   │
│       │   ├── validation/               # Quality Validation Module
│       │   │   ├── __init__.py
│       │   │   ├── validator.py          # Validation engine
│       │   │   ├── basic_checks.py       # Missing data, duplicates
│       │   │   ├── balance_analysis.py   # Class distribution
│       │   │   ├── quality_metrics.py    # Mask quality, etc.
│       │   │   └── report_generator.py   # Generate Markdown reports
│       │   │
│       │   ├── data_splitter/            # Dataset Splitting Module
│       │   │   ├── __init__.py
│       │   │   ├── stratified_splitter.py
│       │   │   ├── temporal_splitter.py  # For video data
│       │   │   ├── random_splitter.py
│       │   │   └── split_config.py       # Configuration management
│       │   │
│       │   ├── trainers/                 # Training Module
│       │   │   ├── __init__.py
│       │   │   ├── base_trainer.py       # Abstract trainer
│       │   │   ├── lora_trainer.py       # LoRA fine-tuning
│       │   │   ├── vl_collator.py        # Vision-language data collator
│       │   │   ├── callbacks.py          # Training callbacks
│       │   │   └── optimizer_config.py   # Optimizer setup
│       │   │
│       │   ├── config/                   # Configuration Module
│       │   │   ├── __init__.py
│       │   │   ├── training_config.py    # Pydantic config schema
│       │   │   ├── model_registry.py     # Supported models
│       │   │   ├── presets.py            # Preset configurations
│       │   │   └── config_validator.py   # Validate configs
│       │   │
│       │   ├── tracking/                 # Experiment Tracking Module
│       │   │   ├── __init__.py
│       │   │   ├── tensorboard_logger.py
│       │   │   ├── custom_metrics.py     # IoU, mAP, etc.
│       │   │   └── visualizer.py         # Plot generation
│       │   │
│       │   ├── checkpoints/              # Checkpoint Management Module
│       │   │   ├── __init__.py
│       │   │   ├── checkpoint_saver.py
│       │   │   ├── best_model_tracker.py
│       │   │   └── checkpoint_pruner.py  # Keep only N best
│       │   │
│       │   ├── export/                   # Model Export Module
│       │   │   ├── __init__.py
│       │   │   ├── hf_exporter.py        # HuggingFace format
│       │   │   ├── lora_exporter.py      # Adapter-only export
│       │   │   └── model_card_generator.py
│       │   │
│       │   └── deployment/               # Deployment Package Module
│       │       ├── __init__.py
│       │       ├── docker_builder.py     # Generate Dockerfile
│       │       ├── api_generator.py      # Create inference API
│       │       └── config_generator.py   # Deployment configs
│       │
│       ├── jobs/                         # Celery Tasks
│       │   ├── __init__.py
│       │   ├── celery_app.py             # Celery configuration
│       │   ├── data_prep_task.py         # Data preparation task
│       │   ├── training_task.py          # Training task
│       │   ├── export_task.py            # Export task
│       │   ├── status_tracker.py         # Job status management
│       │   └── resource_manager.py       # GPU allocation
│       │
│       ├── storage/                      # Storage Abstraction
│       │   ├── __init__.py
│       │   ├── base_storage.py           # Abstract storage interface
│       │   ├── local_storage.py          # Local filesystem
│       │   └── s3_storage.py             # S3/MinIO (optional)
│       │
│       ├── database/                     # Database Layer
│       │   ├── __init__.py
│       │   ├── connection.py             # DB connection management
│       │   ├── migrations/               # Alembic migrations
│       │   │   ├── versions/
│       │   │   └── env.py
│       │   └── repositories/             # Data access layer
│       │       ├── __init__.py
│       │       ├── experiment_repo.py
│       │       ├── job_repo.py
│       │       └── dataset_repo.py
│       │
│       ├── experiments/                  # Experiment Management
│       │   ├── __init__.py
│       │   ├── experiment_db.py          # Experiment database
│       │   └── comparison.py             # Compare experiments
│       │
│       ├── utils/                        # Utilities
│       │   ├── __init__.py
│       │   ├── logger.py                 # Logging setup
│       │   ├── gpu_utils.py              # GPU detection & monitoring
│       │   ├── file_utils.py             # File operations
│       │   └── metrics.py                # Metric calculations
│       │
│       ├── tests/                        # Test Suite
│       │   ├── __init__.py
│       │   ├── conftest.py               # Pytest fixtures
│       │   │
│       │   ├── unit/
│       │   │   ├── test_converters.py
│       │   │   ├── test_validators.py
│       │   │   ├── test_splitters.py
│       │   │   └── test_trainers.py
│       │   │
│       │   ├── integration/
│       │   │   ├── test_api_endpoints.py
│       │   │   ├── test_training_pipeline.py
│       │   │   └── test_export_pipeline.py
│       │   │
│       │   └── e2e/
│       │       └── test_full_workflow.py  # End-to-end test
│       │
│       ├── scripts/                      # Utility scripts
│       │   ├── setup_db.py               # Initialize database
│       │   ├── start_workers.sh          # Start Celery workers
│       │   └── monitor_gpu.sh            # GPU monitoring script
│       │
│       └── docs/                         # Documentation
│           ├── API.md                    # API documentation
│           ├── USER_GUIDE.md             # User guide
│           ├── DEVELOPER_GUIDE.md        # Developer guide
│           └── ARCHITECTURE.md           # This document
│
└── docker/                               # Docker configurations
    ├── training-api.Dockerfile           # Training API service
    ├── training-worker.Dockerfile        # Celery worker
    └── docker-compose.training.yaml      # Orchestration
```

## Module Interactions

### Data Flow Diagram

```
┌──────────────┐
│   User       │
│   Uploads    │
│   SAM2      │
│   Export    │
└──────┬───────┘
       │
       v
┌──────────────────────────────────────────────────────────────┐
│  1. DATA PREPARATION PHASE                                   │
│                                                               │
│  ┌────────────┐      ┌──────────────┐      ┌──────────────┐│
│  │ SAM2       │─────▶│ Format       │─────▶│ Quality      ││
│  │ Parser     │      │ Converter    │      │ Validator    ││
│  └────────────┘      └──────────────┘      └──────┬───────┘│
│                                                     │        │
│                                                     v        │
│                                            ┌──────────────┐ │
│                                            │ Dataset      │ │
│                                            │ Splitter     │ │
│                                            └──────┬───────┘ │
└───────────────────────────────────────────────────┼─────────┘
                                                    │
                                                    v
                                          [Prepared Dataset]
                                                    │
                                                    v
┌──────────────────────────────────────────────────┼──────────┐
│  2. TRAINING PHASE                                │          │
│                                                   v          │
│  ┌────────────┐      ┌──────────────┐      ┌──────────────┐│
│  │ Job Queue  │─────▶│ GPU Worker   │─────▶│ LoRA         ││
│  │ (Celery)   │      │ (Celery)     │      │ Trainer      ││
│  └────────────┘      └──────────────┘      └──────┬───────┘│
│                                                     │        │
│                                                     v        │
│  ┌──────────────┐    ┌──────────────┐      ┌──────────────┐│
│  │ Tensorboard  │◀───│ Metrics      │◀─────│ Training     ││
│  │              │    │ Logger       │      │ Loop         ││
│  └──────────────┘    └──────────────┘      └──────┬───────┘│
│                                                     │        │
│                                                     v        │
│  ┌──────────────┐    ┌──────────────┐      ┌──────────────┐│
│  │ Best Model   │◀───│ Checkpoint   │◀─────│ Checkpoint   ││
│  │ Tracker      │    │ Saver        │      │ Callback     ││
│  └──────────────┘    └──────────────┘      └──────┬───────┘│
└───────────────────────────────────────────────────┼─────────┘
                                                     │
                                                     v
                                            [Trained Model]
                                                     │
                                                     v
┌──────────────────────────────────────────────────┼──────────┐
│  3. EXPORT PHASE                                  │          │
│                                                   v          │
│  ┌────────────┐      ┌──────────────┐      ┌──────────────┐│
│  │ HuggingFace│─────▶│ Model Card   │─────▶│ Deployment   ││
│  │ Exporter   │      │ Generator    │      │ Package      ││
│  └────────────┘      └──────────────┘      └──────┬───────┘│
│                                                     │        │
└─────────────────────────────────────────────────────────────┘
                                                      │
                                                      v
                                              [Model Package]
                                                      │
                                                      v
                                                 ┌─────────┐
                                                 │  User   │
                                                 │Downloads│
                                                 └─────────┘
```

## API Design

### RESTful API Endpoints

#### 1. Data Preparation API

```typescript
// POST /api/data/convert
interface ConvertRequest {
  sam2_export_path: string;      // Path to uploaded ZIP
  target_format: "huggingface" | "llava" | "custom";
  options?: {
    generate_instructions?: boolean;
    instruction_templates?: string[];
  };
}

interface ConvertResponse {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  output_path?: string;
  statistics?: {
    total_samples: number;
    total_classes: number;
    class_distribution: Record<string, number>;
  };
}

// POST /api/data/validate
interface ValidateRequest {
  dataset_path: string;
  validation_level: "basic" | "strict";
}

interface ValidateResponse {
  status: "passed" | "passed_with_warnings" | "failed";
  report: {
    summary: {
      total_samples: number;
      issues_found: number;
      warnings: number;
      errors: number;
    };
    checks: Array<{
      name: string;
      status: "passed" | "warning" | "error";
      message: string;
      details?: any;
    }>;
    recommendations: string[];
  };
}

// POST /api/data/split
interface SplitRequest {
  dataset_path: string;
  strategy: "stratified" | "temporal" | "random";
  ratios: {
    train: number;  // 0.0-1.0
    val: number;
    test: number;
  };
  seed?: number;
}

interface SplitResponse {
  output_paths: {
    train: string;
    val: string;
    test: string;
  };
  statistics: {
    train_samples: number;
    val_samples: number;
    test_samples: number;
    class_distribution: Record<string, {
      train: number;
      val: number;
      test: number;
    }>;
  };
}
```

#### 2. Training API

```typescript
// POST /api/train/start
interface StartTrainingRequest {
  dataset_path: string;
  config: {
    model: {
      name: string;              // e.g., "liuhaotian/llava-v1.5-7b"
      type: "llava" | "qwen-vl" | "instructblip";
    };
    training: {
      method: "lora" | "qlora" | "full";
      lora_rank?: number;
      lora_alpha?: number;
      learning_rate: number;
      batch_size: number;
      gradient_accumulation_steps: number;
      num_epochs: number;
      warmup_ratio: number;
    };
    hardware: {
      device: "cuda" | "cpu";
      mixed_precision: "no" | "fp16" | "bf16";
      gradient_checkpointing: boolean;
    };
    checkpointing: {
      save_steps: number;
      save_total_limit: number;
    };
  };
  experiment_name?: string;
}

interface StartTrainingResponse {
  job_id: string;
  status: "queued" | "running";
  estimated_time_seconds?: number;
  estimated_cost?: number;
}

// GET /api/train/{job_id}/status
interface TrainingStatus {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: {
    current_epoch: number;
    total_epochs: number;
    current_step: number;
    total_steps: number;
    percentage: number;
  };
  metrics: {
    train_loss: number;
    learning_rate: number;
    val_loss?: number;
    val_accuracy?: number;
  };
  eta_seconds?: number;
  error_message?: string;
}

// POST /api/train/{job_id}/cancel
interface CancelTrainingResponse {
  job_id: string;
  status: "cancelled";
  message: string;
}
```

#### 3. Experiment API

```typescript
// GET /api/experiments
interface ListExperimentsResponse {
  experiments: Array<{
    id: string;
    name: string;
    model: string;
    status: string;
    created_at: string;
    completed_at?: string;
    final_metrics?: {
      train_loss: number;
      val_loss: number;
      val_accuracy: number;
    };
  }>;
  total: number;
  page: number;
  page_size: number;
}

// GET /api/experiments/{id}
interface ExperimentDetail {
  id: string;
  name: string;
  config: object;  // Full training config
  status: string;
  metrics_history: Array<{
    step: number;
    epoch: number;
    train_loss: number;
    val_loss?: number;
    learning_rate: number;
  }>;
  checkpoints: Array<{
    step: number;
    path: string;
    metrics: object;
    is_best: boolean;
  }>;
  created_at: string;
  completed_at?: string;
  duration_seconds?: number;
}

// GET /api/experiments/compare?ids=id1,id2,id3
interface CompareExperimentsResponse {
  experiments: Array<ExperimentDetail>;
  comparison: {
    config_diffs: Array<{
      parameter: string;
      values: Record<string, any>;  // experiment_id -> value
    }>;
    final_metrics: Record<string, {
      train_loss: number;
      val_loss: number;
    }>;
    best_experiment_id: string;
    recommendation: string;
  };
}
```

#### 4. Export API

```typescript
// POST /api/export/{job_id}
interface ExportModelRequest {
  format: "huggingface" | "onnx" | "quantized";
  checkpoint?: string;  // Default: best checkpoint
  options?: {
    quantization?: "int8" | "int4";
    include_inference_code?: boolean;
    include_docker?: boolean;
  };
}

interface ExportModelResponse {
  export_id: string;
  status: "processing" | "completed" | "failed";
  output_path?: string;
  download_url?: string;
  model_card_url?: string;
}

// GET /api/export/{export_id}/download
// Returns: File stream (ZIP archive)
```

## Database Schema

### PostgreSQL Tables

```sql
-- Experiments table
CREATE TABLE experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    user_id UUID NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,  -- queued, running, completed, failed
    config JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds INTEGER,
    error_message TEXT,
    INDEX idx_user_id (user_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);

-- Training jobs table
CREATE TABLE training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    celery_task_id VARCHAR(255) UNIQUE,
    status VARCHAR(50) NOT NULL,
    current_epoch INTEGER,
    current_step INTEGER,
    total_steps INTEGER,
    gpu_device INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_celery_task_id (celery_task_id),
    INDEX idx_status (status)
);

-- Metrics table
CREATE TABLE metrics (
    id BIGSERIAL PRIMARY KEY,
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    train_loss FLOAT,
    val_loss FLOAT,
    learning_rate FLOAT,
    custom_metrics JSONB,
    timestamp TIMESTAMP DEFAULT NOW(),
    INDEX idx_experiment_id (experiment_id),
    INDEX idx_step (step)
);

-- Checkpoints table
CREATE TABLE checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size_mb FLOAT,
    metrics JSONB,
    is_best BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_experiment_id (experiment_id),
    INDEX idx_is_best (is_best)
);

-- Datasets table
CREATE TABLE datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    source_type VARCHAR(50) NOT NULL,  -- sam2_export, custom, etc.
    format VARCHAR(50) NOT NULL,       -- huggingface, llava, etc.
    file_path VARCHAR(500) NOT NULL,
    total_samples INTEGER,
    class_distribution JSONB,
    validation_report JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_source_type (source_type),
    INDEX idx_format (format)
);

-- Exports table
CREATE TABLE model_exports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    checkpoint_id UUID REFERENCES checkpoints(id),
    format VARCHAR(50) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size_mb FLOAT,
    download_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_experiment_id (experiment_id)
);
```

## Configuration Files

### docker-compose.training.yaml

```yaml
version: '3.8'

services:
  # Training API service
  training-api:
    build:
      context: .
      dockerfile: docker/training-api.Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/training
      - REDIS_URL=redis://redis:6379/0
      - STORAGE_TYPE=local  # or s3
      - STORAGE_PATH=/data/training
    volumes:
      - ./demo/training:/app
      - training-data:/data/training
    depends_on:
      - postgres
      - redis
    networks:
      - training-network

  # Celery worker for data preparation (CPU)
  worker-dataprep:
    build:
      context: .
      dockerfile: docker/training-worker.Dockerfile
    command: celery -A jobs.celery_app worker -Q dataprep -c 2 --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/training
      - REDIS_URL=redis://redis:6379/0
      - STORAGE_PATH=/data/training
    volumes:
      - ./demo/training:/app
      - training-data:/data/training
    depends_on:
      - redis
      - postgres
    networks:
      - training-network

  # Celery worker for training (GPU)
  worker-training:
    build:
      context: .
      dockerfile: docker/training-worker.Dockerfile
    command: celery -A jobs.celery_app worker -Q training -c 1 --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/training
      - REDIS_URL=redis://redis:6379/0
      - STORAGE_PATH=/data/training
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./demo/training:/app
      - training-data:/data/training
      - model-cache:/root/.cache/huggingface  # Cache for models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis
      - postgres
    networks:
      - training-network

  # PostgreSQL database
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=training
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - training-network

  # Redis for Celery and caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - training-network

  # Tensorboard for metrics visualization
  tensorboard:
    image: tensorflow/tensorflow:latest
    command: tensorboard --logdir=/logs --host=0.0.0.0
    ports:
      - "6006:6006"
    volumes:
      - training-data:/logs
    networks:
      - training-network

volumes:
  postgres-data:
  training-data:
  model-cache:

networks:
  training-network:
    driver: bridge
```

### requirements.txt

```txt
# Core dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Training
torch==2.1.0
transformers==4.36.0
peft==0.7.1
accelerate==0.25.0
bitsandbytes==0.41.3

# Data processing
datasets==2.15.0
pillow==10.1.0
opencv-python==4.8.1
numpy==1.24.3
pycocotools==2.0.7

# Job queue
celery==5.3.4
redis==5.0.1

# Database
sqlalchemy==2.0.23
alembic==1.13.0
psycopg2-binary==2.9.9

# Experiment tracking
tensorboard==2.15.1

# Storage (optional)
boto3==1.34.0
minio==7.2.0

# Utilities
python-dotenv==1.0.0
python-multipart==0.0.6
httpx==0.25.2
aiofiles==23.2.1
tqdm==4.66.1

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.12.0
ruff==0.1.8
mypy==1.7.1
```

## Deployment Strategy

### Development Environment

```bash
# 1. Start SAM2 demo (existing)
cd sam2
docker compose up -d

# 2. Start training services
docker compose -f docker-compose.training.yaml up -d

# 3. Access services
# - SAM2 Demo: http://localhost:7262
# - Training API: http://localhost:8000
# - Tensorboard: http://localhost:6006
# - API Docs: http://localhost:8000/docs
```

### Production Environment

```bash
# Use separate GPU for training
CUDA_VISIBLE_DEVICES=1 docker compose -f docker-compose.training.yaml up -d

# Or use Kubernetes for better resource management
kubectl apply -f k8s/training-deployment.yaml
```

## Performance Considerations

### Resource Allocation

| Component | CPU | RAM | GPU | Storage |
|-----------|-----|-----|-----|---------|
| Training API | 2 cores | 4GB | - | - |
| Data Prep Worker | 4 cores | 8GB | - | 100GB |
| Training Worker | 8 cores | 32GB | 1x24GB | 500GB |
| PostgreSQL | 2 cores | 4GB | - | 50GB |
| Redis | 1 core | 2GB | - | 10GB |

### Scaling Strategy

1. **Horizontal scaling**: Add more data prep workers for concurrent job processing
2. **Vertical scaling**: Use larger GPUs (A100 80GB) for bigger models
3. **Queue prioritization**: High-priority jobs get GPU resources first
4. **Auto-scaling**: Scale workers based on queue length

## Security Considerations

1. **API Authentication**: JWT tokens from SAM2 demo auth service
2. **Input Validation**: Strict validation of all user inputs
3. **Sandboxing**: Training jobs run in isolated containers
4. **Resource Limits**: CPU/memory/GPU limits per job
5. **Data Isolation**: User data separated by user_id
6. **Audit Logging**: All operations logged for compliance

## Monitoring & Observability

### Metrics to Track

1. **System Metrics**:
   - API response times
   - Queue length
   - Worker utilization
   - GPU utilization

2. **Business Metrics**:
   - Training jobs started/completed
   - Average training time
   - Success rate
   - Storage usage

3. **Alerts**:
   - GPU out of memory
   - Training job failures
   - Queue backing up
   - Storage > 80% full

### Logging Stack

```
Application Logs → Fluentd → Elasticsearch → Kibana
                                    ↓
                              Alertmanager → Slack/Email
```

This architecture provides a scalable, maintainable, and production-ready LLM fine-tuning pipeline integrated with your SAM2 demo!
