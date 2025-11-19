# Spec: Model Export and Deployment

## Overview
Exports trained models in standard formats and generates deployment artifacts.

## ADDED Requirements

### Requirement: HuggingFace Format Export

The system SHALL export trained models in HuggingFace-compatible format.

#### Scenario: Export LoRA-fine-tuned Model

**Given** a completed training job with LoRA adapters

**When** the user initiates model export

**Then** the system shall:
1. Save LoRA adapter weights
2. Save adapter configuration (rank, alpha, target modules)
3. Create model card with training details
4. Package all files in HuggingFace directory structure
5. Validate exported model loads with `from_pretrained()`

**And** provide download link for the model package

### Requirement: Model Card Generation

The system SHALL auto-generate model cards with training metadata.

#### Scenario: Generate Model Card for Fine-tuned Model

**Given** a trained model with tracked training metadata

**When** model export completes

**Then** the system shall generate a model card including:
1. Model architecture and base model info
2. Training dataset statistics (samples, classes, split ratios)
3. Training hyperparameters (LR, batch size, epochs)
4. Performance metrics (final loss, validation accuracy)
5. Usage examples with code snippets
6. Training duration and hardware used

**And** save model card as `README.md` in model directory

### Requirement: Deployment Package Creation

The system SHALL create deployment-ready packages with inference code.

#### Scenario: Generate Docker Deployment Package

**Given** an exported model

**When** the user selects "Docker deployment package"

**Then** the system shall:
1. Create Dockerfile with model and dependencies
2. Generate inference API using FastAPI
3. Include example request/response
4. Add docker-compose.yaml for easy deployment
5. Package all files in ZIP archive

**And** provide deployment instructions in README
