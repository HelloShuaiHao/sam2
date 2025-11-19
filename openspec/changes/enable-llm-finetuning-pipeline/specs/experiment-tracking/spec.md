# Spec: Experiment Tracking

## Overview
Tracks training experiments, metrics, and checkpoints for reproducibility and comparison.

## ADDED Requirements

### Requirement: Metrics Logging

The system SHALL log training and validation metrics for visualization and analysis.

#### Scenario: Log Training Metrics to Tensorboard

**Given** a running training job

**When** training progresses through steps

**Then** the system shall:
1. Log scalar metrics to Tensorboard every 10 steps:
   - Training loss
   - Learning rate
   - Gradient norm
2. Log validation metrics every 100 steps:
   - Validation loss
   - IoU score
   - Accuracy

**And** metrics shall be viewable in real-time via Tensorboard web interface

### Requirement: Checkpoint Management

The system SHALL save and manage model checkpoints during training.

#### Scenario: Save Best Checkpoint Based on Validation Loss

**Given** a training job with validation every 100 steps

**When** validation loss improves from 0.45 to 0.42

**Then** the system shall:
1. Save current model state as best checkpoint
2. Include optimizer and scheduler states
3. Tag checkpoint with metadata (epoch, step, metrics)
4. Prune older non-best checkpoints (keep only last 3)

**And** allow users to resume training from best checkpoint

### Requirement: Experiment Comparison

The system SHALL enable comparison of multiple training experiments.

#### Scenario: Compare Two Experiments with Different Learning Rates

**Given** two completed experiments:
- Experiment A: LR=1e-5, final val loss=0.40
- Experiment B: LR=2e-5, final val loss=0.35

**When** the user views experiment comparison dashboard

**Then** the system shall display:
1. Side-by-side configuration differences (LR highlighted)
2. Overlaid loss curves for visual comparison
3. Final metric summary table
4. Recommendation: "Experiment B achieved 12.5% better validation loss"

**And** allow export of comparison report as PDF
