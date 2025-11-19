# Spec: Data Preparation for LLM Fine-tuning

## Overview
This specification defines the data preparation capabilities required to convert SAM2 annotation exports into training-ready formats for vision-language model fine-tuning.

## ADDED Requirements

### Requirement: Format Conversion

The system SHALL convert SAM2 annotation exports (JSON format) into multiple training-compatible formats.

#### Scenario: Convert to HuggingFace Dataset

**Given** a SAM2 export ZIP file containing:
- `annotations.json` with frame annotations and object masks
- `metadata.json` with video information
- Referenced frame images

**When** the user selects "HuggingFace Dataset" as the target format

**Then** the system shall:
1. Parse the JSON structure and extract annotations
2. Create a HuggingFace `Dataset` object with columns:
   - `image`: PIL Image or path
   - `conversations`: List of instruction-response pairs
   - `masks`: RLE-encoded segmentation masks
   - `bounding_boxes`: List of [x, y, width, height] coordinates
3. Save the dataset in Arrow format
4. Return success with dataset statistics (total samples, classes, etc.)

**And** the output dataset shall be loadable via `datasets.load_from_disk()`

#### Scenario: Convert to LLaVA Instruction Format

**Given** a SAM2 export ZIP file with multi-object annotations

**When** the user selects "LLaVA" as the target format

**Then** the system shall:
1. Generate instruction-response pairs for each annotated frame
2. Format as JSONL with structure:
   ```json
   {
     "id": "unique_frame_id",
     "image": "path/to/frame.jpg",
     "conversations": [
       {"from": "human", "value": "<image>\nInstruction text"},
       {"from": "gpt", "value": "Response with segmentation details"}
     ],
     "masks": ["rle_encoded_mask_1", ...],
     "bounding_boxes": [[x, y, w, h], ...]
   }
   ```
3. Write JSONL file to output directory
4. Create manifest file listing all samples

**And** instructions shall be varied (e.g., "Segment the person", "Identify all objects", "Find and segment cars")

### Requirement: Dataset Quality Validation

The system SHALL validate dataset quality and provide actionable feedback before training.

#### Scenario: Detect Missing Annotations

**Given** a dataset where 15 out of 1000 frames have no annotations

**When** quality validation is performed

**Then** the system shall:
1. Detect frames with zero annotations
2. Report error: "15 frames (1.5%) have no annotations"
3. List frame IDs with missing annotations
4. Mark validation status as "FAILED"

**And** prevent training from starting until issue is resolved

#### Scenario: Detect Class Imbalance

**Given** a dataset with class distribution:
- "person": 2800 instances
- "car": 900 instances
- "dog": 300 instances

**When** quality validation is performed

**Then** the system shall:
1. Calculate class ratios: person/dog = 9.3:1 (severe imbalance)
2. Report warning: "Severe class imbalance detected"
3. Recommend data augmentation for minority classes
4. Mark validation status as "PASSED_WITH_WARNINGS"

**And** allow training to proceed with warnings

#### Scenario: Validate Mask Integrity

**Given** a dataset with RLE-encoded masks

**When** quality validation is performed

**Then** the system shall:
1. Decode each RLE mask to verify format correctness
2. Check mask dimensions match image dimensions
3. Verify mask area is non-zero
4. Detect any corrupted or malformed masks

**And** report any masks that fail validation

### Requirement: Dataset Splitting

The system SHALL split datasets into train/validation/test sets while maintaining data integrity.

#### Scenario: Stratified Split for Balanced Training

**Given** a dataset with 1500 annotated frames and class distribution:
- 60% "person"
- 30% "car"
- 10% "dog"

**When** the user configures split ratios: train=0.8, val=0.1, test=0.1

**Then** the system shall:
1. Perform stratified splitting to maintain class ratios in each split
2. Ensure each split has approximately the same class distribution (±5%)
3. Create three separate datasets:
   - Train: 1200 frames (60% person, 30% car, 10% dog)
   - Val: 150 frames (similar ratios)
   - Test: 150 frames (similar ratios)

**And** save split configuration for reproducibility

#### Scenario: Temporal Split to Prevent Data Leakage

**Given** a video dataset with 900 frames from 3 videos (300 frames each)

**When** the user selects "temporal" split strategy

**Then** the system shall:
1. Split by video segments, not individual frames
2. Ensure no frames from the same video appear in different splits
3. Distribute videos across splits to achieve target ratios

**And** prevent data leakage from temporally adjacent frames

#### Scenario: Reproducible Random Split

**Given** a dataset and split configuration with seed=42

**When** dataset splitting is performed multiple times

**Then** the system shall:
1. Use the same random seed for reproducibility
2. Generate identical splits across multiple runs
3. Save seed in split metadata

**And** allow users to reproduce exact train/val/test splits

### Requirement: Data Preparation Pipeline Orchestration

The system SHALL orchestrate the complete data preparation workflow from SAM2 export to training-ready dataset.

#### Scenario: End-to-End Data Preparation

**Given** a SAM2 export ZIP file uploaded by the user

**When** the user initiates data preparation with configuration:
- Target format: "HuggingFace"
- Split ratios: 0.8/0.1/0.1
- Validation level: "strict"

**Then** the system shall:
1. Extract and parse the ZIP file
2. Perform quality validation and generate report
3. Convert annotations to HuggingFace format
4. Split dataset into train/val/test
5. Save all outputs to user's workspace
6. Return preparation summary with statistics

**And** the entire process shall complete within 5 minutes for 10,000 frames

#### Scenario: Incremental Data Addition

**Given** an existing prepared dataset with 1000 samples

**When** the user uploads a new SAM2 export with 500 additional samples

**Then** the system shall:
1. Detect existing dataset
2. Merge new samples with existing data
3. Re-validate combined dataset
4. Optionally re-split to maintain ratios
5. Update dataset metadata

**And** preserve existing data integrity during merge

## Cross-References

- Related to `training-orchestration`: Prepared datasets are consumed by training pipeline
- Related to `experiment-tracking`: Dataset metadata is logged with each experiment
- Depends on SAM2 export format from `add-frame-export-annotation` change

## Acceptance Criteria

1. All format converters (HuggingFace, LLaVA) successfully convert sample SAM2 exports
2. Quality validation detects known issues (missing annotations, imbalance, corrupted masks)
3. Dataset splitting maintains class distribution within 5% deviation
4. End-to-end pipeline completes in under 5 minutes for 10,000 frames on standard hardware
5. All validation scenarios pass with appropriate error handling
