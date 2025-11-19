# Dataset Quality Validation Report

**Dataset:** Example Dataset
**Generated:** 2025-11-19 16:25:26
**Status:** PASSED_WITH_WARNINGS

## Summary

- **Total Checks:** 6
- **Passed:** ✅ 5
- **Warnings:** ⚠️  1
- **Errors:** ❌ 0

## ✅ Passed Checks

- **missing_annotations_check:** All 40 frames have annotations
- **duplicate_frames_check:** No duplicate frames found (40 unique frames)
- **mask_integrity_check:** All 120 masks have valid RLE encoding
- **bounding_box_check:** All 120 bounding boxes are valid
- **class_balance_analysis:** Classes are reasonably balanced across 3 classes

## ⚠️  Warnings

### quality_metrics
Quality issues found: 1 warnings

**Details:**
- total_frames: 40
- total_objects: 120
- avg_objects_per_frame: 3
- min_objects_per_frame: 3
- max_objects_per_frame: 3
- avg_object_area: 2543.383333333333
- median_object_area: 2821.5
- small_objects_count: 42

## 💡 Recommendations

1. Dataset is ready for training

## 📊 Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| S1 | 40 | 33.3% |
| s2 | 40 | 33.3% |
| a1 | 40 | 33.3% |
