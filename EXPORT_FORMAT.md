# SAM2 Export Format Documentation

This document describes the export formats for video annotations in the SAM2 system.

## Format Versions

The SAM2 system supports two export formats:

- **Version 1.0**: Global object export (全局对象导出)
- **Version 2.0**: Action segment export (动作片段导出)

---

## Version 1.0: Global Object Export

### Overview

Exports all annotated objects across the entire video with frame sampling based on target FPS.

### ZIP Structure

```
export_<timestamp>.zip
├── annotations.json          # Main annotation data
├── metadata.json             # Export metadata
└── images/                   # Sampled frames (optional)
    ├── frame_000000.jpg
    ├── frame_000030.jpg
    └── ...
```

### metadata.json

```json
{
  "format_version": "1.0",
  "export_type": "global_objects",
  "session_id": "abc123...",
  "created_at": "2025-11-28T10:30:00Z",
  "video_info": {
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "total_frames": 900
  },
  "export_config": {
    "target_fps": 5,
    "total_exported_frames": 150,
    "include_visualizations": true
  }
}
```

### annotations.json

```json
{
  "objects": [
    {
      "object_id": 1,
      "label": "Person",
      "color": "#ff5733",
      "frames": [
        {
          "frame_index": 0,
          "timestamp": 0.0,
          "mask": {
            "size": [1080, 1920],
            "counts": "eNp...",
            "order": "F"
          },
          "bbox": [100, 150, 200, 300]
        },
        {
          "frame_index": 30,
          "timestamp": 1.0,
          "mask": {
            "size": [1080, 1920],
            "counts": "eNq...",
            "order": "F"
          },
          "bbox": [105, 155, 205, 305]
        }
      ]
    },
    {
      "object_id": 2,
      "label": "Ball",
      "color": "#33ff57",
      "frames": [...]
    }
  ]
}
```

---

## Version 2.0: Action Segment Export

### Overview

Exports annotations organized by action segments (time-bounded regions). Each segment contains only the objects and frames within that segment's time range.

### ZIP Structure

```
action_segments_export_<timestamp>.zip
├── segments_index.json       # Index of all segments
├── metadata.json             # Global export metadata
├── segment_<id1>/           # First action segment
│   ├── segment_metadata.json
│   ├── annotations.json
│   └── images/              # Frames only within segment range
│       ├── frame_000100.jpg
│       ├── frame_000130.jpg
│       └── ...
├── segment_<id2>/           # Second action segment
│   ├── segment_metadata.json
│   ├── annotations.json
│   └── images/
│       └── ...
└── ...
```

### metadata.json (Root Level)

```json
{
  "format_version": "2.0",
  "export_type": "action_segments",
  "session_id": "abc123...",
  "created_at": "2025-11-28T10:30:00Z",
  "video_info": {
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "total_frames": 900
  },
  "export_config": {
    "target_fps": 5,
    "include_visualizations": true
  },
  "total_segments": 3
}
```

### segments_index.json

```json
{
  "segments": [
    {
      "segment_id": "seg_1701234567890",
      "name": "Walking",
      "frame_start": 100,
      "frame_end": 250,
      "duration_seconds": 5.0,
      "object_count": 2,
      "folder": "segment_seg_1701234567890"
    },
    {
      "segment_id": "seg_1701234567891",
      "name": "Running",
      "frame_start": 300,
      "frame_end": 500,
      "duration_seconds": 6.67,
      "object_count": 1,
      "folder": "segment_seg_1701234567891"
    },
    {
      "segment_id": "seg_1701234567892",
      "name": "Jumping",
      "frame_start": 550,
      "frame_end": 700,
      "duration_seconds": 5.0,
      "object_count": 3,
      "folder": "segment_seg_1701234567892"
    }
  ]
}
```

### segment_<id>/segment_metadata.json

```json
{
  "segment_id": "seg_1701234567890",
  "name": "Walking",
  "frame_start": 100,
  "frame_end": 250,
  "created_at": 1701234567890,
  "duration_frames": 150,
  "duration_seconds": 5.0,
  "objects": [
    {
      "object_id": 1,
      "label": "Person",
      "color": "#ff5733"
    },
    {
      "object_id": 2,
      "label": "Dog",
      "color": "#33ff57"
    }
  ],
  "export_info": {
    "exported_frames": 30,
    "target_fps": 5,
    "actual_fps": 30
  }
}
```

### segment_<id>/annotations.json

```json
{
  "segment_id": "seg_1701234567890",
  "segment_name": "Walking",
  "frame_range": {
    "start": 100,
    "end": 250
  },
  "objects": [
    {
      "object_id": 1,
      "label": "Person",
      "color": "#ff5733",
      "frames": [
        {
          "frame_index": 100,
          "timestamp": 3.33,
          "relative_frame": 0,
          "mask": {
            "size": [1080, 1920],
            "counts": "eNp...",
            "order": "F"
          },
          "bbox": [100, 150, 200, 300]
        },
        {
          "frame_index": 130,
          "timestamp": 4.33,
          "relative_frame": 30,
          "mask": {
            "size": [1080, 1920],
            "counts": "eNq...",
            "order": "F"
          },
          "bbox": [105, 155, 205, 305]
        }
      ]
    },
    {
      "object_id": 2,
      "label": "Dog",
      "color": "#33ff57",
      "frames": [...]
    }
  ]
}
```

---

## Key Differences

| Feature | Version 1.0 | Version 2.0 |
|---------|-------------|-------------|
| **Scope** | Entire video | Time-bounded segments |
| **Structure** | Flat (all objects in one file) | Hierarchical (per-segment folders) |
| **Frame Range** | All frames (sampled) | Only frames within segment |
| **Use Case** | Global object tracking | Action-specific annotation |
| **Index File** | Not applicable | `segments_index.json` |
| **Metadata** | Single `metadata.json` | Root + per-segment metadata |

---

## Mask Encoding

Both formats use **RLE (Run-Length Encoding)** for masks via `pycocotools`:

```json
{
  "size": [height, width],
  "counts": "eNp...",  // Base64-encoded RLE data
  "order": "F"         // Fortran order (column-major)
}
```

### Decoding Example (Python)

```python
from pycocotools import mask as mask_utils
import numpy as np

# Decode RLE mask
rle_mask = {
    "size": [1080, 1920],
    "counts": "eNp...",
    "order": "F"
}
binary_mask = mask_utils.decode(rle_mask)  # Returns numpy array
```

---

## Bounding Box Format

Bounding boxes use `[x, y, width, height]` format:

```json
{
  "bbox": [100, 150, 200, 300]
}
```

Where:
- `x, y`: Top-left corner coordinates
- `width, height`: Box dimensions

---

## Timestamp Calculation

```python
timestamp = frame_index / video_fps
```

For a 30 FPS video:
- Frame 0 → 0.0 seconds
- Frame 30 → 1.0 seconds
- Frame 100 → 3.33 seconds

---

## Frame Sampling

Frame sampling is based on `target_fps`:

```python
sample_interval = video_fps / target_fps
```

**Example**: Video at 30 FPS, target 5 FPS
- Sample interval: 30 / 5 = 6 frames
- Sampled frames: 0, 6, 12, 18, 24, 30, ...

For action segments, sampling is limited to `[frame_start, frame_end]`.

---

## Export API Usage

### Global Object Export (v1.0)

```graphql
mutation ExportGlobalObjects {
  exportVideoAnnotations(input: {
    sessionId: "abc123..."
    targetFps: 5
    objectNames: {
      "1": "Custom Name for Object 1",
      "2": "Custom Name for Object 2"
    }
  }) {
    jobId
    status
    message
    estimatedFrames
  }
}
```

### Action Segment Export (v2.0)

```graphql
mutation ExportActionSegments {
  exportActionSegments(input: {
    sessionId: "abc123..."
    targetFps: 5
    includeVisualizations: true
    actionSegments: [
      {
        id: "seg_1701234567890"
        name: "Walking"
        frame_start: 100
        frame_end: 250
        created_at: 1701234567890
        objects: [
          {
            object_id: 1
            label: "Person"
            color: "#ff5733"
          }
        ]
      }
    ]
  }) {
    jobId
    status
    message
    segmentCount
  }
}
```

---

## Polling Export Status

Both export types use the same job status polling:

```graphql
query CheckExportStatus {
  exportJobStatus(jobId: "job_abc123") {
    jobId
    sessionId
    status
    progress
    totalFrames
    processedFrames
    downloadUrl
    fileSize
    errorMessage
  }
}
```

**Status Values**:
- `pending`: Job queued
- `processing`: Export in progress
- `completed`: Ready for download
- `failed`: Error occurred

---

## Best Practices

### For Global Object Export (v1.0)

1. Use lower `target_fps` (1-5) for large videos to reduce file size
2. Enable visualizations only when needed for review
3. Provide custom `objectNames` for clarity in exported data
4. Monitor `progress` to estimate completion time

### For Action Segment Export (v2.0)

1. Export segments individually for focused analysis
2. Use batch export for training data preparation
3. Set `includeVisualizations: false` to save disk space
4. Review `segments_index.json` to understand export structure
5. Use `relative_frame` in segment annotations for easier processing

---

## Backward Compatibility

- **Version 1.0** format remains unchanged
- Existing tools parsing v1.0 exports continue to work
- **Version 2.0** is a new format, not a replacement
- Both formats can coexist in the same project

---

## Error Handling

Common export errors:

| Error | Cause | Solution |
|-------|-------|----------|
| `session_not_found` | Invalid session ID | Verify session is active |
| `no_annotations` | No objects tracked | Annotate objects first |
| `invalid_fps` | target_fps <= 0 | Use positive FPS value |
| `disk_space_full` | Insufficient storage | Free up disk space |
| `segment_empty` | Segment has no objects | Add objects to segment |

---

## File Size Estimation

### Version 1.0

```
Total Size ≈ (exported_frames × avg_mask_size) + visualization_size
```

**Example**: 150 frames, 50KB avg mask, with visualizations
- Annotation data: ~7.5 MB
- Images (1920×1080 JPEG): ~45 MB
- **Total**: ~52.5 MB

### Version 2.0

```
Total Size ≈ Σ(segment_frames × avg_mask_size × object_count) + visualizations
```

**Example**: 3 segments, avg 30 frames each, 2 objects, with visualizations
- Annotation data per segment: ~3 MB
- Images per segment: ~9 MB
- **Total per segment**: ~12 MB
- **Total (3 segments)**: ~36 MB

---

## Support

For issues or questions:

1. Check `TODO.md` for known issues
2. Review `IMPLEMENTATION_SUMMARY.md` for architecture details
3. Verify export job status via GraphQL query
4. Check backend logs for error details

---

## Changelog

### Version 2.0 (2025-11-28)
- Added action segment export format
- Introduced hierarchical folder structure
- Added `segments_index.json` for batch exports
- Added `relative_frame` field in annotations
- Per-segment metadata files

### Version 1.0 (Initial Release)
- Global object export format
- Single `annotations.json` file
- Optional visualization images
- Custom object naming support
