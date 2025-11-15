# Architecture - Video Export Feature

## Overview

This document describes the architecture and design decisions for the SAM2 video annotation export feature.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                             │
├─────────────────────────────────────────────────────────────────┤
│  React Frontend (port 7262)                                      │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │ ExportButton │→ │ ExportModal     │→ │ ExportProgress   │   │
│  │              │  │ (Config FPS)    │  │ (Polling Status) │   │
│  └──────────────┘  └─────────────────┘  └──────────────────┘   │
│         │                   │                      ↑             │
│         └───────────────────┴──────────────────────┘             │
│                             │                                    │
│                      useExport Hook                              │
│                    (State Management)                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │ GraphQL / HTTP
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│  Flask Backend (port 7263)                                       │
├─────────────────────────────────────────────────────────────────┤
│  GraphQL API (Strawberry)                                        │
│  ┌──────────────────────┐  ┌─────────────────────────────────┐ │
│  │ Mutation             │  │ Query                           │ │
│  │ exportVideoAnnotations│  │ exportJobStatus                │ │
│  └──────────┬───────────┘  └──────────┬──────────────────────┘ │
│             │                          │                         │
│             ↓                          ↓                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            ExportService (Job Manager)                   │   │
│  │  - Create job → Background thread → Update status       │   │
│  │  - In-memory job storage (dict)                         │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         Export Processing Pipeline                       │   │
│  │  ┌──────────────┐  ┌────────────────┐  ┌─────────────┐ │   │
│  │  │ FrameSampler │→ │ SAM2Predictor  │→ │ RLEEncoder  │ │   │
│  │  │ (Time-based) │  │ (Get masks)    │  │ (Compress)  │ │   │
│  │  └──────────────┘  └────────────────┘  └─────────────┘ │   │
│  │         │                   │                  │         │   │
│  │         └───────────────────┴──────────────────┘         │   │
│  │                           ↓                               │   │
│  │              ┌───────────────────────────┐               │   │
│  │              │ AnnotationSerializer      │               │   │
│  │              │ (JSON with metadata)      │               │   │
│  │              └────────────┬──────────────┘               │   │
│  │                           ↓                               │   │
│  │              ┌───────────────────────────┐               │   │
│  │              │ ZIP Archive Generator     │               │   │
│  │              └────────────┬──────────────┘               │   │
│  └───────────────────────────┼──────────────────────────────┘   │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ File Storage: /data/exports/{job_id}.zip                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Download Endpoint: /api/download/export/{job_id}        │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP Download
                               ↓
                        ┌──────────────┐
                        │  User's Disk │
                        └──────────────┘
```

## Component Design

### Frontend Components

#### 1. ExportButton (Integration Component)

**Location**: `demo/frontend/src/common/components/export/ExportButton.tsx`

**Responsibilities**:
- Entry point for export feature
- Manages modal and progress visibility
- Disabled when no tracked objects exist
- Passes video metadata to child components

**Props**:
```typescript
{
  sessionId: string | null;
  videoMetadata: {
    duration: number;    // seconds
    fps: number;         // source FPS
    totalFrames: number;
    width: number;       // pixels
    height: number;
  };
  hasTrackedObjects: boolean;
}
```

**State Flow**:
```
Idle → Click → Modal Open → Configure → Export Started →
  Polling → Progress Updates → Completed → Download → Idle
```

#### 2. ExportConfigModal (Configuration UI)

**Location**: `demo/frontend/src/common/components/export/ExportConfigModal.tsx`

**Responsibilities**:
- Display video metadata (duration, resolution, source FPS)
- Frame rate selector integration
- Show export estimates (frame count, file size)
- Trigger export mutation

**Key Features**:
- Warning for large exports (>1000 frames)
- FPS validation (0.5 - 30 range)
- Responsive grid layout for metadata

#### 3. FrameRateSelector (FPS Configuration)

**Location**: `demo/frontend/src/common/components/export/FrameRateSelector.tsx`

**Responsibilities**:
- Radio button group for FPS options (0.5, 1, 2, 5, 10, 15, 30)
- Calculate and display estimated frame count
- Recommend appropriate FPS based on video duration
- Visual indicators for high frame counts

**Calculation**:
```typescript
estimatedFrames = Math.floor(videoDuration * selectedFps);
```

#### 4. ExportProgress (Status Indicator)

**Location**: `demo/frontend/src/common/components/export/ExportProgress.tsx`

**Responsibilities**:
- Fixed position progress panel
- Real-time progress bar (0-100%)
- Frame count display (processed / total)
- Status messages (processing, completed, failed)
- Auto-hide after 3 seconds on success
- Download button on completion
- Retry button on failure

**State Transitions**:
```
PENDING → PROCESSING → COMPLETED (auto-hide)
                    ↘ FAILED (show retry)
```

#### 5. useExport Hook (State Management)

**Location**: `demo/frontend/src/common/components/export/useExport.ts`

**Responsibilities**:
- GraphQL mutation and query execution
- Job status polling (1 second interval)
- Download URL handling
- Error state management
- Reset export state

**State Interface**:
```typescript
{
  isExporting: boolean;
  status: 'idle' | 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;           // 0.0 - 1.0
  processedFrames: number;
  totalFrames: number;
  downloadUrl: string | null;
  errorMessage: string | null;
  fileSizeMb: number | null;
}
```

**Polling Logic**:
```typescript
const startPolling = (jobId: string) => {
  const interval = setInterval(async () => {
    const status = await queryJobStatus(jobId);
    if (status.status === 'completed' || status.status === 'failed') {
      clearInterval(interval);
    }
  }, 1000);  // Poll every 1 second
};
```

### Backend Components

#### 1. GraphQL Schema (API Layer)

**Location**: `demo/backend/server/data/schema.py`, `data_types.py`

**Mutations**:
- `exportVideoAnnotations`: Create export job

**Queries**:
- `exportJobStatus`: Get job status for polling

**Types**:
- `ExportVideoAnnotationsInput`
- `ExportResult`
- `ExportJobInfo`
- `ExportJobStatus` (enum)

**Design Decision**: GraphQL for type safety and single endpoint simplicity.

#### 2. ExportService (Job Manager)

**Location**: `demo/backend/server/data/export_service.py`

**Responsibilities**:
- Job lifecycle management (create, track, cleanup)
- Background thread spawning for async processing
- In-memory job storage (dict: job_id → ExportJobInfo)
- Progress updates during processing

**Key Methods**:

```python
class ExportService:
    def create_export_job(
        self,
        session_id: str,
        target_fps: float,
        inference_api: InferenceAPI
    ) -> ExportResult:
        # 1. Validate session and objects
        # 2. Create job ID
        # 3. Initialize job status
        # 4. Start background thread
        # 5. Return job info

    def _process_export_job(self, job_id: str, ...):
        # Background thread function
        # 1. Sample frames
        # 2. Get masks for each frame
        # 3. Serialize to JSON
        # 4. Create ZIP
        # 5. Update job status

    def get_job_status(self, job_id: str) -> Optional[ExportJobInfo]:
        # Return current job info (for polling)

    def get_export_file_path(self, job_id: str) -> Path:
        # Return path to ZIP file (for download)
```

**Thread Safety**: Uses Python's GIL for dict operations (safe for read/write).

#### 3. FrameSampler (Frame Selection)

**Location**: `demo/backend/server/utils/frame_sampler.py`

**Responsibilities**:
- Calculate frame indices for export based on time
- Handle variable frame rate (VFR) videos
- Ensure consistent sampling regardless of source FPS

**Algorithm** (Time-based Sampling):

```python
class FrameSampler:
    def __init__(self, video_duration_sec: float, source_fps: float):
        self.duration = video_duration_sec
        self.source_fps = source_fps

    def calculate_frame_indices(self, target_fps: float) -> List[int]:
        time_interval = 1.0 / target_fps  # e.g., 0.2s for 5 FPS
        timestamps = []
        current_time = 0.0

        while current_time < self.duration:
            timestamps.append(current_time)
            current_time += time_interval

        # Convert timestamps to frame indices
        frame_indices = [
            int(round(ts * self.source_fps))
            for ts in timestamps
        ]

        # Clamp to valid range
        max_frame = int(self.duration * self.source_fps)
        return [min(idx, max_frame - 1) for idx in frame_indices]
```

**Design Decision**: Time-based (not index-based) to handle VFR videos correctly.

**Example**:
- 30s video @ 30 FPS → 900 frames
- Export @ 5 FPS → 150 frames
- Timestamps: 0.0, 0.2, 0.4, ..., 29.8
- Indices: 0, 6, 12, ..., 894

#### 4. RLEEncoder (Compression)

**Location**: `demo/backend/server/utils/rle_encoder.py`

**Responsibilities**:
- Encode binary masks to Run-Length Encoding (RLE)
- COCO-compatible format
- Calculate bounding boxes from RLE
- Lossless roundtrip encoding/decoding

**Algorithm** (COCO RLE):

```python
class RLEEncoder:
    @staticmethod
    def encode(mask: np.ndarray) -> Dict[str, Any]:
        # Input: H x W binary mask (0 or 1)
        # Output: {"counts": "RLE_string", "size": [H, W]}

        # Flatten mask in Fortran order (column-major)
        flat_mask = mask.T.flatten()

        # Run-length encode
        runs = []
        prev_val = 0
        count = 0

        for val in flat_mask:
            if val == prev_val:
                count += 1
            else:
                runs.append(count)
                count = 1
                prev_val = val

        runs.append(count)

        # Encode to base64 (COCO format)
        rle_string = _encode_runs_to_base64(runs)

        return {
            "counts": rle_string,
            "size": [mask.shape[0], mask.shape[1]]
        }
```

**Compression Ratio**: Typically 10-20× for segmentation masks.

**Design Decision**: RLE chosen for:
- COCO compatibility (widely supported)
- Excellent compression for sparse masks
- Fast encoding/decoding
- Lossless

#### 5. AnnotationSerializer (JSON Generation)

**Location**: `demo/backend/server/utils/annotation_serializer.py`

**Responsibilities**:
- Build JSON structure with metadata
- Serialize frame annotations
- Include RLE masks, bounding boxes, object IDs

**JSON Structure**:

```json
{
  "video": {
    "file_name": "video.mp4",
    "duration_sec": 30.0,
    "width": 1920,
    "height": 1080,
    "source_fps": 30.0,
    "total_frames": 900
  },
  "export_config": {
    "target_fps": 5.0,
    "frame_indices": [0, 6, 12, ...],
    "total_exported_frames": 150,
    "exported_at": "2025-11-15T10:30:00Z"
  },
  "annotations": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "objects": [
        {
          "object_id": 1,
          "label": "object_1",
          "mask": {
            "counts": "eNp1...",  // RLE encoded
            "size": [1080, 1920]
          },
          "bbox": [100, 200, 300, 400],  // [x, y, w, h]
          "area": 120000,
          "confidence": 0.95
        }
      ]
    },
    ...
  ]
}
```

## Data Flow

### Export Workflow

```
1. User clicks "Export" button
   ↓
2. ExportConfigModal opens → User selects FPS → Clicks "Export"
   ↓
3. useExport.startExport() → GraphQL mutation
   ↓
4. Backend: schema.exportVideoAnnotations()
   ↓
5. ExportService.create_export_job()
   - Validate session
   - Generate job_id
   - Spawn background thread
   - Return ExportResult (status: PENDING)
   ↓
6. Background thread: _process_export_job()
   - FrameSampler.calculate_frame_indices()
   - For each frame index:
     * Get mask from SAM2VideoPredictor
     * RLEEncoder.encode(mask)
     * Add to AnnotationSerializer
   - AnnotationSerializer.serialize() → JSON
   - Create ZIP with JSON + metadata
   - Save to /data/exports/{job_id}.zip
   - Update job status to COMPLETED
   ↓
7. Frontend: useExport polling (every 1s)
   - Query exportJobStatus(job_id)
   - Update progress UI
   - When COMPLETED: show download button
   ↓
8. User clicks "Download"
   ↓
9. Browser fetches /api/download/export/{job_id}
   ↓
10. Backend: app.download_export()
    - Validate job_id
    - Stream ZIP file
    ↓
11. User receives sam2_export_{job_id}.zip
```

## Design Decisions

### 1. Why Time-Based Frame Sampling?

**Problem**: Index-based sampling (every Nth frame) fails for VFR videos.

**Solution**: Calculate timestamps at regular intervals, then find nearest frames.

**Benefit**: Consistent export regardless of source FPS or dropped frames.

**Trade-off**: Slightly more complex calculation, but negligible performance impact.

### 2. Why RLE Encoding?

**Alternatives Considered**:
- PNG images: Larger file size, harder to parse
- Raw binary: Not human-readable, no tooling support
- Polygon contours: Lossy, complex shapes need many points

**RLE Advantages**:
- 10-20× compression for typical masks
- COCO-compatible (industry standard)
- Fast encode/decode
- Lossless
- Widely supported by CV tools

**Trade-off**: Slightly slower than raw binary, but compression wins.

### 3. Why GraphQL Polling Instead of WebSockets?

**Decision**: Use GraphQL query polling (1s interval) for status updates.

**Reasons**:
- Simpler implementation (no WebSocket infrastructure)
- Existing GraphQL endpoint (no additional server setup)
- Low overhead for 1s polling
- Easier to debug and test

**Future**: Can upgrade to GraphQL subscriptions or SSE if needed.

**Trade-off**: Slightly higher latency (up to 1s), but acceptable for export UX.

### 4. Why In-Memory Job Storage?

**Decision**: Store job status in Python dict (ExportService._jobs).

**Reasons**:
- Simple implementation for MVP
- Fast read/write (no DB overhead)
- Sufficient for single-server deployment
- Auto-cleanup on restart (no stale jobs)

**Limitations**:
- Lost on server restart (acceptable for temporary jobs)
- Not suitable for multi-server deployment

**Future**: Migrate to Redis or DB for production scaling.

### 5. Why Background Threading?

**Decision**: Use Python threading for async export processing.

**Reasons**:
- Non-blocking API response (instant job creation)
- Flask-compatible (no async/await needed)
- Simple thread management with threading.Thread

**Alternatives Considered**:
- Celery: Overkill for single async task, requires Redis/RabbitMQ
- asyncio: Flask doesn't natively support async routes well

**Trade-off**: Threading in Python has GIL overhead, but export is I/O-bound (not CPU-bound).

### 6. Why JSON-Only Export?

**Decision**: Phase 1 only supports JSON annotation format.

**Reasons**:
- Matches user requirement ("JSON 标注数据")
- Most flexible format (can convert to others later)
- Small file size with RLE compression
- Easy to parse and inspect

**Future** (Phase 2+):
- PNG image sequences
- COCO dataset format
- Video overlay export

## Scalability Considerations

### Current Limitations

1. **Single Server**: In-memory job storage doesn't scale to multiple servers
2. **Local File Storage**: /data/exports/ is local disk (not distributed)
3. **No Rate Limiting**: Users can create unlimited concurrent exports
4. **Manual Cleanup**: Old export files need manual removal

### Production Scaling Path

1. **Distributed Job Storage**:
   - Migrate to Redis for job status
   - Use celery for distributed task processing

2. **Cloud Storage**:
   - Upload ZIP files to S3/GCS
   - Generate signed download URLs
   - Auto-expire after 24 hours

3. **Rate Limiting**:
   - Max 5 concurrent exports per user
   - Max 50 exports per day per user

4. **Auto-Cleanup**:
   - Cron job to delete exports older than 1 hour
   - Or use S3 lifecycle policies

5. **Horizontal Scaling**:
   - Load balancer across multiple Flask instances
   - Shared Redis for job coordination
   - Shared storage (S3) for exports

## Security Considerations

### Phase 1 (Current)

- ✅ Input validation (FPS range, session existence)
- ✅ Path traversal prevention (job_id sanitization)
- ❌ No authentication (anyone can export any session)
- ❌ No authorization (no user ownership of sessions)
- ❌ No quota limits (can exhaust disk space)

### Phase 2 (Planned)

- [ ] JWT authentication for all export endpoints
- [ ] User-session association (only export your own videos)
- [ ] Quota enforcement (track export usage)
- [ ] Rate limiting (prevent abuse)
- [ ] File size limits (prevent disk exhaustion)

## Performance Metrics

### Target Performance

| Metric | Target | Current (Estimated) |
|--------|--------|---------------------|
| Export creation latency | < 100ms | ~50ms |
| 30s video @ 5 FPS export time | < 30s | ~13s* |
| JSON file size (30s, 5 FPS) | < 50MB | ~2MB* |
| RLE compression ratio | > 10× | 10-20× |
| Status polling overhead | < 5% CPU | ~1% CPU |

*Based on theoretical calculations and code analysis

### Bottlenecks

1. **Mask Generation**: SAM2VideoPredictor inference (~100ms per frame)
2. **RLE Encoding**: Fast (~5ms per mask)
3. **JSON Serialization**: Fast (~10ms for 150 frames)
4. **ZIP Compression**: Fast (~50ms for 2MB JSON)

**Total for 30s @ 5 FPS (150 frames)**: ~100ms × 150 = 15s (dominated by inference)

## Testing Strategy

See `docs/export-feature/testing/testing-guide.md` for detailed testing procedures.

**Key Test Categories**:
1. Unit tests (frame sampling, RLE encoding)
2. Integration tests (GraphQL API end-to-end)
3. Performance tests (export time, file size)
4. Concurrency tests (multiple simultaneous exports)
5. Error handling tests (network failures, invalid inputs)

## Deployment

### Docker Compose

Services:
- **frontend**: nginx serving built React app (port 7262)
- **backend**: Gunicorn + Flask + GraphQL (port 7263)

Volumes:
- `./demo/data/:/data/:rw` (persistent export storage)

See `docs/export-feature/guides/implementation-guide.md` for deployment instructions.

## Future Enhancements

### Phase 2: Auth & Quota (Planned)

- JWT authentication with iDoctor auth_service
- Quota enforcement with quota_service
- User-session association
- Usage tracking and metrics

### Phase 3: Additional Formats (Future)

- PNG image sequences
- COCO dataset format
- Video overlay export (masks rendered on video)
- Custom frame range selection

### Phase 4: Advanced Features (Future)

- Batch export (multiple videos)
- Export templates (saved configurations)
- Cloud storage integration (S3/GCS)
- Webhook notifications on completion

## Related Documentation

- **Implementation Guide**: `docs/export-feature/guides/implementation-guide.md`
- **API Reference**: `docs/export-feature/api/graphql-api.md`
- **Testing Guide**: `docs/export-feature/testing/testing-guide.md`
- **OpenSpec Proposal**: `openspec/changes/add-frame-export-annotation/proposal.md`
