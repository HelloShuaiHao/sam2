# Design: Add Frame Export and Annotation Features

## Why

The SAM2 demo currently showcases impressive interactive video segmentation capabilities but lacks critical production features that would make it useful for real-world annotation workflows. This limitation prevents users from:

1. **Creating training datasets**: Without export functionality, annotated videos exist only in the demo UI and cannot be used to train ML models
2. **Managing dataset size**: Exporting all frames of high-FPS videos creates massive datasets that are impractical to store and process
3. **Integrating with ML pipelines**: No standardized output format means annotations cannot feed into existing data processing tools
4. **Deploying to production**: Lack of authentication and quota management prevents commercial deployment and monetization

**Core insight**: SAM2's video object tracking is transformative for annotation efficiency - users can click once on an object and have it tracked across an entire video. However, this power is wasted if the results cannot be exported at a practical frame rate (e.g., 5 FPS instead of 30 FPS) in a usable format.

**Business driver**: The iDoctor team has an existing authentication and quota infrastructure (from the commercial backend). By integrating the SAM2 demo with these systems and adding export capabilities, we can rapidly productionize the demo without building a new application from scratch (as attempted with the failed iSeg project).

## What

We will extend the existing SAM2 demo (demo/frontend + demo/backend) with three core capabilities:

### 1. Video Annotation Export (Phase 1 Priority)
- **Frame rate selector UI**: Let users choose export density (0.5 to 30 FPS)
- **JSON export format**: Structured output with RLE-encoded masks, bounding boxes, metadata
- **Time-based frame sampling**: Sample frames at regular time intervals, not just every Nth frame
- **Progress feedback**: Real-time updates during export processing
- **Download packaging**: ZIP archive with JSON and metadata files

### 2. Frame Rate Control (Phase 1 Priority)
- **Preset options**: Common FPS values (0.5, 1, 2, 5, 10, 15, 30)
- **Smart defaults**: Auto-select based on source video FPS
- **Size estimation**: Show estimated export size and frame count before processing
- **Validation**: Prevent invalid selections (export FPS > source FPS)
- **Educational tooltips**: Explain trade-offs between FPS choices

### 3. Authentication & Quota Integration (Phase 2 Final Step)
- **JWT validation middleware**: Verify tokens from existing iDoctor auth service
- **Quota checking**: Validate sufficient quota before video processing
- **Usage tracking**: Measure video duration, export operations, frame counts
- **Quota deduction**: Update quota service after successful operations
- **UI visibility**: Display quota balance and consumption estimates

## How

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     SAM2 Demo Frontend                           │
│  (React + TypeScript + Vite - demo/frontend)                     │
│                                                                   │
│  ┌───────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │ Video Editor      │  │ Export Config    │  │ Quota        │ │
│  │ - Object tracking │  │ - FPS selector   │  │ - Balance    │ │
│  │ - Frame timeline  │  │ - Size estimator │  │ - Cost preview│ │
│  │ - Mask overlay    │  │ - Progress UI    │  │ - Warnings   │ │
│  └───────────────────┘  └──────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │ GraphQL
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              SAM2 Demo Backend (Flask + Strawberry)              │
│  (Python GraphQL - demo/backend)                                 │
│                                                                   │
│  ┌─────────────────┐  ┌────────────────┐  ┌──────────────────┐ │
│  │ Auth Middleware │  │ Quota Check    │  │ Export Service   │ │
│  │ - JWT validate  │  │ - Pre-check    │  │ - Frame sampler  │ │
│  │ - Extract user  │  │ - Deduct usage │  │ - RLE encoder    │ │
│  │                 │  │ - Track metrics│  │ - JSON generator │ │
│  └─────────────────┘  └────────────────┘  └──────────────────┘ │
│           │                    │                                 │
│           ▼                    ▼                                 │
│  ┌───────────────────┐  ┌────────────────┐                      │
│  │ SAM2VideoPredictor│  │ Export Jobs    │                      │
│  │ - Object tracking │  │ - Job queue    │                      │
│  │ - Mask generation │  │ - Status track │                      │
│  └───────────────────┘  └────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────────┐   ┌────────────────────────┐
│ iDoctor Auth Service │   │ iDoctor Quota Service  │
│ - JWT validation     │   │ - Balance tracking     │
│ - User management    │   │ - Usage metrics        │
└──────────────────────┘   └────────────────────────┘
```

### Key Design Decisions

**1. Why extend the existing demo instead of building a new app?**
- **Decision**: Extend demo/frontend and demo/backend
- **Rationale**:
  - The demo already has a polished video editor UI with SAM2VideoPredictor integration
  - Previous attempt to build standalone app (iSeg) failed
  - Faster time-to-market by building on existing infrastructure
  - Less maintenance burden (one codebase instead of two)
- **Trade-off**: Less architectural flexibility, but significantly lower development cost

**2. Why time-based frame sampling instead of index-based?**
- **Decision**: Sample frames at regular time intervals (e.g., every 0.2s for 5 FPS)
- **Rationale**:
  - Consistent results across variable frame rate (VFR) videos
  - More intuitive for users ("5 FPS" is clearer than "every 6th frame")
  - Handles dropped frames gracefully
- **Trade-off**: Slightly more complex logic, but much better user experience

**3. Why RLE encoding for masks?**
- **Decision**: Use Run-Length Encoding (COCO format) for binary masks
- **Rationale**:
  - 10-20× size reduction vs. raw binary masks
  - Standard format compatible with ML tools (pycocotools)
  - Lossless encoding/decoding
- **Trade-off**: Requires encoding/decoding step, but file size savings are essential

**4. Why JSON-only export in Phase 1?**
- **Decision**: Support only JSON annotation format initially
- **Rationale**:
  - User explicitly requested JSON only (based on user questions)
  - Simplifies scope and accelerates delivery
  - JSON is sufficient for most ML pipelines
  - Other formats (PNG sequences, COCO, video overlays) can be added later
- **Trade-off**: Less flexibility, but faster implementation

**5. Why auth/quota in Phase 2, not Phase 1?**
- **Decision**: Implement core export functionality first, add auth/quota later
- **Rationale**:
  - User explicitly requested putting auth/quota last
  - Core export features provide immediate value for testing
  - Auth/quota requires coordination with external services
  - Phased approach allows user validation before production integration
- **Trade-off**: Phase 1 deployment won't be production-ready, but enables rapid iteration

**6. Why background job queue for exports?**
- **Decision**: Process exports asynchronously with job tracking
- **Rationale**:
  - Export processing can take 30+ seconds for long videos
  - Keeps GraphQL requests lightweight and responsive
  - Allows progress updates during processing
  - Prevents timeout issues
- **Trade-off**: More complex infrastructure (job queue, status polling), but necessary for UX

### Data Flow: Export Workflow

```
1. User configures export (5 FPS) → Frontend
2. Frontend calls exportVideoAnnotations mutation → Backend GraphQL
3. Backend creates export job → Job queue
4. Background worker picks up job:
   a. Calculate target timestamps: [0.0s, 0.2s, 0.4s, ...]
   b. Find closest frames to each timestamp
   c. For each frame:
      - Get mask from SAM2VideoPredictor
      - Encode mask to RLE
      - Extract bounding box and metadata
   d. Serialize all annotations to JSON
   e. Package JSON + metadata into ZIP
   f. Store ZIP at download URL
5. Frontend polls job status → Backend
6. Job completes → Frontend receives download URL
7. User clicks download → Browser downloads ZIP
```

### JSON Export Schema

```typescript
interface VideoExport {
  video: {
    filename: string;
    width: number;
    height: number;
    source_fps: number;
    total_frames: number;
    duration_sec: number;
  };
  export_config: {
    target_fps: number;
    total_exported_frames: number;
    frame_indices: number[];
    export_timestamp: string;
  };
  annotations: Array<{
    frame_index: number;
    timestamp_sec: number;
    objects: Array<{
      object_id: number;
      label: string;
      mask_rle: string;  // COCO RLE format
      bbox: [number, number, number, number];  // [x, y, w, h]
      area: number;
      confidence: number;
    }>;
  }>;
}
```

### Performance Considerations

**Export processing time budget**:
- Frame sampling: < 1s (in-memory calculation)
- Mask generation: ~0.05s per frame (GPU-bound, SAM2VideoPredictor cached)
- RLE encoding: ~0.01s per mask (CPU-bound)
- JSON serialization: < 1s (even for large exports)
- ZIP compression: < 2s

**For 30s video at 5 FPS (150 frames, 2 objects)**:
- Mask generation: 150 frames × 0.05s = 7.5s
- RLE encoding: 300 masks × 0.01s = 3s
- Other overhead: ~3s
- **Total: ~13.5s** (well under 30s target)

**Bottleneck**: Mask generation is GPU-bound. Optimization strategies:
- Cache masks already generated during annotation (reduces redundant work)
- Batch frame processing (amortize GPU overhead)
- Use smaller SAM2 model (base_plus vs large) for faster inference

### Scalability: Auth & Quota Integration

**Phase 2 quota metrics**:
- **Video processing**: Deduct based on video duration (1 minute of video = 1 quota minute)
- **Export operations**: Flat cost per export (e.g., 5 quota minutes per export regardless of FPS)
  - Rationale: Prevents abuse, keeps quota simple
  - Alternative: Cost per frame exported (more fair but complex)

**Quota enforcement points**:
1. **Before video upload**: Check if user has quota for video duration
2. **Before export**: Check if user has quota for export operation
3. **After success**: Deduct quota from user balance
4. **On failure**: Refund quota if system error (not user error like corrupt video)

**Auth integration**:
- **JWT validation**: Middleware extracts user ID from token claims
- **User context**: All operations tagged with user ID for auditing
- **Session expiry**: Frontend handles token refresh or re-login flow

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Large export files exceed user expectations | Users frustrated by long downloads | Show size estimate before export, warn if > 50MB, recommend lower FPS |
| Export processing times out for very long videos | Poor UX, failed exports | Background job queue, progress feedback, stream processing instead of loading all frames |
| RLE encoding doesn't provide expected compression | Large JSON files despite encoding | Fallback to binary mask + gzip, or warn user and recommend reducing FPS |
| Auth service integration breaks demo during Phase 2 | Development blocked | Design middleware interface to allow mock auth during dev, graceful degradation |
| Quota service is slow, adds latency to requests | Sluggish UX | Cache quota balance in backend, async quota updates, timeout handling |

## Alternatives Considered

### Alternative A: Build standalone application (like iSeg proposal)
- **Pros**: Clean architecture, full control, independent deployment
- **Cons**: Significant dev effort, duplicate UI work, previous attempt failed
- **Decision**: Rejected - user explicitly requested extending demo instead

### Alternative B: Support multiple export formats (PNG, COCO, video overlay)
- **Pros**: Flexibility for different use cases
- **Cons**: Longer dev time, larger file sizes, more complex UI
- **Decision**: Deferred - start with JSON, add formats based on user feedback

### Alternative C: Implement auth/quota in Phase 1
- **Pros**: Production-ready from start
- **Cons**: Blocks core feature development, requires external service coordination
- **Decision**: Rejected - user explicitly requested auth/quota as final step

### Alternative D: Index-based frame sampling (every Nth frame)
- **Pros**: Simpler logic
- **Cons**: Inconsistent results for VFR videos, less intuitive
- **Decision**: Rejected - time-based sampling provides better UX

## Success Metrics

**Phase 1 (Core Export)**:
- [ ] 100% of test videos successfully export at configured FPS
- [ ] RLE compression achieves ≥10× size reduction vs. raw binary
- [ ] Export processing completes in < 30s for 30s video at 5 FPS
- [ ] JSON validates against schema, compatible with pycocotools

**Phase 2 (Auth & Quota)**:
- [ ] JWT validation blocks 100% of invalid tokens
- [ ] Quota enforcement prevents processing when balance = 0
- [ ] Quota deduction accurate within 1% of actual usage
- [ ] Auth middleware adds < 10ms latency per request

**User Experience**:
- [ ] Users can export a fully annotated video in < 3 clicks
- [ ] Export size estimates are accurate within 20%
- [ ] Progress feedback updates at least every 2 seconds
- [ ] Error messages are actionable and user-friendly
