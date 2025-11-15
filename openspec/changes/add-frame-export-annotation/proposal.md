# Proposal: Add Frame Export and Annotation Features to SAM2 Demo

## Metadata
- **Change ID**: `add-frame-export-annotation`
- **Status**: Proposed
- **Author**: iDoctor Team
- **Created**: 2025-11-15
- **Related Changes**: None (initial enhancement to existing demo)

## Problem Statement

The current SAM2 demo (demo/frontend and demo/backend) provides interactive video segmentation and object tracking capabilities, but lacks essential production features for practical annotation workflows:

**Key problems:**
1. **No frame export capability**: Users can annotate and track objects in videos but cannot export the results for downstream use (training datasets, validation, etc.)
2. **Missing frame rate control**: No ability to control export density - exporting all frames creates massive datasets, but there's no way to sample at configurable frame rates (e.g., 1 FPS, 5 FPS)
3. **No structured annotation output**: Results are only viewable in the UI, not exportable in standard formats like JSON for integration with ML pipelines
4. **Lack of authentication and quota management**: No integration with existing auth/quota systems for production deployment and monetization
5. **Limited production readiness**: The demo is great for showcasing SAM2 capabilities but not ready for real-world annotation workflows

## Business Value

Adding frame export and annotation features with auth/quota integration will:
1. **Enable production workflows**: Transform the demo into a usable annotation tool for creating training datasets
2. **Reduce annotation effort**: Leverage SAM2's video tracking to create frame-sampled datasets efficiently (e.g., annotate at 1 FPS instead of 30 FPS)
3. **Standardized output**: JSON export enables integration with existing ML pipelines and dataset management tools
4. **Revenue generation**: Integration with existing quota system enables monetization of compute-intensive video processing
5. **Minimal development cost**: Build on existing demo infrastructure rather than creating new project from scratch

## Proposed Solution

### Overview
Extend the existing SAM2 demo (demo/frontend + demo/backend) with:
- **Frame export controls**: UI for configuring export frame rate (frames per second)
- **JSON annotation export**: Export tracking results in structured JSON format with masks, bounding boxes, and metadata
- **Backend export API**: GraphQL mutations for processing and packaging export data
- **Authentication middleware** (final step): Integrate with existing iDoctor auth service for JWT validation
- **Quota enforcement** (final step): Track and limit usage based on existing quota system

### Architecture Integration

**Current Demo Stack:**
- **Frontend**: React + TypeScript + Vite (demo/frontend)
- **Backend**: Python Flask + Strawberry GraphQL (demo/backend)
- **Deployment**: Docker Compose

**New Components:**
- Export controls in existing video editor UI
- GraphQL export mutations in Flask backend
- Frame sampling and JSON serialization logic
- Auth middleware integration (final phase)
- Quota tracking middleware (final phase)

### Key Features (Phased Approach)

**Phase 1: Core Export Functionality** (Priority)
1. **Frame Rate Selector**: UI control to set export frame rate (0.5, 1, 2, 5, 10, 15, 30 FPS)
2. **JSON Export**:
   - Per-frame annotations with timestamps
   - Object tracking IDs and labels
   - Mask RLE (Run-Length Encoding) for efficient storage
   - Bounding boxes and confidence scores
   - Video metadata (resolution, total frames, source FPS)
3. **Export Processing**:
   - Sample frames based on configured FPS
   - Generate masks for selected frames only
   - Package JSON + metadata into downloadable archive

**Phase 2: Auth & Quota Integration** (Final)
4. **Authentication**: JWT validation via existing auth service
5. **Quota Management**: Track video processing and export operations against user quotas

### User Workflow

1. **Upload video** → SAM2 processes and displays in editor
2. **Annotate objects** → Click to add object masks, SAM2 tracks across frames
3. **Configure export**:
   - Select frame rate (e.g., 5 FPS for a 30 FPS video = export every 6th frame)
   - Choose output format (JSON annotations)
4. **Export** → Backend samples frames at configured rate, generates JSON with masks
5. **Download** → Receive ZIP file with:
   - `annotations.json` (all tracking data)
   - `metadata.json` (video info, export settings)
   - Optional: sampled frame images (if requested)

### Export JSON Format

```json
{
  "video": {
    "filename": "example.mp4",
    "width": 1920,
    "height": 1080,
    "source_fps": 30,
    "total_frames": 900,
    "duration_sec": 30
  },
  "export_config": {
    "target_fps": 5,
    "total_exported_frames": 150,
    "frame_indices": [0, 6, 12, 18, ...]
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
          "bbox": [x, y, width, height],
          "area": 123456,
          "confidence": 0.95
        }
      ]
    },
    ...
  ]
}
```

### Port and Service Integration

**Development:**
- Frontend: http://localhost:7262 (existing demo port)
- Backend: http://localhost:7263 (existing demo port)

**Auth/Quota Integration (Phase 2):**
- Auth Service: Existing iDoctor auth service endpoint (to be configured)
- Quota Service: Existing iDoctor quota service endpoint (to be configured)

## Scope

### In Scope (Phase 1: Core Export)
1. ✅ Frame rate selector UI component in demo/frontend
2. ✅ GraphQL mutation for export request (`exportVideoAnnotations`)
3. ✅ Frame sampling logic based on target FPS
4. ✅ JSON serialization with RLE mask encoding
5. ✅ ZIP archive generation for download
6. ✅ Export progress feedback in UI
7. ✅ Error handling for export failures

### In Scope (Phase 2: Auth & Quota)
8. ✅ JWT authentication middleware integration
9. ✅ Quota tracking for video processing and exports
10. ✅ User account linkage for usage limits

### Out of Scope (Future Enhancements)
1. ❌ Multiple export formats (PNG sequences, COCO format, video overlays) - only JSON in v1
2. ❌ Advanced frame selection (custom frame index ranges)
3. ❌ Batch export of multiple videos
4. ❌ Cloud storage integration (S3, GCS)
5. ❌ Real-time collaboration features
6. ❌ Dataset versioning and management
7. ❌ Custom annotation schemas

## Dependencies

### Existing Infrastructure
- **SAM2 Demo**: demo/frontend (React) and demo/backend (Flask + GraphQL)
- **SAM2VideoPredictor**: Already implemented for object tracking
- **Docker Setup**: docker-compose.yaml for containerized deployment

### New Dependencies
- **Python Libraries**:
  - `pycocotools` or custom RLE encoder for mask compression
  - No additional frontend dependencies needed

### External Services (Phase 2 Only)
- **Auth Service**: Existing iDoctor commercial/auth_service
- **Quota Service**: Existing iDoctor commercial/quota_service

## Risks and Mitigation

### Technical Risks
1. **Large Export Size**
   - *Risk*: JSON files with RLE masks can still be large for high-FPS exports
   - *Mitigation*: Default to conservative FPS (1-5), show estimated export size before processing, implement compression

2. **Export Processing Time**
   - *Risk*: Generating masks for many frames can be slow
   - *Mitigation*: Show progress indicator, implement background task queue for large exports, cache already-processed frames

3. **Memory Usage**
   - *Risk*: Loading many frames for export can exhaust memory
   - *Mitigation*: Stream frame processing instead of loading all at once, implement frame batching

4. **Auth Service Coupling** (Phase 2)
   - *Risk*: Dependency on external auth service creates tight coupling
   - *Mitigation*: Design middleware interface to allow mock auth during development, implement graceful degradation

### Operational Risks
1. **Export Rate Limiting**
   - *Risk*: Users may abuse export feature, exhausting server resources
   - *Mitigation*: Phase 2 quota enforcement, implement per-user rate limits

## Success Criteria

### Functional Requirements Met
- [ ] Users can select export frame rate (1, 5, 10 FPS options minimum)
- [ ] Export generates valid JSON with RLE masks for all tracked objects
- [ ] JSON includes video metadata and export configuration
- [ ] Download delivers ZIP file with JSON annotations
- [ ] Progress feedback shows export status
- [ ] Phase 2: Authentication validates JWT tokens correctly
- [ ] Phase 2: Quota system tracks and limits export operations

### Performance Targets
- [ ] Export processing completes within 30 seconds for 30-second video at 5 FPS
- [ ] JSON file size < 50MB for typical video (1 minute, 2-3 objects, 5 FPS)
- [ ] UI remains responsive during export processing
- [ ] Export download starts within 5 seconds of completion

### Quality Gates
- [ ] No TypeScript errors in frontend export components
- [ ] No Python errors in backend export mutations
- [ ] Export JSON validates against documented schema
- [ ] Error messages are user-friendly for all failure cases
- [ ] Phase 2: Auth middleware passes security review
- [ ] Phase 2: Quota tracking accurately reflects usage

## Timeline Estimate

**Phase 1: Core Export Functionality (Priority)**

*Week 1-2: Frontend Export UI*
- Frame rate selector component
- Export configuration modal
- Progress indicator and download handler
- Error state handling

*Week 2-3: Backend Export API*
- GraphQL mutation implementation
- Frame sampling logic
- JSON serialization with RLE encoding
- ZIP archive generation
- Testing with various videos

**Phase 2: Auth & Quota Integration (Final)**

*Week 4: Authentication*
- JWT middleware integration
- Auth service connection setup
- Token validation testing

*Week 5: Quota System*
- Quota tracking middleware
- Usage metrics collection
- Limit enforcement logic
- End-to-end testing

**Total estimated effort**: 5 weeks (1 developer, can be parallelized)

## Open Questions

1. **RLE Encoding Library**
   - *Question*: Should we use pycocotools RLE or implement custom encoder?
   - *Recommendation*: Use pycocotools if already in dependencies, otherwise lightweight custom implementation

2. **Frame Sampling Strategy**
   - *Question*: Should frame sampling be exact (every Nth frame) or time-based (closest frame to target timestamps)?
   - *Recommendation*: Time-based for consistency across variable FPS videos

3. **Export Size Limits**
   - *Question*: Should we enforce maximum export size or frame count?
   - *Recommendation*: Start with warning at 10MB, hard limit at 100MB until quota system is in place

4. **Mask Format**
   - *Question*: RLE vs binary masks vs polygon vertices?
   - *Recommendation*: RLE for space efficiency, add polygon conversion as future enhancement

5. **Auth Service Endpoint**
   - *Question*: What is the exact endpoint and JWT validation method for the existing auth service?
   - *Decision*: To be determined during Phase 2 implementation based on existing service docs

6. **Quota Metrics**
   - *Question*: Should quotas be based on video duration, frame count, or export operations?
   - *Recommendation*: Track multiple metrics: video processing time, export operations, total frames processed

## Alternatives Considered

### Alternative 1: Create New Standalone Application (like iSeg)
- **Pros**: Clean architecture, independent scaling, full control over stack
- **Cons**: Significant development effort, duplicate UI work, maintenance burden
- **Decision**: Rejected - user explicitly requested extending existing demo

### Alternative 2: Support Multiple Export Formats (PNG, COCO, Video)
- **Pros**: Maximum flexibility for different use cases
- **Cons**: Increased complexity, longer development time, larger export files
- **Decision**: Deferred to future - start with JSON only, add formats based on user feedback

### Alternative 3: Real-time Export (Stream Results)
- **Pros**: No waiting for export to complete, better UX for large videos
- **Cons**: Complex implementation, requires WebSocket or SSE
- **Decision**: Deferred - use simple download approach first, optimize later if needed

### Alternative 4: Auth Integration in Phase 1
- **Pros**: Production-ready from start
- **Cons**: Blocks core feature development, requires external service coordination
- **Decision**: Rejected - user explicitly requested auth/quota as final step

## References

- Previous iSeg Proposal: `/Users/mbp/Desktop/Work/Life/IDoctor/iSeg(自动分割)/openspec/changes/build-iseg-app/proposal.md`
- SAM2 Demo: `demo/frontend` and `demo/backend`
- SAM2VideoPredictor: `sam2/sam2_video_predictor.py`
- Demo README: `demo/README.md`
- Existing Auth Service: iDoctor commercial/auth_service (details TBD in Phase 2)
- Existing Quota Service: iDoctor commercial/quota_service (details TBD in Phase 2)
