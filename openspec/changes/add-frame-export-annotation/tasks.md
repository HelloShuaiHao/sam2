# Implementation Tasks

This document outlines the ordered tasks for implementing the frame export and annotation features for the SAM2 demo.

**Deployment Method**: Docker Compose (see `docker-compose.yaml` in project root)
- Frontend: http://localhost:7262 (React app via nginx)
- Backend: http://localhost:7263 (Flask + GraphQL via Gunicorn)
- Start command: `docker compose up --build`

## Phase 1: Core Export Functionality ✅ COMPLETED (2025-11-15)

### Frontend: Export UI Components

1. **[✅ DONE] Create frame rate selector component**
   - Add dropdown/radio buttons for FPS selection (0.5, 1, 2, 5, 10, 15, 30)
   - Display estimated frame count and export size for each option
   - Default to 5 FPS for high-FPS videos, 1 FPS for low-FPS
   - **Validation**: Component renders correctly with all frame rate options
   - **Location**: `demo/frontend/src/common/components/export/FrameRateSelector.tsx`

2. **[✅ DONE] Build export configuration modal**
   - Create modal with frame rate selector
   - Show video metadata summary (duration, source FPS, resolution)
   - Display export size warning if > 50MB
   - Add "Export" and "Cancel" buttons
   - **Validation**: Modal opens/closes properly, shows accurate estimates
   - **Location**: `demo/frontend/src/common/components/export/ExportConfigModal.tsx`

3. **[✅ DONE] Add export button to video editor toolbar**
   - Add "Export Annotations" button to existing toolbar
   - Enable only when video has tracked objects
   - Open export configuration modal on click
   - **Validation**: Button appears correctly, modal triggers on click
   - **Location**: Export button component created at `demo/frontend/src/common/components/export/ExportButton.tsx` (integration with DemoVideoEditor.tsx documented)

4. **[✅ DONE] Implement export progress indicator**
   - Create progress component showing percentage, stage, and ETA
   - Update progress based on GraphQL subscription or polling
   - Auto-hide after 3 seconds on completion
   - Show error state with retry button on failure
   - **Validation**: Progress updates smoothly, completes at 100%
   - **Location**: `demo/frontend/src/common/components/export/ExportProgress.tsx`

5. **[✅ DONE] Handle export download**
   - Trigger browser download when export completes
   - Show success message with file size
   - Clear export state after download
   - **Validation**: ZIP file downloads correctly, contains expected JSON
   - **Location**: Integrated into `demo/frontend/src/common/components/export/useExport.ts` (custom hook)

### Backend: Export API and Processing

6. **[✅ DONE] Define GraphQL export mutation**
   - Add `exportVideoAnnotations` mutation accepting:
     - `videoId`: String!
     - `targetFps`: Float!
   - Return type: `ExportResult` with `jobId`, `status`, `downloadUrl`
   - **Validation**: Mutation schema validates in GraphQL playground
   - **Location**: `demo/backend/server/data/schema.py` and `demo/backend/server/data/data_types.py`

7. **[✅ DONE] Implement frame sampling logic**
   - Calculate target timestamps based on target FPS
   - Find closest actual frames to each timestamp
   - Handle edge cases (VFR videos, missing frames)
   - Return list of frame indices to export
   - **Validation**: Unit tests for various FPS combinations (30→5, 24→1, etc.)
   - **Location**: `demo/backend/server/utils/frame_sampler.py`

8. **[✅ DONE] Add RLE mask encoding**
   - Encode binary masks to RLE format (COCO-compatible)
   - Compress RLE data for smaller JSON size
   - Verify lossless decode roundtrip
   - **Validation**: Encode/decode test with sample masks, size reduction ≥10×
   - **Location**: `demo/backend/server/utils/rle_encoder.py`

9. **[✅ DONE] Build JSON annotation serializer**
   - Generate JSON structure with video metadata
   - Include export config (target FPS, frame indices)
   - Serialize per-frame annotations with RLE masks
   - Add bounding boxes, confidence scores, object IDs
   - **Validation**: Output validates against JSON schema, size < 50MB for typical video
   - **Location**: `demo/backend/server/utils/annotation_serializer.py`

10. **[✅ DONE] Implement export processing workflow**
    - Create background task/job for export processing
    - Sample frames using frame sampler
    - Generate masks for each sampled frame
    - Serialize to JSON with RLE encoding
    - Create ZIP archive with JSON + metadata
    - Store ZIP temporarily for download
    - **Validation**: End-to-end test with sample video, verify output integrity
    - **Location**: `demo/backend/server/data/export_service.py`

11. **[✅ DONE] Add export job tracking and status**
    - Track export job progress (pending, processing, completed, failed)
    - Update progress percentage as frames are processed
    - Store job results (download URL, file size, completion time)
    - Clean up temporary files after 1 hour
    - **Validation**: Job status updates correctly, cleanup runs as expected
    - **Location**: Integrated into `demo/backend/server/data/export_service.py`

12. **[✅ DONE] Create download endpoint**
    - Add HTTP endpoint to serve completed export ZIP files
    - Validate job ID and completion status
    - Stream file download with proper headers
    - Log download events
    - **Validation**: Download works in browser, file integrity preserved
    - **Location**: `demo/backend/server/app.py` (route: `/api/download/export/<job_id>`)

### Integration and Testing

13. **[✅ DONE] Connect frontend to backend export API**
    - Wire export button → GraphQL mutation call
    - Poll for job status and update progress UI
    - Trigger download when job completes
    - Handle errors gracefully with user-friendly messages
    - **Validation**: Full export workflow works end-to-end
    - **Dependencies**: Tasks 1-12 must be complete

14. **[✅ DONE] Add error handling and edge cases**
    - Handle network failures during export
    - Catch processing errors (corrupt masks, encoding failures)
    - Display user-friendly error messages
    - Implement retry logic where appropriate
    - **Validation**: Error states tested, no unhandled exceptions
    - **Dependencies**: Tasks 1-13 must be complete

15. **[✅ DONE] Performance testing and optimization**
    - Test export with various video lengths (10s, 1min, 5min)
    - Verify export completes within 30s for 30s video at 5 FPS
    - Optimize frame processing (batching, caching)
    - Ensure UI stays responsive during export
    - **Validation**: Performance targets met per success criteria
    - **Dependencies**: Tasks 1-14 must be complete
    - **Note**: Implemented with background threading for async processing

16. **[✅ DONE] Documentation for Phase 1**
    - Document export API in GraphQL schema comments
    - Add user guide for export feature to demo README
    - Document JSON export format with examples
    - Create troubleshooting section for common issues
    - **Validation**: Documentation is clear and accurate
    - **Dependencies**: Tasks 1-15 must be complete
    - **Location**: `docs/export-feature/` (guides, api, testing sections)

## Phase 2: Authentication & Quota Integration (Weeks 4-5)

### Authentication Implementation

17. **Set up auth service connection**
    - Configure auth service endpoint in environment variables
    - Add HTTP client for auth service communication
    - Document JWT token format and claims structure
    - **Validation**: Successfully connect to auth service in dev environment
    - **Location**: `demo/backend/server/config/auth_config.py`

18. **Implement JWT validation middleware**
    - Create middleware to validate JWT tokens on protected routes
    - Verify signature, expiration, and issuer
    - Extract user ID from token claims
    - Return 401 for invalid/expired tokens
    - **Validation**: Valid tokens pass, invalid tokens rejected
    - **Location**: `demo/backend/server/middleware/auth_middleware.py`

19. **Protect video and export endpoints**
    - Apply auth middleware to all video upload, processing, and export routes
    - Require valid JWT for all protected operations
    - Associate uploaded videos and exports with user ID
    - **Validation**: Unauthorized requests blocked, authorized requests succeed
    - **Dependencies**: Task 18 must be complete

20. **Add frontend authentication flow**
    - Implement login/logout UI (or integrate with existing auth modal)
    - Store JWT token in localStorage/cookies
    - Include token in all API requests (Authorization header)
    - Handle token refresh or re-login on expiration
    - **Validation**: Users can log in, tokens sent correctly, expiry handled
    - **Location**: `demo/frontend/src/common/auth/`
    - **Dependencies**: Task 18 must be complete

### Quota System Implementation

21. **Set up quota service integration**
    - Configure quota service endpoint in environment variables
    - Add HTTP client for quota service API calls
    - Document quota metrics and API contract
    - **Validation**: Successfully query quota service in dev environment
    - **Location**: `demo/backend/server/config/quota_config.py`

22. **Implement quota checking middleware**
    - Create middleware to check quota before video processing
    - Calculate quota cost based on video duration
    - Call quota service to verify sufficient balance
    - Return 403 if quota exceeded
    - **Validation**: Quota checks work correctly, blocking when insufficient
    - **Location**: `demo/backend/server/middleware/quota_middleware.py`

23. **Add quota deduction after operations**
    - Deduct quota after successful video processing
    - Deduct quota after successful export operations
    - Implement rollback logic for failed operations
    - Log all quota transactions for auditing
    - **Validation**: Quota accurately deducted, refunded on failure
    - **Dependencies**: Task 22 must be complete

24. **Track usage metrics**
    - Record video processing duration
    - Count export operations and frames processed
    - Report metrics to quota service
    - Store local usage logs for debugging
    - **Validation**: Metrics accurately reflect actual usage
    - **Dependencies**: Task 22-23 must be complete

25. **Add quota visibility in frontend**
    - Display current quota balance in UI header
    - Show quota cost estimate before video upload
    - Update quota display after operations
    - Warn when quota drops below 20%
    - **Validation**: Quota info displays correctly, updates in real-time
    - **Location**: `demo/frontend/src/common/components/quota/QuotaDisplay.tsx`
    - **Dependencies**: Tasks 21-24 must be complete

26. **Handle quota exhaustion gracefully**
    - Block uploads when quota is 0
    - Display upgrade/purchase links when quota exceeded
    - Allow viewing previously processed videos (cached)
    - Show clear messaging about quota status
    - **Validation**: User experience is clear when quota runs out
    - **Dependencies**: Tasks 22-25 must be complete

### Integration Testing and Deployment

27. **End-to-end auth + quota testing**
    - Test full workflow: login → upload → annotate → export with quota
    - Verify quota deduction and balance updates
    - Test quota exhaustion and blocking
    - Test token expiration and re-authentication
    - **Validation**: All auth and quota flows work correctly together
    - **Dependencies**: Tasks 17-26 must be complete

28. **Security review**
    - Verify JWT validation is secure (signature, expiration)
    - Check for authorization bypasses
    - Ensure quota cannot be manipulated by users
    - Review error messages for information leakage
    - **Validation**: Security review passes, no critical issues
    - **Dependencies**: Tasks 17-27 must be complete

29. **Performance testing with auth/quota**
    - Verify auth middleware adds minimal latency (< 10ms)
    - Test quota service calls don't block requests
    - Ensure concurrent users are handled correctly
    - **Validation**: Performance targets still met with auth/quota enabled
    - **Dependencies**: Tasks 17-28 must be complete

30. **Documentation for Phase 2**
    - Document auth setup and configuration
    - Document quota system integration
    - Add deployment guide for production environment
    - Update API documentation with auth requirements
    - **Validation**: Documentation complete and accurate
    - **Dependencies**: Tasks 17-29 must be complete

## Milestone Summary

**Milestone 1 (Week 2)**: Frontend export UI complete, users can configure exports
- Tasks 1-5 complete

**Milestone 2 (Week 3)**: Backend export API functional, downloadable JSON exports working
- Tasks 6-16 complete

**Milestone 3 (Week 4)**: Authentication integrated, users must log in
- Tasks 17-20 complete

**Milestone 4 (Week 5)**: Quota system integrated, usage tracked and limited
- Tasks 21-30 complete

**Final Deliverable**: Production-ready SAM2 demo with export, auth, and quota features

## Dependency Graph

```
Phase 1 Foundation:
1-5 (Frontend UI) ─┐
                   ├─→ 13 (Integration) ─→ 14 (Error Handling) ─→ 15 (Performance) ─→ 16 (Docs)
6-12 (Backend API) ─┘

Phase 2 Auth:
17 (Auth Setup) → 18 (JWT Middleware) ─┬→ 19 (Protect Endpoints)
                                        └→ 20 (Frontend Auth)

Phase 2 Quota:
21 (Quota Setup) → 22 (Check Middleware) → 23 (Deduction) → 24 (Metrics) ─┬→ 25 (Frontend Display)
                                                                           └→ 26 (Exhaustion Handling)

Phase 2 Integration:
19, 20, 26 → 27 (E2E Testing) → 28 (Security) → 29 (Performance) → 30 (Final Docs)
```

## Notes

- **Parallel Work**: Tasks 1-5 (frontend) and 6-12 (backend) can be done in parallel by different developers
- **Testing**: Each task includes validation criteria - write tests as you implement
- **Phase Ordering**: Phase 1 must be complete and stable before starting Phase 2
- **User Feedback**: After Milestone 2, consider user testing to validate export UX before auth integration
