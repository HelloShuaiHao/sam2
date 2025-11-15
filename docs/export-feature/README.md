# SAM2 Video Export Feature - Documentation Index

## Overview

This directory contains comprehensive documentation for the SAM2 video annotation export feature.

## Documentation Structure

```
docs/export-feature/
├── README.md                           # This file (documentation index)
├── SUMMARY.md                          # Implementation completion summary
├── guides/
│   ├── implementation-guide.md         # Docker Compose setup and integration
│   └── architecture.md                 # System architecture and design decisions
├── api/
│   └── graphql-api.md                  # GraphQL API reference
└── testing/
    └── testing-guide.md                # Testing procedures and validation
```

## Quick Start

**New to the export feature?** Start here:

1. **[Implementation Guide](guides/implementation-guide.md)**
   - Docker Compose startup instructions
   - Integration steps for DemoVideoEditor
   - Configuration options
   - Troubleshooting guide

2. **[API Reference](api/graphql-api.md)**
   - GraphQL schema documentation
   - Mutation and query examples
   - Request/response formats
   - Error codes

3. **[Testing Guide](testing/testing-guide.md)**
   - Manual test cases
   - Automated testing scripts
   - Performance benchmarks
   - Validation procedures

4. **[Architecture](guides/architecture.md)**
   - System design overview
   - Component descriptions
   - Design decisions and trade-offs
   - Scalability considerations

5. **[Summary](SUMMARY.md)**
   - Implementation completion status
   - Deliverables and metrics
   - Known limitations
   - Future enhancements

## Feature Overview

The video export feature allows users to:

- ✅ Export video annotations with tracked object masks
- ✅ Configure export frame rate (0.5 - 30 FPS)
- ✅ Download annotations as JSON with RLE-encoded masks
- ✅ Track export progress in real-time
- ✅ Automatic ZIP file packaging

## Deployment

**Quick Start**:

```bash
# Start services
docker compose up --build

# Access application
# Frontend: http://localhost:7262
# Backend: http://localhost:7263/graphql
```

See [Implementation Guide](guides/implementation-guide.md) for detailed instructions.

## API Example

```graphql
# Create export job
mutation {
  exportVideoAnnotations(input: {
    session_id: "my_session"
    target_fps: 5.0
  }) {
    job_id
    status
  }
}

# Poll for status
query {
  exportJobStatus(jobId: "export_123") {
    status
    progress
    download_url
  }
}
```

See [API Reference](api/graphql-api.md) for complete documentation.

## Testing

```bash
# Run full test suite
docker compose up -d
python test_export_api.py

# Manual testing
# 1. Upload video at http://localhost:7262
# 2. Add object annotations
# 3. Click "Export" button
# 4. Configure FPS and export
# 5. Download ZIP file
```

See [Testing Guide](testing/testing-guide.md) for detailed procedures.

## Implementation Status

**Phase 1**: ✅ COMPLETED (2025-11-15)
- All core export functionality implemented
- Frontend: 5 components (~950 lines TypeScript/React)
- Backend: 7 files (~1,090 lines Python)
- Documentation: Complete

**Phase 2**: ⏳ PLANNED
- JWT authentication integration
- Quota enforcement
- Usage tracking

See [Summary](SUMMARY.md) for full status.

## Key Files

### Frontend
- `demo/frontend/src/common/components/export/ExportButton.tsx` - Integration component
- `demo/frontend/src/common/components/export/ExportConfigModal.tsx` - Configuration UI
- `demo/frontend/src/common/components/export/ExportProgress.tsx` - Progress indicator
- `demo/frontend/src/common/components/export/FrameRateSelector.tsx` - FPS selector
- `demo/frontend/src/common/components/export/useExport.ts` - State management hook

### Backend
- `demo/backend/server/data/export_service.py` - Export job manager
- `demo/backend/server/data/schema.py` - GraphQL schema
- `demo/backend/server/data/data_types.py` - GraphQL types
- `demo/backend/server/utils/frame_sampler.py` - Frame sampling logic
- `demo/backend/server/utils/rle_encoder.py` - RLE mask encoding
- `demo/backend/server/utils/annotation_serializer.py` - JSON serialization
- `demo/backend/server/app.py` - Download endpoint

### OpenSpec
- `openspec/changes/add-frame-export-annotation/proposal.md` - Feature proposal
- `openspec/changes/add-frame-export-annotation/design.md` - Design decisions
- `openspec/changes/add-frame-export-annotation/tasks.md` - Implementation tasks
- `openspec/changes/add-frame-export-annotation/specs/` - Detailed specifications

## Support

For issues or questions:

1. Check the [Troubleshooting](guides/implementation-guide.md#troubleshooting) section
2. Review [Testing Guide](testing/testing-guide.md) for validation procedures
3. Consult [Architecture](guides/architecture.md) for design decisions

## Related Resources

- **SAM2 GitHub**: https://github.com/facebookresearch/segment-anything-2
- **GraphQL Strawberry**: https://strawberry.rocks/
- **Docker Compose**: https://docs.docker.com/compose/
- **COCO RLE Format**: https://github.com/cocodataset/cocoapi

## Version History

- **v1.0.0** (2025-11-15): Initial release
  - Phase 1 core export functionality
  - Docker Compose deployment
  - Complete documentation
