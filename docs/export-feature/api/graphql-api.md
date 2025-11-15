# GraphQL API Reference - Video Export

## Overview

This document describes the GraphQL API for the SAM2 video annotation export feature.

**Endpoint**: `http://localhost:7263/graphql` (Docker Compose setup)

## Types

### Enums

#### ExportJobStatus

Export job status enumeration.

```graphql
enum ExportJobStatus {
  PENDING
  PROCESSING
  COMPLETED
  FAILED
}
```

**Values**:
- `PENDING`: Job created, waiting to start processing
- `PROCESSING`: Currently generating export
- `COMPLETED`: Export finished successfully, ready for download
- `FAILED`: Export encountered an error

### Input Types

#### ExportVideoAnnotationsInput

Input parameters for creating an export job.

```graphql
input ExportVideoAnnotationsInput {
  session_id: String!
  target_fps: Float!
}
```

**Fields**:
- `session_id` (String!, required): SAM2 session ID containing the video and tracked objects
- `target_fps` (Float!, required): Target frame rate for export (0.5 - 30 FPS)

**Constraints**:
- `target_fps` must be between 0.5 and 30.0
- Session must exist and have at least one tracked object

### Output Types

#### ExportResult

Result of initiating an export job.

```graphql
type ExportResult {
  job_id: String!
  status: ExportJobStatus!
  message: String
  download_url: String
}
```

**Fields**:
- `job_id`: Unique identifier for the export job (use for status polling)
- `status`: Current job status (typically `PENDING` when first created)
- `message`: Optional status message (e.g., error description)
- `download_url`: Download URL (only present when status is `COMPLETED`)

#### ExportJobInfo

Detailed information about an export job (for status polling).

```graphql
type ExportJobInfo {
  job_id: String!
  status: ExportJobStatus!
  progress: Float!
  processed_frames: Int!
  total_frames: Int!
  download_url: String
  error_message: String
  file_size_mb: Float
}
```

**Fields**:
- `job_id`: Job identifier
- `status`: Current job status
- `progress`: Completion percentage (0.0 - 1.0)
- `processed_frames`: Number of frames processed so far
- `total_frames`: Total frames to be exported
- `download_url`: Download URL (only when `COMPLETED`)
- `error_message`: Error details (only when `FAILED`)
- `file_size_mb`: Export file size in MB (only when `COMPLETED`)

## Operations

### Mutations

#### exportVideoAnnotations

Creates a new export job for video annotations.

```graphql
mutation ExportVideoAnnotations($input: ExportVideoAnnotationsInput!) {
  exportVideoAnnotations(input: $input) {
    job_id
    status
    message
    download_url
  }
}
```

**Variables**:
```json
{
  "input": {
    "session_id": "abc123",
    "target_fps": 5.0
  }
}
```

**Success Response**:
```json
{
  "data": {
    "exportVideoAnnotations": {
      "job_id": "export_1234567890",
      "status": "PENDING",
      "message": "Export job created successfully",
      "download_url": null
    }
  }
}
```

**Error Response** (no tracked objects):
```json
{
  "data": {
    "exportVideoAnnotations": {
      "job_id": "",
      "status": "FAILED",
      "message": "No tracked objects found in session",
      "download_url": null
    }
  }
}
```

**Error Response** (session not found):
```json
{
  "errors": [
    {
      "message": "Session not found: abc123",
      "path": ["exportVideoAnnotations"]
    }
  ]
}
```

### Queries

#### exportJobStatus

Retrieves the current status of an export job (for polling).

```graphql
query ExportJobStatus($jobId: String!) {
  exportJobStatus(jobId: $jobId) {
    job_id
    status
    progress
    processed_frames
    total_frames
    download_url
    error_message
    file_size_mb
  }
}
```

**Variables**:
```json
{
  "jobId": "export_1234567890"
}
```

**Response** (processing):
```json
{
  "data": {
    "exportJobStatus": {
      "job_id": "export_1234567890",
      "status": "PROCESSING",
      "progress": 0.45,
      "processed_frames": 67,
      "total_frames": 150,
      "download_url": null,
      "error_message": null,
      "file_size_mb": null
    }
  }
}
```

**Response** (completed):
```json
{
  "data": {
    "exportJobStatus": {
      "job_id": "export_1234567890",
      "status": "COMPLETED",
      "progress": 1.0,
      "processed_frames": 150,
      "total_frames": 150,
      "download_url": "http://localhost:7263/api/download/export/export_1234567890",
      "error_message": null,
      "file_size_mb": 3.2
    }
  }
}
```

**Response** (failed):
```json
{
  "data": {
    "exportJobStatus": {
      "job_id": "export_1234567890",
      "status": "FAILED",
      "progress": 0.0,
      "processed_frames": 0,
      "total_frames": 0,
      "download_url": null,
      "error_message": "Failed to encode mask for frame 42",
      "file_size_mb": null
    }
  }
}
```

**Response** (job not found):
```json
{
  "data": {
    "exportJobStatus": null
  }
}
```

## REST Endpoint

### Download Export File

**Endpoint**: `GET /api/download/export/{job_id}`

Downloads the completed export ZIP file.

**Example**:
```bash
curl -O http://localhost:7263/api/download/export/export_1234567890
```

**Response**:
- **Success (200)**: ZIP file with annotation data
- **Not Found (404)**: Job ID not found or export not completed
- **Headers**:
  - `Content-Type: application/zip`
  - `Content-Disposition: attachment; filename="sam2_export_{job_id}.zip"`

## Usage Workflow

### 1. Create Export Job

```graphql
mutation {
  exportVideoAnnotations(input: {
    session_id: "my_session_123"
    target_fps: 5.0
  }) {
    job_id
    status
    message
  }
}
```

Save the returned `job_id` for polling.

### 2. Poll for Status

Poll every 1-2 seconds until status is `COMPLETED` or `FAILED`:

```graphql
query {
  exportJobStatus(jobId: "export_1234567890") {
    status
    progress
    processed_frames
    total_frames
    download_url
  }
}
```

### 3. Download File

When status is `COMPLETED`, use the `download_url`:

```javascript
window.location.href = downloadUrl;
// or
fetch(downloadUrl).then(res => res.blob()).then(blob => {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'annotations.zip';
  a.click();
});
```

## Error Codes

| Error | Cause | Solution |
|-------|-------|----------|
| Session not found | Invalid `session_id` | Verify session exists |
| No tracked objects | Session has no annotations | Add object tracking first |
| Invalid FPS | `target_fps` out of range | Use 0.5 - 30 FPS |
| Job not found | Invalid `job_id` in status query | Check job ID spelling |
| Export failed | Processing error | Check `error_message` in response |

## Rate Limiting

Currently no rate limiting is implemented. For production, consider:
- Max 10 concurrent exports per user
- Max 100 exports per hour per user
- Auto-cleanup of old jobs after 1 hour

## Testing with GraphQL Playground

Access the interactive GraphQL playground at `http://localhost:7263/graphql` (when using Docker Compose).

**Example Test Workflow**:

1. Create export:
```graphql
mutation {
  exportVideoAnnotations(input: {
    session_id: "test_session"
    target_fps: 2.0
  }) {
    job_id
    status
  }
}
```

2. Check status:
```graphql
query {
  exportJobStatus(jobId: "YOUR_JOB_ID_HERE") {
    status
    progress
    download_url
  }
}
```

3. Download from returned URL in browser.

## Implementation Files

- **GraphQL Schema**: `demo/backend/server/data/schema.py`
- **Type Definitions**: `demo/backend/server/data/data_types.py`
- **Export Service**: `demo/backend/server/data/export_service.py`
- **Download Route**: `demo/backend/server/app.py`

## Related Documentation

- **Implementation Guide**: `docs/export-feature/guides/implementation-guide.md`
- **Testing Guide**: `docs/export-feature/testing/testing-guide.md`
- **Architecture**: `docs/export-feature/guides/architecture.md`
