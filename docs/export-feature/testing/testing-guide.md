# Testing Guide - Video Export Feature

## Overview

This guide provides comprehensive testing instructions for the SAM2 video annotation export feature when deployed via Docker Compose.

## Prerequisites

### Required Software

- Docker Desktop or Docker Engine with Compose V2
- NVIDIA GPU with nvidia-docker runtime (recommended, but not required)
- 8GB+ RAM
- 10GB+ free disk space

### Verify Installation

```bash
# Check Docker version
docker --version  # Should be 20.10+
docker compose version  # Should be v2.0+

# Check NVIDIA GPU (optional)
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start Testing

### 1. Start Services

```bash
# From project root
cd /path/to/sam2

# Build and start all services
docker compose up --build

# Or run in background
docker compose up -d --build

# Verify services are running
docker compose ps
```

Expected output:
```
NAME                COMMAND                  SERVICE    STATUS
sam2-backend-1      "gunicorn ..."           backend    running
sam2-frontend-1     "nginx -g ..."           frontend   running
```

### 2. Verify Endpoints

```bash
# Check frontend
curl http://localhost:7262
# Should return HTML

# Check backend health
curl http://localhost:7263/healthy
# Should return {"status": "healthy"}

# Check GraphQL endpoint
curl -X POST http://localhost:7263/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ __schema { types { name } } }"}'
# Should return schema types
```

### 3. Access Application

Open browser: http://localhost:7262

## Manual Testing Workflow

### Test Case 1: Basic Export (Happy Path)

**Objective**: Verify end-to-end export workflow

**Steps**:

1. **Upload Video**
   - Navigate to http://localhost:7262
   - Click "Upload Video" or use default video
   - Wait for video to load

2. **Add Annotations**
   - Click on video to add positive points
   - Track object through video (should see mask overlay)
   - Verify at least one object is being tracked

3. **Configure Export**
   - Click "Export" button in toolbar
   - Verify button is enabled (requires tracked objects)
   - Modal should open showing video metadata
   - Select target FPS: **5 FPS**
   - Verify estimated frame count is displayed
   - Click "Export"

4. **Monitor Progress**
   - Progress indicator should appear
   - Watch percentage increase (0% → 100%)
   - Verify processed frames count updates
   - Wait for completion (should auto-hide after 3s)

5. **Download and Verify**
   - Download should trigger automatically
   - File should be named `sam2_export_{job_id}.zip`
   - Extract ZIP file:
     ```bash
     unzip sam2_export_*.zip -d /tmp/export_test
     ls /tmp/export_test
     # Should contain: annotations.json, metadata.txt
     ```

6. **Validate JSON**
   ```bash
   # Check JSON structure
   cat /tmp/export_test/annotations.json | jq '.'

   # Verify required fields
   cat /tmp/export_test/annotations.json | jq '.video, .export_config, .annotations'

   # Count frames
   cat /tmp/export_test/annotations.json | jq '.annotations | length'
   # Should match estimated frame count
   ```

**Expected Results**:
- ✅ Export completes in < 30s for 30s video
- ✅ JSON file size < 50MB
- ✅ All frames have RLE-encoded masks
- ✅ Bounding boxes are present for all objects

### Test Case 2: Different Frame Rates

**Objective**: Verify FPS options produce correct frame counts

**Test Matrix**:

| Video Duration | Source FPS | Target FPS | Expected Frames |
|----------------|------------|------------|-----------------|
| 10s            | 30         | 0.5        | 5               |
| 10s            | 30         | 1          | 10              |
| 10s            | 30         | 5          | 50              |
| 10s            | 30         | 30         | 300             |
| 30s            | 24         | 2          | 60              |

**For Each Test**:

1. Upload video of specified duration
2. Add object tracking
3. Export with target FPS
4. Verify exported frame count:
   ```bash
   jq '.annotations | length' annotations.json
   ```

**Pass Criteria**: Frame count matches expected ± 1 frame

### Test Case 3: Multiple Objects

**Objective**: Verify export handles multiple tracked objects

**Steps**:

1. Upload video
2. Add **3 different objects** with unique positive points
3. Verify all objects are tracked in timeline
4. Export at 5 FPS
5. Verify JSON contains all objects:
   ```bash
   jq '.annotations[0].objects | length' annotations.json
   # Should be 3
   ```

**Expected Results**:
- ✅ All objects present in every exported frame
- ✅ Each object has unique `object_id`
- ✅ RLE masks are different for each object

### Test Case 4: Error Handling

**Objective**: Verify graceful error handling

#### 4a. Export Without Tracked Objects

1. Upload video (don't add any annotations)
2. Click "Export" button
3. **Expected**: Button is disabled with tooltip "Add object annotations before exporting"

#### 4b. Network Interruption

1. Start export with long video
2. Stop backend container during processing:
   ```bash
   docker compose stop backend
   ```
3. **Expected**: Frontend shows error message, retry option
4. Restart backend:
   ```bash
   docker compose start backend
   ```
5. Click retry
6. **Expected**: Export resumes or restarts successfully

#### 4c. Invalid Session

Test via GraphQL playground (http://localhost:7263/graphql):

```graphql
mutation {
  exportVideoAnnotations(input: {
    session_id: "invalid_session_id"
    target_fps: 5.0
  }) {
    job_id
    status
    message
  }
}
```

**Expected**: Error message "Session not found"

### Test Case 5: Progress Tracking

**Objective**: Verify accurate progress reporting

**Steps**:

1. Upload **long video** (2+ minutes)
2. Add object tracking
3. Export at 15 FPS (many frames)
4. **Monitor**:
   - Progress percentage should increase smoothly
   - `processed_frames / total_frames` should match percentage
   - No backwards progress
   - No stuck progress (> 5s without update)

**Validation**:
```bash
# Watch backend logs
docker compose logs -f backend | grep -i "export\|progress"
```

### Test Case 6: Concurrent Exports

**Objective**: Verify multiple simultaneous exports

**Steps**:

1. Open **3 browser tabs** to http://localhost:7262
2. In each tab:
   - Upload different videos
   - Add object tracking
   - Start export simultaneously
3. **Monitor** all three:
   - All should process independently
   - No cross-contamination of progress
   - All complete successfully

**Expected**: All exports complete, unique job IDs, correct files

## Automated Testing with GraphQL

### Setup Test Environment

```bash
# Install test dependencies (in a Python virtualenv)
pip install pytest requests

# Or use Docker container
docker compose exec backend bash
```

### GraphQL Test Script

Create `test_export_api.py`:

```python
import requests
import time
import json

BASE_URL = "http://localhost:7263"

def test_export_workflow():
    """Test complete export workflow via GraphQL"""

    # Step 1: Create export job
    mutation = """
    mutation ExportVideo($input: ExportVideoAnnotationsInput!) {
      exportVideoAnnotations(input: $input) {
        job_id
        status
        message
      }
    }
    """

    variables = {
        "input": {
            "session_id": "test_session_123",  # Replace with valid session
            "target_fps": 5.0
        }
    }

    response = requests.post(
        f"{BASE_URL}/graphql",
        json={"query": mutation, "variables": variables}
    )

    assert response.status_code == 200
    data = response.json()["data"]["exportVideoAnnotations"]
    job_id = data["job_id"]
    assert data["status"] in ["PENDING", "PROCESSING"]

    print(f"✅ Created job: {job_id}")

    # Step 2: Poll for completion
    status_query = """
    query JobStatus($jobId: String!) {
      exportJobStatus(jobId: $jobId) {
        status
        progress
        processed_frames
        total_frames
        download_url
      }
    }
    """

    max_polls = 60
    for i in range(max_polls):
        response = requests.post(
            f"{BASE_URL}/graphql",
            json={"query": status_query, "variables": {"jobId": job_id}}
        )

        status_data = response.json()["data"]["exportJobStatus"]
        status = status_data["status"]
        progress = status_data["progress"]

        print(f"Poll {i+1}: {status} - {progress*100:.1f}%")

        if status == "COMPLETED":
            download_url = status_data["download_url"]
            print(f"✅ Export completed: {download_url}")

            # Step 3: Download file
            download_response = requests.get(download_url)
            assert download_response.status_code == 200
            assert download_response.headers["Content-Type"] == "application/zip"

            print(f"✅ Downloaded {len(download_response.content)} bytes")
            return True

        elif status == "FAILED":
            print(f"❌ Export failed")
            return False

        time.sleep(1)

    print(f"❌ Timeout after {max_polls}s")
    return False

if __name__ == "__main__":
    test_export_workflow()
```

**Run Test**:
```bash
python test_export_api.py
```

## Performance Testing

### Benchmarking Export Speed

Create `benchmark_export.sh`:

```bash
#!/bin/bash

echo "SAM2 Export Performance Benchmark"
echo "================================="

# Test configurations
CONFIGS=(
  "10s,30fps,5fps"
  "30s,30fps,5fps"
  "60s,30fps,10fps"
  "120s,24fps,15fps"
)

for config in "${CONFIGS[@]}"; do
  IFS=',' read -r duration source_fps target_fps <<< "$config"

  echo ""
  echo "Test: ${duration} video @ ${source_fps} → ${target_fps}"

  start_time=$(date +%s)

  # Trigger export (replace with actual session ID)
  job_id=$(curl -s -X POST http://localhost:7263/graphql \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"mutation { exportVideoAnnotations(input: {session_id: \\\"test\\\", target_fps: ${target_fps}}) { job_id } }\"}" \
    | jq -r '.data.exportVideoAnnotations.job_id')

  echo "Job ID: $job_id"

  # Poll until complete
  while true; do
    status=$(curl -s -X POST http://localhost:7263/graphql \
      -H "Content-Type: application/json" \
      -d "{\"query\": \"query { exportJobStatus(jobId: \\\"${job_id}\\\") { status } }\"}" \
      | jq -r '.data.exportJobStatus.status')

    if [ "$status" == "COMPLETED" ]; then
      end_time=$(date +%s)
      duration=$((end_time - start_time))
      echo "✅ Completed in ${duration}s"
      break
    elif [ "$status" == "FAILED" ]; then
      echo "❌ Failed"
      break
    fi

    sleep 1
  done
done
```

**Run Benchmark**:
```bash
chmod +x benchmark_export.sh
./benchmark_export.sh
```

### Performance Targets

| Video Length | Target FPS | Max Export Time | Status |
|--------------|------------|-----------------|--------|
| 10s          | 5          | 10s             | ✅      |
| 30s          | 5          | 30s             | ✅      |
| 60s          | 10         | 60s             | ✅      |
| 120s         | 15         | 120s            | ✅      |

## Validation Scripts

### JSON Schema Validation

```bash
# Validate annotations.json structure
cat annotations.json | jq '
  if (.video and .export_config and .annotations) then
    "✅ Valid structure"
  else
    "❌ Missing required fields"
  end
'

# Check RLE encoding
cat annotations.json | jq '
  .annotations[0].objects[0].mask |
  if (.counts and .size) then
    "✅ Valid RLE format"
  else
    "❌ Invalid RLE"
  end
'
```

### File Size Validation

```bash
# Check export file size
FILE_SIZE=$(stat -f%z sam2_export_*.zip)  # macOS
# FILE_SIZE=$(stat -c%s sam2_export_*.zip)  # Linux

if [ $FILE_SIZE -lt 52428800 ]; then  # 50MB
  echo "✅ File size OK: $(($FILE_SIZE / 1024 / 1024))MB"
else
  echo "⚠️  File size large: $(($FILE_SIZE / 1024 / 1024))MB"
fi
```

## Troubleshooting Tests

### Issue: Export Never Completes

**Debug**:
```bash
# Check backend logs
docker compose logs backend | grep -i export

# Check for Python errors
docker compose logs backend | grep -i error

# Verify export job exists
docker compose exec backend ls -la /data/exports/
```

### Issue: Download Returns 404

**Debug**:
```bash
# Check if file was created
docker compose exec backend ls /data/exports/

# Verify job ID
# Check GraphQL response for correct download_url
```

### Issue: JSON Invalid or Corrupted

**Debug**:
```bash
# Validate JSON syntax
jq '.' annotations.json

# Check file encoding
file annotations.json
# Should be: "ASCII text" or "UTF-8 Unicode text"

# Check for truncation
tail annotations.json
# Should end with: }
```

## Continuous Testing

### Pre-Deployment Checklist

- [ ] All 6 manual test cases pass
- [ ] GraphQL automated test passes
- [ ] Performance benchmark meets targets
- [ ] JSON validation succeeds
- [ ] No errors in backend logs
- [ ] File sizes within limits
- [ ] Multiple concurrent exports work

### Regression Testing

Run after any code changes:

```bash
# Rebuild and restart
docker compose down
docker compose build --no-cache
docker compose up -d

# Run test suite
python test_export_api.py

# Check logs for warnings
docker compose logs backend | grep -i "warn\|error"
```

## Related Documentation

- **Implementation Guide**: `docs/export-feature/guides/implementation-guide.md`
- **API Reference**: `docs/export-feature/api/graphql-api.md`
- **Architecture**: `docs/export-feature/guides/architecture.md`
