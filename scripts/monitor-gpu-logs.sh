#!/bin/bash
# Monitor GPU memory logs from SAM2 backend

echo "=== Monitoring SAM2 Backend GPU Memory Logs ==="
echo "Press Ctrl+C to stop"
echo ""

# Get container name from docker-compose
CONTAINER=$(docker-compose ps -q | head -1)

if [ -z "$CONTAINER" ]; then
    echo "Error: No running container found. Please run 'docker compose up' first."
    exit 1
fi

# Follow logs and filter for GPU-related messages
docker logs -f $CONTAINER 2>&1 | grep --line-buffered -E "GPU memory|Clearing|Deleted|session|allocated|reserved|MiB"
