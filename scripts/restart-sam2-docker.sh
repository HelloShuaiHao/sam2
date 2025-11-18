#!/bin/bash

##############################################################################
# SAM2 Demo - Docker Restart Script with GPU Memory Cleanup
#
# This script stops, rebuilds, and restarts the SAM2 Docker containers
# with the latest code changes (including GPU memory leak fixes)
#
# Usage:
#   chmod +x restart-sam2-docker.sh
#   ./restart-sam2-docker.sh
##############################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "=========================================="
echo "  SAM2 Docker - Restart & Rebuild"
echo "=========================================="
echo ""

# Check if docker-compose.yaml exists
if [ ! -f "docker-compose.yaml" ]; then
    echo -e "${RED}Error: docker-compose.yaml not found!${NC}"
    echo "Please run this script from the sam2 directory."
    exit 1
fi

# Step 1: Stop all containers
echo -e "${YELLOW}Step 1/4: Stopping all containers...${NC}"
docker-compose down
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to stop containers!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Containers stopped${NC}"
echo ""

# Wait for containers to fully stop
sleep 3

# Step 2: Show current GPU memory usage
echo -e "${BLUE}Current GPU Memory Status:${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
else
    echo -e "${YELLOW}nvidia-smi not available (may be inside Docker)${NC}"
fi
echo ""

# Step 3: Rebuild backend image with latest code
echo -e "${YELLOW}Step 2/4: Rebuilding backend image with GPU memory leak fixes...${NC}"
echo -e "${BLUE}This includes the following fixes:${NC}"
echo "  • Session close cleanup (predictor.py)"
echo "  • Reset state GPU cache clearing (sam2_video_predictor.py)"
echo "  • Propagation completion cleanup"
echo "  • Frame output tensor cleanup"
echo "  • Auto-close old sessions before starting new ones"
echo ""

docker-compose build backend
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to rebuild backend!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Backend rebuilt with memory leak fixes${NC}"
echo ""

# Step 4: Start all services
echo -e "${YELLOW}Step 3/4: Starting all services...${NC}"
docker-compose up -d
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to start services!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Services started${NC}"
echo ""

# Wait for services to initialize
echo -e "${YELLOW}Waiting for services to initialize...${NC}"
sleep 5

# Step 5: Show service status
echo -e "${YELLOW}Step 4/4: Checking service status...${NC}"
docker-compose ps
echo ""

# Show GPU memory after restart
echo -e "${BLUE}GPU Memory After Restart:${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
fi
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✅ SAM2 Demo Restarted Successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Access the demo at:"
echo "  Frontend: http://ai.bygpu.com:55305/sam2/"
echo "  Backend API: http://ai.bygpu.com:55305/api/sam2"
echo ""
echo -e "${BLUE}To view backend logs:${NC}"
echo "  docker-compose logs -f backend"
echo ""
echo -e "${BLUE}To view all logs:${NC}"
echo "  docker-compose logs -f"
echo ""
echo -e "${YELLOW}GPU Memory Leak Fixes Applied:${NC}"
echo "  ✅ Sessions now properly clean up GPU memory on close"
echo "  ✅ Old sessions auto-close when uploading new videos"
echo "  ✅ Periodic GPU cache clearing during propagation"
echo "  ✅ Frame output tensors cleaned up immediately"
echo ""
echo -e "${GREEN}Test by uploading multiple videos consecutively!${NC}"
echo ""
