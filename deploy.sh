#!/bin/bash
# SAM2 Training API Quick Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Docker is running
check_docker() {
    print_info "Checking Docker..."
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running!"
        exit 1
    fi
    print_success "Docker is running"
}

# Check if docker-compose is available
check_compose() {
    print_info "Checking Docker Compose..."
    if ! command -v docker compose &> /dev/null; then
        print_error "Docker Compose not found!"
        exit 1
    fi
    print_success "Docker Compose is available"
}

# Check GPU
check_gpu() {
    print_info "Checking GPU..."
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi not found. GPU may not be available."
    else
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        print_success "GPU detected"
    fi
}

# Create necessary directories
create_dirs() {
    print_info "Creating necessary directories..."
    mkdir -p checkpoints
    mkdir -p demo/training/output
    print_success "Directories created"
}

# Build images
build_images() {
    print_info "Building Docker images..."
    docker compose build training-api
    print_success "Images built successfully"
}

# Start services
start_services() {
    print_info "Starting services..."
    docker compose up -d
    print_success "Services started"
}

# Check service health
check_health() {
    print_info "Checking service health..."

    sleep 5  # Wait for services to start

    # Check training API
    if curl -f http://localhost:7264/health > /dev/null 2>&1; then
        print_success "Training API is healthy"
    else
        print_warning "Training API is not responding yet (this may take a minute)"
    fi

    # Check backend
    if curl -f http://localhost:7263 > /dev/null 2>&1; then
        print_success "Backend is healthy"
    else
        print_warning "Backend is not responding yet"
    fi

    # Check frontend
    if curl -f http://localhost:7262 > /dev/null 2>&1; then
        print_success "Frontend is healthy"
    else
        print_warning "Frontend is not responding yet"
    fi
}

# Show logs
show_logs() {
    print_info "Showing logs (Ctrl+C to exit)..."
    docker compose logs -f training-api
}

# Show status
show_status() {
    print_info "Service status:"
    docker compose ps

    echo ""
    print_info "GPU usage:"
    docker compose exec training-api nvidia-smi 2>/dev/null || print_warning "Could not access GPU"

    echo ""
    print_info "Access URLs:"
    echo "  Frontend:    http://ai.bygpu.com:7262"
    echo "  Backend:     http://ai.bygpu.com:7263"
    echo "  Training API: http://ai.bygpu.com:7264"
    echo "  API Docs:    http://ai.bygpu.com:7264/docs"
}

# Stop services
stop_services() {
    print_info "Stopping services..."
    docker compose down
    print_success "Services stopped"
}

# Restart training API
restart_training() {
    print_info "Restarting training API..."
    docker compose restart training-api
    print_success "Training API restarted"
}

# Main menu
show_menu() {
    echo ""
    echo "========================================"
    echo " SAM2 Training Deployment Manager"
    echo "========================================"
    echo "1) Full deployment (build + start)"
    echo "2) Build images only"
    echo "3) Start services"
    echo "4) Stop services"
    echo "5) Restart training API"
    echo "6) Show status"
    echo "7) Show logs"
    echo "8) Check health"
    echo "9) Exit"
    echo "========================================"
    read -p "Select option: " choice

    case $choice in
        1)
            check_docker
            check_compose
            check_gpu
            create_dirs
            build_images
            start_services
            check_health
            show_status
            ;;
        2)
            check_docker
            check_compose
            build_images
            ;;
        3)
            check_docker
            start_services
            check_health
            show_status
            ;;
        4)
            stop_services
            ;;
        5)
            restart_training
            check_health
            ;;
        6)
            show_status
            ;;
        7)
            show_logs
            ;;
        8)
            check_health
            ;;
        9)
            exit 0
            ;;
        *)
            print_error "Invalid option"
            show_menu
            ;;
    esac
}

# Run menu
while true; do
    show_menu
done
