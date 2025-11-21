#!/bin/bash

# Fix missing sentencepiece dependency
# This script provides two options:
# 1. Quick fix: Install in running container (temporary, lost on restart)
# 2. Permanent fix: Rebuild Docker image

set -e

echo "🔧 Fixing missing sentencepiece dependency"
echo ""

# Check if docker compose is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

echo "Choose fix method:"
echo "  1) Quick fix - Install in running container (immediate, lost on restart)"
echo "  2) Permanent fix - Rebuild Docker image (takes 5-10 min, permanent)"
echo ""
read -p "Enter choice [1 or 2]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 Quick fix: Installing sentencepiece in running container..."
        echo ""

        # Install in running container
        docker compose exec training-api pip install sentencepiece protobuf

        echo ""
        echo "✅ Done! You can now start training immediately."
        echo ""
        echo "⚠️  Note: This fix is temporary. If you restart the container,"
        echo "   you'll need to run this again OR do option 2 (rebuild)."
        ;;

    2)
        echo ""
        echo "🔨 Permanent fix: Rebuilding Docker image..."
        echo "This will take 5-10 minutes..."
        echo ""

        # Rebuild image
        docker compose build training-api

        echo ""
        echo "🔄 Restarting training-api..."
        docker compose up -d training-api

        echo ""
        echo "✅ Done! The fix is now permanent."
        echo ""
        echo "Waiting for service to be ready..."
        sleep 5

        # Check health
        if docker compose ps | grep training-api | grep -q "Up"; then
            echo "✅ Service is running"
        else
            echo "⚠️  Service may still be starting. Check with:"
            echo "   docker compose logs -f training-api"
        fi
        ;;

    *)
        echo "❌ Invalid choice. Please run again and enter 1 or 2."
        exit 1
        ;;
esac

echo ""
echo "📊 Next steps:"
echo "  1. Go to the training UI"
echo "  2. Click 'Start Training'"
echo "  3. Monitor progress in logs:"
echo "     docker compose logs -f training-api"
echo ""
