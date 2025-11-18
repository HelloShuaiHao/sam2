#!/bin/bash

##############################################################################
# SAM2 Demo - macOS Safari Launcher
#
# This script launches Safari and opens the SAM2 demo URL.
# Note: Safari requires manual configuration for HTTP development features.
#
# Usage:
#   chmod +x start-sam2-safari-macos.sh
#   ./start-sam2-safari-macos.sh
##############################################################################

# Configuration
SAM2_URL="http://ai.bygpu.com:55305/sam2/"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "=========================================="
echo "  SAM2 Demo - Safari Launcher (macOS)"
echo "=========================================="
echo ""

# Check if Safari is installed
if [ ! -d "/Applications/Safari.app" ]; then
    echo -e "${RED}Error: Safari is not installed!${NC}"
    exit 1
fi

# Check if Develop menu is enabled
DEVELOP_MENU_ENABLED=$(defaults read com.apple.Safari IncludeDevelopMenu 2>/dev/null)

if [ "$DEVELOP_MENU_ENABLED" != "1" ]; then
    echo -e "${YELLOW}⚠️  Safari Develop Menu is not enabled!${NC}"
    echo ""
    echo "Enabling Develop Menu automatically..."
    defaults write com.apple.Safari IncludeDevelopMenu -bool true
    defaults write com.apple.Safari WebKitDeveloperExtrasEnabledPreferenceKey -bool true
    defaults write com.apple.Safari com.apple.Safari.ContentPageGroupIdentifier.WebKit2DeveloperExtrasEnabled -bool true
    echo -e "${GREEN}✅ Develop Menu enabled!${NC}"
    echo ""
fi

# Enable WebRTC and Media Capture for HTTP (insecure origins)
echo -e "${YELLOW}Configuring Safari for HTTP development...${NC}"
defaults write com.apple.Safari AllowInsecureMediaCaptureInMainFrame -bool true

echo -e "${GREEN}✅ Safari configured for development!${NC}"
echo ""

# Launch Safari
echo -e "${GREEN}Launching Safari for SAM2 Demo...${NC}"
echo ""
echo "URL: $SAM2_URL"
echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  IMPORTANT: Manual Configuration${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo "After Safari opens, you need to:"
echo ""
echo "1. Enable 'Disable Cross-Origin Restrictions' (required for HTTP):"
echo "   Develop → Disable Cross-Origin Restrictions"
echo ""
echo "2. (Optional) Open Web Inspector to check console:"
echo "   Press: Cmd + Option + C"
echo ""
echo "3. (Optional) To verify WebCodecs support, run in console:"
echo "   console.log('VideoEncoder' in window)"
echo ""
echo -e "${YELLOW}Note: Safari doesn't clear your browsing data or cache!${NC}"
echo -e "${YELLOW}Your regular Safari sessions remain intact.${NC}"
echo ""

# Open Safari with the URL
open -a Safari "$SAM2_URL"

sleep 2
echo -e "${GREEN}Safari launched successfully!${NC}"
echo ""
echo -e "${YELLOW}If you encounter issues with video processing:${NC}"
echo "- Make sure 'Disable Cross-Origin Restrictions' is enabled"
echo "- Check Safari → Preferences → Privacy → Website tracking"
echo "- Try reloading the page after enabling developer features"
echo ""
