#!/bin/bash

##############################################################################
# SAM2 Demo - macOS Chrome Launcher
#
# This script launches Google Chrome with special flags to allow the SAM2 demo
# to work over HTTP (instead of requiring HTTPS).
#
# Usage:
#   chmod +x start-sam2-chrome-macos.sh
#   ./start-sam2-chrome-macos.sh
##############################################################################

# Configuration
SAM2_URL="http://ai.bygpu.com:55305/sam2/"
CHROME_USER_DATA_DIR="/tmp/chrome-sam2"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "=========================================="
echo "  SAM2 Demo - Chrome Launcher (macOS)"
echo "=========================================="
echo ""

# Check if Chrome is installed
if [ ! -d "/Applications/Google Chrome.app" ]; then
    echo -e "${RED}Error: Google Chrome is not installed!${NC}"
    echo "Please install Google Chrome from: https://www.google.com/chrome/"
    exit 1
fi

# Close existing Chrome instances
echo -e "${YELLOW}Closing existing Chrome instances...${NC}"
killall "Google Chrome" 2>/dev/null
sleep 2

# Clean up old user data directory
if [ -d "$CHROME_USER_DATA_DIR" ]; then
    echo -e "${YELLOW}Cleaning up previous session data...${NC}"
    rm -rf "$CHROME_USER_DATA_DIR"
fi

# Launch Chrome with special flags
echo -e "${GREEN}Launching Chrome for SAM2 Demo...${NC}"
echo ""
echo "URL: $SAM2_URL"
echo "User Data Dir: $CHROME_USER_DATA_DIR"
echo ""
echo -e "${YELLOW}Note: This Chrome instance uses special flags for development.${NC}"
echo -e "${YELLOW}Do NOT use it for regular browsing!${NC}"
echo ""

/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --unsafely-treat-insecure-origin-as-secure="http://ai.bygpu.com:55305" \
  --user-data-dir="$CHROME_USER_DATA_DIR" \
  --disable-features=SecureContextCheck \
  --no-first-run \
  --no-default-browser-check \
  "$SAM2_URL" \
  2>/dev/null &

sleep 2
echo -e "${GREEN}Chrome launched successfully!${NC}"
echo ""
echo "To verify WebCodecs is working, press F12 in Chrome and run:"
echo "  console.log('VideoEncoder' in window)"
echo ""
echo "Should return: true"
echo ""
