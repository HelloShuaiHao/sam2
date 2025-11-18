#!/bin/bash

##############################################################################
# SAM2 Demo - macOS Firefox Launcher
#
# This script launches Firefox with a separate profile for SAM2 demo,
# without affecting your regular Firefox browsing data.
#
# Usage:
#   chmod +x start-sam2-firefox-macos.sh
#   ./start-sam2-firefox-macos.sh
##############################################################################

# Configuration
SAM2_URL="http://ai.bygpu.com:55305/sam2/"
FIREFOX_PROFILE_DIR="$HOME/.mozilla/firefox-sam2-profile"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "=========================================="
echo "  SAM2 Demo - Firefox Launcher (macOS)"
echo "=========================================="
echo ""

# Check if Firefox is installed
FIREFOX_PATH="/Applications/Firefox.app/Contents/MacOS/firefox"
if [ ! -f "$FIREFOX_PATH" ]; then
    echo -e "${RED}Error: Firefox is not installed!${NC}"
    echo "Please install Firefox from: https://www.mozilla.org/firefox/"
    exit 1
fi

# Create profile directory if it doesn't exist
if [ ! -d "$FIREFOX_PROFILE_DIR" ]; then
    echo -e "${YELLOW}Creating dedicated Firefox profile for SAM2...${NC}"
    mkdir -p "$FIREFOX_PROFILE_DIR"

    # Create basic profile configuration
    cat > "$FIREFOX_PROFILE_DIR/prefs.js" << 'EOF'
// SAM2 Demo Firefox Profile Configuration
user_pref("security.fileuri.strict_origin_policy", false);
user_pref("privacy.file_unique_origin", false);
user_pref("media.autoplay.default", 0);
user_pref("media.autoplay.blocking_policy", 0);
user_pref("permissions.default.camera", 1);
user_pref("permissions.default.microphone", 1);
user_pref("browser.startup.homepage", "about:blank");
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("devtools.chrome.enabled", true);
user_pref("devtools.debugger.remote-enabled", true);
EOF

    echo -e "${GREEN}✅ Profile created!${NC}"
fi

echo -e "${GREEN}Launching Firefox for SAM2 Demo...${NC}"
echo ""
echo "URL: $SAM2_URL"
echo "Profile Dir: $FIREFOX_PROFILE_DIR"
echo ""
echo -e "${YELLOW}Note: This uses a separate Firefox profile.${NC}"
echo -e "${YELLOW}Your regular Firefox browsing data remains intact!${NC}"
echo ""

# Launch Firefox with dedicated profile
"$FIREFOX_PATH" --profile "$FIREFOX_PROFILE_DIR" --no-remote "$SAM2_URL" &

sleep 2
echo -e "${GREEN}Firefox launched successfully!${NC}"
echo ""
echo -e "${BLUE}To verify WebCodecs support:${NC}"
echo "1. Press F12 to open Developer Tools"
echo "2. Go to Console tab"
echo "3. Run: console.log('VideoEncoder' in window)"
echo ""
echo "Should return: true"
echo ""
