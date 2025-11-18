# SAM2 Demo - Browser Launcher Scripts

This folder contains scripts to launch different browsers with configurations that allow the SAM2 demo to work over HTTP (without HTTPS).

## Why are these scripts needed?

The SAM2 demo uses **WebCodecs API** (VideoEncoder, VideoDecoder, VideoFrame) which requires a **Secure Context**:
- ✅ `https://` (any domain with SSL)
- ✅ `http://localhost` (local development)
- ❌ `http://` (non-localhost HTTP)

Since our demo is deployed at `http://ai.bygpu.com:55305/sam2/` without HTTPS, we need to configure browsers properly.

---

## 🌐 Browser Comparison

| Browser | Clears Cache? | Your Data Safe? | Setup Difficulty | Recommended For |
|---------|---------------|-----------------|------------------|-----------------|
| **Safari** 🦁 | ❌ No | ✅ Yes | Easy (auto-config) | **Daily use - Best choice!** |
| **Firefox** 🦊 | ❌ No | ✅ Yes | Easy (separate profile) | **Alternative option** |
| **Chrome** 🔵 | ⚠️ **Yes** | ⚠️ Isolated | Easy (temp profile) | Already have clean browser |

### 🎯 Recommendation:
- **Use Safari or Firefox** if you don't want to affect your Chrome cache/data
- **Use Chrome** only if you don't mind temporary cache clearing

---

## Available Scripts

### 🦁 macOS Safari: `start-sam2-safari-macos.sh` ⭐ RECOMMENDED

**✅ Advantages:**
- **Does NOT clear your cache or browsing data**
- Uses your regular Safari (no separate instance)
- Auto-configures developer settings
- Lightweight and fast

**How to use:**
```bash
# Make it executable (first time only)
chmod +x start-sam2-safari-macos.sh

# Run the script
./start-sam2-safari-macos.sh
```

**What it does:**
1. Enables Safari Develop Menu automatically
2. Configures HTTP development settings
3. Opens SAM2 demo in Safari
4. Shows manual configuration steps

**After Safari opens, you need to:**
1. Click **Develop** menu → **Disable Cross-Origin Restrictions**
2. Reload the page if needed

---

### 🦊 macOS Firefox: `start-sam2-firefox-macos.sh`

**✅ Advantages:**
- **Does NOT affect your regular Firefox data**
- Uses a separate profile (isolated from daily browsing)
- Keeps your regular Firefox untouched
- Auto-configures all settings

**How to use:**
```bash
# Make it executable (first time only)
chmod +x start-sam2-firefox-macos.sh

# Run the script
./start-sam2-firefox-macos.sh
```

**What it does:**
1. Creates a dedicated Firefox profile for SAM2
2. Configures media permissions automatically
3. Launches Firefox with the SAM2 URL
4. Keeps your regular Firefox browsing data safe

---

### 🔵 macOS Chrome: `start-sam2-chrome-macos.sh`

**⚠️ Warning:**
- **WILL clear temporary cache** (uses `/tmp/chrome-sam2`)
- Isolated from your regular Chrome
- Does NOT affect your main Chrome profile data

**How to use:**
```bash
# Make it executable (first time only)
chmod +x start-sam2-chrome-macos.sh

# Run the script
./start-sam2-chrome-macos.sh
```

**What it does:**
1. Closes all existing Chrome instances
2. Cleans up previous session data
3. Launches Chrome with special flags
4. Opens SAM2 demo automatically

---

### 🪟 Windows: `start-sam2-chrome-windows.bat`

**How to use:**
- **Option 1**: Double-click `start-sam2-chrome-windows.bat`
- **Option 2**: Run from Command Prompt:
  ```cmd
  start-sam2-chrome-windows.bat
  ```

**What it does:**
1. Finds Chrome installation (32-bit or 64-bit)
2. Closes all existing Chrome instances
3. Cleans up previous session data
4. Launches Chrome with special flags
5. Opens SAM2 demo automatically

---

## Important Notes

### ⚠️ Security Warning

The Chrome instance launched by these scripts uses flags that **reduce security**:
- `--unsafely-treat-insecure-origin-as-secure`
- `--disable-features=SecureContextCheck`

**DO NOT use this Chrome instance for:**
- ❌ Regular web browsing
- ❌ Online banking
- ❌ Entering passwords or sensitive data
- ❌ Accessing other websites

**ONLY use it for:**
- ✅ SAM2 Demo at `http://ai.bygpu.com:55305/sam2/`

### 📝 How it works

The scripts use a **separate user data directory** (`/tmp/chrome-sam2` or `%TEMP%\chrome-sam2`), so:
- ✅ Your normal Chrome browsing data is safe
- ✅ No cookies/passwords from your normal Chrome
- ✅ Independent settings and state

### ✅ Verify it's working

After Chrome opens, press **F12** to open Developer Console and run:
```javascript
console.log('VideoEncoder' in window);  // Should return: true
console.log(window.isSecureContext);     // Should return: true
```

If both return `true`, WebCodecs is working!

---

## Troubleshooting

### Problem: "Chrome is not installed"
**Solution**: Install Google Chrome from https://www.google.com/chrome/

### Problem: Still shows "Browser not supported"
**Solution**:
1. Make sure ALL Chrome windows are closed before running the script
2. Check Windows Task Manager (Ctrl+Shift+Esc) for `chrome.exe` processes
3. Manually kill all Chrome processes
4. Run the script again

### Problem: Script doesn't work on macOS
**Solution**: Make sure the script is executable:
```bash
chmod +x start-sam2-chrome-macos.sh
```

### Problem: "Access Denied" on Windows
**Solution**: Right-click the `.bat` file and select "Run as Administrator"

---

## Alternative Solutions

If you prefer not to use these scripts, you can:

### 1. SSH Tunnel (Recommended for developers)
```bash
ssh -L 7262:localhost:7262 -L 7263:localhost:7263 user@ai.bygpu.com
# Then visit: http://localhost:7262/sam2/
```

### 2. Configure HTTPS (Recommended for production)
- Use Cloudflare Tunnel (free)
- Use Let's Encrypt + Certbot
- Use Nginx with SSL certificate

---

## Files in this folder

```
scripts/
├── README.md                          # This file
├── start-sam2-safari-macos.sh        # ⭐ Safari launcher (recommended)
├── start-sam2-firefox-macos.sh       # Firefox launcher (alternative)
├── start-sam2-chrome-macos.sh        # Chrome launcher (clears cache)
└── start-sam2-chrome-windows.bat     # Windows Chrome launcher
```

---

## Support

If you encounter any issues, please check:
1. Chrome is properly installed
2. All Chrome windows are closed before running the script
3. You have proper permissions to execute the script

For more information about the SAM2 demo, visit:
https://github.com/facebookresearch/sam2
