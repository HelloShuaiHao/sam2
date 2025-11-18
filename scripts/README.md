# SAM2 Demo - Chrome Launcher Scripts

This folder contains scripts to launch Google Chrome with special flags that allow the SAM2 demo to work over HTTP (without HTTPS).

## Why are these scripts needed?

The SAM2 demo uses **WebCodecs API** (VideoEncoder, VideoDecoder, VideoFrame) which requires a **Secure Context**:
- ✅ `https://` (any domain with SSL)
- ✅ `http://localhost` (local development)
- ❌ `http://` (non-localhost HTTP)

Since our demo is deployed at `http://ai.bygpu.com:55305/sam2/` without HTTPS, we need to tell Chrome to treat it as secure.

## Available Scripts

### 🍎 macOS: `start-sam2-chrome-macos.sh`

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
├── start-sam2-chrome-macos.sh        # macOS launcher script
└── start-sam2-chrome-windows.bat     # Windows launcher script
```

---

## Support

If you encounter any issues, please check:
1. Chrome is properly installed
2. All Chrome windows are closed before running the script
3. You have proper permissions to execute the script

For more information about the SAM2 demo, visit:
https://github.com/facebookresearch/sam2
