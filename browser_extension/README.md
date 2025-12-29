# PhishNet Browser Extension

Real-time phishing protection for Chrome/Edge browsers.

## Installation (Development Mode)

### Chrome/Edge:

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `browser_extension` folder
5. The PhishNet icon will appear in your toolbar

### Usage:

1. **Make sure PhishNet API is running:**
   ```bash
   cd /Users/eeshanbhanap/Desktop/PDF
   docker-compose up
   # OR
   uvicorn src.api.app:app --host 0.0.0.0 --port 8000
   ```

2. **Visit any website**

3. **Click the PhishNet extension icon**
   - It will automatically check the current page
   - Shows ✅ SAFE or 🚨 PHISHING DETECTED
   - Displays risk score and recommendations

## Features

- ✅ One-click URL checking
- ✅ Real-time phishing detection
- ✅ Visual risk indicators
- ✅ Actionable safety recommendations
- ✅ Works on any website

## How It Works

The extension:
1. Gets the URL of your current tab
2. Sends it to PhishNet API (localhost:8000)
3. Displays the result in a clean popup
4. Shows risk score and safety recommendations

## Screenshots

### Safe Website:
```
✅ SAFE
Risk Score: 0.2%
This website appears to be legitimate
```

### Phishing Website:
```
🚨 PHISHING DETECTED
Risk Score: 98.7%
❌ DO NOT enter passwords or personal info
❌ DO NOT download anything
✅ Close this page immediately
```

## For Distribution (Production):

To publish to Chrome Web Store:
1. Add real icons (replace placeholder icons/)
2. Test thoroughly
3. Create privacy policy
4. Update manifest with production API URL
5. Submit to Chrome Web Store

## Troubleshooting

**Extension shows "Error":**
- Make sure API is running at http://localhost:8000
- Check: `curl http://localhost:8000/health`

**Extension not loading:**
- Check Developer mode is enabled
- Reload extension after code changes
- Check console for errors (chrome://extensions/ → Details → Inspect views)
