# FastAPI CORS Configuration for Web Demo

This guide explains how to configure CORS (Cross-Origin Resource Sharing) in your FastAPI backend to allow the web demo to call your API.

---

## 🎯 Why CORS is Needed

The web demo is hosted on GitHub Pages (e.g., `https://username.github.io`), but your FastAPI backend runs on a different domain (e.g., `http://localhost:8000` or `https://your-api.com`).

Browsers block requests between different origins by default for security. CORS configuration tells the browser: "It's okay for this web page to call this API."

---

## ✅ Quick Setup

### 1. Install FastAPI CORS Middleware

CORS middleware is built into FastAPI, no installation needed!

### 2. Add CORS to Your FastAPI App

**File**: `main.py` (or wherever you create your FastAPI app)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS Configuration
# IMPORTANT: Update allowed origins based on your deployment
origins = [
    "http://localhost:3000",              # Local development
    "http://localhost:8080",              # Alternative local port
    "https://YOUR_USERNAME.github.io",    # GitHub Pages (UPDATE THIS!)
]

# For development only - allows all origins
# NEVER use this in production!
# origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,               # Which websites can call this API
    allow_credentials=True,              # Allow cookies/auth headers
    allow_methods=["*"],                 # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],                 # Allow all headers
)

# Your API routes here
@app.post("/api/check")
async def check_url(request: dict):
    url = request.get("url")
    # Your phishing detection logic here
    return {
        "url": url,
        "prediction": "safe",
        "confidence": 0.95,
        "model_scores": {
            "url_model": 0.92,
            "dns_model": 0.96,
            "whois_model": 0.94,
            "ensemble_model": 0.95
        },
        "features": {
            "has_https": True,
            "has_ip": False,
            "dns_record": True,
            "domain_age": 365
        }
    }

@app.get("/api/stats")
async def get_stats():
    # Return system statistics
    return {
        "total_urls": 50000,
        "accuracy": 96.5,
        "last_updated": "2025-12-29T00:00:00Z"
    }
```

### 3. Update GitHub Pages URL

**IMPORTANT**: After deploying to GitHub Pages, update the `origins` list with your actual GitHub Pages URL:

```python
origins = [
    "https://YOUR_USERNAME.github.io",  # Replace YOUR_USERNAME with your GitHub username
]
```

For example, if your username is `eeshanbhanap`:
```python
origins = [
    "https://eeshanbhanap.github.io",
]
```

---

## 🔒 Security Best Practices

### Development vs Production

**Development** (on your local machine):
```python
# Allow localhost for testing
origins = [
    "http://localhost:3000",
    "http://localhost:8080",
]
```

**Production** (deployed to server):
```python
# ONLY allow your GitHub Pages URL
origins = [
    "https://YOUR_USERNAME.github.io",
]

# Or use environment variable
import os
origins = [
    os.getenv("ALLOWED_ORIGIN", "https://YOUR_USERNAME.github.io")
]
```

### Never Do This in Production

```python
# ❌ BAD - Allows ANY website to call your API
origins = ["*"]
```

This is a security risk! Only use `["*"]` for local testing.

---

## 🧪 Testing CORS Configuration

### 1. Start Your FastAPI Backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Test from Browser Console

Open your web demo page and run this in the browser console:

```javascript
fetch('http://localhost:8000/api/check', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url: 'https://google.com' })
})
.then(r => r.json())
.then(data => console.log('Success:', data))
.catch(err => console.error('Error:', err));
```

**If CORS is working**: You'll see the response data

**If CORS is NOT working**: You'll see an error like:
```
Access to fetch at 'http://localhost:8000/api/check' from origin 'https://username.github.io'
has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present
```

---

## 🌐 Deployment Scenarios

### Scenario 1: Local Testing

**FastAPI**: `http://localhost:8000`
**Web Demo**: Open `web/demo/index.html` directly in browser

```python
# main.py
origins = [
    "null",  # For file:// protocol (local HTML files)
    "http://localhost:3000",
]
```

```javascript
// web/demo/index.html
const API_BASE_URL = 'http://localhost:8000';
```

### Scenario 2: GitHub Pages + Local FastAPI

**FastAPI**: `http://localhost:8000`
**Web Demo**: `https://username.github.io/repo-name`

```python
# main.py
origins = [
    "https://username.github.io",
]
```

```javascript
// web/demo/index.html
const API_BASE_URL = 'http://localhost:8000';
```

**Note**: This won't work from GitHub Pages because browsers block HTTP requests from HTTPS pages. You'll need to use `ngrok` or deploy your API to HTTPS.

### Scenario 3: GitHub Pages + Cloud FastAPI

**FastAPI**: `https://your-api.com` (deployed to GCP/AWS/Heroku)
**Web Demo**: `https://username.github.io/repo-name`

```python
# main.py
origins = [
    "https://username.github.io",
]
```

```javascript
// web/demo/index.html
const API_BASE_URL = 'https://your-api.com';
```

This is the **recommended production setup**.

---

## 🔧 Using ngrok for Local HTTPS Testing

If you want to test the GitHub Pages demo with your local FastAPI backend:

### 1. Install ngrok

```bash
# macOS
brew install ngrok

# Or download from https://ngrok.com/download
```

### 2. Start FastAPI Normally

```bash
uvicorn main:app --reload --port 8000
```

### 3. Create HTTPS Tunnel

```bash
ngrok http 8000
```

You'll get an HTTPS URL like: `https://abc123.ngrok.io`

### 4. Update CORS and Web Demo

```python
# main.py
origins = [
    "https://username.github.io",
    "https://abc123.ngrok.io",  # ngrok URL
]
```

```javascript
// web/demo/index.html
const API_BASE_URL = 'https://abc123.ngrok.io';
```

Now your GitHub Pages demo can call your local API via HTTPS!

---

## 📊 API Response Format

Your FastAPI endpoints should return this format for the web demo to work correctly:

### `/api/check` (POST)

**Request**:
```json
{
  "url": "https://example.com"
}
```

**Response**:
```json
{
  "url": "https://example.com",
  "prediction": "safe",  // or "phishing"
  "confidence": 0.95,
  "model_scores": {
    "url_model": 0.92,
    "dns_model": 0.96,
    "whois_model": 0.94,
    "ensemble_model": 0.95
  },
  "features": {
    "has_https": true,
    "has_ip": false,
    "has_at_symbol": false,
    "long_url": false,
    "short_url": false,
    "dns_record": true,
    "suspicious_tld": false,
    "subdomain_count": 1,
    "special_chars": 2,
    "domain_age": 365
  }
}
```

### `/api/stats` (GET)

**Response**:
```json
{
  "total_urls": 50000,
  "accuracy": 96.5,
  "last_updated": "2025-12-29T00:00:00Z"
}
```

---

## 🐛 Common Issues

### Issue 1: CORS Error in Console

**Error**: `No 'Access-Control-Allow-Origin' header is present`

**Solution**:
- Make sure CORS middleware is added to FastAPI
- Check that the GitHub Pages URL is in the `origins` list
- Restart FastAPI after adding CORS

### Issue 2: Mixed Content (HTTP/HTTPS)

**Error**: `Mixed Content: The page at 'https://...' was loaded over HTTPS, but requested an insecure resource 'http://...'`

**Solution**:
- Deploy FastAPI to HTTPS endpoint
- Or use ngrok for local testing
- GitHub Pages is always HTTPS, so API must be HTTPS too

### Issue 3: API Returns 404

**Error**: Response status 404

**Solution**:
- Check that `/api/check` and `/api/stats` routes exist in FastAPI
- Verify `API_BASE_URL` in `web/demo/index.html` is correct
- Test API directly: `curl http://localhost:8000/api/stats`

### Issue 4: Preflight Request Failed

**Error**: `Response to preflight request doesn't pass access control check`

**Solution**:
- Make sure `allow_methods=["*"]` is set in CORS middleware
- Check that `allow_headers=["*"]` is set
- This usually happens with POST requests

---

## 📝 Complete Example

Here's a complete working example:

**main.py** (FastAPI Backend):
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="PhishNet API")

# CORS - Update with your GitHub Pages URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://eeshanbhanap.github.io",  # UPDATE THIS
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your ML models
# url_model = joblib.load("models/url_model.pkl")
# dns_model = joblib.load("models/dns_model.pkl")
# whois_model = joblib.load("models/whois_model.pkl")
# ensemble_model = joblib.load("models/ensemble_model.pkl")

class URLRequest(BaseModel):
    url: str

@app.post("/api/check")
async def check_url(request: URLRequest):
    url = request.url

    # Extract features (your existing code)
    # features = extract_features(url)

    # Get predictions from models
    # url_pred = url_model.predict_proba([features])[0][1]
    # dns_pred = dns_model.predict_proba([features])[0][1]
    # whois_pred = whois_model.predict_proba([features])[0][1]
    # ensemble_pred = ensemble_model.predict_proba([features])[0][1]

    # For demo purposes (replace with actual predictions)
    return {
        "url": url,
        "prediction": "safe",
        "confidence": 0.95,
        "model_scores": {
            "url_model": 0.92,
            "dns_model": 0.96,
            "whois_model": 0.94,
            "ensemble_model": 0.95
        },
        "features": {
            "has_https": True,
            "has_ip": False,
            "dns_record": True,
            "domain_age": 365
        }
    }

@app.get("/api/stats")
async def get_stats():
    return {
        "total_urls": 50000,
        "accuracy": 96.5,
        "last_updated": "2025-12-29T00:00:00Z"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## ✅ Checklist

Before deploying:

- [ ] CORS middleware added to FastAPI
- [ ] GitHub Pages URL added to `origins` list
- [ ] `API_BASE_URL` updated in `web/demo/index.html`
- [ ] FastAPI backend is running
- [ ] Test API with curl: `curl http://localhost:8000/api/stats`
- [ ] Test from browser console (see Testing section above)
- [ ] Deploy FastAPI to HTTPS endpoint for production
- [ ] Update `API_BASE_URL` to production URL

---

**Author**: Eeshan Bhanap (eb3658@columbia.edu)
**Date**: December 29, 2025
