# API Usage Examples

This directory contains examples of how to call the Phishing Detection API in different programming languages and frameworks.

## Starting the API Server

First, make sure your API is running:

```bash
cd /Users/eeshanbhanap/Desktop/PDF
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

Interactive docs (Swagger UI): `http://localhost:8000/docs`

## Available Endpoints

| Endpoint | Method | Description | Speed | Accuracy |
|----------|--------|-------------|-------|----------|
| `/health` | GET | Health check | Instant | N/A |
| `/predict/url` | POST | URL features only | ~10-50ms | 90% |
| `/predict/whois` | POST | WHOIS features only | ~1-10s | 90% |
| `/predict/ensemble` | POST | URL + WHOIS combined | ~1-10s | **Best** |

## Request/Response Format

**Request:**
```json
{
  "url": "https://example.com"
}
```

**Response:**
```json
{
  "url": "https://example.com",
  "risk_score": 0.08,
  "is_phishing": false,
  "threshold": 0.5,
  "model_name": "ensemble_url+whois_v1",
  "latency_ms": 1234.5,
  "debug": {
    "url_prediction": {
      "risk_score": 0.05,
      "latency_ms": 42.3,
      "model": "catboost_url_v1",
      "weight": 0.6
    },
    "whois_prediction": {
      "risk_score": 0.12,
      "latency_ms": 1192.2,
      "model": "catboost_whois_v1",
      "weight": 0.4
    },
    "ensemble": {
      "weighted_score": 0.08,
      "formula": "0.6*0.0500 + 0.4*0.1200"
    }
  }
}
```

## Examples

### 1. cURL (Command Line)

**Quick test:**
```bash
curl -X POST "http://localhost:8000/predict/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.google.com"}'
```

**Run all examples:**
```bash
chmod +x api_examples.sh
./api_examples.sh
```

### 2. Python

**Install requirements:**
```bash
pip install requests
```

**Run example:**
```bash
python3 api_client.py
```

**Use in your code:**
```python
from api_client import PhishingDetectionClient

client = PhishingDetectionClient("http://localhost:8000")

# Quick check
result = client.predict_url("https://example.com")
print(f"Risk: {result['risk_score']:.1%}")

# Best accuracy (slower)
result = client.predict_ensemble("https://example.com")
print(f"Risk: {result['risk_score']:.1%}")

# Simple boolean check
is_safe = client.check_url_safe("https://example.com")
print(f"Safe: {is_safe}")
```

### 3. JavaScript/Node.js

**Run example:**
```bash
node api_client.js
```

**Use in your code:**
```javascript
const PhishingDetectionClient = require('./api_client.js');

const client = new PhishingDetectionClient('http://localhost:8000');

// Quick check
const result = await client.predictURL('https://example.com');
console.log(`Risk: ${(result.risk_score * 100).toFixed(1)}%`);

// Best accuracy (slower)
const ensembleResult = await client.predictEnsemble('https://example.com');
console.log(`Risk: ${(ensembleResult.risk_score * 100).toFixed(1)}%`);

// Simple boolean check
const isSafe = await client.isURLSafe('https://example.com');
console.log(`Safe: ${isSafe}`);
```

### 4. React Component

**Install in your React app:**
```bash
# Copy PhishingChecker.jsx to your src/components/
cp PhishingChecker.jsx /path/to/your/react-app/src/components/
```

**Use in your app:**
```jsx
import PhishingChecker from './components/PhishingChecker';

function App() {
  return (
    <div className="App">
      <PhishingChecker />
    </div>
  );
}
```

## Production Deployment

When deploying to production:

### 1. Add Authentication

Update [app.py](../src/api/app.py):
```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header()):
    if x_api_key != "your-secret-api-key":
        raise HTTPException(status_code=403, detail="Invalid API key")

@app.post("/predict/ensemble", dependencies=[Depends(verify_api_key)])
def predict_ensemble(request: EnsemblePredictRequest):
    # ... existing code
```

### 2. Deploy to Cloud

**Option A: Docker + AWS/GCP/Azure**
```bash
# Build Docker image
docker build -t phishing-api .

# Run container
docker run -p 8000:8000 phishing-api

# Deploy to cloud (AWS ECS, GCP Cloud Run, Azure Container Instances)
```

**Option B: Heroku**
```bash
heroku create phishing-detection-api
git push heroku main
```

**Option C: Railway/Render**
- Connect GitHub repo
- Auto-deploy on push

### 3. Update Client URLs

```python
# Development
client = PhishingDetectionClient("http://localhost:8000")

# Production
client = PhishingDetectionClient(
    "https://api.yourcompany.com",
    api_key="your-secret-key"
)
```

## Rate Limiting

For production, add rate limiting:

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict/ensemble")
@limiter.limit("10/minute")  # 10 requests per minute
def predict_ensemble(request: Request, data: EnsemblePredictRequest):
    # ... existing code
```

## Monitoring

Add logging and monitoring:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@app.post("/predict/ensemble")
def predict_ensemble(request: EnsemblePredictRequest):
    logging.info(f"Ensemble prediction for {request.url}")
    # ... existing code
```

## Browser Extension Integration

You can integrate this API into a browser extension:

```javascript
// background.js (Chrome Extension)
chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
  const response = await fetch('http://localhost:8000/predict/url', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url: details.url })
  });

  const result = await response.json();

  if (result.is_phishing) {
    // Show warning popup
    chrome.tabs.update(details.tabId, {
      url: 'warning.html?url=' + encodeURIComponent(details.url)
    });
  }
});
```

## Testing URLs

**Legitimate URLs (should show low risk):**
- https://www.google.com
- https://github.com/microsoft/vscode
- https://stackoverflow.com/questions/11227809
- https://www.oracle.com
- https://docs.python.org

**Phishing URLs (should show high risk):**
- https://paypal-verify-account.com/login
- https://microsoft-security-alert.com/update
- https://appleid-unlock.com/verify
- https://amazon-prize-winner.com/claim

## Support

For issues or questions:
- Check Swagger UI: `http://localhost:8000/docs`
- Check API logs: Look at terminal where uvicorn is running
- Test with cURL first to isolate client vs server issues
