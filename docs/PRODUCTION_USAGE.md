# Production Usage Guide

Quick guide for using PhishNet in production with feature alignment.

---

## Quick Start

### 1. Predict Single URL

```python
from src.features.url_features import extract_single_url_features
from src.api.feature_aligner import predict_url_aligned

url = "https://suspicious-site.com"
url_features = extract_single_url_features(url)
phish_prob = predict_url_aligned(url_features, model_name="catboost")

print(f"Phishing probability: {phish_prob:.2%}")
```

### 2. Predict with URL + DNS Ensemble

```python
from src.features.url_features import extract_single_url_features
from src.features.dns_ipwhois import extract_single_domain_features
from src.api.feature_aligner import predict_url_aligned, predict_dns_aligned

url = "https://suspicious-site.com"

# Extract features
url_features = extract_single_url_features(url)
dns_features = extract_single_domain_features(url)

# Get predictions
url_prob = predict_url_aligned(url_features)
dns_prob = predict_dns_aligned(dns_features)

# Ensemble (70% URL, 30% DNS)
ensemble_prob = 0.7 * url_prob + 0.3 * dns_prob
verdict = "PHISHING" if ensemble_prob > 0.5 else "LEGITIMATE"

print(f"Verdict: {verdict} ({ensemble_prob:.2%})")
```

### 3. Full 3-Way Ensemble (URL + DNS + WHOIS)

```python
from src.features.url_features import extract_single_url_features
from src.features.dns_ipwhois import extract_single_domain_features
from src.features.whois import extract_single_whois_features
from src.api.feature_aligner import predict_ensemble_aligned

url = "https://suspicious-site.com"

# Extract all features
url_features = extract_single_url_features(url)
dns_features = extract_single_domain_features(url)
whois_features = extract_single_whois_features(url, live_lookup=False)

# Get ensemble prediction with auto-alignment
result = predict_ensemble_aligned(
    url_features,
    dns_features,
    whois_features,
    weights={"url": 0.5, "dns": 0.2, "whois": 0.3},
    model_names={"url": "catboost", "dns": "catboost", "whois": "catboost"}
)

print(result)
# {
#   "url_probability": 0.03,
#   "dns_probability": 0.97,
#   "whois_probability": 0.45,
#   "ensemble_probability": 0.32,
#   "verdict": "LEGITIMATE",
#   "confidence": 0.36
# }
```

---

## FastAPI Integration

Update `src/api/app.py` to use feature alignment:

```python
from fastapi import FastAPI
from src.features.url_features import extract_single_url_features
from src.features.dns_ipwhois import extract_single_domain_features
from src.api.feature_aligner import predict_ensemble_aligned

app = FastAPI()

@app.post("/predict")
async def predict_url(url: str):
    """Predict if URL is phishing with feature alignment."""
    try:
        # Extract features
        url_features = extract_single_url_features(url)
        dns_features = extract_single_domain_features(url)

        # For now, use URL-only (DNS has limited training data)
        url_prob = predict_url_aligned(url_features)

        return {
            "url": url,
            "phishing_probability": url_prob,
            "verdict": "PHISHING" if url_prob > 0.5 else "LEGITIMATE",
            "confidence": abs(url_prob - 0.5) * 2
        }
    except Exception as e:
        return {"error": str(e)}
```

---

## Model Selection

### Available Models

For each feature type (url, dns, whois):
- `catboost` - Best accuracy (recommended)
- `xgb` - Fast and accurate
- `lgbm` - Fastest inference
- `rf` - Good baseline
- `logreg_elasticnet` - Most interpretable

### Usage

```python
# Use XGBoost instead of CatBoost
phish_prob = predict_url_aligned(url_features, model_name="xgb")

# Or specify different models for ensemble
result = predict_ensemble_aligned(
    url_features, dns_features, whois_features,
    model_names={"url": "catboost", "dns": "xgb", "whois": "lgbm"}
)
```

---

## Weekly Retraining

Models automatically retrain every Sunday 3 AM UTC via GitHub Actions.

**What happens**:
1. Download new data from GCS
2. Merge with existing datasets
3. Check for data growth (10% threshold)
4. If growth detected: retrain all 45 models
5. Save models + feature column order files
6. Commit to repository

**Manual trigger**:
```bash
gh workflow run weekly_model_retrain.yml
```

---

## Deployment Checklist

Before deploying to production:

- [ ] Verify feature alignment works: `python3 /tmp/e2e_pipeline_test.py`
- [ ] Test prediction on new URL
- [ ] Check model files exist: `ls models/*_catboost*`
- [ ] Verify feature order files exist: `ls models/*_feature_cols.pkl`
- [ ] Test FastAPI endpoint locally
- [ ] Set up monitoring/logging
- [ ] Configure error alerts

---

## Troubleshooting

### "Feature column order file not found"
**Fix**: Retrain models to generate feature order files
```bash
PYTHONPATH=/Users/eeshanbhanap/Desktop/PDF python3 src/training/url_train.py
```

### "Column mismatch" errors
**Fix**: Use `feature_aligner.py` instead of loading models directly

### Low accuracy on new URLs
**Expected**: Models trained on small dataset (1000 URLs)
**Solution**: Let data accumulate, weekly retraining will improve models

### DNS/WHOIS features missing
**Expected**: Only 20 samples with DNS/WHOIS features
**Solution**: Use URL-only model until more data collected

---

## Monitoring

Track these metrics in production:

```python
# Log every prediction
{
    "url": "https://example.com",
    "timestamp": "2025-12-30T12:00:00Z",
    "phishing_probability": 0.03,
    "verdict": "LEGITIMATE",
    "latency_ms": 12.5,
    "model_version": "20251230",
    "user_feedback": null  # Collect if available
}
```

**Key Metrics**:
- Prediction latency (target: <100ms)
- Accuracy on labeled URLs
- False positive rate (target: <1%)
- False negative rate (target: <5%)

---

## Next Steps

1. **Deploy** URL-only model first (proven 80% accuracy)
2. **Collect** production predictions and user feedback
3. **Wait** for DNS/WHOIS data to accumulate (daily collection running)
4. **Retrain** weekly to incorporate new data
5. **Test** URL+DNS+WHOIS ensemble once enough data available
6. **Optimize** weights based on production performance

---

*For technical details, see [E2E_VERIFICATION_RESULTS.md](E2E_VERIFICATION_RESULTS.md)*
*For model information, see [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)*
