# Deployment & Testing Plan
## PhishNet Ensemble System

*Last Updated: 2025-12-30*

---

## Current System State

### Trained Models ✅
- **Location**: `models/` directory
- **Total Models**: 45 models (15 per feature type)
- **Format**: `.pkl` files (joblib serialization)

**URL Models** (15):
- rf, extratrees, gb, histgb, xgb, lgbm, catboost
- logreg_l2, logreg_elasticnet, sgd_log, linear_svm_cal
- cnb, knn, mlp, svm_rbf

**DNS Models** (15):
- Same 15 model types as URL

**WHOIS Models** (15):
- Same 15 model types as URL

### Data Infrastructure ✅
- **Main Datasets**:
  - `data/phishing_features_complete_url.csv`
  - `data/phishing_features_complete_dns.csv`
  - `data/phishing_features_complete_whois.csv`

- **Model-Ready Datasets** (for testing):
  - `data/processed/url_features_modelready_imputed.csv`
  - `data/processed/dns_features_modelready_imputed.csv`
  - `data/processed/whois_features_modelready_imputed.csv`

### API Infrastructure ✅
- **Main API**: `src/api/app.py` (FastAPI)
- **Prediction Utils**: `src/api/predict_utils.py`
- **Parallel Prediction**: `src/api/parallel_predict.py`
- **Model Loader**: `src/api/model_loader.py`
- **A/B Testing Framework**: `src/api/ab_testing.py`
- **Caching**: `src/api/cache.py`

### Ensemble Testing Scripts ✅
1. **`scripts/ensemble_comparison.py`**
   - Tests 7 predefined ensemble strategies
   - Loads models from `models/` directory
   - Benchmarks latency and accuracy
   - Generates comparison reports

2. **`scripts/test_ensemble_combinations.py`**
   - Tests all combinations of top 3 models per feature type
   - More exhaustive than ensemble_comparison.py
   - Saves best ensemble to production metadata

3. **`scripts/simple_ensemble_test.py`**
   - Quick benchmark using existing API functions
   - Good for rapid testing

---

## What Needs to be Done

### 1. Fix Ensemble Testing Scripts

**Problem**: Scripts expect models in subdirectories (`models/url/*.pkl`) but they're stored flat (`models/*.pkl`)

**Solution**: Update ensemble_comparison.py model loading logic

**File to modify**: `scripts/ensemble_comparison.py`

```python
# Current (line 184-195):
models_dir = Path("models") / feature_type
for model_file in models_dir.glob("*.pkl"):
    model_name = model_file.stem
    models[model_name] = joblib.load(model_file)

# Should be:
models_dir = Path("models")
for model_file in models_dir.glob(f"{feature_type}_*.pkl"):
    model_name = model_file.stem.replace(f"{feature_type}_", "")
    models[model_name] = joblib.load(model_file)
```

### 2. Run Comprehensive Ensemble Testing

Once fixed, run all three testing approaches:

#### A. Quick Test (5 minutes)
```bash
python3 scripts/simple_ensemble_test.py
```
- Tests URL-only, DNS-only, WHOIS-only, and current ensemble
- Fast benchmark on 200 URLs
- Good for sanity check

#### B. Predefined Strategies (15 minutes)
```bash
python3 scripts/ensemble_comparison.py --test-size 1000 --iterations 100
```
- Tests 7 ensemble strategies (E1-E7)
- Comprehensive latency benchmarking
- Saves results to `analysis/ensemble_comparison/`

#### C. Exhaustive Search (30-60 minutes)
```bash
python3 scripts/test_ensemble_combinations.py
```
- Tests all combinations of top models
- Tests different weight configurations
- Updates `models/production_metadata.json` with best ensemble

### 3. Analyze Results

Expected outputs in `analysis/ensemble_comparison/`:
- `comparison_YYYYMMDD_HHMMSS.json` - Full metrics
- `comparison_YYYYMMDD_HHMMSS.csv` - Tabular view
- `contributions_YYYYMMDD_HHMMSS.json` - Model contribution analysis

**Key Metrics to Compare**:
1. **Accuracy**: Minimize false negatives (missing phishing)
2. **False Positive Rate**: Minimize false positives (blocking legit sites)
3. **F1 Score**: Balance between precision and recall
4. **Latency (p95)**: 95th percentile response time
5. **Composite Score**: Weighted combination of above

**Decision Criteria**:
- If latency < 100ms: Prioritize accuracy (choose highest F1)
- If latency 100-200ms: Balance accuracy and speed
- If latency > 200ms: This is too slow for production - choose faster models

### 4. Select Production Ensemble

Based on test results, select ensemble configuration:

**Format**:
```json
{
  "ensemble": {
    "models": {
      "url": "url_catboost",
      "dns": "dns_lgbm",
      "whois": "whois_xgb"
    },
    "weights": {
      "url": 0.50,
      "dns": 0.25,
      "whois": 0.25
    },
    "performance": {
      "accuracy": 0.9965,
      "f1": 0.9958,
      "fpr": 0.0012,
      "avg_latency_ms": 87.3,
      "composite_score": 0.9912
    }
  }
}
```

This will be saved to `models/production_metadata.json`

### 5. Deploy to Production API

#### Option A: Direct Deployment (No A/B Testing)
Update `src/api/predict_utils.py` to load ensemble from metadata:

```python
def load_production_ensemble():
    """Load production ensemble configuration"""
    with open("models/production_metadata.json", "r") as f:
        metadata = json.load(f)

    ensemble_config = metadata["ensemble"]

    return {
        "url_model": joblib.load(f"models/{ensemble_config['models']['url']}.pkl"),
        "dns_model": joblib.load(f"models/{ensemble_config['models']['dns']}.pkl"),
        "whois_model": joblib.load(f"models/{ensemble_config['models']['whois']}.pkl"),
        "weights": ensemble_config["weights"]
    }
```

#### Option B: A/B Testing (Recommended)
Use existing `src/api/ab_testing.py`:

```python
# In src/api/app.py startup
from src.api.ab_testing import ABTestManager

ab_manager = ABTestManager()

# Configure A/B test: 90% current, 10% new ensemble
ab_manager.add_variant("control", current_ensemble_config, traffic_pct=90)
ab_manager.add_variant("challenger", new_ensemble_config, traffic_pct=10)

@app.post("/predict/ab")
async def predict_with_ab(url: str):
    return ab_manager.route_request(url)

@app.get("/ab/metrics")
async def get_ab_results():
    return ab_manager.get_variant_metrics()
```

**A/B Test Duration**: Run for 7 days or 10,000 requests (whichever comes first)

**Rollout Plan**:
1. Day 1-3: 90% control, 10% challenger
2. Day 4-5: 75% control, 25% challenger (if metrics good)
3. Day 6-7: 50% control, 50% challenger (if metrics good)
4. Day 8+: 100% champion (best performer)

### 6. Monitor Production Performance

**Metrics to Track**:
```python
{
  "ensemble_id": "E6_production",
  "timestamp": "2025-12-30T14:00:00",

  # Accuracy Metrics (from user feedback / labeled data)
  "accuracy": 0.9965,
  "false_positives": 12,
  "false_negatives": 3,
  "total_predictions": 10000,

  # Latency Metrics (from API logs)
  "latency_p50_ms": 42.1,
  "latency_p95_ms": 87.3,
  "latency_p99_ms": 156.8,
  "latency_avg_ms": 55.2,

  # Resource Metrics
  "requests_per_minute": 150,
  "cache_hit_rate": 0.67,
  "feature_extraction_failures": 5,

  # Model-Specific
  "url_model_calls": 10000,
  "dns_model_calls": 9500,  # 500 DNS failures
  "whois_model_calls": 8200  # 1800 WHOIS failures
}
```

**Alert Thresholds**:
- FPR > 2%: Warning
- FPR > 5%: Critical - consider rollback
- Accuracy < 95%: Warning
- Accuracy < 90%: Critical - rollback
- p95 Latency > 250ms: Warning
- p95 Latency > 500ms: Critical

---

## Testing Checklist

Before deploying to production:

- [ ] Fix ensemble_comparison.py model loading paths
- [ ] Run quick test (simple_ensemble_test.py)
- [ ] Run comprehensive test (ensemble_comparison.py)
- [ ] Run exhaustive search (test_ensemble_combinations.py)
- [ ] Analyze results and select best ensemble
- [ ] Document selected ensemble in production metadata
- [ ] Test selected ensemble with sample URLs
- [ ] Verify latency meets requirements (<200ms p95)
- [ ] Set up A/B test configuration
- [ ] Deploy A/B test to production
- [ ] Monitor for 7 days
- [ ] Analyze A/B test results
- [ ] Rollout winning ensemble to 100%
- [ ] Archive old ensemble for rollback capability

---

## Expected Timeline

**Week 1: Testing & Selection**
- Day 1: Fix scripts, run tests
- Day 2: Analyze results, select ensemble
- Day 3: Validate selection with additional testing
- Day 4-5: Prepare deployment configuration
- Day 6-7: Set up monitoring and alerts

**Week 2: Deployment**
- Day 1-2: Deploy A/B test (90/10 split)
- Day 3-4: Monitor results, adjust traffic if needed
- Day 5-6: Gradual rollout (25% → 50%)
- Day 7: Full rollout (100%) if metrics good

**Week 3: Optimization**
- Monitor production performance
- Fine-tune based on real-world data
- Document lessons learned
- Plan next iteration

---

## Rollback Plan

If new ensemble performs worse in production:

1. **Automatic Rollback Triggers**:
   - FPR > 5% after 1000 requests
   - 3 consecutive API errors
   - p95 latency > 500ms for 5 minutes

2. **Manual Rollback**:
   ```bash
   # Revert to previous ensemble
   cp models/production_metadata_backup.json models/production_metadata.json

   # Restart API
   pkill -f "uvicorn src.api.app"
   uvicorn src.api.app:app --reload
   ```

3. **Gradual Rollback**:
   - Reduce challenger traffic: 10% → 5% → 0%
   - Keep control at 100%

---

## Success Criteria

Deployment is successful if:

✅ **Accuracy**: >= 96% (better than current URL-only)
✅ **FPR**: <= 1% (low false positives)
✅ **Latency (p95)**: <= 150ms (acceptable for production)
✅ **Composite Score**: >= 0.99 (best overall balance)
✅ **Stability**: No crashes or errors in 7 days
✅ **User Satisfaction**: No complaints about false positives/negatives

---

## Next Steps After Deployment

1. **Collect Production Data**:
   - User feedback on predictions
   - False positive/negative reports
   - Feature extraction failure patterns

2. **Retrain with Production Data**:
   - Weekly retraining includes production feedback
   - Continuous improvement cycle

3. **Explore Advanced Ensembles**:
   - Stacking (meta-learner on top of base models)
   - Blending (weighted combination of model types)
   - Cascading (fast model first, complex model if uncertain)

4. **Optimize Latency Further**:
   - Model quantization/compression
   - Batch prediction for bulk requests
   - GPU acceleration for deep models
   - Feature caching improvements

---

*For detailed model information, see [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)*
*For MLOps pipeline details, see [AUTOMATED_MLOPS_PIPELINE.md](AUTOMATED_MLOPS_PIPELINE.md)*
