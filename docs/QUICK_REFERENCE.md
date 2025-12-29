# 🚀 Quick Reference Guide - Multi-Model Ensembles & A/B Testing

## One-Line Commands

### Check System Status
```bash
# MLOps pipeline status
ps aux | grep mlops_pipeline | grep -v grep

# VM collector status
./scripts/vm_manager.sh status

# Check trained models
ls -lh models/*.pkl | wc -l  # Should be ≥12

# View pipeline logs
tail -100 logs/mlops_bg.log | grep -E "(STAGE|✅|❌|Training)"
```

### Run Ensemble Comparison
```bash
# Quick test (100 URLs, 50 iterations)
python3 scripts/ensemble_comparison.py --test-size 100 --iterations 50

# Full test (2000 URLs, 200 iterations)
python3 scripts/ensemble_comparison.py --test-size 2000 --iterations 200

# View latest results
ls -lt analysis/ensemble_comparison/ | head -5
```

### A/B Testing
```bash
# Check A/B test status
python3 src/api/ab_testing.py --status

# Export metrics
python3 src/api/ab_testing.py --export analysis/ab_metrics_$(date +%Y%m%d).json
```

---

## Expected Model Count

After MLOps pipeline runs successfully, you should have:

**12 Primary Models** (4 per feature type):
```
models/url_catboost.pkl
models/url_lgbm.pkl
models/url_xgb.pkl
models/url_rf.pkl

models/dns_catboost.pkl
models/dns_lgbm.pkl
models/dns_xgb.pkl
models/dns_rf.pkl

models/whois_catboost.pkl
models/whois_lgbm.pkl
models/whois_xgb.pkl
models/whois_rf.pkl
```

---

## Ensemble Configurations Cheat Sheet

### E1: URL Only (Fastest)
```python
{
    "models": {"url": "url_catboost.pkl"},
    "weights": {"url": 1.0}
}
```
**Best for**: Ultra-low latency, no API dependencies

### E2: URL + DNS
```python
{
    "models": {
        "url": "url_catboost.pkl",
        "dns": "dns_lgbm.pkl"
    },
    "weights": {"url": 0.70, "dns": 0.30}
}
```
**Best for**: Fast + network signals

### E3: URL + WHOIS (Current Production)
```python
{
    "models": {
        "url": "url_catboost.pkl",
        "whois": "whois_catboost.pkl"
    },
    "weights": {"url": 0.60, "whois": 0.40}
}
```
**Best for**: Current baseline

### E6: All 3 Optimized (Recommended)
```python
{
    "models": {
        "url": "url_catboost.pkl",
        "dns": "dns_xgb.pkl",
        "whois": "whois_lgbm.pkl"
    },
    "weights": {"url": 0.50, "dns": 0.20, "whois": 0.30}
}
```
**Best for**: Maximum accuracy

### E7: All 3 Speed
```python
{
    "models": {
        "url": "url_lgbm.pkl",
        "dns": "dns_lgbm.pkl",
        "whois": "whois_lgbm.pkl"
    },
    "weights": {"url": 0.50, "dns": 0.25, "whois": 0.25}
}
```
**Best for**: Balance speed + accuracy

---

## A/B Test Setup (Copy-Paste)

```python
from src.api.ab_testing import ABTestManager

# Initialize
ab = ABTestManager()

# Add control (current production: URL + WHOIS)
ab.add_variant(
    "control",
    {
        "models": {"url": "url_catboost.pkl", "whois": "whois_catboost.pkl"},
        "weights": {"url": 0.60, "whois": 0.40}
    },
    traffic_pct=90,
    is_control=True
)

# Add variant (new: All 3 Optimized)
ab.add_variant(
    "optimized",
    {
        "models": {
            "url": "url_catboost.pkl",
            "dns": "dns_xgb.pkl",
            "whois": "whois_lgbm.pkl"
        },
        "weights": {"url": 0.50, "dns": 0.20, "whois": 0.30}
    },
    traffic_pct=10,
    is_control=False
)

# Route requests
variant_id, result = ab.route_request("https://test.com")
print(f"Routed to: {variant_id}")

# Check if ready to promote
if ab.should_promote("optimized"):
    ab.promote_variant("optimized")
    print("✅ Promoted!")
```

---

## Troubleshooting Quick Fixes

### Pipeline not training DNS models?
```bash
# Check if DNS data exists
ls -lh data/processed/dns_features_modelready*.csv

# If missing, run data prep manually
python3 src/data_prep/dataset_builder.py

# Then restart pipeline
pkill -f mlops_pipeline
nohup python3 scripts/mlops_pipeline.py --continuous > logs/mlops_bg.log 2>&1 &
```

### Ensemble comparison fails?
```bash
# Check all required files exist
for f in url dns whois; do
    echo "=== $f models ==="
    ls models/${f}_*.pkl 2>/dev/null || echo "MISSING!"
done

# Check data files
ls -lh data/processed/*_features_modelready_imputed.csv
```

### A/B test errors?
```python
# Reset A/B test config
import os
os.remove("logs/ab_test_config.json")

# Re-initialize
ab = ABTestManager()
```

---

## Performance Targets

| Metric | Target | Excellent |
|--------|--------|-----------|
| Accuracy | ≥ 85% | ≥ 99% |
| FPR | ≤ 5% | ≤ 1% |
| P95 Latency | ≤ 200ms | ≤ 100ms |
| Error Rate | ≤ 1% | ≤ 0.1% |

---

## Next MLOps Run

Check when next training will occur:
```bash
grep "Next pipeline check" logs/mlops_bg.log | tail -1
```

Expected output:
```
⏰ Next pipeline check in 6 hours...
```

Or manually trigger:
```bash
pkill -f mlops_pipeline
python3 scripts/mlops_pipeline.py --once
```

---

## File Locations

| What | Where |
|------|-------|
| Models | `models/*.pkl` |
| Training logs | `logs/mlops_bg.log` |
| Ensemble results | `analysis/ensemble_comparison/` |
| A/B config | `logs/ab_test_config.json` |
| A/B metrics | `analysis/ab_metrics_*.json` |
| Pipeline state | `logs/pipeline_state.json` |

---

## Key Metrics to Monitor

**From pipeline state**:
```bash
cat logs/pipeline_state.json | python3 -m json.tool
```

**From A/B test**:
```bash
python3 src/api/ab_testing.py --status
```

**From ensemble comparison**:
```bash
cat analysis/ensemble_comparison/*.json | python3 -m json.tool | grep -A5 "composite_score"
```
