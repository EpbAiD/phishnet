# 🎯 Ensemble Optimization & A/B Testing Plan

## Overview

We now have DNS and WHOIS features being collected 24/7. This document outlines the plan to:
1. Train models for all 3 feature types (URL, DNS, WHOIS)
2. Test different ensemble combinations
3. Measure latency for each ensemble
4. Implement A/B testing for production deployment

## Current Status

### ✅ What Works:
- **Data Collection**: VM collecting DNS + WHOIS features 24/7
- **Feature Extraction**: URL, DNS, WHOIS extractors ready
- **Training Scripts**: `url_train.py`, `dns_train.py`, `whois_train.py` exist
- **Model Zoo**: 16 algorithms available for each feature type
- **MLOps Pipeline**: Automated training every 6 hours

### ❌ What's Missing:
- DNS models NOT trained in MLOps pipeline
- No systematic ensemble comparison
- No latency benchmarking
- No A/B testing framework

## Architecture: 4 Categories × 3 Feature Types = 12 Models

### Model Categories (Best 4 per feature type):
Based on exploration findings, we'll train these 4 for each feature:

1. **CatBoost** - Best overall performance (ROC: 0.9977 for URL)
2. **LightGBM** - Fast inference, good performance
3. **XGBoost** - Industry standard, robust
4. **Random Forest** - Reliable baseline, interpretable

### Feature Types:
1. **URL Features** (30 features)
2. **DNS Features** (from dns_ipwhois.py)
3. **WHOIS Features** (14 features)

### Total Models: 4 × 3 = 12 models

## Ensemble Combinations to Test

We'll test 7 different ensemble strategies:

| Ensemble ID | Models Used | Weights | Use Case |
|------------|-------------|---------|----------|
| E1 | URL only | 100% URL | Fastest (no API calls) |
| E2 | URL + DNS | 70% URL, 30% DNS | Fast + network signals |
| E3 | URL + WHOIS | 60% URL, 40% WHOIS | Current production |
| E4 | DNS + WHOIS | 50% DNS, 50% WHOIS | Infrastructure-based |
| E5 | All 3 equal | 33% each | Balanced |
| E6 | All 3 optimized | TBD (grid search) | Best accuracy |
| E7 | All 3 speed | Lightweight models only | Lowest latency |

## Implementation Plan

### Phase 1: Enable DNS Model Training (Week 1)

**File**: `scripts/mlops_pipeline.py`

```python
# Add DNS training to ModelTrainingStage
def run() -> bool:
    # ... existing code ...

    # Train URL models
    logger.info("Training URL models...")
    from src.training.url_train import train_all_url_models
    train_all_url_models(subset=["catboost", "lgbm", "xgb", "rf"])

    # Train DNS models (NEW)
    logger.info("Training DNS models...")
    from src.training.dns_train import train_all_dns_models
    train_all_dns_models(subset=["catboost", "lgbm", "xgb", "rf"])

    # Train WHOIS models
    logger.info("Training WHOIS models...")
    from src.training.whois_train import train_all_whois_models
    # TODO: Add subset parameter to whois_train.py
    train_all_whois_models()
```

### Phase 2: Ensemble Comparison Framework (Week 1)

**New File**: `scripts/ensemble_comparison.py`

Features:
- Test all 7 ensemble combinations
- Measure prediction latency (avg, p50, p95, p99)
- Calculate accuracy, precision, recall, F1, FPR
- Generate comparison report
- Recommend best ensemble for production

**Metrics Tracked**:
```python
{
    "ensemble_id": "E6",
    "models_used": ["url_catboost", "dns_lgbm", "whois_xgb"],
    "weights": [0.5, 0.3, 0.2],

    # Performance Metrics
    "accuracy": 0.9964,
    "precision": 0.9980,
    "recall": 0.9905,
    "f1_score": 0.9942,
    "fpr": 0.0009,

    # Latency Metrics (milliseconds)
    "latency_avg": 45.2,
    "latency_p50": 42.1,
    "latency_p95": 78.3,
    "latency_p99": 125.7,

    # Resource Usage
    "model_size_mb": 12.5,
    "memory_usage_mb": 180.0,

    # Feature Availability
    "requires_dns": true,
    "requires_whois": true,
    "fallback_strategy": "URL-only if features unavailable"
}
```

### Phase 3: Latency Benchmarking (Week 1)

**New File**: `scripts/benchmark_latency.py`

```python
import time
import numpy as np
from src.api.predict_utils import predict_ensemble_risk

def benchmark_ensemble(urls, ensemble_config, n_iterations=100):
    """
    Benchmark ensemble latency.

    Args:
        urls: List of test URLs
        ensemble_config: Dict with model names and weights
        n_iterations: Number of prediction runs

    Returns:
        Latency statistics (avg, median, p95, p99)
    """
    latencies = []

    for _ in range(n_iterations):
        url = np.random.choice(urls)

        start = time.perf_counter()
        predict_ensemble_risk(url, config=ensemble_config)
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # Convert to ms

    return {
        "avg": np.mean(latencies),
        "median": np.median(latencies),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "min": np.min(latencies),
        "max": np.max(latencies)
    }
```

### Phase 4: A/B Testing Framework (Week 2)

**New File**: `src/api/ab_testing.py`

Features:
- Traffic splitting (e.g., 90% Model A, 10% Model B)
- Canary deployment
- Performance monitoring per variant
- Statistical significance testing
- Automatic rollback if variant performs worse

```python
class ABTestManager:
    def __init__(self):
        self.variants = {}
        self.metrics = {}

    def add_variant(self, variant_id, ensemble_config, traffic_pct):
        """Register A/B test variant"""
        self.variants[variant_id] = {
            "config": ensemble_config,
            "traffic_pct": traffic_pct,
            "requests": 0,
            "correct_predictions": 0,
            "total_latency": 0.0
        }

    def route_request(self, url):
        """Route request to variant based on traffic split"""
        rand = random.random() * 100

        cumulative = 0
        for variant_id, variant in self.variants.items():
            cumulative += variant["traffic_pct"]
            if rand < cumulative:
                return self.predict_with_tracking(variant_id, url)

        # Fallback to control
        return self.predict_with_tracking("control", url)

    def predict_with_tracking(self, variant_id, url):
        """Make prediction and track metrics"""
        start = time.perf_counter()
        result = predict_ensemble_risk(url, self.variants[variant_id]["config"])
        latency = time.perf_counter() - start

        # Track metrics
        self.variants[variant_id]["requests"] += 1
        self.variants[variant_id]["total_latency"] += latency

        return result

    def get_variant_metrics(self):
        """Get performance metrics for all variants"""
        return {
            vid: {
                "requests": v["requests"],
                "avg_latency": v["total_latency"] / v["requests"] if v["requests"] > 0 else 0,
                "traffic_pct": v["traffic_pct"]
            }
            for vid, v in self.variants.items()
        }
```

### Phase 5: Production Integration (Week 2)

**Modified File**: `src/api/app.py`

Add new endpoints:
```python
# New A/B testing endpoints
ab_manager = ABTestManager()

@app.post("/predict/ab")
async def predict_with_ab_test(url: str):
    """Route prediction through A/B test"""
    return ab_manager.route_request(url)

@app.get("/ab/metrics")
async def get_ab_metrics():
    """Get A/B test performance metrics"""
    return ab_manager.get_variant_metrics()

@app.post("/ab/configure")
async def configure_ab_test(config: ABTestConfig):
    """Configure new A/B test"""
    for variant in config.variants:
        ab_manager.add_variant(
            variant.id,
            variant.ensemble_config,
            variant.traffic_pct
        )
    return {"status": "configured"}
```

## Evaluation Criteria

### Ranking Formula:
```
Score = (Accuracy × 0.4) +
        ((1 - FPR) × 0.3) +
        ((1 - Normalized_Latency) × 0.2) +
        (F1_Score × 0.1)
```

### Constraints:
- Minimum accuracy: 85%
- Maximum FPR: 5%
- Maximum p95 latency: 200ms (production requirement)

## Expected Outcomes

### Latency Predictions:

| Ensemble | Expected Latency (p95) | Rationale |
|----------|------------------------|-----------|
| E1 (URL only) | 20-30ms | No external API calls |
| E2 (URL+DNS) | 80-100ms | DNS cached, fast |
| E3 (URL+WHOIS) | 150-200ms | WHOIS slower, caching helps |
| E4 (DNS+WHOIS) | 180-250ms | Both APIs needed |
| E5 (All 3 equal) | 200-300ms | All features required |
| E6 (All 3 optimized) | 180-280ms | Optimized weights |
| E7 (All 3 speed) | 100-150ms | Lightweight models |

### Accuracy Predictions:

| Ensemble | Expected Accuracy | Rationale |
|----------|------------------|-----------|
| E1 (URL only) | 95-97% | Current URL model: 99.64% |
| E2 (URL+DNS) | 96-98% | DNS adds network signals |
| E3 (URL+WHOIS) | 98-99% | Current production: 99.64% |
| E4 (DNS+WHOIS) | 92-94% | Missing URL signals |
| E5 (All 3 equal) | 98-99.5% | All signals combined |
| E6 (All 3 optimized) | 99-99.7% | Best possible |
| E7 (All 3 speed) | 97-98% | Lightweight sacrifice |

## Timeline

**Week 1**:
- Day 1-2: Add DNS training to MLOps pipeline
- Day 3-4: Build ensemble comparison framework
- Day 5-7: Run comprehensive benchmarks

**Week 2**:
- Day 1-3: Implement A/B testing framework
- Day 4-5: Production integration and testing
- Day 6-7: Deploy best ensemble to production

## Success Metrics

1. ✅ All 12 models trained successfully (4 per feature type)
2. ✅ All 7 ensembles benchmarked (latency + accuracy)
3. ✅ Best ensemble identified (highest score)
4. ✅ A/B test running in production (90/10 split)
5. ✅ Automatic deployment if new ensemble wins

## Monitoring & Alerts

**Key Metrics to Track**:
- Prediction latency (p50, p95, p99)
- Accuracy degradation (>2% drop triggers alert)
- False positive rate (>5% triggers rollback)
- Model inference errors
- Feature extraction failures
- A/B test statistical significance

## Rollback Strategy

If new ensemble performs worse:
1. Automatic rollback after 1000 requests if FPR > 5%
2. Manual rollback option via API
3. Gradual traffic shift (10% → 25% → 50% → 100%)
4. Keep previous ensemble for 7 days before deletion
