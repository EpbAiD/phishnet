# SHAP-Based Explainability Implementation Plan

## Current Status

✅ **System is running** with URL + WHOIS models
✅ **Feature Importance** approach is working (current implementation)
🔧 **SHAP Implementation** in progress (more powerful but needs work)

---

## Why SHAP is Better Than Feature Importance

### Feature Importance (Current)
- Shows **global** importance: which features the model uses most
- Cannot show **direction** (positive vs negative contribution)
- Cannot show **magnitude** for a specific URL
- Example: "domain_age_days is important" (but is it high or low?)

### SHAP (Proposed)
- Shows **per-sample** contributions: why THIS specific URL was flagged
- Shows **direction**: "domain_age_days = 3 days → +0.35 phishing score"
- Shows **magnitude**: how much each feature contributed
- More interpretable: "This feature added +0.3 to the phishing score because..."

---

## The Problem We Hit

**Error**: Feature mismatch between model and test data

```
CatBoostError: At position 8 should be feature with name has_at_symbol
(found has_double_slash_redirect)
```

### Root Cause
1. Models were trained with a specific feature order/set
2. Test data has different features or different order
3. CatBoost is very strict about feature matching

### Why This Happens
During training, feature engineering may evolve:
- New features added
- Features removed
- Feature order changed
- Models saved before data preprocessing finalized

---

## Solution: Proper ML Pipeline

To implement SHAP correctly, we need to:

### Step 1: **Retrain Models with Feature Tracking**
Modify training scripts to save:
```python
{
    "model": trained_model,
    "features": list_of_feature_names,  # Exact order
    "threshold": 0.5,
    "preprocessing": {
        "imputation": "mean",
        "scaling": None
    }
}
```

###Step 2: **Compute SHAP on Test Data** (One-Time)
After training, run SHAP computation:
```bash
python src/training/compute_shap_importance.py --all
```

This will:
- Load each trained model
- Load test data with EXACT same features
- Compute SHAP values on 100-1000 test samples
- Calculate global feature importance (mean |SHAP|)
- Save top N features to JSON

Output: `models/shap_importance_url.json`
```json
{
  "model_type": "url",
  "features": [
    {"rank": 1, "feature": "brand_in_subdomain", "mean_abs_shap": 0.145},
    {"rank": 2, "feature": "domain_age_days", "mean_abs_shap": 0.132},
    ...
  ]
}
```

### Step 3: **Update Inference to Use SHAP**
Modify `shap_explainer.py` to:
1. Load global important features (from Step 2)
2. Compute SHAP for single URL during inference
3. Focus on top N globally important features
4. Return SHAP values with direction and magnitude

---

## Current Workaround

Since we hit the feature mismatch issue, here are options:

### Option A: Use Feature Importance (Keep Current System)
**Pros:**
- Already working
- Fast (no SHAP computation per request)
- Good enough for most users

**Cons:**
- Less interpretable (no per-sample contributions)
- Cannot show direction of impact

### Option B: Fix and Implement SHAP
**Pros:**
- More powerful explanations
- Shows why THIS specific URL was flagged
- Research-grade explainability

**Cons:**
- Requires retraining models with proper feature tracking
- Adds ~100-500ms latency per request
- More complex implementation

### Option C: Hybrid Approach (Recommended)
**Use both:**
- **Feature Importance**: Fast fallback, always available
- **SHAP**: Compute when explicitly requested or for flagged URLs

Implementation:
```python
if use_shap and shap_available:
    return compute_shap_explanation(url)
else:
    return compute_feature_importance_explanation(url)
```

---

## Next Steps (Your Choice)

### Path 1: Stick with Feature Importance ✅
**No changes needed** - system is working and explanations are good

### Path 2: Implement SHAP Properly
1. Retrain URL model with feature tracking
2. Compute SHAP on test data
3. Update inference to use SHAP
4. Test and compare

### Path 3: Hybrid (Best of Both)
1. Keep Feature Importance as default
2. Add SHAP as optional (for detailed analysis)
3. No retraining needed immediately

---

## Performance Comparison

| Method | Latency | Interpretability | Complexity |
|--------|---------|------------------|------------|
| Feature Importance | <10ms | Good | Low |
| SHAP (cached explainer) | 100-300ms | Excellent | Medium |
| SHAP (no cache) | 500-2000ms | Excellent | High |

---

## Recommendation

For **production deployment**, I recommend:

1. **Short term**: Keep Feature Importance (current system works great)
2. **Medium term**: Fix feature mismatch and compute SHAP as background job
3. **Long term**: Hybrid approach with SHAP on-demand

The current system with Feature Importance is already providing **good explanations** and is **fast**. SHAP would make it even better, but requires proper ML pipeline setup first.

---

## What We've Built So Far

✅ **Created**: `src/training/compute_shap_importance.py`
   - Computes SHAP on test data
   - Saves global feature importance
   - Ready to use once feature mismatch is fixed

✅ **Already Have**: `src/api/feature_importance_explainer.py`
   - Working Feature Importance implementation
   - Fast and reliable
   - Good enough for most use cases

✅ **System Running**: URL + WHOIS ensemble with LLM explanations

---

**Decision**: What approach would you like to take?

- **A**: Keep current Feature Importance system (no changes)
- **B**: Fix feature mismatch and implement SHAP
- **C**: Hybrid approach (both methods available)
