# ===============================================================
# shap_explainer_realtime.py
# ---------------------------------------------------------------
# ✅ Per-request SHAP computation for runtime explanations
# ✅ Computes SHAP values for single URL during inference
# ✅ Translates to layperson-friendly language
# ✅ Faster than training-time SHAP (only one sample)
# ===============================================================

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Optional

# Reuse the same translation mappings from feature_importance_explainer
from src.api.feature_importance_explainer import (
    URL_FEATURE_TRANSLATIONS,
    DNS_FEATURE_TRANSLATIONS,
    WHOIS_FEATURE_TRANSLATIONS,
    translate_feature_value_to_plain_language
)

# ---------------------------------------------------------------
# SHAP Explainer Cache (for performance)
# ---------------------------------------------------------------
_explainer_cache = {}


def extract_shap_features_realtime(
    model,
    X: pd.DataFrame,
    model_type: str,
    top_n: int = 5
) -> List[Dict]:
    """
    Extract SHAP values for a single URL in real-time during inference.

    This computes SHAP values on-the-fly to show:
    - Which features contributed to THIS specific prediction
    - How much each feature contributed (magnitude)
    - Direction of contribution (positive = more phishing, negative = more legit)

    Args:
        model: Trained model
        X: Feature dataframe (single row)
        model_type: 'url', 'dns', or 'whois'
        top_n: Number of top features to return

    Returns:
        List of dicts with SHAP feature contributions in plain language
    """
    try:
        # Create explainer if not cached
        cache_key = f"{model_type}_{id(model)}"
        if cache_key not in _explainer_cache:
            print(f"   Creating SHAP explainer for {model_type} model...")
            _explainer_cache[cache_key] = shap.TreeExplainer(model)

        explainer = _explainer_cache[cache_key]

        # Compute SHAP values for this single sample
        shap_obj = explainer(X, check_additivity=False)

        # Get SHAP values for positive class (phishing)
        if len(shap_obj.shape) == 3:  # Multi-output (binary classification)
            shap_vals = shap_obj.values[:, :, 1]  # Positive class
        else:
            shap_vals = shap_obj.values

        # Extract top contributing features
        if len(shap_vals.shape) == 2:
            shap_vals = shap_vals[0]  # Single sample

        # Get top N features by absolute SHAP value
        abs_vals = np.abs(shap_vals)
        top_idx = np.argsort(abs_vals)[-top_n:][::-1]

        feat_names = X.columns.tolist()
        top_features = []

        for idx in top_idx:
            feat_name = feat_names[idx]
            feat_value = X.iloc[0, idx]
            shap_contribution = float(shap_vals[idx])

            # Skip features with negligible contribution
            if abs(shap_contribution) < 0.01:
                continue

            # Skip features with missing/sentinel values that aren't meaningful
            if feat_value == -999 or pd.isna(feat_value):
                continue

            # Translate to plain language
            plain_language = translate_feature_value_to_plain_language(
                feat_name, feat_value, model_type
            )

            top_features.append({
                "feature": feat_name,
                "value": float(feat_value),
                "shap_contribution": shap_contribution,
                "impact": "increases phishing risk" if shap_contribution > 0 else "decreases phishing risk",
                "plain_language": plain_language
            })

        return top_features

    except Exception as e:
        print(f"⚠️ SHAP extraction failed for {model_type}: {e}")
        # Fallback to Feature Importance if SHAP fails
        from src.api.feature_importance_explainer import extract_important_features_with_values
        return extract_important_features_with_values(model, X, model_type, top_n=top_n)


def clear_explainer_cache():
    """Clear the SHAP explainer cache to free memory."""
    _explainer_cache.clear()
