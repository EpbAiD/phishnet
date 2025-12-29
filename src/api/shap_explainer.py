# ===============================================================
# shap_explainer.py
# ---------------------------------------------------------------
# ✅ Real-time SHAP feature extraction for URL/DNS/WHOIS models
# ✅ Extracts top contributing features per inference
# ✅ Translates technical features to plain language
# ✅ Used by LLM explainer for evidence-based explanations
# ===============================================================

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Optional
from urllib.parse import urlparse

# ---------------------------------------------------------------
# Feature Translation Mappings
# ---------------------------------------------------------------

# URL Model Feature Translations
URL_FEATURE_TRANSLATIONS = {
    # Domain features
    "has_ip": "uses a raw IP address instead of a domain name",
    "url_length": "has an unusually long URL",
    "domain_length": "has an unusually long domain name",
    "num_dots": "has many dots in the domain",
    "num_hyphens": "has multiple hyphens in the domain",
    "num_underscores": "has underscores in the domain",
    "num_percent": "has percent-encoded characters",
    "num_query_params": "has many query parameters",
    "num_ampersand": "has multiple ampersands in the URL",
    "num_hash": "has multiple hash symbols",
    "num_at": "uses @ symbol (redirection trick)",
    # Suspicious patterns
    "suspicious_tld": "uses a suspicious domain extension (like .xyz, .tk)",
    "brand_in_subdomain": "contains a brand name in a suspicious position",
    "has_punycode": "uses international characters (punycode)",
    "has_subdomain": "has a subdomain structure",
    "num_subdomains": "has multiple subdomains",
    # Entropy and randomness
    "domain_entropy": "has random-looking characters in the domain",
    "path_entropy": "has random-looking characters in the path",
    # Protocol
    "is_https": "uses HTTPS protocol",
    "port_in_url": "specifies a custom port number",
}

# DNS Model Feature Translations
DNS_FEATURE_TRANSLATIONS = {
    # TTL features
    "min_ttl": "has very short DNS cache time (TTL)",
    "max_ttl": "has unusual DNS cache time",
    "avg_ttl": "has suspicious average DNS cache settings",
    # IP features
    "num_ips": "resolves to multiple IP addresses",
    "num_unique_countries": "has servers in multiple countries",
    "has_private_ip": "uses a private IP address",
    # ASN features
    "asn_reputation_score": "is hosted by a provider with poor reputation",
    "is_cloud_provider": "is hosted on cloud infrastructure",
    # Geographic
    "country_risk_score": "is hosted in a high-risk country",
    "multiple_nameservers": "uses multiple nameservers",
}

# WHOIS Model Feature Translations
WHOIS_FEATURE_TRANSLATIONS = {
    # Age features
    "domain_age_days": "domain registration age",
    "days_until_expiry": "days until domain expires",
    "registration_length_days": "registration period length",
    # Status features
    "is_recently_registered": "was registered very recently",
    "expires_soon": "is about to expire",
    "is_privacy_protected": "uses privacy protection (hides owner info)",
    # Registrar features
    "registrar_abuse_score": "is registered with a provider known for abuse",
    "free_registration": "uses a free domain registration service",
    # Update features
    "recently_updated": "was recently updated",
    "update_frequency": "has been updated frequently",
}

# ---------------------------------------------------------------
# SHAP Explainer Cache (for performance)
# ---------------------------------------------------------------
_explainer_cache = {}


def get_estimator(model_entry):
    """Extract actual sklearn estimator if model is wrapped in dict."""
    if isinstance(model_entry, dict) and "model" in model_entry:
        return model_entry["model"]
    return model_entry


def align_to_model_schema(X: pd.DataFrame, model) -> pd.DataFrame:
    """Ensure X has all columns expected by model; fill missing with 0."""
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        missing = [c for c in expected if c not in X.columns]
        if missing:
            for c in missing:
                X[c] = 0
        X = X[expected]
    return X


def numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric columns, fill NaNs."""
    df = df.select_dtypes(include=[np.number])
    return df.fillna(0)


def extract_shap_features(
    model, X: pd.DataFrame, model_type: str, top_n: int = 5
) -> List[Dict]:
    """
    Extract SHAP values and return top contributing features.

    Args:
        model: Trained sklearn model
        X: Feature dataframe (single row)
        model_type: 'url', 'dns', or 'whois'
        top_n: Number of top features to return

    Returns:
        List of dicts with feature info:
        [
            {
                "feature": "domain_age_days",
                "value": 3.0,
                "shap_contribution": 0.25,
                "impact": "increases phishing score",
                "plain_language": "domain was registered 3 days ago"
            },
            ...
        ]
    """
    try:
        # Create explainer if not cached
        cache_key = f"{model_type}_{id(model)}"
        if cache_key not in _explainer_cache:
            if model_type == "whois":
                # Use KernelExplainer for WHOIS (safer for some models)
                # Note: In production, you may want to create background sample once
                background = X.sample(n=min(10, len(X)), random_state=42)
                _explainer_cache[cache_key] = shap.KernelExplainer(
                    model.predict_proba, background
                )
            else:
                # Use TreeExplainer for tree-based models (faster)
                _explainer_cache[cache_key] = shap.TreeExplainer(model)

        explainer = _explainer_cache[cache_key]

        # Compute SHAP values
        if model_type == "whois":
            shap_values = explainer.shap_values(X)
            # For binary classification, use positive class SHAP values
            shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        else:
            shap_obj = explainer(X, check_additivity=False)
            # Get SHAP values for positive class (phishing)
            if len(shap_obj.shape) == 3:  # Multi-output
                shap_vals = shap_obj.values[:, :, 1]
            else:
                shap_vals = shap_obj.values

        # Extract top contributing features
        if len(shap_vals.shape) == 2:
            shap_vals = shap_vals[0]  # Single sample

        abs_vals = np.abs(shap_vals)
        top_idx = np.argsort(abs_vals)[-top_n:][::-1]  # Top N features

        feat_names = X.columns.tolist()
        top_features = []

        for idx in top_idx:
            feat_name = feat_names[idx]
            feat_value = X.iloc[0, idx]
            shap_contribution = float(shap_vals[idx])

            # Skip features with negligible contribution
            if abs(shap_contribution) < 0.01:
                continue

            # Translate to plain language
            plain_language = translate_feature_to_plain_language(
                feat_name, feat_value, model_type
            )

            top_features.append(
                {
                    "feature": feat_name,
                    "value": float(feat_value),
                    "shap_contribution": shap_contribution,
                    "impact": (
                        "increases phishing score"
                        if shap_contribution > 0
                        else "decreases phishing score"
                    ),
                    "plain_language": plain_language,
                }
            )

        return top_features

    except Exception as e:
        print(f"⚠️ SHAP extraction failed for {model_type}: {e}")
        return []


def translate_feature_to_plain_language(
    feature_name: str, feature_value: float, model_type: str
) -> str:
    """
    Translate a technical feature name and value to plain language.

    Args:
        feature_name: Technical feature name (e.g., "domain_age_days")
        feature_value: Feature value (e.g., 3.0)
        model_type: 'url', 'dns', or 'whois'

    Returns:
        Plain language description (e.g., "domain was registered just 3 days ago")
    """
    # Select appropriate translation map
    if model_type == "url":
        translations = URL_FEATURE_TRANSLATIONS
    elif model_type == "dns":
        translations = DNS_FEATURE_TRANSLATIONS
    elif model_type == "whois":
        translations = WHOIS_FEATURE_TRANSLATIONS
    else:
        translations = {}

    # Get base translation
    base_translation = translations.get(feature_name, feature_name.replace("_", " "))

    # Add specific value context for certain features
    if "age" in feature_name.lower() and feature_value < 30:
        return f"domain was registered just {int(feature_value)} days ago"
    elif "age" in feature_name.lower() and feature_value > 365:
        return f"domain has been registered for {int(feature_value / 365)} years"
    elif "ttl" in feature_name.lower() and feature_value < 600:
        return f"DNS cache time is very short ({int(feature_value)} seconds)"
    elif "length" in feature_name.lower() and feature_value > 30:
        return f"the {feature_name.replace('_length', '')} is unusually long ({int(feature_value)} characters)"
    elif "num_" in feature_name.lower() and feature_value > 2:
        clean_name = feature_name.replace("num_", "").replace("_", " ")
        return f"has {int(feature_value)} {clean_name}"
    elif feature_value == 1 and feature_name.startswith("has_"):
        return base_translation
    elif feature_value == 1 and feature_name.startswith("is_"):
        return base_translation
    elif "suspicious_tld" in feature_name and feature_value == 1:
        return "uses a domain extension commonly used in phishing (.xyz, .tk, etc.)"
    elif "brand_in_subdomain" in feature_name and feature_value == 1:
        return (
            "contains a brand name (like PayPal, Amazon) but isn't the official website"
        )
    else:
        # Generic translation
        if isinstance(feature_value, float):
            if feature_value.is_integer():
                return f"{base_translation} ({int(feature_value)})"
            else:
                return f"{base_translation} ({feature_value:.2f})"
        return base_translation


def clear_explainer_cache():
    """Clear the SHAP explainer cache to free memory."""
    _explainer_cache.clear()
