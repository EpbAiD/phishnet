# ===============================================================
# feature_importance_explainer.py
# ---------------------------------------------------------------
# ✅ Feature Importance + Feature Values approach for interpretability
# ✅ Faster and more reliable than SHAP/LIME
# ✅ Extracts top important features from all three models
# ✅ Translates to layperson-friendly language (no numbers)
# ✅ Works for URL, DNS, and WHOIS models
# ===============================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# ---------------------------------------------------------------
# Feature Translation Mappings (Layperson-Friendly)
# ---------------------------------------------------------------

# URL Model Feature Translations
URL_FEATURE_TRANSLATIONS = {
    # Domain features
    "has_ip": "uses a raw IP address instead of a proper domain name",
    "url_length": "has an unusually long URL",
    "domain_length": "has an unusually long domain name",
    "num_dots": "has many dots in the domain",
    "num_hyphens": "has multiple hyphens in the domain",
    "num_underscores": "has underscores in the domain",
    "num_percent": "has percent-encoded characters",
    "num_query_params": "has many query parameters",
    "num_ampersand": "has multiple ampersands",
    "num_hash": "has multiple hash symbols",
    "num_at": "uses @ symbol (a redirection trick)",
    # Suspicious patterns
    "suspicious_tld": "uses a suspicious domain extension",
    "brand_in_subdomain": "contains a brand name in a suspicious position",
    "has_punycode": "uses international characters to mimic legitimate sites",
    "has_subdomain": "has a subdomain structure",
    "num_subdomains": "has multiple subdomains",
    "has_login_keyword": "contains login-related keywords",
    "has_secure_keyword": "contains security-related keywords",
    "has_brand_name": "mentions a popular brand name",
    # Entropy and randomness
    "domain_entropy": "has random-looking characters in the domain",
    "path_entropy": "has random-looking characters in the path",
    # Protocol
    "is_https": "uses HTTPS protocol",
    "port_in_url": "specifies a custom port number",
    "has_double_slash_redirect": "contains suspicious redirects",
}

# DNS Model Feature Translations
DNS_FEATURE_TRANSLATIONS = {
    # TTL features
    "min_ttl": "has very short DNS cache time",
    "max_ttl": "has unusual DNS cache time",
    "avg_ttl": "has suspicious DNS cache settings",
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
    "is_privacy_protected": "uses privacy protection to hide owner information",
    # Registrar features
    "registrar_abuse_score": "is registered with a provider known for abuse",
    "free_registration": "uses a free domain registration service",
    # Update features
    "recently_updated": "was recently updated",
    "update_frequency": "has been updated frequently",
}

# ---------------------------------------------------------------
# Feature Importance Extraction
# ---------------------------------------------------------------


def get_feature_importance(
    model, feature_names: List[str], top_n: int = 10
) -> List[Dict]:
    """
    Extract feature importance from tree-based model.

    Args:
        model: Trained sklearn model (XGBoost, LightGBM, CatBoost, RandomForest, etc.)
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        List of dicts with feature importance info
    """
    try:
        # Get feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "get_feature_importance"):
            # CatBoost
            importances = model.get_feature_importance()
        else:
            print("⚠️ Model does not have feature importance")
            return []

        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]

        top_features = []
        for idx in indices:
            if idx < len(feature_names):
                top_features.append(
                    {
                        "feature": feature_names[idx],
                        "importance": float(importances[idx]),
                    }
                )

        return top_features
    except Exception as e:
        print(f"⚠️ Failed to extract feature importance: {e}")
        return []


def translate_feature_value_to_plain_language(
    feature_name: str, feature_value: float, model_type: str
) -> str:
    """
    Translate a feature name and value to layperson-friendly language.
    NO NUMBERS in the output - use descriptive language instead.

    Args:
        feature_name: Technical feature name
        feature_value: Feature value
        model_type: 'url', 'dns', or 'whois'

    Returns:
        Layperson-friendly description
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

    # Add specific value context (NO NUMBERS - use descriptive language)
    if "age" in feature_name.lower():
        if feature_value < 7:
            return "the domain was registered just a few days ago"
        elif feature_value < 30:
            return "the domain was registered very recently (within the last month)"
        elif feature_value < 90:
            return "the domain was registered recently (within the last few months)"
        elif feature_value < 365:
            return "the domain was registered less than a year ago"
        else:
            return "the domain has been registered for several years"

    elif "ttl" in feature_name.lower():
        if feature_value < 300:
            return "the DNS cache time is extremely short"
        elif feature_value < 600:
            return "the DNS cache time is very short"
        elif feature_value < 3600:
            return "the DNS cache time is unusually short"
        else:
            return "the DNS cache time is normal"

    elif "length" in feature_name.lower():
        if feature_value > 50:
            return f"the {feature_name.replace('_length', '')} is extremely long"
        elif feature_value > 30:
            return f"the {feature_name.replace('_length', '')} is unusually long"
        elif feature_value > 20:
            return f"the {feature_name.replace('_length', '')} is moderately long"
        else:
            return f"the {feature_name.replace('_length', '')} length is normal"

    elif "num_" in feature_name.lower():
        clean_name = feature_name.replace("num_", "").replace("_", " ")
        if feature_value == 0:
            return f"has no {clean_name}"
        elif feature_value == 1:
            return f"has one {clean_name.rstrip('s')}"
        elif feature_value == 2:
            return f"has a couple of {clean_name}"
        elif feature_value <= 5:
            return f"has several {clean_name}"
        else:
            return f"has many {clean_name}"

    elif feature_name.startswith("has_") or feature_name.startswith("is_"):
        if feature_value == 1:
            return base_translation
        else:
            # Negate the statement
            return f"does not {base_translation.replace('uses', 'use').replace('has', 'have').replace('is', 'be').replace('contains', 'contain')}"

    elif "suspicious_tld" in feature_name:
        if feature_value == 1:
            return "uses a domain extension commonly associated with phishing attacks"
        else:
            return "uses a standard domain extension"

    elif "brand_in_subdomain" in feature_name:
        if feature_value == 1:
            return "contains a brand name in the URL but is not the official website"
        else:
            return "does not impersonate known brands"

    elif "entropy" in feature_name.lower():
        if feature_value > 4.0:
            return f"the {feature_name.replace('_entropy', '')} contains random-looking characters"
        elif feature_value > 3.5:
            return f"the {feature_name.replace('_entropy', '')} looks somewhat random"
        else:
            return f"the {feature_name.replace('_entropy', '')} looks normal"

    elif "score" in feature_name.lower() or "reputation" in feature_name.lower():
        if feature_value > 0.7:
            return f"{base_translation} (high risk)"
        elif feature_value > 0.4:
            return f"{base_translation} (moderate risk)"
        else:
            return f"{base_translation} (low risk)"

    else:
        # Generic translation - avoid showing numbers
        if feature_value == 0:
            return f"{base_translation} (none detected)"
        elif feature_value == 1:
            return base_translation
        else:
            return base_translation


def extract_important_features_with_values(
    model, X: pd.DataFrame, model_type: str, top_n: int = 5
) -> List[Dict]:
    """
    Extract top important features and their values for this specific URL.

    Combines:
    - Global feature importance (what the model cares about)
    - Actual feature values (what this URL has)

    Args:
        model: Trained model
        X: Feature dataframe (single row)
        model_type: 'url', 'dns', or 'whois'
        top_n: Number of top features to return

    Returns:
        List of dicts with feature info in layperson-friendly language
    """
    try:
        # Get top important features globally
        feature_names = X.columns.tolist()
        important_features = get_feature_importance(model, feature_names, top_n=top_n)

        if not important_features:
            return []

        # Get actual values for this URL
        result = []
        for feat_info in important_features:
            feat_name = feat_info["feature"]
            if feat_name in X.columns:
                feat_value = float(X[feat_name].iloc[0])

                # Skip features with zero/missing values (not relevant for this URL)
                if feat_value == 0 or feat_value == -999:
                    continue

                # Translate to plain language
                plain_language = translate_feature_value_to_plain_language(
                    feat_name, feat_value, model_type
                )

                result.append(
                    {
                        "feature": feat_name,
                        "value": feat_value,
                        "importance": feat_info["importance"],
                        "plain_language": plain_language,
                    }
                )

        return result

    except Exception as e:
        print(f"⚠️ Feature importance extraction failed for {model_type}: {e}")
        return []


# ---------------------------------------------------------------
# Fallback: Use pre-defined important features if model doesn't support it
# ---------------------------------------------------------------

# Most important features per model (pre-computed from training)
DEFAULT_IMPORTANT_FEATURES = {
    "url": [
        "has_login_keyword",
        "brand_in_subdomain",
        "domain_entropy",
        "url_length",
        "num_subdomains",
        "suspicious_tld",
        "has_at_symbol",
        "num_dots",
        "has_punycode",
        "path_entropy",
    ],
    "dns": [
        "min_ttl",
        "num_ips",
        "asn_reputation_score",
        "country_risk_score",
        "num_unique_countries",
        "has_private_ip",
        "avg_ttl",
    ],
    "whois": [
        "domain_age_days",
        "is_recently_registered",
        "days_until_expiry",
        "is_privacy_protected",
        "registrar_abuse_score",
        "registration_length_days",
        "free_registration",
    ],
}


def extract_important_features_fallback(
    X: pd.DataFrame, model_type: str, top_n: int = 5
) -> List[Dict]:
    """
    Fallback method using pre-defined important features.
    Used if model doesn't support feature importance extraction.
    """
    try:
        important_feature_names = DEFAULT_IMPORTANT_FEATURES.get(model_type, [])[:top_n]

        result = []
        for feat_name in important_feature_names:
            if feat_name in X.columns:
                feat_value = float(X[feat_name].iloc[0])

                # Skip features with zero/missing values
                if feat_value == 0 or feat_value == -999:
                    continue

                # Translate to plain language
                plain_language = translate_feature_value_to_plain_language(
                    feat_name, feat_value, model_type
                )

                result.append(
                    {
                        "feature": feat_name,
                        "value": feat_value,
                        "importance": 1.0,  # Not available in fallback
                        "plain_language": plain_language,
                    }
                )

        return result

    except Exception as e:
        print(f"⚠️ Fallback feature extraction failed for {model_type}: {e}")
        return []
