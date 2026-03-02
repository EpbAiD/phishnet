# ===============================================================
# src/api/model_loader.py
# ---------------------------------------------------------------
# Unified loader for phishing models (URL, DNS, WHOIS).
# - Reads ensemble config from production_metadata.json to
#   determine which model architecture to load (e.g. lightgbm)
# - Loads models only once (global cache)
# - Stores feature list + threshold for each model
# - Falls back to catboost if config is missing
# ===============================================================

import os
import json
import joblib

# ---------- GLOBAL CACHES ----------
# Stored as tuple: (model_object, feature_list, threshold)
_URL_MODEL_CACHE = None
_WHOIS_MODEL_CACHE = None
_DNS_MODEL_CACHE = None

# Resolved model names (set during loading)
_URL_MODEL_NAME = "catboost"
_DNS_MODEL_NAME = "catboost"
_WHOIS_MODEL_NAME = "catboost"

# ---------- DEFAULTS ----------
DEFAULT_THRESHOLD = 0.5
MODELS_DIR = "models"


def _get_ensemble_model_names():
    """Read model names from production_metadata.json ensemble config."""
    metadata_path = os.path.join(MODELS_DIR, "production_metadata.json")
    defaults = {"url": "catboost", "dns": "catboost", "whois": "catboost"}
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        models = metadata.get("ensemble", {}).get("models", {})
        return {
            "url": models.get("url", defaults["url"]),
            "dns": models.get("dns", defaults["dns"]),
            "whois": models.get("whois", defaults["whois"]),
        }
    except Exception:
        return defaults


def _load_feature_cols(model, model_type, model_name):
    """Load feature columns, trying feature_cols.pkl first, then model attributes."""
    # Try saved feature columns file (most reliable)
    feature_cols_path = os.path.join(
        MODELS_DIR, f"{model_type}_{model_name}_feature_cols.pkl"
    )
    if os.path.exists(feature_cols_path):
        feature_cols = joblib.load(feature_cols_path)
        return [str(c).strip() for c in feature_cols]

    # Fall back to model attributes (CatBoost: feature_names_, LightGBM: feature_name_)
    for attr in ("feature_names_", "feature_name_"):
        if hasattr(model, attr):
            feature_cols = getattr(model, attr)
            return [str(c).strip() for c in feature_cols]

    raise ValueError(
        f"Cannot determine feature columns for {model_type}_{model_name}. "
        f"No feature_cols.pkl found and model has no feature_names_ or feature_name_."
    )


def get_model_names():
    """Return the currently loaded model names for each feature type."""
    return {"url": _URL_MODEL_NAME, "dns": _DNS_MODEL_NAME, "whois": _WHOIS_MODEL_NAME}


def load_url_model():
    """
    Load URL phishing model with feature columns.

    Returns:
        tuple: (model, feature_cols, threshold)
    """
    global _URL_MODEL_CACHE, _URL_MODEL_NAME

    if _URL_MODEL_CACHE is not None:
        return _URL_MODEL_CACHE

    model_name = _get_ensemble_model_names()["url"]
    model_path = os.path.join(MODELS_DIR, f"url_{model_name}.pkl")

    # Fall back to catboost if selected model file doesn't exist
    if not os.path.exists(model_path):
        print(f"⚠️  url_{model_name}.pkl not found, falling back to url_catboost.pkl")
        model_name = "catboost"
        model_path = os.path.join(MODELS_DIR, "url_catboost.pkl")

    model = joblib.load(model_path)
    feature_cols = _load_feature_cols(model, "url", model_name)
    threshold = DEFAULT_THRESHOLD

    _URL_MODEL_NAME = model_name
    _URL_MODEL_CACHE = (model, feature_cols, threshold)
    print(
        f"🔐 Loaded URL model: {model_name} ({type(model).__name__}) "
        f"| {len(feature_cols)} features | threshold={threshold}"
    )

    return _URL_MODEL_CACHE


def load_whois_model():
    """
    Load WHOIS phishing model with feature columns.

    Returns:
        tuple: (model, feature_cols, threshold)
    """
    global _WHOIS_MODEL_CACHE, _WHOIS_MODEL_NAME

    if _WHOIS_MODEL_CACHE is not None:
        return _WHOIS_MODEL_CACHE

    model_name = _get_ensemble_model_names()["whois"]
    model_path = os.path.join(MODELS_DIR, f"whois_{model_name}.pkl")

    if not os.path.exists(model_path):
        print(
            f"⚠️  whois_{model_name}.pkl not found, falling back to whois_catboost.pkl"
        )
        model_name = "catboost"
        model_path = os.path.join(MODELS_DIR, "whois_catboost.pkl")

    model = joblib.load(model_path)
    feature_cols = _load_feature_cols(model, "whois", model_name)
    threshold = DEFAULT_THRESHOLD

    _WHOIS_MODEL_NAME = model_name
    _WHOIS_MODEL_CACHE = (model, feature_cols, threshold)
    print(
        f"🔐 Loaded WHOIS model: {model_name} ({type(model).__name__}) "
        f"| {len(feature_cols)} features | threshold={threshold}"
    )

    return _WHOIS_MODEL_CACHE


def load_dns_model():
    """
    Load DNS phishing model with feature columns.

    Returns:
        tuple: (model, feature_cols, threshold)
    """
    global _DNS_MODEL_CACHE, _DNS_MODEL_NAME

    if _DNS_MODEL_CACHE is not None:
        return _DNS_MODEL_CACHE

    model_name = _get_ensemble_model_names()["dns"]
    model_path = os.path.join(MODELS_DIR, f"dns_{model_name}.pkl")

    if not os.path.exists(model_path):
        print(f"⚠️  dns_{model_name}.pkl not found, falling back to dns_catboost.pkl")
        model_name = "catboost"
        model_path = os.path.join(MODELS_DIR, "dns_catboost.pkl")

    model = joblib.load(model_path)
    feature_cols = _load_feature_cols(model, "dns", model_name)
    threshold = DEFAULT_THRESHOLD

    _DNS_MODEL_NAME = model_name
    _DNS_MODEL_CACHE = (model, feature_cols, threshold)
    print(
        f"🔐 Loaded DNS model: {model_name} ({type(model).__name__}) "
        f"| {len(feature_cols)} features | threshold={threshold}"
    )

    return _DNS_MODEL_CACHE


def clear_model_cache():
    """
    Clear all model caches to force reload on next access.
    Used by hot reload to pick up new models from disk.
    """
    global _URL_MODEL_CACHE, _WHOIS_MODEL_CACHE, _DNS_MODEL_CACHE
    global _URL_MODEL_NAME, _DNS_MODEL_NAME, _WHOIS_MODEL_NAME
    _URL_MODEL_CACHE = None
    _WHOIS_MODEL_CACHE = None
    _DNS_MODEL_CACHE = None
    _URL_MODEL_NAME = "catboost"
    _DNS_MODEL_NAME = "catboost"
    _WHOIS_MODEL_NAME = "catboost"
    print("🔄 Model caches cleared")
