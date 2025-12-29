# ===============================================================
# src/api/model_loader.py
# ---------------------------------------------------------------
# Unified loader for URL and WHOIS phishing models.
# - Loads models only once (global cache)
# - Stores feature list + threshold for each model
# - Provides safe fallback if missing
# ===============================================================

import os
import joblib
import numpy as np
import pandas as pd

# ---------- GLOBAL CACHES ----------
# Stored as tuple: (model_object, feature_list, threshold)
_URL_MODEL_CACHE = None
_WHOIS_MODEL_CACHE = None
_DNS_MODEL_CACHE = None

# ---------- DEFAULTS ----------
DEFAULT_THRESHOLD = 0.5
MODELS_DIR = "models"


def load_url_model():
    """
    Load URL phishing model with feature columns.

    Returns:
        tuple: (model, feature_cols, threshold)
    """
    global _URL_MODEL_CACHE

    if _URL_MODEL_CACHE is not None:
        return _URL_MODEL_CACHE

    model_path = "models/url_catboost.pkl"
    model = joblib.load(model_path)

    # Extract feature names from CatBoost model
    feature_cols = model.feature_names_

    # Convert from strings if needed
    feature_cols = [str(c).strip() for c in feature_cols]

    # Default threshold
    threshold = 0.5

    _URL_MODEL_CACHE = (model, feature_cols, threshold)
    print(f"🔐 Loaded URL model: {type(model).__name__} | {len(feature_cols)} features | threshold={threshold}")

    return _URL_MODEL_CACHE


def load_whois_model():
    """
    Load WHOIS phishing model with feature columns.

    Returns:
        tuple: (model, feature_cols, threshold)
    """
    global _WHOIS_MODEL_CACHE

    if _WHOIS_MODEL_CACHE is not None:
        return _WHOIS_MODEL_CACHE

    model_path = "models/whois_catboost.pkl"
    model = joblib.load(model_path)

    # Extract feature names from CatBoost model
    feature_cols = model.feature_names_

    # Convert from strings if needed
    feature_cols = [str(c).strip() for c in feature_cols]

    # Default threshold
    threshold = 0.5

    _WHOIS_MODEL_CACHE = (model, feature_cols, threshold)
    print(f"🔐 Loaded WHOIS model: {type(model).__name__} | {len(feature_cols)} features | threshold={threshold}")

    return _WHOIS_MODEL_CACHE


def load_dns_model():
    """
    Load DNS phishing model with feature columns.

    Returns:
        tuple: (model, feature_cols, threshold)
    """
    global _DNS_MODEL_CACHE

    if _DNS_MODEL_CACHE is not None:
        return _DNS_MODEL_CACHE

    model_path = "models/dns_catboost.pkl"
    model = joblib.load(model_path)

    # Extract feature names from CatBoost model
    feature_cols = model.feature_names_

    # Convert from strings if needed
    feature_cols = [str(c).strip() for c in feature_cols]

    # Default threshold
    threshold = 0.5

    _DNS_MODEL_CACHE = (model, feature_cols, threshold)
    print(f"🔐 Loaded DNS model: {type(model).__name__} | {len(feature_cols)} features | threshold={threshold}")

    return _DNS_MODEL_CACHE