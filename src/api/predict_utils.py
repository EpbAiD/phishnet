# ===============================================================
# src/api/predict_utils.py
# Feature extraction + prediction helpers for URL and WHOIS models
# ⚙️ Uses the same extractors and preprocessing as training
# ===============================================================

import time
from typing import Tuple

import pandas as pd

from src.api.model_loader import load_url_model, load_whois_model, load_dns_model
from src.features.url_features import extract_single_url_features
from src.features.whois import extract_single_whois_features
from src.features.dns_ipwhois import extract_single_domain_features
from src.data_prep.dataset_builder import preprocess_features_for_inference


# ---------------------------------------------------------------
# 🔧 Helper to convert feature dict to DataFrame for model
# ---------------------------------------------------------------
def _features_dict_to_dataframe(features: dict, feature_cols: list) -> pd.DataFrame:
    """
    Convert feature dictionary to DataFrame matching model's expected columns.

    Args:
        features: dict of features (preprocessed, flattened, imputed)
        feature_cols: list of column names expected by model

    Returns:
        DataFrame with single row, matching model's feature order
    """
    # Create DataFrame from dict
    df = pd.DataFrame([features])

    # Ensure all model features exist (fill missing with 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Restrict to exact feature order expected by model
    df = df[feature_cols]

    # Encode object columns to numeric (like training)
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).fillna("MISSING")
        df[col], _ = pd.factorize(df[col])

    return df


# ---------------------------------------------------------------
# URL Prediction
# ---------------------------------------------------------------
def predict_url_risk(url: str) -> Tuple[float, float, str, dict]:
    """
    Compute P(phishing) for a single URL using the trained URL model.

    Uses the inference pipeline:
    1. extract_single_url_features(url) → dict
    2. preprocess_features_for_inference() → flatten + impute
    3. Convert to DataFrame → model.predict_proba()

    Returns:
        prob (float): P(phishing) in [0, 1]
        latency_ms (float): End-to-end latency
        model_name (str): Model identifier
        debug (dict): Debug info
    """
    t0 = time.time()

    # Step 1: Extract URL features
    url_feats = extract_single_url_features(url)

    # Step 2: Preprocess (flatten lists + impute)
    features = preprocess_features_for_inference(url_features=url_feats)

    # Step 3: Load model and convert to DataFrame
    model, feature_cols, threshold = load_url_model()
    df_feats = _features_dict_to_dataframe(features, feature_cols)

    # Step 4: Predict
    proba = model.predict_proba(df_feats)[:, 1]
    latency_ms = (time.time() - t0) * 1000.0

    debug = {
        "expected_features": len(feature_cols),
        "input_features": int(df_feats.shape[1]),
        "features_extracted": len(url_feats),
        "features_after_preprocessing": len(features),
    }

    return float(proba[0]), float(latency_ms), "catboost_url_v1", debug


# ---------------------------------------------------------------
# WHOIS Prediction
# ---------------------------------------------------------------
def predict_whois_risk(url: str) -> Tuple[float, float, str, dict]:
    """
    Compute P(phishing) for a single URL using WHOIS features.

    Uses the inference pipeline:
    1. extract_single_whois_features(url) → dict
    2. preprocess_features_for_inference() → flatten + impute
    3. Convert to DataFrame → model.predict_proba()

    Returns:
        prob (float): P(phishing) in [0, 1]
        latency_ms (float): End-to-end latency
        model_name (str): Model identifier
        debug (dict): Debug info
    """
    t0 = time.time()

    # Step 1: Extract WHOIS features
    whois_feats = extract_single_whois_features(url, live_lookup=True)

    # Step 2: Preprocess (flatten lists + impute)
    # Note: pass empty dict for url_features since we're only using WHOIS
    features = preprocess_features_for_inference(
        url_features={},
        whois_features=whois_feats
    )

    # Step 3: Load model and convert to DataFrame
    model, feature_cols, threshold = load_whois_model()
    df_feats = _features_dict_to_dataframe(features, feature_cols)

    # Step 4: Predict
    proba = model.predict_proba(df_feats)[:, 1]
    latency_ms = (time.time() - t0) * 1000.0

    debug = {
        "expected_features": len(feature_cols),
        "input_features": int(df_feats.shape[1]),
        "features_extracted": len(whois_feats),
        "features_after_preprocessing": len(features),
    }

    return float(proba[0]), float(latency_ms), "catboost_whois_v1", debug


# ---------------------------------------------------------------
# DNS Prediction
# ---------------------------------------------------------------
def predict_dns_risk(url: str) -> Tuple[float, float, str, dict]:
    """
    Compute P(phishing) for a single URL using DNS features.

    Uses the inference pipeline:
    1. extract_single_domain_features(url) → dict
    2. preprocess_features_for_inference() → flatten + impute
    3. Convert to DataFrame → model.predict_proba()

    Returns:
        prob (float): P(phishing) in [0, 1]
        latency_ms (float): End-to-end latency
        model_name (str): Model identifier
        debug (dict): Debug info
    """
    t0 = time.time()

    # Step 1: Extract DNS features
    dns_feats = extract_single_domain_features(url)

    # Step 2: Preprocess (flatten lists + impute)
    # Note: pass empty dict for url_features since we're only using DNS
    features = preprocess_features_for_inference(
        url_features={},
        dns_features=dns_feats
    )

    # Step 3: Load model and convert to DataFrame
    model, feature_cols, threshold = load_dns_model()
    df_feats = _features_dict_to_dataframe(features, feature_cols)

    # Step 4: Predict
    proba = model.predict_proba(df_feats)[:, 1]
    latency_ms = (time.time() - t0) * 1000.0

    debug = {
        "expected_features": len(feature_cols),
        "input_features": int(df_feats.shape[1]),
        "features_extracted": len(dns_feats),
        "features_after_preprocessing": len(features),
    }

    return float(proba[0]), float(latency_ms), "catboost_dns_v1", debug


# ---------------------------------------------------------------
# Ensemble Prediction (URL + WHOIS + DNS)
# ---------------------------------------------------------------
def _load_ensemble_config():
    """Load ensemble configuration from production_metadata.json."""
    import json
    import os

    metadata_path = "models/production_metadata.json"

    # Default fallback configuration
    default_config = {
        "weights": {"url": 0.60, "dns": 0.15, "whois": 0.25},
        "models": {
            "url": "catboost",
            "dns": "catboost",
            "whois": "catboost"
        }
    }

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Check if ensemble config exists
            if "ensemble" in metadata:
                return metadata["ensemble"]
        except Exception as e:
            print(f"⚠️  Error loading ensemble config: {e}")

    print("⚠️  Using default ensemble configuration")
    return default_config


def predict_ensemble_risk_with_weights(url: str, weights: dict = None) -> Tuple[float, float, str, dict]:
    """
    Compute P(phishing) using ensemble with custom weights.

    This version allows specifying custom weights for meta-learning and testing.
    Used by meta_learning_ensemble_weights.py to search for optimal weights.

    Args:
        url: URL to predict
        weights: Optional dict with keys "url", "dns", "whois" (will be normalized)
                If None, loads from production_metadata.json

    Returns:
        prob (float): Ensemble P(phishing) in [0, 1]
        latency_ms (float): Total end-to-end latency
        model_name (str): Model identifier
        debug (dict): Debug info including individual predictions
    """
    t0 = time.time()

    # Load weights
    if weights is None:
        config = _load_ensemble_config()
        weights = config.get("weights", {"url": 0.60, "dns": 0.15, "whois": 0.25})
    else:
        # Normalize custom weights to sum to 1.0
        total = weights.get("url", 0) + weights.get("dns", 0) + weights.get("whois", 0)
        if total > 0:
            weights = {
                "url": weights.get("url", 0) / total,
                "dns": weights.get("dns", 0) / total,
                "whois": weights.get("whois", 0) / total,
            }
        else:
            weights = {"url": 0.60, "dns": 0.15, "whois": 0.25}

    # Run all three models
    url_prob, url_latency, url_model, url_debug = predict_url_risk(url)
    whois_prob, whois_latency, whois_model, whois_debug = predict_whois_risk(url)
    dns_prob, dns_latency, dns_model, dns_debug = predict_dns_risk(url)

    # Weighted ensemble
    url_weight = weights["url"]
    whois_weight = weights["whois"]
    dns_weight = weights["dns"]
    ensemble_prob = (url_weight * url_prob) + (whois_weight * whois_prob) + (dns_weight * dns_prob)

    total_latency_ms = (time.time() - t0) * 1000.0

    debug = {
        "url_prediction": {
            "risk_score": url_prob,
            "latency_ms": url_latency,
            "model": url_model,
            "weight": url_weight,
            "features": url_debug
        },
        "whois_prediction": {
            "risk_score": whois_prob,
            "latency_ms": whois_latency,
            "model": whois_model,
            "weight": whois_weight,
            "features": whois_debug
        },
        "dns_prediction": {
            "risk_score": dns_prob,
            "latency_ms": dns_latency,
            "model": dns_model,
            "weight": dns_weight,
            "features": dns_debug
        },
        "ensemble": {
            "weighted_score": ensemble_prob,
            "formula": f"{url_weight:.0%}*{url_prob:.4f} + {whois_weight:.0%}*{whois_prob:.4f} + {dns_weight:.0%}*{dns_prob:.4f}",
            "config_source": "custom_weights" if weights else "models/production_metadata.json",
        }
    }

    return float(ensemble_prob), float(total_latency_ms), f"ensemble_custom_weights", debug


def predict_ensemble_risk(url: str) -> Tuple[float, float, str, dict]:
    """
    Compute P(phishing) using ensemble of URL, WHOIS, and DNS models.

    Loads optimal ensemble configuration from models/production_metadata.json,
    which is automatically updated daily by test_ensemble_combinations.py.

    The ensemble uses weighted averaging of predictions from:
    - URL model: Fast, comprehensive URL structure features
    - WHOIS model: Domain registration and reputation signals
    - DNS model: Infrastructure and network-level indicators

    Returns:
        prob (float): Ensemble P(phishing) in [0, 1]
        latency_ms (float): Total end-to-end latency
        model_name (str): Model identifier
        debug (dict): Debug info including individual predictions
    """
    # Delegate to version with custom weights (using None = load from config)
    return predict_ensemble_risk_with_weights(url, weights=None)
