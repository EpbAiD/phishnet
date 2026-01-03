"""
Feature Column Alignment for Deployment
========================================
Ensures features extracted at runtime match the column order used during training.
"""
import joblib
import pandas as pd
from pathlib import Path


def align_features(features_dict: dict, model_type: str, model_name: str) -> pd.DataFrame:
    """
    Align extracted features to match training column order.

    Args:
        features_dict: Dictionary of feature name -> value from extraction
        model_type: 'url', 'dns', or 'whois'
        model_name: Model name (e.g., 'catboost', 'lgbm', 'xgb')

    Returns:
        DataFrame with columns in correct order for the model

    Example:
        >>> url_features = extract_single_url_features("https://google.com")
        >>> aligned_df = align_features(url_features, "url", "catboost")
        >>> model = joblib.load("models/url_catboost.pkl")
        >>> prediction = model.predict_proba(aligned_df)
    """
    # Load saved feature column order
    feature_cols_path = Path("models") / f"{model_type}_{model_name}_feature_cols.pkl"

    if not feature_cols_path.exists():
        raise FileNotFoundError(
            f"Feature column order file not found: {feature_cols_path}\n"
            f"This file should have been created during training.\n"
            f"Please retrain the model to generate it."
        )

    expected_cols = joblib.load(feature_cols_path)

    # Create DataFrame from features dict
    df = pd.DataFrame([features_dict])

    # Check for missing columns
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        # Fill missing columns with zeros (or appropriate defaults)
        for col in missing_cols:
            df[col] = 0

    # Reorder columns to match training order
    df = df[expected_cols]

    # Handle categorical encoding (same as training)
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).fillna("MISSING")
        df[col], _ = pd.factorize(df[col])

    return df


def load_model_and_align(model_type: str, model_name: str):
    """
    Load model and return a prediction function that auto-aligns features.

    Args:
        model_type: 'url', 'dns', or 'whois'
        model_name: Model name (e.g., 'catboost')

    Returns:
        Tuple of (model, predict_func)

    Example:
        >>> model, predict = load_model_and_align("url", "catboost")
        >>> url_features = extract_single_url_features("https://google.com")
        >>> phish_prob = predict(url_features)
    """
    model_path = Path("models") / f"{model_type}_{model_name}.pkl"
    model = joblib.load(model_path)

    def predict(features_dict: dict) -> float:
        """Predict phishing probability with automatic feature alignment."""
        aligned_df = align_features(features_dict, model_type, model_name)
        proba = model.predict_proba(aligned_df)[0][1]  # Phishing probability
        return proba

    return model, predict


# Convenience functions for each model type
def predict_url_aligned(url_features: dict, model_name: str = "catboost") -> float:
    """Make URL prediction with automatic feature alignment."""
    _, predict = load_model_and_align("url", model_name)
    return predict(url_features)


def predict_dns_aligned(dns_features: dict, model_name: str = "catboost") -> float:
    """Make DNS prediction with automatic feature alignment."""
    _, predict = load_model_and_align("dns", model_name)
    return predict(dns_features)


def predict_whois_aligned(whois_features: dict, model_name: str = "catboost") -> float:
    """Make WHOIS prediction with automatic feature alignment."""
    _, predict = load_model_and_align("whois", model_name)
    return predict(whois_features)


def predict_ensemble_aligned(
    url_features: dict,
    dns_features: dict,
    whois_features: dict,
    weights: dict = None,
    model_names: dict = None
) -> dict:
    """
    Make ensemble prediction with automatic feature alignment.

    Args:
        url_features: URL features dictionary
        dns_features: DNS features dictionary
        whois_features: WHOIS features dictionary
        weights: Dict with keys 'url', 'dns', 'whois' (default: 50%, 20%, 30%)
        model_names: Dict with keys 'url', 'dns', 'whois' (default: all 'catboost')

    Returns:
        Dict with individual probabilities, ensemble probability, and verdict
    """
    if weights is None:
        weights = {"url": 0.5, "dns": 0.2, "whois": 0.3}

    if model_names is None:
        model_names = {"url": "catboost", "dns": "catboost", "whois": "catboost"}

    # Get individual predictions
    url_prob = predict_url_aligned(url_features, model_names["url"])
    dns_prob = predict_dns_aligned(dns_features, model_names["dns"])
    whois_prob = predict_whois_aligned(whois_features, model_names["whois"])

    # Weighted ensemble
    ensemble_prob = (
        weights["url"] * url_prob +
        weights["dns"] * dns_prob +
        weights["whois"] * whois_prob
    )

    verdict = "PHISHING" if ensemble_prob > 0.5 else "LEGITIMATE"

    return {
        "url_probability": float(url_prob),
        "dns_probability": float(dns_prob),
        "whois_probability": float(whois_prob),
        "ensemble_probability": float(ensemble_prob),
        "verdict": verdict,
        "confidence": abs(ensemble_prob - 0.5) * 2  # 0 to 1 scale
    }
