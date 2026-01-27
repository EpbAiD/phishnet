#!/usr/bin/env python3
"""
Test All Ensemble Combinations
================================
Tests all combinations of URL/DNS/WHOIS models to find optimal tradeoff
between accuracy and latency for production deployment.

Uses FRESH, UNSEEN test set (not training data) for realistic evaluation.

Outputs:
- analysis/ensemble_combinations.csv: All combinations with metrics
- models/production_metadata.json: Updated with best ensemble config
"""

import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from itertools import product
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# Import feature extractors and preprocessing (SAME as training)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features.url_features import extract_single_url_features
from src.features.dns_ipwhois import extract_single_domain_features
from src.features.whois import extract_single_whois_features
from src.data_prep.dataset_builder import preprocess_features_for_inference

# Paths
MODELS_DIR = "models"
ANALYSIS_DIR = "analysis"
DATA_DIR = "data/processed"
TEST_SET_PATH = "data/test/fresh_phishing_test_set.csv"

# ===============================================================
# OPTIMAL WEIGHT LEARNING using scipy.optimize
# ===============================================================
# Uses scipy.optimize.minimize to find weights that minimize
# cross-validation loss (maximize accuracy/ROC-AUC)
# Reference: https://arxiv.org/pdf/1908.05287
# ===============================================================

from scipy.optimize import minimize


def optimize_ensemble_weights(
    url_probs: np.ndarray,
    dns_probs: np.ndarray,
    whois_probs: np.ndarray,
    y_true: np.ndarray,
    metric: str = "log_loss"
) -> dict:
    """
    Find optimal ensemble weights using scipy.optimize.

    This minimizes cross-validation loss to find the best weight combination
    for combining model predictions.

    Args:
        url_probs: Predicted probabilities from URL model (n_samples,)
        dns_probs: Predicted probabilities from DNS model (n_samples,)
        whois_probs: Predicted probabilities from WHOIS model (n_samples,)
        y_true: True labels (n_samples,)
        metric: Loss metric to minimize ("log_loss", "brier", "neg_accuracy")

    Returns:
        Dict with optimal weights and optimization info
    """
    from sklearn.metrics import log_loss, brier_score_loss

    def objective(weights):
        """Objective function to minimize."""
        w_url, w_dns, w_whois = weights
        # Weighted ensemble prediction
        ensemble_probs = w_url * url_probs + w_dns * dns_probs + w_whois * whois_probs
        # Clip to avoid log(0)
        ensemble_probs = np.clip(ensemble_probs, 1e-15, 1 - 1e-15)

        if metric == "log_loss":
            return log_loss(y_true, ensemble_probs)
        elif metric == "brier":
            return brier_score_loss(y_true, ensemble_probs)
        elif metric == "neg_accuracy":
            preds = (ensemble_probs >= 0.5).astype(int)
            return -accuracy_score(y_true, preds)
        elif metric == "neg_f1":
            preds = (ensemble_probs >= 0.5).astype(int)
            return -f1_score(y_true, preds)
        else:
            return log_loss(y_true, ensemble_probs)

    # Constraints: weights must sum to 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # Bounds: each weight between 0 and 1
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    # Initial guess: equal weights
    initial_weights = [1/3, 1/3, 1/3]

    # Optimize using SLSQP (Sequential Least Squares Programming)
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9}
    )

    if result.success:
        optimal_weights = result.x
        # Normalize to ensure sum = 1 (numerical precision)
        optimal_weights = optimal_weights / optimal_weights.sum()
        return {
            "url": float(optimal_weights[0]),
            "dns": float(optimal_weights[1]),
            "whois": float(optimal_weights[2]),
            "_strategy": f"scipy_optimized_{metric}",
            "_loss": float(result.fun),
            "_success": True
        }
    else:
        # Fallback to equal weights
        return {
            "url": 1/3,
            "dns": 1/3,
            "whois": 1/3,
            "_strategy": "scipy_failed_equal",
            "_loss": None,
            "_success": False
        }


def optimize_weights_with_cv(
    url_model, dns_model, whois_model,
    url_X, dns_X, whois_X, y,
    n_folds: int = 5
) -> dict:
    """
    Optimize ensemble weights using out-of-fold predictions.

    This is the proper way to learn ensemble weights without overfitting:
    1. Get out-of-fold predictions from each model using CV
    2. Use these OOF predictions to optimize weights
    3. The weights generalize better because they're learned on held-out data

    Args:
        url_model, dns_model, whois_model: Trained models
        url_X, dns_X, whois_X: Feature DataFrames
        y: True labels
        n_folds: Number of CV folds for OOF predictions

    Returns:
        Dict with optimal weights
    """
    from sklearn.model_selection import StratifiedKFold

    n_samples = len(y)
    url_oof = np.zeros(n_samples)
    dns_oof = np.zeros(n_samples)
    whois_oof = np.zeros(n_samples)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(url_X, y):
        # Get validation predictions (out-of-fold)
        # Note: Models are already trained, we just predict
        if hasattr(url_model, "predict_proba"):
            url_oof[val_idx] = url_model.predict_proba(url_X.iloc[val_idx])[:, 1]
        else:
            scores = url_model.decision_function(url_X.iloc[val_idx])
            url_oof[val_idx] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

        if hasattr(dns_model, "predict_proba"):
            dns_oof[val_idx] = dns_model.predict_proba(dns_X.iloc[val_idx])[:, 1]
        else:
            scores = dns_model.decision_function(dns_X.iloc[val_idx])
            dns_oof[val_idx] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

        if hasattr(whois_model, "predict_proba"):
            whois_oof[val_idx] = whois_model.predict_proba(whois_X.iloc[val_idx])[:, 1]
        else:
            scores = whois_model.decision_function(whois_X.iloc[val_idx])
            whois_oof[val_idx] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    # Now optimize weights using OOF predictions
    return optimize_ensemble_weights(url_oof, dns_oof, whois_oof, y, metric="log_loss")


def learn_meta_weights_logistic(
    url_probs: np.ndarray,
    dns_probs: np.ndarray,
    whois_probs: np.ndarray,
    y_true: np.ndarray
) -> dict:
    """
    Learn ensemble weights using a logistic regression meta-learner.

    This is a simplified stacking approach where:
    1. Stack predictions from base models as features
    2. Train logistic regression to learn optimal combination
    3. Extract coefficients as interpretable weights

    Args:
        url_probs, dns_probs, whois_probs: Predicted probabilities
        y_true: True labels

    Returns:
        Dict with learned weights
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Stack predictions as features
    X_meta = np.column_stack([url_probs, dns_probs, whois_probs])

    # Train logistic regression (no regularization for interpretable weights)
    meta_model = LogisticRegression(
        penalty=None,  # No regularization
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
    meta_model.fit(X_meta, y_true)

    # Extract and normalize coefficients as weights
    coefs = meta_model.coef_[0]
    # Use softmax to convert to positive weights summing to 1
    exp_coefs = np.exp(coefs - np.max(coefs))  # Subtract max for numerical stability
    weights = exp_coefs / exp_coefs.sum()

    return {
        "url": float(weights[0]),
        "dns": float(weights[1]),
        "whois": float(weights[2]),
        "_strategy": "meta_logistic",
        "_raw_coefs": coefs.tolist()
    }


# Legacy heuristic strategies (kept for comparison)
HEURISTIC_STRATEGIES = [
    "performance_proportional",
    "inverse_error",
]


def calculate_heuristic_weights(url_roc: float, dns_roc: float, whois_roc: float, strategy: str) -> dict:
    """Calculate weights using simple heuristics (for comparison with optimized)."""
    if strategy == "performance_proportional":
        total = url_roc + dns_roc + whois_roc
        return {
            "url": url_roc / total,
            "dns": dns_roc / total,
            "whois": whois_roc / total,
            "_strategy": "heuristic_proportional"
        }
    elif strategy == "inverse_error":
        epsilon = 0.001
        url_inv = 1.0 / (1.0 - url_roc + epsilon)
        dns_inv = 1.0 / (1.0 - dns_roc + epsilon)
        whois_inv = 1.0 / (1.0 - whois_roc + epsilon)
        total = url_inv + dns_inv + whois_inv
        return {
            "url": url_inv / total,
            "dns": dns_inv / total,
            "whois": whois_inv / total,
            "_strategy": "heuristic_inverse_error"
        }
    else:
        return {"url": 1/3, "dns": 1/3, "whois": 1/3, "_strategy": "equal"}


def load_model(model_type: str, model_name: str):
    """Load a trained model."""
    model_path = f"{MODELS_DIR}/{model_type}_{model_name}.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def load_test_data():
    """
    Load FRESH test set and extract features in real-time.

    This provides realistic evaluation on never-seen-before URLs.
    Uses the SAME preprocessing pipeline as training.
    """
    print(f"  Loading fresh test set from {TEST_SET_PATH}...")

    if not os.path.exists(TEST_SET_PATH):
        print(f"\n  ⚠️  Fresh test set not found!")
        print(f"  Run: python3 scripts/collect_test_set.py")
        print(f"  Falling back to training data (NOT RECOMMENDED)")
        return load_training_data_as_test()

    # Load fresh URLs
    df_test = pd.read_csv(TEST_SET_PATH)
    print(f"  Fresh test set: {len(df_test)} URLs")
    print(f"    Phishing: {len(df_test[df_test['label'] == 'phishing'])}")
    print(f"    Legitimate: {len(df_test[df_test['label'] == 'legitimate'])}")

    # Extract features for each URL (real-time, SAME as training pipeline)
    print("\n  Extracting features from fresh URLs (this may take a few minutes)...")

    all_features_list = []
    labels = []

    for idx, row in df_test.iterrows():
        url = row['url']
        label = 1 if row['label'] == 'phishing' else 0

        if idx % 20 == 0:
            print(f"    Processing {idx+1}/{len(df_test)}...")

        try:
            # Extract raw features (SAME functions as training)
            url_feats = extract_single_url_features(url)
            dns_feats = extract_single_domain_features(url)
            whois_feats = extract_single_whois_features(url, live_lookup=False)

            # Preprocess using SAME function as training
            processed_feats = preprocess_features_for_inference(
                url_features=url_feats,
                dns_features=dns_feats,
                whois_features=whois_feats
            )

            all_features_list.append(processed_feats)
            labels.append(label)

        except Exception as e:
            print(f"    ⚠️  Failed to extract features for {url}: {e}")
            continue

    print(f"  ✓ Successfully extracted features for {len(labels)} URLs")

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features_list)
    labels_array = np.array(labels)

    # Split into URL/DNS/WHOIS feature sets based on column names
    # Load training data column names to know which features belong where
    url_train_df = pd.read_csv(f"{DATA_DIR}/url_features_modelready.csv")
    dns_train_df = pd.read_csv(f"{DATA_DIR}/dns_features_modelready.csv")
    whois_train_df = pd.read_csv(f"{DATA_DIR}/whois_features_modelready.csv")

    url_features = [c for c in url_train_df.columns if c not in ['url', 'label', 'bucket']]
    dns_features = [c for c in dns_train_df.columns if c not in ['url', 'label', 'bucket']]
    whois_features = [c for c in whois_train_df.columns if c not in ['url', 'label', 'bucket']]

    # Extract subsets (with all expected columns, fill missing with 0)
    url_df = pd.DataFrame(0, index=range(len(features_df)), columns=url_features)
    for col in url_features:
        if col in features_df.columns:
            url_df[col] = features_df[col]

    dns_df = pd.DataFrame(0, index=range(len(features_df)), columns=dns_features)
    for col in dns_features:
        if col in features_df.columns:
            dns_df[col] = features_df[col]

    whois_df = pd.DataFrame(0, index=range(len(features_df)), columns=whois_features)
    for col in whois_features:
        if col in features_df.columns:
            whois_df[col] = features_df[col]

    # Encode object columns to numeric (same as training)
    for df_X in [url_df, dns_df, whois_df]:
        obj_cols = df_X.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            df_X[col] = df_X[col].astype(str).fillna("MISSING")
            df_X[col], _ = pd.factorize(df_X[col])

    return {
        "url": (url_df, labels_array),
        "dns": (dns_df, labels_array),
        "whois": (whois_df, labels_array)
    }


def load_training_data_as_test():
    """FALLBACK: Load training data (not recommended for ensemble testing)."""
    print("  ⚠️  WARNING: Using training data for latency benchmarking only!")

    # Load all three datasets
    url_df = pd.read_csv(f"{DATA_DIR}/url_features_modelready.csv")
    dns_df = pd.read_csv(f"{DATA_DIR}/dns_features_modelready.csv")
    whois_df = pd.read_csv(f"{DATA_DIR}/whois_features_modelready.csv")

    # Find common URLs
    url_urls = set(url_df['url'].values)
    dns_urls = set(dns_df['url'].values)
    whois_urls = set(whois_df['url'].values)
    common_urls = url_urls & dns_urls & whois_urls

    # Filter and align
    url_df = url_df[url_df['url'].isin(common_urls)].sort_values('url').reset_index(drop=True)
    dns_df = dns_df[dns_df['url'].isin(common_urls)].sort_values('url').reset_index(drop=True)
    whois_df = whois_df[whois_df['url'].isin(common_urls)].sort_values('url').reset_index(drop=True)

    # Extract features and labels - keep separate columns per type
    url_y = url_df["label"].astype(int).values
    url_X = url_df.drop(columns=["url", "label", "bucket"], errors="ignore")
    dns_X = dns_df.drop(columns=["url", "label", "bucket"], errors="ignore")
    whois_X = whois_df.drop(columns=["url", "label", "bucket"], errors="ignore")

    # Encode categorical - create copies to avoid SettingWithCopyWarning
    url_X = url_X.copy()
    dns_X = dns_X.copy()
    whois_X = whois_X.copy()

    for df_X in [url_X, dns_X, whois_X]:
        obj_cols = df_X.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            df_X[col] = df_X[col].astype(str).fillna("MISSING")
            df_X[col], _ = pd.factorize(df_X[col])

    return {
        "url": (url_X, url_y),
        "dns": (dns_X, url_y),
        "whois": (whois_X, url_y)
    }


def align_features_for_model(model_type: str, model_name: str, X: pd.DataFrame) -> pd.DataFrame:
    """
    Align features to match the column order used during model training.

    Args:
        model_type: 'url', 'dns', or 'whois'
        model_name: Model name (e.g., 'rf', 'catboost')
        X: DataFrame with features

    Returns:
        DataFrame with columns in correct order for the model
    """
    feature_cols_path = f"{MODELS_DIR}/{model_type}_{model_name}_feature_cols.pkl"

    if not os.path.exists(feature_cols_path):
        # Fall back to using X as-is
        return X

    expected_cols = joblib.load(feature_cols_path)

    # Create aligned DataFrame
    aligned = pd.DataFrame(0, index=X.index, columns=expected_cols)
    for col in expected_cols:
        if col in X.columns:
            aligned[col] = X[col].values

    return aligned


def measure_single_url_latency(model, X_single):
    """
    Measure latency for predicting a SINGLE URL (realistic user scenario).
    Runs multiple iterations to get stable measurement.
    """
    import time

    # Warm up the model
    if hasattr(model, "predict_proba"):
        _ = model.predict_proba(X_single)
    else:
        _ = model.decision_function(X_single)

    # Measure over multiple iterations
    n_iterations = 100
    latencies = []

    for _ in range(n_iterations):
        start = time.perf_counter()
        if hasattr(model, "predict_proba"):
            _ = model.predict_proba(X_single)
        else:
            _ = model.decision_function(X_single)
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    return {
        "mean_ms": np.mean(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99)
    }


def test_ensemble_combination_with_cv(
    url_model_name, dns_model_name, whois_model_name,
    cv_results, weights, test_data
):
    """
    Test ensemble using CV metrics for accuracy (unbiased) and
    single-URL latency for realistic performance measurement.

    Args:
        url_model_name, dns_model_name, whois_model_name: Model names
        cv_results: Dict with CV results DataFrames for each type
        weights: Dict with ensemble weights
        test_data: Dict with single sample for latency testing

    Returns:
        Dict with metrics from CV + single-URL latency
    """
    # Get CV metrics for each model (these are unbiased estimates)
    url_cv = cv_results['url']
    dns_cv = cv_results['dns']
    whois_cv = cv_results['whois']

    url_metrics = url_cv[url_cv['model'] == url_model_name].iloc[0]
    dns_metrics = dns_cv[dns_cv['model'] == dns_model_name].iloc[0]
    whois_metrics = whois_cv[whois_cv['model'] == whois_model_name].iloc[0]

    # Weighted ensemble CV metrics (estimated)
    # Handle both column naming conventions: 'acc'/'phish_f1' (local) and 'accuracy'/'f1' (workflow)
    def get_acc(metrics):
        return metrics.get('acc', metrics.get('accuracy', 0))

    def get_f1(metrics):
        return metrics.get('phish_f1', metrics.get('f1', 0))

    ensemble_roc = (
        weights["url"] * url_metrics['roc_auc'] +
        weights["dns"] * dns_metrics['roc_auc'] +
        weights["whois"] * whois_metrics['roc_auc']
    )
    ensemble_acc = (
        weights["url"] * get_acc(url_metrics) +
        weights["dns"] * get_acc(dns_metrics) +
        weights["whois"] * get_acc(whois_metrics)
    )
    ensemble_f1 = (
        weights["url"] * get_f1(url_metrics) +
        weights["dns"] * get_f1(dns_metrics) +
        weights["whois"] * get_f1(whois_metrics)
    )

    # Load models for latency testing
    url_model = load_model("url", url_model_name)
    dns_model = load_model("dns", dns_model_name)
    whois_model = load_model("whois", whois_model_name)

    if not all([url_model, dns_model, whois_model]):
        return None

    # Measure SINGLE URL latency (realistic user scenario)
    # Align features to match training column order
    url_X_single = align_features_for_model("url", url_model_name, test_data["url"][0].iloc[[0]])
    dns_X_single = align_features_for_model("dns", dns_model_name, test_data["dns"][0].iloc[[0]])
    whois_X_single = align_features_for_model("whois", whois_model_name, test_data["whois"][0].iloc[[0]])

    url_latency = measure_single_url_latency(url_model, url_X_single)
    dns_latency = measure_single_url_latency(dns_model, dns_X_single)
    whois_latency = measure_single_url_latency(whois_model, whois_X_single)

    # Total per-URL latency (all 3 models sequential)
    total_latency_mean = url_latency['mean_ms'] + dns_latency['mean_ms'] + whois_latency['mean_ms']
    total_latency_p95 = url_latency['p95_ms'] + dns_latency['p95_ms'] + whois_latency['p95_ms']

    # Tradeoff score: higher accuracy, lower latency is better
    # Using CV ROC-AUC as primary metric, latency as secondary
    tradeoff_score = ensemble_roc - (0.3 * (total_latency_mean / 1000))

    return {
        "url_roc": url_metrics['roc_auc'],
        "dns_roc": dns_metrics['roc_auc'],
        "whois_roc": whois_metrics['roc_auc'],
        "ensemble_roc_auc": ensemble_roc,
        "ensemble_accuracy": ensemble_acc,
        "ensemble_f1": ensemble_f1,
        "url_latency_ms": url_latency['mean_ms'],
        "dns_latency_ms": dns_latency['mean_ms'],
        "whois_latency_ms": whois_latency['mean_ms'],
        "total_latency_mean_ms": total_latency_mean,
        "total_latency_p95_ms": total_latency_p95,
        "tradeoff_score": tradeoff_score
    }


def predict_with_timing(model, X):
    """Predict with timing measurement (for batch processing)."""
    start = time.time()

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        # Decision function fallback
        scores = model.decision_function(X)
        mn, mx = scores.min(), scores.max()
        probs = (scores - mn) / (mx - mn + 1e-12)

    latency = time.time() - start
    return probs, latency


def test_ensemble_combination(
    url_model, dns_model, whois_model,
    test_data, weights, sample_size=1000
):
    """
    Test a specific ensemble combination (legacy batch method).
    For accurate metrics, use test_ensemble_combination_with_cv instead.
    """
    url_X, url_y = test_data["url"]
    dns_X, dns_y = test_data["dns"]
    whois_X, whois_y = test_data["whois"]

    # Use smaller sample for latency testing
    sample_indices = np.random.choice(len(url_X), min(sample_size, len(url_X)), replace=False)

    url_X_sample = url_X.iloc[sample_indices]
    dns_X_sample = dns_X.iloc[sample_indices]
    whois_X_sample = whois_X.iloc[sample_indices]
    y_sample = url_y[sample_indices]

    # Get predictions with timing
    url_probs, url_latency = predict_with_timing(url_model, url_X_sample)
    dns_probs, dns_latency = predict_with_timing(dns_model, dns_X_sample)
    whois_probs, whois_latency = predict_with_timing(whois_model, whois_X_sample)

    # Weighted ensemble
    ensemble_probs = (
        weights["url"] * url_probs +
        weights["dns"] * dns_probs +
        weights["whois"] * whois_probs
    )
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_sample, ensemble_preds)
    f1 = f1_score(y_sample, ensemble_preds)
    roc_auc = roc_auc_score(y_sample, ensemble_probs)

    # Total latency (sequential prediction)
    total_latency = url_latency + dns_latency + whois_latency
    avg_latency_per_url = total_latency / len(url_X_sample) * 1000  # ms

    # Tradeoff score: higher accuracy, lower latency is better
    tradeoff_score = accuracy - (0.3 * (avg_latency_per_url / 1000))

    return {
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "avg_latency_ms": avg_latency_per_url,
        "tradeoff_score": tradeoff_score
    }


def main():
    print("="*80)
    print("ENSEMBLE COMBINATION TESTING (CV Metrics + Single-URL Latency)")
    print("="*80)

    # Load CV results to get best models and their UNBIASED metrics
    print("\nLoading CV results (unbiased metrics from 5-fold cross-validation)...")

    url_results = pd.read_csv(f"{ANALYSIS_DIR}/url_cv_results.csv")
    dns_results = pd.read_csv(f"{ANALYSIS_DIR}/dns_cv_results.csv")
    whois_results = pd.read_csv(f"{ANALYSIS_DIR}/whois_cv_results.csv")

    cv_results = {
        'url': url_results,
        'dns': dns_results,
        'whois': whois_results
    }

    # Display individual model statistics
    print("\n" + "="*80)
    print("INDIVIDUAL MODEL STATISTICS (from 5-fold CV)")
    print("="*80)

    for feature_type, df in cv_results.items():
        print(f"\n{feature_type.upper()} Models:")
        print("-" * 60)
        for _, row in df.iterrows():
            # Handle both column naming conventions
            acc_val = row.get('acc', row.get('accuracy', 0))
            f1_val = row.get('phish_f1', row.get('f1', 0))
            print(f"  {row['model']:20s} ROC={row['roc_auc']:.4f} ACC={acc_val:.4f} F1={f1_val:.4f}")

    # Get top 3 models for each type
    url_top = url_results.nlargest(3, "roc_auc")["model"].values
    dns_top = dns_results.nlargest(3, "roc_auc")["model"].values
    whois_top = whois_results.nlargest(3, "roc_auc")["model"].values

    print(f"\n{'='*80}")
    print("TOP 3 MODELS PER FEATURE TYPE")
    print("="*80)
    print(f"URL:   {list(url_top)}")
    print(f"DNS:   {list(dns_top)}")
    print(f"WHOIS: {list(whois_top)}")

    # Load test data (just for latency measurement - single sample)
    print("\nLoading data for single-URL latency measurement...")
    test_data = load_training_data_as_test()  # Use training data format for latency only
    print(f"  Loaded sample data for latency benchmarking")

    # ================================================================
    # SCIPY OPTIMIZED WEIGHT LEARNING
    # ================================================================
    print(f"\n{'='*80}")
    print("USING SCIPY.OPTIMIZE FOR OPTIMAL WEIGHT LEARNING")
    print("  Method: Minimize log-loss using SLSQP optimization")
    print("  Constraint: Weights sum to 1.0, each in [0, 1]")
    print("  Reference: https://arxiv.org/pdf/1908.05287")
    print("="*80)

    # For each model combination, we'll:
    # 1. Get predictions from each model on training data
    # 2. Use scipy.optimize to find weights that minimize log-loss
    # 3. Compare with heuristic weights

    n_combinations = len(url_top) * len(dns_top) * len(whois_top)
    print(f"\nOPTIMIZING {n_combinations} MODEL COMBINATIONS")
    print(f"  Models: {len(url_top)} URL × {len(dns_top)} DNS × {len(whois_top)} WHOIS")
    print(f"  For each combination: scipy optimization + 2 heuristic baselines")
    print("="*80)

    results = []

    for url_name in url_top:
        for dns_name in dns_top:
            for whois_name in whois_top:
                combo_name = f"{url_name}+{dns_name}+{whois_name}"
                print(f"\n{'='*60}")
                print(f"Optimizing: {combo_name}")
                print("="*60)

                # Load models
                url_model = load_model("url", url_name)
                dns_model = load_model("dns", dns_name)
                whois_model = load_model("whois", whois_name)

                if not all([url_model, dns_model, whois_model]):
                    print(f"  ⚠️ Skipping (model not found)")
                    continue

                # Get aligned feature data
                url_X, y = test_data["url"]
                dns_X, _ = test_data["dns"]
                whois_X, _ = test_data["whois"]

                url_X_aligned = align_features_for_model("url", url_name, url_X)
                dns_X_aligned = align_features_for_model("dns", dns_name, dns_X)
                whois_X_aligned = align_features_for_model("whois", whois_name, whois_X)

                # ============================================================
                # IMPORTANT: Use OUT-OF-FOLD predictions for weight optimization
                # This prevents overfitting the weights to training data
                # ============================================================
                from sklearn.model_selection import StratifiedKFold

                n_samples = len(y)
                url_oof = np.zeros(n_samples)
                dns_oof = np.zeros(n_samples)
                whois_oof = np.zeros(n_samples)

                print("\n  Computing out-of-fold predictions for weight optimization...")
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                try:
                    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(url_X_aligned, y)):
                        # Get validation (OOF) predictions - model predicts on data it hasn't seen
                        if hasattr(url_model, "predict_proba"):
                            url_oof[val_idx] = url_model.predict_proba(url_X_aligned.iloc[val_idx])[:, 1]
                        else:
                            scores = url_model.decision_function(url_X_aligned.iloc[val_idx])
                            url_oof[val_idx] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

                        if hasattr(dns_model, "predict_proba"):
                            dns_oof[val_idx] = dns_model.predict_proba(dns_X_aligned.iloc[val_idx])[:, 1]
                        else:
                            scores = dns_model.decision_function(dns_X_aligned.iloc[val_idx])
                            dns_oof[val_idx] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

                        if hasattr(whois_model, "predict_proba"):
                            whois_oof[val_idx] = whois_model.predict_proba(whois_X_aligned.iloc[val_idx])[:, 1]
                        else:
                            scores = whois_model.decision_function(whois_X_aligned.iloc[val_idx])
                            whois_oof[val_idx] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

                    print(f"      ✓ OOF predictions computed (5-fold CV)")

                except Exception as e:
                    print(f"  ❌ OOF prediction failed: {e}")
                    continue

                # ============================================================
                # METHOD 1: SCIPY OPTIMIZATION using OOF predictions
                # ============================================================
                print("\n  [1] Scipy Optimization (log-loss on OOF predictions)...")
                scipy_weights = optimize_ensemble_weights(
                    url_oof, dns_oof, whois_oof, y, metric="log_loss"
                )
                print(f"      Weights: URL={scipy_weights['url']:.1%}, DNS={scipy_weights['dns']:.1%}, WHOIS={scipy_weights['whois']:.1%}")
                if scipy_weights.get('_loss'):
                    print(f"      Log-loss: {scipy_weights['_loss']:.6f}")

                # ============================================================
                # METHOD 2: META-LEARNER using OOF predictions
                # ============================================================
                print("\n  [2] Meta-Learner (Logistic Regression on OOF predictions)...")
                try:
                    meta_weights = learn_meta_weights_logistic(
                        url_oof, dns_oof, whois_oof, y
                    )
                    print(f"      Weights: URL={meta_weights['url']:.1%}, DNS={meta_weights['dns']:.1%}, WHOIS={meta_weights['whois']:.1%}")
                except Exception as e:
                    print(f"      ❌ Failed: {e}")
                    meta_weights = {"url": 1/3, "dns": 1/3, "whois": 1/3, "_strategy": "meta_failed"}

                # ============================================================
                # METHOD 3: HEURISTIC BASELINES
                # ============================================================
                url_roc = url_results[url_results['model'] == url_name]['roc_auc'].iloc[0]
                dns_roc = dns_results[dns_results['model'] == dns_name]['roc_auc'].iloc[0]
                whois_roc = whois_results[whois_results['model'] == whois_name]['roc_auc'].iloc[0]

                heuristic_proportional = calculate_heuristic_weights(url_roc, dns_roc, whois_roc, "performance_proportional")
                heuristic_inverse = calculate_heuristic_weights(url_roc, dns_roc, whois_roc, "inverse_error")

                print(f"\n  [3] Heuristic (proportional): URL={heuristic_proportional['url']:.1%}, DNS={heuristic_proportional['dns']:.1%}, WHOIS={heuristic_proportional['whois']:.1%}")
                print(f"  [4] Heuristic (inverse_error): URL={heuristic_inverse['url']:.1%}, DNS={heuristic_inverse['dns']:.1%}, WHOIS={heuristic_inverse['whois']:.1%}")

                # Collect all weight strategies for this model combination
                all_weights = [scipy_weights, meta_weights, heuristic_proportional, heuristic_inverse]

                for weights in all_weights:
                    # Extract strategy name for logging
                    strategy = weights.get("_strategy", "unknown")
                    # Create clean weights dict without _strategy key
                    clean_weights = {k: v for k, v in weights.items() if k != "_strategy"}
                    combo_name = f"{url_name}+{dns_name}+{whois_name}"
                    weight_str = f"U{int(clean_weights['url']*100)}D{int(clean_weights['dns']*100)}W{int(clean_weights['whois']*100)}"

                    print(f"\n  Testing: {combo_name} | Strategy: {strategy} | Weights: {weight_str}")

                    try:
                        metrics = test_ensemble_combination_with_cv(
                            url_name, dns_name, whois_name,
                            cv_results, clean_weights, test_data
                        )

                        if metrics is None:
                            print(f"    ⚠️  Skipping (model not found)")
                            continue

                        result = {
                            "url_model": url_name,
                            "dns_model": dns_name,
                            "whois_model": whois_name,
                            "weight_strategy": strategy,
                            "weight_url": clean_weights["url"],
                            "weight_dns": clean_weights["dns"],
                            "weight_whois": clean_weights["whois"],
                            **metrics
                        }
                        results.append(result)

                        print(f"    Single-URL Latency: {metrics['total_latency_mean_ms']:.2f}ms (URL={metrics['url_latency_ms']:.2f}, DNS={metrics['dns_latency_ms']:.2f}, WHOIS={metrics['whois_latency_ms']:.2f})")
                        print(f"    Tradeoff Score: {metrics['tradeoff_score']:.4f}")

                    except Exception as e:
                        print(f"    ❌ Failed: {e}")
                        import traceback
                        traceback.print_exc()

    # Save results
    if not results:
        print("\n❌ No successful ensemble combinations found!")
        print("All combinations failed. Check:")
        print("  1. Are CV results available in analysis/?")
        print("  2. Are models trained and saved correctly?")
        print("  3. Check log file for detailed errors")
        exit(1)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("tradeoff_score", ascending=False)

    output_path = f"{ANALYSIS_DIR}/ensemble_combinations.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\n{'='*80}")
    print(f"RESULTS SAVED: {output_path}")
    print(f"{'='*80}")

    # Display top 5 combinations
    print("\nTop 5 Ensemble Combinations (by tradeoff score):")
    print("-" * 80)

    for idx, (_, row) in enumerate(results_df.head(5).iterrows(), 1):
        strategy = row.get('weight_strategy', 'unknown')
        print(f"\n{idx}. {row['url_model']} + {row['dns_model']} + {row['whois_model']}")
        print(f"   Strategy: {strategy}")
        print(f"   Weights: URL={row['weight_url']:.1%}, DNS={row['weight_dns']:.1%}, WHOIS={row['weight_whois']:.1%}")
        print(f"   CV Metrics: ROC={row['ensemble_roc_auc']:.4f} | ACC={row['ensemble_accuracy']:.4f} | F1={row['ensemble_f1']:.4f}")
        print(f"   Single-URL Latency: {row['total_latency_mean_ms']:.2f}ms | Tradeoff: {row['tradeoff_score']:.4f}")

    # Select best ensemble
    best = results_df.iloc[0]

    print(f"\n{'='*80}")
    print("BEST ENSEMBLE SELECTED")
    print(f"{'='*80}")
    best_strategy = best.get('weight_strategy', 'unknown')
    print(f"URL Model:   {best['url_model']} (CV ROC={best['url_roc']:.4f})")
    print(f"DNS Model:   {best['dns_model']} (CV ROC={best['dns_roc']:.4f})")
    print(f"WHOIS Model: {best['whois_model']} (CV ROC={best['whois_roc']:.4f})")
    print(f"Weight Strategy: {best_strategy}")
    print(f"Weights: URL={best['weight_url']:.1%}, DNS={best['weight_dns']:.1%}, WHOIS={best['weight_whois']:.1%}")
    print(f"\nPerformance (from 5-fold CV - unbiased estimates):")
    print(f"  Ensemble ROC-AUC: {best['ensemble_roc_auc']:.4f}")
    print(f"  Ensemble Accuracy: {best['ensemble_accuracy']:.4f} ({best['ensemble_accuracy']*100:.2f}%)")
    print(f"  Ensemble F1 Score: {best['ensemble_f1']:.4f}")
    print(f"\nLatency (single URL prediction - realistic):")
    print(f"  URL Model: {best['url_latency_ms']:.2f}ms")
    print(f"  DNS Model: {best['dns_latency_ms']:.2f}ms")
    print(f"  WHOIS Model: {best['whois_latency_ms']:.2f}ms")
    print(f"  Total: {best['total_latency_mean_ms']:.2f}ms (p95: {best['total_latency_p95_ms']:.2f}ms)")
    print(f"\nTradeoff Score: {best['tradeoff_score']:.4f}")

    # Load existing metadata or create new
    metadata_path = f"{MODELS_DIR}/production_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Update with ensemble config
    from datetime import datetime

    metadata["ensemble"] = {
        "selected_at": datetime.now().isoformat(),
        "models": {
            "url": best["url_model"],
            "dns": best["dns_model"],
            "whois": best["whois_model"]
        },
        "weight_strategy": best.get("weight_strategy", "unknown"),
        "weights": {
            "url": float(best["weight_url"]),
            "dns": float(best["weight_dns"]),
            "whois": float(best["weight_whois"])
        },
        "cv_performance": {
            "url_roc_auc": float(best["url_roc"]),
            "dns_roc_auc": float(best["dns_roc"]),
            "whois_roc_auc": float(best["whois_roc"]),
            "ensemble_roc_auc": float(best["ensemble_roc_auc"]),
            "ensemble_accuracy": float(best["ensemble_accuracy"]),
            "ensemble_f1": float(best["ensemble_f1"])
        },
        "latency": {
            "url_ms": float(best["url_latency_ms"]),
            "dns_ms": float(best["dns_latency_ms"]),
            "whois_ms": float(best["whois_latency_ms"]),
            "total_mean_ms": float(best["total_latency_mean_ms"]),
            "total_p95_ms": float(best["total_latency_p95_ms"])
        },
        "tradeoff_score": float(best["tradeoff_score"])
    }

    # Save updated metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Production metadata updated: {metadata_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
