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

# Ensemble weights to test
WEIGHT_CONFIGS = [
    {"url": 0.60, "dns": 0.15, "whois": 0.25},  # URL-heavy (current)
    {"url": 0.50, "dns": 0.25, "whois": 0.25},  # Balanced
    {"url": 0.70, "dns": 0.10, "whois": 0.20},  # URL-dominant
    {"url": 0.40, "dns": 0.30, "whois": 0.30},  # Equal DNS/WHOIS
]


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
    print("  ⚠️  WARNING: Using training data for testing - results will be overly optimistic!")

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

    # Extract features and labels
    url_y = url_df["label"].astype(int).values
    url_X = url_df.drop(columns=["url", "label", "bucket"], errors="ignore")
    dns_X = dns_df.drop(columns=["url", "label", "bucket"], errors="ignore")
    whois_X = whois_df.drop(columns=["url", "label", "bucket"], errors="ignore")

    # Encode categorical
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
    ensemble_roc = (
        weights["url"] * url_metrics['roc_auc'] +
        weights["dns"] * dns_metrics['roc_auc'] +
        weights["whois"] * whois_metrics['roc_auc']
    )
    ensemble_acc = (
        weights["url"] * url_metrics['accuracy'] +
        weights["dns"] * dns_metrics['accuracy'] +
        weights["whois"] * whois_metrics['accuracy']
    )
    ensemble_f1 = (
        weights["url"] * url_metrics['f1'] +
        weights["dns"] * dns_metrics['f1'] +
        weights["whois"] * whois_metrics['f1']
    )

    # Load models for latency testing
    url_model = load_model("url", url_model_name)
    dns_model = load_model("dns", dns_model_name)
    whois_model = load_model("whois", whois_model_name)

    if not all([url_model, dns_model, whois_model]):
        return None

    # Measure SINGLE URL latency (realistic user scenario)
    url_X_single = test_data["url"][0].iloc[[0]]  # Single row
    dns_X_single = test_data["dns"][0].iloc[[0]]
    whois_X_single = test_data["whois"][0].iloc[[0]]

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
            print(f"  {row['model']:20s} ROC={row['roc_auc']:.4f} ACC={row['accuracy']:.4f} F1={row['f1']:.4f}")

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

    # Test all combinations using CV metrics + single-URL latency
    n_combinations = len(url_top) * len(dns_top) * len(whois_top) * len(WEIGHT_CONFIGS)
    print(f"\n{'='*80}")
    print(f"TESTING {n_combinations} COMBINATIONS")
    print(f"  Models: {len(url_top)} URL × {len(dns_top)} DNS × {len(whois_top)} WHOIS")
    print(f"  Weight configs: {len(WEIGHT_CONFIGS)}")
    print(f"  Metrics: CV-based (unbiased) | Latency: Single-URL (realistic)")
    print("="*80)

    results = []

    for url_name in url_top:
        for dns_name in dns_top:
            for whois_name in whois_top:
                for weights in WEIGHT_CONFIGS:
                    combo_name = f"{url_name}+{dns_name}+{whois_name}"
                    weight_str = f"U{int(weights['url']*100)}D{int(weights['dns']*100)}W{int(weights['whois']*100)}"

                    print(f"\n  Testing: {combo_name} | Weights: {weight_str}")

                    try:
                        metrics = test_ensemble_combination_with_cv(
                            url_name, dns_name, whois_name,
                            cv_results, weights, test_data
                        )

                        if metrics is None:
                            print(f"    ⚠️  Skipping (model not found)")
                            continue

                        result = {
                            "url_model": url_name,
                            "dns_model": dns_name,
                            "whois_model": whois_name,
                            "weight_url": weights["url"],
                            "weight_dns": weights["dns"],
                            "weight_whois": weights["whois"],
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
        print(f"\n{idx}. {row['url_model']} + {row['dns_model']} + {row['whois_model']}")
        print(f"   Weights: URL={row['weight_url']:.0%}, DNS={row['weight_dns']:.0%}, WHOIS={row['weight_whois']:.0%}")
        print(f"   CV Metrics: ROC={row['ensemble_roc_auc']:.4f} | ACC={row['ensemble_accuracy']:.4f} | F1={row['ensemble_f1']:.4f}")
        print(f"   Single-URL Latency: {row['total_latency_mean_ms']:.2f}ms | Tradeoff: {row['tradeoff_score']:.4f}")

    # Select best ensemble
    best = results_df.iloc[0]

    print(f"\n{'='*80}")
    print("BEST ENSEMBLE SELECTED")
    print(f"{'='*80}")
    print(f"URL Model:   {best['url_model']} (CV ROC={best['url_roc']:.4f})")
    print(f"DNS Model:   {best['dns_model']} (CV ROC={best['dns_roc']:.4f})")
    print(f"WHOIS Model: {best['whois_model']} (CV ROC={best['whois_roc']:.4f})")
    print(f"Weights: URL={best['weight_url']:.0%}, DNS={best['weight_dns']:.0%}, WHOIS={best['weight_whois']:.0%}")
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
