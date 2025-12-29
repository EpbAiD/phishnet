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


def predict_with_timing(model, X):
    """Predict with timing measurement."""
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
    Test a specific ensemble combination.

    Args:
        url_model, dns_model, whois_model: Trained models
        test_data: Dict with test data for each type
        weights: Dict with ensemble weights
        sample_size: Number of samples to test (for latency measurement)

    Returns:
        Dict with metrics: accuracy, f1, roc_auc, latency, tradeoff_score
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
    # Normalize: accuracy [0-1], latency [0-5000ms typical]
    # Weight accuracy more heavily (latency weight = 0.3)
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
    print("ENSEMBLE COMBINATION TESTING")
    print("="*80)

    # Load CV results to get best models
    print("\nLoading CV results to identify best models...")

    url_results = pd.read_csv(f"{ANALYSIS_DIR}/url_cv_results.csv")
    dns_results = pd.read_csv(f"{ANALYSIS_DIR}/dns_cv_results.csv")
    whois_results = pd.read_csv(f"{ANALYSIS_DIR}/whois_cv_results.csv")

    # Get top 3 models for each type
    url_top = url_results.nlargest(3, "roc_auc")["model"].values
    dns_top = dns_results.nlargest(3, "roc_auc")["model"].values
    whois_top = whois_results.nlargest(3, "roc_auc")["model"].values

    print(f"\nTop URL models: {list(url_top)}")
    print(f"Top DNS models: {list(dns_top)}")
    print(f"Top WHOIS models: {list(whois_top)}")

    # Load test data
    print("\nLoading test data...")
    test_data = load_test_data()
    print(f"  URL: {len(test_data['url'][0])} samples")
    print(f"  DNS: {len(test_data['dns'][0])} samples")
    print(f"  WHOIS: {len(test_data['whois'][0])} samples")

    # Test all combinations
    print(f"\nTesting {len(url_top) * len(dns_top) * len(whois_top) * len(WEIGHT_CONFIGS)} combinations...")
    print(f"  Models: {len(url_top)} URL × {len(dns_top)} DNS × {len(whois_top)} WHOIS")
    print(f"  Weight configs: {len(WEIGHT_CONFIGS)}")

    results = []

    for url_name in url_top:
        url_model = load_model("url", url_name)
        if not url_model:
            print(f"  ⚠️  Skipping {url_name} (model not found)")
            continue

        for dns_name in dns_top:
            dns_model = load_model("dns", dns_name)
            if not dns_model:
                print(f"  ⚠️  Skipping {dns_name} (model not found)")
                continue

            for whois_name in whois_top:
                whois_model = load_model("whois", whois_name)
                if not whois_model:
                    print(f"  ⚠️  Skipping {whois_name} (model not found)")
                    continue

                for weights in WEIGHT_CONFIGS:
                    combo_name = f"{url_name}+{dns_name}+{whois_name}"
                    weight_str = f"U{int(weights['url']*100)}D{int(weights['dns']*100)}W{int(weights['whois']*100)}"

                    print(f"\n  Testing: {combo_name} | Weights: {weight_str}")

                    try:
                        metrics = test_ensemble_combination(
                            url_model, dns_model, whois_model,
                            test_data, weights, sample_size=1000
                        )

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

                        print(f"    Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
                        print(f"    F1 Score: {metrics['f1']:.4f}")
                        print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
                        print(f"    Latency: {metrics['avg_latency_ms']:.2f}ms/URL")
                        print(f"    Tradeoff: {metrics['tradeoff_score']:.4f}")

                    except Exception as e:
                        print(f"    ❌ Failed: {e}")

    # Save results
    if not results:
        print("\n❌ No successful ensemble combinations found!")
        print("All combinations failed. Check:")
        print("  1. Do all three datasets have common URLs?")
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

    for i, row in results_df.head(5).iterrows():
        print(f"\n{i+1}. {row['url_model']} + {row['dns_model']} + {row['whois_model']}")
        print(f"   Weights: URL={row['weight_url']:.0%}, DNS={row['weight_dns']:.0%}, WHOIS={row['weight_whois']:.0%}")
        print(f"   Accuracy: {row['accuracy']:.4f} | F1: {row['f1']:.4f} | ROC-AUC: {row['roc_auc']:.4f}")
        print(f"   Latency: {row['avg_latency_ms']:.2f}ms | Tradeoff: {row['tradeoff_score']:.4f}")

    # Select best ensemble
    best = results_df.iloc[0]

    print(f"\n{'='*80}")
    print("BEST ENSEMBLE SELECTED")
    print(f"{'='*80}")
    print(f"URL Model: {best['url_model']}")
    print(f"DNS Model: {best['dns_model']}")
    print(f"WHOIS Model: {best['whois_model']}")
    print(f"Weights: URL={best['weight_url']:.0%}, DNS={best['weight_dns']:.0%}, WHOIS={best['weight_whois']:.0%}")
    print(f"\nPerformance:")
    print(f"  Accuracy: {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
    print(f"  F1 Score: {best['f1']:.4f}")
    print(f"  ROC-AUC: {best['roc_auc']:.4f}")
    print(f"  Avg Latency: {best['avg_latency_ms']:.2f}ms per URL")
    print(f"  Tradeoff Score: {best['tradeoff_score']:.4f}")

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
        "performance": {
            "accuracy": float(best["accuracy"]),
            "f1": float(best["f1"]),
            "roc_auc": float(best["roc_auc"]),
            "avg_latency_ms": float(best["avg_latency_ms"]),
            "tradeoff_score": float(best["tradeoff_score"])
        }
    }

    # Save updated metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Production metadata updated: {metadata_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
