#!/usr/bin/env python3
"""
🎯 Ensemble Comparison Framework
=================================

Systematically tests different ensemble combinations to find:
1. Best accuracy ensemble
2. Lowest latency ensemble
3. Best balanced ensemble (accuracy + speed)

Tests 7 ensemble strategies:
- E1: URL only (fastest)
- E2: URL + DNS
- E3: URL + WHOIS (current production)
- E4: DNS + WHOIS
- E5: All 3 equal weights
- E6: All 3 optimized weights
- E7: All 3 speed-optimized (lightweight models)

Outputs:
- Comparison report (JSON + CSV)
- Latency benchmarks
- Recommendation for production deployment
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================
# Ensemble Configurations
# ============================================

# ============================================
# 3-Model Ensemble Configurations (URL + DNS + WHOIS)
# ============================================
# Focus: Finding the best combination of all 3 model types
# Each ensemble uses URL + DNS + WHOIS models with different algorithms

ENSEMBLE_CONFIGS = {
    "E1_CATBOOST_ALL": {
        "name": "CatBoost Ensemble (All 3)",
        "description": "All 3 models using CatBoost - best for categorical features",
        "models": {
            "url": "url_catboost.pkl",
            "dns": "dns_catboost.pkl",
            "whois": "whois_catboost.pkl"
        },
        "weights": {
            "url": 0.50,
            "dns": 0.25,
            "whois": 0.25
        },
        "requires_dns": True,
        "requires_whois": True,
        "expected_latency_ms": 100
    },

    "E2_LIGHTGBM_ALL": {
        "name": "LightGBM Ensemble (All 3)",
        "description": "All 3 models using LightGBM - fastest inference",
        "models": {
            "url": "url_lightgbm.pkl",
            "dns": "dns_lightgbm.pkl",
            "whois": "whois_lightgbm.pkl"
        },
        "weights": {
            "url": 0.50,
            "dns": 0.25,
            "whois": 0.25
        },
        "requires_dns": True,
        "requires_whois": True,
        "expected_latency_ms": 80
    },

    "E3_XGBOOST_ALL": {
        "name": "XGBoost Ensemble (All 3)",
        "description": "All 3 models using XGBoost - robust performance",
        "models": {
            "url": "url_xgboost.pkl",
            "dns": "dns_xgboost.pkl",
            "whois": "whois_xgboost.pkl"
        },
        "weights": {
            "url": 0.50,
            "dns": 0.25,
            "whois": 0.25
        },
        "requires_dns": True,
        "requires_whois": True,
        "expected_latency_ms": 90
    },

    "E4_RANDOM_FOREST_ALL": {
        "name": "Random Forest Ensemble (All 3)",
        "description": "All 3 models using Random Forest - interpretable",
        "models": {
            "url": "url_random_forest.pkl",
            "dns": "dns_random_forest.pkl",
            "whois": "whois_random_forest.pkl"
        },
        "weights": {
            "url": 0.50,
            "dns": 0.25,
            "whois": 0.25
        },
        "requires_dns": True,
        "requires_whois": True,
        "expected_latency_ms": 120
    },

    "E5_MIXED_BEST": {
        "name": "Mixed Best Models (All 3)",
        "description": "Best algorithm for each feature type",
        "models": {
            "url": "url_catboost.pkl",      # CatBoost for URL (handles categories well)
            "dns": "dns_lightgbm.pkl",       # LightGBM for DNS (fast)
            "whois": "whois_xgboost.pkl"     # XGBoost for WHOIS (robust)
        },
        "weights": {
            "url": 0.50,
            "dns": 0.25,
            "whois": 0.25
        },
        "requires_dns": True,
        "requires_whois": True,
        "expected_latency_ms": 95
    },

    "E6_URL_HEAVY": {
        "name": "URL-Heavy Ensemble (All 3)",
        "description": "Higher weight on URL features",
        "models": {
            "url": "url_catboost.pkl",
            "dns": "dns_catboost.pkl",
            "whois": "whois_catboost.pkl"
        },
        "weights": {
            "url": 0.60,
            "dns": 0.20,
            "whois": 0.20
        },
        "requires_dns": True,
        "requires_whois": True,
        "expected_latency_ms": 100
    },

    "E7_BALANCED": {
        "name": "Balanced Ensemble (All 3)",
        "description": "Equal weights for all 3 models",
        "models": {
            "url": "url_catboost.pkl",
            "dns": "dns_catboost.pkl",
            "whois": "whois_catboost.pkl"
        },
        "weights": {
            "url": 0.333,
            "dns": 0.333,
            "whois": 0.334
        },
        "requires_dns": True,
        "requires_whois": True,
        "expected_latency_ms": 100
    }
}


# ============================================
# Ensemble Predictor
# ============================================

class EnsemblePredictor:
    """Loads models and makes weighted ensemble predictions"""

    def __init__(self, ensemble_config: Dict):
        self.config = ensemble_config
        self.models = {}
        self.load_models()

    def load_models(self):
        """Load all models specified in config"""
        import joblib

        models_dir = Path("models")
        self.feature_cols = {}  # Store expected feature columns for each model type

        for feature_type, model_file in self.config["models"].items():
            model_path = models_dir / model_file

            if not model_path.exists():
                logger.warning(f"Model not found: {model_path} - skipping {feature_type}")
                continue

            try:
                self.models[feature_type] = joblib.load(model_path)
                logger.info(f"Loaded {feature_type} model: {model_file}")

                # Try to load the feature columns file
                feature_cols_file = model_path.with_name(model_file.replace('.pkl', '_feature_cols.pkl'))
                if feature_cols_file.exists():
                    self.feature_cols[feature_type] = joblib.load(feature_cols_file)
                    logger.info(f"  Loaded feature columns: {len(self.feature_cols[feature_type])} features")
            except Exception as e:
                logger.error(f"Failed to load {model_path}: {e}")

    def predict(self, features: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """
        Make weighted ensemble prediction.

        Args:
            features: Dict with keys 'url', 'dns', 'whois' containing feature arrays
                     that match the expected feature order for each model

        Returns:
            (phishing_prob, legitimate_prob)
        """
        if not self.models:
            raise ValueError("No models loaded!")

        weighted_phish_prob = 0.0
        weighted_legit_prob = 0.0
        total_weight = 0.0

        for feature_type, model in self.models.items():
            if feature_type not in features:
                logger.warning(f"Features missing for {feature_type} - skipping")
                continue

            weight = self.config["weights"].get(feature_type, 0.0)
            if weight == 0:
                continue

            # Get feature data
            feat_data = features[feature_type]

            # Get prediction probabilities [legit_prob, phish_prob]
            proba = model.predict_proba(feat_data)

            if len(proba.shape) == 1:
                # Binary output
                phish_prob = proba[0]
                legit_prob = 1 - phish_prob
            else:
                # Multi-class output [legit, phish]
                legit_prob = proba[0][0]
                phish_prob = proba[0][1]

            weighted_phish_prob += phish_prob * weight
            weighted_legit_prob += legit_prob * weight
            total_weight += weight

        # Normalize if total weight != 1.0
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            weighted_phish_prob /= total_weight
            weighted_legit_prob /= total_weight

        return weighted_phish_prob, weighted_legit_prob


# ============================================
# Benchmarking
# ============================================

def benchmark_latency(
    ensemble_predictor: EnsemblePredictor,
    test_features: List[Dict],
    n_iterations: int = 100
) -> Dict:
    """
    Benchmark prediction latency.

    Args:
        ensemble_predictor: Configured predictor
        test_features: List of feature dicts to test
        n_iterations: Number of iterations

    Returns:
        Latency statistics (ms)
    """
    latencies = []

    for i in range(n_iterations):
        # Random sample
        features = test_features[i % len(test_features)]

        start = time.perf_counter()
        ensemble_predictor.predict(features)
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # Convert to ms

    return {
        "avg_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "std_ms": float(np.std(latencies))
    }


def evaluate_accuracy(
    ensemble_predictor: EnsemblePredictor,
    test_features: List[Dict],
    test_labels: List[int]
) -> Dict:
    """
    Evaluate prediction accuracy.

    Args:
        ensemble_predictor: Configured predictor
        test_features: List of feature dicts
        test_labels: True labels (0=legit, 1=phish)

    Returns:
        Accuracy metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    predictions = []
    actuals = []

    for features, label in zip(test_features, test_labels):
        try:
            phish_prob, _ = ensemble_predictor.predict(features)
            prediction = 1 if phish_prob > 0.5 else 0

            predictions.append(prediction)
            actuals.append(label)
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            continue

    # Calculate metrics
    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions, pos_label=1, zero_division=0)
    recall = recall_score(actuals, predictions, pos_label=1, zero_division=0)
    f1 = f1_score(actuals, predictions, pos_label=1, zero_division=0)

    # False positive rate
    fp = sum(1 for a, p in zip(actuals, predictions) if a == 0 and p == 1)
    tn = sum(1 for a, p in zip(actuals, predictions) if a == 0 and p == 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "fpr": float(fpr),
        "test_size": len(actuals)
    }


# ============================================
# Main Comparison
# ============================================

def load_test_data(test_size: int = 1000) -> Tuple[List[Dict], List[int]]:
    """
    Load test data for all feature types.

    IMPORTANT: The current models were trained on COMBINED features (84 features each).
    This function loads the complete feature dataset and prepares features that match
    what the models expect.

    Returns:
        (test_features, test_labels)
    """
    import joblib

    processed_dir = Path("data/processed")
    models_dir = Path("models")

    # Load the complete feature dataset
    complete_data_path = processed_dir / "phishing_features_complete.csv"
    if not complete_data_path.exists():
        # Fallback to seed file if complete doesn't exist
        complete_data_path = processed_dir / "phishing_features_master_seed.csv"
    if not complete_data_path.exists():
        raise FileNotFoundError("No complete feature data found in data/processed/")

    logger.info(f"Loading complete feature data from {complete_data_path}...")
    df = pd.read_csv(complete_data_path)

    # Check for label column
    if 'label' in df.columns:
        df['label_encoded'] = (df['label'] == 'phishing').astype(int)
    label_col = 'label_encoded' if 'label_encoded' in df.columns else 'label'

    # Filter valid labels
    df = df[df[label_col].isin([0, 1])]
    logger.info(f"Loaded {len(df)} samples with valid labels")

    # Encode categorical columns to numeric
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['url', 'label']:
            df[col] = df[col].fillna('MISSING')
            df[col], _ = pd.factorize(df[col])

    # Try to load feature columns from actual model files
    # This supports SEPARATE feature architecture where each model has different features
    feature_cols = {}

    for feature_type in ['url', 'dns', 'whois']:
        # Try to load from model feature_cols file
        feature_cols_file = models_dir / f"{feature_type}_catboost_feature_cols.pkl"
        if feature_cols_file.exists():
            try:
                loaded_cols = joblib.load(feature_cols_file)
                feature_cols[feature_type] = loaded_cols
                logger.info(f"Loaded {feature_type} feature columns: {len(loaded_cols)} features")
                continue
            except Exception as e:
                logger.warning(f"Could not load {feature_cols_file}: {e}")

        # Fallback to hardcoded feature names from original extractors
        logger.info(f"Using hardcoded feature names for {feature_type}")

    # Define fallback feature groups from ORIGINAL extractors
    url_feature_names = [
        'url_length', 'hostname_length', 'path_length', 'num_subdomains', 'num_dots',
        'num_special_chars', 'num_digits', 'num_uppercase_chars', 'has_at_symbol',
        'has_double_slash_redirect', 'has_dash_in_domain', 'is_ip_address', 'ip_category',
        'has_encoded_chars', 'has_non_ascii_chars', 'url_entropy', 'hostname_entropy',
        'digit_to_letter_ratio', 'domain_quality', 'tld_length', 'subdomain_entropy',
        'subdomain_length', 'has_login_keyword', 'has_suspicious_words', 'has_brand_mismatch',
        'file_type', 'is_file_download', 'is_script_file', 'is_shortened', 'num_fragments',
        'num_query_params', 'num_directories', 'port', 'is_risky_port', 'protocol_mismatch',
        'is_unknown_port', 'contains_hex_encoding', 'starts_with_https_but_contains_http',
        'missing_hostname_flag'
    ]
    dns_feature_names = [
        'domain', 'has_A', 'num_A', 'has_AAAA', 'num_AAAA', 'has_MX', 'num_MX',
        'has_NS', 'num_NS', 'has_TXT', 'num_TXT', 'has_CNAME', 'cname_chain_length',
        'has_SOA', 'ttl_min', 'ttl_max', 'ttl_mean', 'ttl_var', 'mx_priority_min',
        'mx_priority_max', 'num_distinct_ips', 'txt_entropy', 'has_SPF', 'has_DKIM',
        'has_DMARC', 'has_wildcard_dns', 'dnssec_enabled', 'asn_list', 'asn_org_list',
        'asn_country_list', 'cidr_list', 'error_type'
    ]
    whois_feature_names = [
        'registrar', 'whois_server', 'creation_date', 'expiration_date', 'updated_date',
        'domain_age_days', 'registration_length_days', 'status', 'registrant_country',
        'has_privacy_protection', 'whois_success', 'error_msg'
    ]

    # Use fallbacks for any missing feature types
    if 'url' not in feature_cols:
        feature_cols['url'] = [c for c in url_feature_names if c in df.columns]
    if 'dns' not in feature_cols:
        feature_cols['dns'] = [c for c in dns_feature_names if c in df.columns]
    if 'whois' not in feature_cols:
        feature_cols['whois'] = [c for c in whois_feature_names if c in df.columns]

    logger.info(f"Feature counts: URL={len(feature_cols['url'])}, DNS={len(feature_cols['dns'])}, WHOIS={len(feature_cols['whois'])}")

    # Limit to test_size samples
    n_samples = min(len(df), test_size)
    logger.info(f"Using {n_samples} samples for evaluation")

    test_features = []
    test_labels = []

    for idx in range(n_samples):
        row = df.iloc[idx]
        label = int(row[label_col])

        # Get features for each type, filling missing with 0
        url_feats = row[feature_cols['url']].fillna(0).values.reshape(1, -1).astype(float)
        dns_feats = row[feature_cols['dns']].fillna(0).values.reshape(1, -1).astype(float)
        whois_feats = row[feature_cols['whois']].fillna(0).values.reshape(1, -1).astype(float)

        test_features.append({
            "url": url_feats,
            "dns": dns_feats,
            "whois": whois_feats
        })
        test_labels.append(label)

    logger.info(f"Loaded {len(test_features)} test samples")
    return test_features, test_labels


def run_ensemble_comparison(test_size: int = 1000, n_iterations: int = 100):
    """
    Run comprehensive ensemble comparison.

    Args:
        test_size: Number of test URLs
        n_iterations: Latency benchmark iterations
    """
    logger.info("=" * 80)
    logger.info("ENSEMBLE COMPARISON FRAMEWORK")
    logger.info("=" * 80)

    # Load test data
    test_features, test_labels = load_test_data(test_size)

    results = []

    # Test each ensemble
    for ensemble_id, config in ENSEMBLE_CONFIGS.items():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Testing {ensemble_id}: {config['name']}")
        logger.info(f"Description: {config['description']}")
        logger.info(f"{'=' * 80}")

        try:
            # Load ensemble
            predictor = EnsemblePredictor(config)

            # Benchmark latency
            logger.info("Benchmarking latency...")
            latency_stats = benchmark_latency(predictor, test_features, n_iterations)

            # Evaluate accuracy
            logger.info("Evaluating accuracy...")
            accuracy_stats = evaluate_accuracy(predictor, test_features, test_labels)

            # Calculate composite score
            # Score = (Accuracy × 0.4) + ((1 - FPR) × 0.3) + ((1 - Normalized_Latency) × 0.2) + (F1 × 0.1)
            max_latency = 300  # ms
            normalized_latency = min(latency_stats['p95_ms'] / max_latency, 1.0)

            composite_score = (
                accuracy_stats['accuracy'] * 0.4 +
                (1 - accuracy_stats['fpr']) * 0.3 +
                (1 - normalized_latency) * 0.2 +
                accuracy_stats['f1_score'] * 0.1
            )

            # Combine results
            result = {
                "ensemble_id": ensemble_id,
                "name": config['name'],
                "description": config['description'],
                "models": config['models'],
                "weights": config['weights'],
                "requires_dns": config['requires_dns'],
                "requires_whois": config['requires_whois'],

                # Accuracy metrics
                **accuracy_stats,

                # Latency metrics
                **latency_stats,

                # Overall score
                "composite_score": float(composite_score)
            }

            results.append(result)

            # Print summary
            logger.info(f"\n📊 Results for {ensemble_id}:")
            logger.info(f"   Accuracy: {accuracy_stats['accuracy']:.4f}")
            logger.info(f"   F1 Score: {accuracy_stats['f1_score']:.4f}")
            logger.info(f"   FPR: {accuracy_stats['fpr']:.4f}")
            logger.info(f"   Latency (p95): {latency_stats['p95_ms']:.2f} ms")
            logger.info(f"   Composite Score: {composite_score:.4f}")

        except Exception as e:
            logger.error(f"Failed to test {ensemble_id}: {e}", exc_info=True)
            continue

    # Sort by composite score
    results.sort(key=lambda x: x['composite_score'], reverse=True)

    # Save results
    output_dir = Path("analysis/ensemble_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = output_dir / f"comparison_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n💾 Saved JSON results: {json_path}")

    # Save CSV
    csv_path = output_dir / f"comparison_{timestamp}.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    logger.info(f"💾 Saved CSV results: {csv_path}")

    # Print recommendations
    logger.info(f"\n{'=' * 80}")
    logger.info("🎯 RECOMMENDATIONS")
    logger.info(f"{'=' * 80}")

    if not results:
        logger.warning("No ensemble results to report - all ensembles failed")
        return results

    best_overall = results[0]
    logger.info(f"\n✅ BEST OVERALL: {best_overall['name']}")
    logger.info(f"   Score: {best_overall['composite_score']:.4f}")
    logger.info(f"   Accuracy: {best_overall['accuracy']:.4f}")
    logger.info(f"   Latency (p95): {best_overall['p95_ms']:.2f} ms")

    fastest = min(results, key=lambda x: x['p95_ms'])
    logger.info(f"\n⚡ FASTEST: {fastest['name']}")
    logger.info(f"   Latency (p95): {fastest['p95_ms']:.2f} ms")
    logger.info(f"   Accuracy: {fastest['accuracy']:.4f}")

    most_accurate = max(results, key=lambda x: x['accuracy'])
    logger.info(f"\n🎯 MOST ACCURATE: {most_accurate['name']}")
    logger.info(f"   Accuracy: {most_accurate['accuracy']:.4f}")
    logger.info(f"   Latency (p95): {most_accurate['p95_ms']:.2f} ms")

    return results


# ============================================
# CLI Entry Point
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble Comparison Framework")
    parser.add_argument("--test-size", type=int, default=1000, help="Number of test URLs")
    parser.add_argument("--iterations", type=int, default=100, help="Latency benchmark iterations")

    args = parser.parse_args()

    results = run_ensemble_comparison(
        test_size=args.test_size,
        n_iterations=args.iterations
    )

    logger.info("\n✅ Ensemble comparison complete!")
