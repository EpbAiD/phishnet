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

ENSEMBLE_CONFIGS = {
    "E1_URL_ONLY": {
        "name": "URL Only",
        "description": "Fastest - no external API calls needed",
        "models": {
            "url": "url_catboost.pkl"
        },
        "weights": {
            "url": 1.0
        },
        "requires_dns": False,
        "requires_whois": False,
        "expected_latency_ms": 25
    },

    "E2_URL_DNS": {
        "name": "URL + DNS",
        "description": "Fast with network signals",
        "models": {
            "url": "url_catboost.pkl",
            "dns": "dns_lgbm.pkl"  # LightGBM fastest for DNS
        },
        "weights": {
            "url": 0.70,
            "dns": 0.30
        },
        "requires_dns": True,
        "requires_whois": False,
        "expected_latency_ms": 90
    },

    "E3_URL_WHOIS": {
        "name": "URL + WHOIS (Current Production)",
        "description": "Current production ensemble",
        "models": {
            "url": "url_catboost.pkl",
            "whois": "whois_catboost.pkl"
        },
        "weights": {
            "url": 0.60,
            "whois": 0.40
        },
        "requires_dns": False,
        "requires_whois": True,
        "expected_latency_ms": 175
    },

    "E4_DNS_WHOIS": {
        "name": "DNS + WHOIS",
        "description": "Infrastructure-based detection",
        "models": {
            "dns": "dns_catboost.pkl",
            "whois": "whois_catboost.pkl"
        },
        "weights": {
            "dns": 0.50,
            "whois": 0.50
        },
        "requires_dns": True,
        "requires_whois": True,
        "expected_latency_ms": 220
    },

    "E5_ALL_EQUAL": {
        "name": "All 3 Equal Weights",
        "description": "Balanced ensemble - all features equal",
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
        "expected_latency_ms": 250
    },

    "E6_ALL_OPTIMIZED": {
        "name": "All 3 Optimized Weights",
        "description": "Best accuracy - optimized weights via grid search",
        "models": {
            "url": "url_catboost.pkl",
            "dns": "dns_xgb.pkl",
            "whois": "whois_lgbm.pkl"
        },
        "weights": {
            "url": 0.50,
            "dns": 0.20,
            "whois": 0.30
        },
        "requires_dns": True,
        "requires_whois": True,
        "expected_latency_ms": 230
    },

    "E7_ALL_SPEED": {
        "name": "All 3 Speed-Optimized",
        "description": "Lowest latency - lightweight models",
        "models": {
            "url": "url_lgbm.pkl",
            "dns": "dns_lgbm.pkl",
            "whois": "whois_lgbm.pkl"
        },
        "weights": {
            "url": 0.50,
            "dns": 0.25,
            "whois": 0.25
        },
        "requires_dns": True,
        "requires_whois": True,
        "expected_latency_ms": 120
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
        for feature_type, model_file in self.config["models"].items():
            model_path = models_dir / model_file

            if not model_path.exists():
                logger.warning(f"Model not found: {model_path} - skipping {feature_type}")
                continue

            try:
                self.models[feature_type] = joblib.load(model_path)
                logger.info(f"Loaded {feature_type} model: {model_file}")
            except Exception as e:
                logger.error(f"Failed to load {model_path}: {e}")

    def predict(self, features: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """
        Make weighted ensemble prediction.

        Args:
            features: Dict with keys 'url', 'dns', 'whois' (DataFrames or arrays)

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

            # Get prediction probabilities [legit_prob, phish_prob]
            proba = model.predict_proba(features[feature_type])

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

    Returns:
        (test_features, test_labels)
    """
    logger.info("Loading test data...")

    # Load URL features
    url_df = pd.read_csv("data/processed/url_features_modelready_imputed.csv")
    url_df = url_df[url_df['label'].isin([0, 1])]  # Filter valid labels
    url_df = url_df.sample(n=min(test_size, len(url_df)), random_state=42)

    # Load DNS features
    dns_df = pd.read_csv("data/processed/dns_features_modelready_imputed.csv")

    # Load WHOIS features
    whois_df = pd.read_csv("data/processed/whois_features_modelready_imputed.csv")

    # Merge on URL
    test_features = []
    test_labels = []

    for idx, url_row in url_df.iterrows():
        url_val = url_row['url']
        label = int(url_row['label'])

        # Get URL features (exclude url and label columns)
        url_feats = url_row.drop(['url', 'label']).values.reshape(1, -1)

        # Get DNS features (match by URL)
        dns_row = dns_df[dns_df['url'] == url_val]
        if not dns_row.empty:
            dns_feats = dns_row.drop(['url', 'label'], axis=1, errors='ignore').values[:1]
        else:
            # Use zeros if DNS features missing
            dns_feats = np.zeros((1, len(dns_df.columns) - 2))

        # Get WHOIS features (match by URL)
        whois_row = whois_df[whois_df['url'] == url_val]
        if not whois_row.empty:
            whois_feats = whois_row.drop(['url', 'label'], axis=1, errors='ignore').values[:1]
        else:
            # Use zeros if WHOIS features missing
            whois_feats = np.zeros((1, len(whois_df.columns) - 2))

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
