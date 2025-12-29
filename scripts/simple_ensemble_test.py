#!/usr/bin/env python3
"""
Simple Ensemble Test
Loads test URLs and benchmarks different ensemble configurations using existing predict_utils
"""
import sys
sys.path.insert(0, '/Users/eeshanbhanap/Desktop/PDF')

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load test data
logger.info("Loading test URLs...")
url_df = pd.read_csv("data/processed/url_features_modelready_imputed.csv")
url_df = url_df[url_df['label'].isin([0, 1])]
test_urls = url_df.sample(n=min(200, len(url_df)), random_state=42)

logger.info(f"Loaded {len(test_urls)} test URLs")

# Import prediction functions
from src.api.predict_utils import predict_url_risk, predict_whois_risk, predict_dns_risk, predict_ensemble_risk

def benchmark_single(predict_func, urls, name):
    """Benchmark a single prediction function"""
    logger.info(f"\nTesting {name}...")

    latencies = []
    predictions = []
    actuals = []

    for idx, row in urls.iterrows():
        url = row['url']
        label = int(row['label'])

        try:
            start = time.perf_counter()
            phish_prob, legit_prob, verdict, details = predict_func(url)
            latency_ms = (time.perf_counter() - start) * 1000

            latencies.append(latency_ms)
            predictions.append(1 if phish_prob > 0.5 else 0)
            actuals.append(label)
        except Exception as e:
            logger.warning(f"Prediction failed for {url}: {e}")
            continue

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions, pos_label=1, zero_division=0)
    recall = recall_score(actuals, predictions, pos_label=1, zero_division=0)
    f1 = f1_score(actuals, predictions, pos_label=1, zero_division=0)

    fp = sum(1 for a, p in zip(actuals, predictions) if a == 0 and p == 1)
    tn = sum(1 for a, p in zip(actuals, predictions) if a == 0 and p == 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    results = {
        "name": name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "fpr": float(fpr),
        "latency_avg_ms": float(np.mean(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "test_size": len(actuals)
    }

    # Calculate composite score
    max_latency = 300
    normalized_latency = min(results['latency_p95_ms'] / max_latency, 1.0)
    results['composite_score'] = (
        accuracy * 0.4 +
        (1 - fpr) * 0.3 +
        (1 - normalized_latency) * 0.2 +
        f1 * 0.1
    )

    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  FPR: {fpr:.4f}")
    logger.info(f"  Latency (p95): {results['latency_p95_ms']:.2f} ms")
    logger.info(f"  Composite Score: {results['composite_score']:.4f}")

    return results

# Run tests
logger.info("=" * 80)
logger.info("ENSEMBLE COMPARISON")
logger.info("=" * 80)

results = []

# Test 1: URL only
results.append(benchmark_single(predict_url_risk, test_urls, "URL Only (E1)"))

# Test 2: WHOIS only
results.append(benchmark_single(predict_whois_risk, test_urls, "WHOIS Only"))

# Test 3: DNS only
results.append(benchmark_single(predict_dns_risk, test_urls, "DNS Only"))

# Test 4: Ensemble (current production: URL + WHOIS)
results.append(benchmark_single(predict_ensemble_risk, test_urls, "Ensemble: URL + WHOIS (E3)"))

# Sort by composite score
results.sort(key=lambda x: x['composite_score'], reverse=True)

# Save results
output_dir = Path("analysis/ensemble_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"simple_comparison_{timestamp}.json"

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

logger.info(f"\n💾 Results saved to: {output_file}")

# Print recommendations
logger.info("\n" + "=" * 80)
logger.info("🎯 RECOMMENDATIONS")
logger.info("=" * 80)

best = results[0]
logger.info(f"\n✅ BEST OVERALL: {best['name']}")
logger.info(f"   Score: {best['composite_score']:.4f}")
logger.info(f"   Accuracy: {best['accuracy']:.4f}")
logger.info(f"   Latency (p95): {best['latency_p95_ms']:.2f} ms")

fastest = min(results, key=lambda x: x['latency_p95_ms'])
logger.info(f"\n⚡ FASTEST: {fastest['name']}")
logger.info(f"   Latency (p95): {fastest['latency_p95_ms']:.2f} ms")
logger.info(f"   Accuracy: {fastest['accuracy']:.4f}")

most_accurate = max(results, key=lambda x: x['accuracy'])
logger.info(f"\n🎯 MOST ACCURATE: {most_accurate['name']}")
logger.info(f"   Accuracy: {most_accurate['accuracy']:.4f}")
logger.info(f"   Latency (p95): {most_accurate['latency_p95_ms']:.2f} ms")

logger.info("\n✅ Ensemble comparison complete!")
