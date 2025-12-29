#!/usr/bin/env python3
"""
Benchmark Optimized Parallel Prediction System
===============================================
Compares:
1. Sequential (current): URL → WHOIS → DNS
2. Parallel + Cache: All features simultaneously with caching
"""

import sys
sys.path.insert(0, '/Users/eeshanbhanap/Desktop/PDF')

import time
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load test URLs
logger.info("Loading test URLs...")
url_df = pd.read_csv("data/processed/url_features_modelready_imputed.csv")
url_df = url_df[url_df['label'].isin([0, 1])]
test_urls = url_df.sample(n=min(50, len(url_df)), random_state=42)['url'].tolist()

logger.info(f"Loaded {len(test_urls)} test URLs")

# Import prediction functions
from src.api.predict_utils import predict_ensemble_risk  # Sequential
from src.api.fast_predict import predict_fast  # Parallel + Cache


async def benchmark_parallel(urls, use_cache=True):
    """
    Benchmark parallel prediction with caching.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"BENCHMARKING PARALLEL + CACHE ({'cache ON' if use_cache else 'cache OFF'})")
    logger.info(f"{'='*80}")

    latencies = []
    predictions = []
    actuals = []

    url_df_test = pd.read_csv("data/processed/url_features_modelready_imputed.csv")

    for url in urls:
        # Get label
        label_row = url_df_test[url_df_test['url'] == url]
        if label_row.empty:
            continue
        label = int(label_row.iloc[0]['label'])

        try:
            start = time.perf_counter()
            phish_prob, legit_prob, verdict, details = await predict_fast(
                url,
                timeout_whois=0.1,
                timeout_dns=0.1,
                use_cache=use_cache
            )
            latency_ms = (time.perf_counter() - start) * 1000

            latencies.append(latency_ms)
            predictions.append(1 if phish_prob > 0.5 else 0)
            actuals.append(label)

            features_used = details['features_used']
            logger.info(f"  {url[:50]:50s} | {latency_ms:6.2f}ms | URL:{features_used['url']} WHOIS:{features_used['whois']} DNS:{features_used['dns']}")

        except Exception as e:
            logger.error(f"Prediction failed for {url}: {e}")
            continue

    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score

    accuracy = accuracy_score(actuals, predictions)
    f1 = f1_score(actuals, predictions, pos_label=1, zero_division=0)

    fp = sum(1 for a, p in zip(actuals, predictions) if a == 0 and p == 1)
    tn = sum(1 for a, p in zip(actuals, predictions) if a == 0 and p == 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    results = {
        "method": f"Parallel + Cache ({'ON' if use_cache else 'OFF'})",
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "fpr": float(fpr),
        "latency_avg_ms": float(np.mean(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "latency_min_ms": float(np.min(latencies)),
        "latency_max_ms": float(np.max(latencies)),
        "test_size": len(actuals)
    }

    logger.info(f"\n📊 RESULTS:")
    logger.info(f"   Accuracy: {accuracy:.4f}")
    logger.info(f"   F1 Score: {f1:.4f}")
    logger.info(f"   FPR: {fpr:.4f}")
    logger.info(f"   Latency (avg): {results['latency_avg_ms']:.2f} ms")
    logger.info(f"   Latency (p50): {results['latency_p50_ms']:.2f} ms")
    logger.info(f"   Latency (p95): {results['latency_p95_ms']:.2f} ms")
    logger.info(f"   Latency (p99): {results['latency_p99_ms']:.2f} ms")
    logger.info(f"   Latency (min): {results['latency_min_ms']:.2f} ms")
    logger.info(f"   Latency (max): {results['latency_max_ms']:.2f} ms")

    return results


def benchmark_sequential(urls):
    """
    Benchmark current sequential prediction (baseline).
    """
    logger.info(f"\n{'='*80}")
    logger.info("BENCHMARKING SEQUENTIAL (CURRENT)")
    logger.info(f"{'='*80}")

    latencies = []
    predictions = []
    actuals = []

    url_df_test = pd.read_csv("data/processed/url_features_modelready_imputed.csv")

    for url in urls:
        # Get label
        label_row = url_df_test[url_df_test['url'] == url]
        if label_row.empty:
            continue
        label = int(label_row.iloc[0]['label'])

        try:
            start = time.perf_counter()
            phish_prob, legit_prob, verdict, details = predict_ensemble_risk(url)
            latency_ms = (time.perf_counter() - start) * 1000

            latencies.append(latency_ms)
            predictions.append(1 if phish_prob > 0.5 else 0)
            actuals.append(label)

            logger.info(f"  {url[:50]:50s} | {latency_ms:6.2f}ms")

        except Exception as e:
            logger.error(f"Prediction failed for {url}: {e}")
            continue

    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score

    accuracy = accuracy_score(actuals, predictions)
    f1 = f1_score(actuals, predictions, pos_label=1, zero_division=0)

    fp = sum(1 for a, p in zip(actuals, predictions) if a == 0 and p == 1)
    tn = sum(1 for a, p in zip(actuals, predictions) if a == 0 and p == 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    results = {
        "method": "Sequential (Current)",
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "fpr": float(fpr),
        "latency_avg_ms": float(np.mean(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "latency_min_ms": float(np.min(latencies)),
        "latency_max_ms": float(np.max(latencies)),
        "test_size": len(actuals)
    }

    logger.info(f"\n📊 RESULTS:")
    logger.info(f"   Accuracy: {accuracy:.4f}")
    logger.info(f"   F1 Score: {f1:.4f}")
    logger.info(f"   FPR: {fpr:.4f}")
    logger.info(f"   Latency (avg): {results['latency_avg_ms']:.2f} ms")
    logger.info(f"   Latency (p50): {results['latency_p50_ms']:.2f} ms")
    logger.info(f"   Latency (p95): {results['latency_p95_ms']:.2f} ms")
    logger.info(f"   Latency (p99): {results['latency_p99_ms']:.2f} ms")
    logger.info(f"   Latency (min): {results['latency_min_ms']:.2f} ms")
    logger.info(f"   Latency (max): {results['latency_max_ms']:.2f} ms")

    return results


async def main():
    """Run benchmarks"""
    logger.info("=" * 80)
    logger.info("PARALLEL PREDICTION OPTIMIZATION BENCHMARK")
    logger.info("=" * 80)

    all_results = []

    # Benchmark 1: Sequential (baseline)
    sequential_results = benchmark_sequential(test_urls[:10])  # Use fewer URLs for sequential (slow)
    all_results.append(sequential_results)

    # Benchmark 2: Parallel WITHOUT cache (first run)
    parallel_nocache_results = await benchmark_parallel(test_urls[:20], use_cache=False)
    all_results.append(parallel_nocache_results)

    # Benchmark 3: Parallel WITH cache (second run - cache warmed)
    parallel_cache_results = await benchmark_parallel(test_urls[:20], use_cache=True)
    all_results.append(parallel_cache_results)

    # Save results
    output_dir = Path("analysis/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"parallel_benchmark_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n💾 Results saved to: {output_file}")

    # Print comparison
    logger.info(f"\n{'='*80}")
    logger.info("🎯 COMPARISON SUMMARY")
    logger.info(f"{'='*80}")

    for result in all_results:
        logger.info(f"\n{result['method']}:")
        logger.info(f"   Latency (p95): {result['latency_p95_ms']:.2f} ms")
        logger.info(f"   Accuracy: {result['accuracy']:.4f}")

    # Calculate speedup
    if len(all_results) >= 3:
        speedup_nocache = sequential_results['latency_p95_ms'] / parallel_nocache_results['latency_p95_ms']
        speedup_cache = sequential_results['latency_p95_ms'] / parallel_cache_results['latency_p95_ms']

        logger.info(f"\n⚡ SPEEDUP:")
        logger.info(f"   Parallel (no cache): {speedup_nocache:.2f}x faster")
        logger.info(f"   Parallel (with cache): {speedup_cache:.2f}x faster")

    # Get cache stats
    from src.api.cache import get_cache
    cache = get_cache()
    cache_stats = cache.get_stats()

    logger.info(f"\n📊 CACHE STATISTICS:")
    logger.info(f"   Enabled: {cache_stats.get('enabled', False)}")
    if 'hit_rates' in cache_stats:
        hit_rates = cache_stats['hit_rates']
        logger.info(f"   WHOIS hit rate: {hit_rates['whois']['hit_rate']:.2%}")
        logger.info(f"   DNS hit rate: {hit_rates['dns']['hit_rate']:.2%}")
        logger.info(f"   Total keys: {cache_stats.get('total_keys', 0)}")

    logger.info("\n✅ Benchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())
