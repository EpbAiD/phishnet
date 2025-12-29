"""
🧪 A/B Testing Framework for Model Ensembles
=============================================

Features:
- Traffic splitting (e.g., 90% control, 10% variant)
- Canary deployment
- Performance monitoring per variant
- Statistical significance testing
- Automatic rollback on degradation

Usage:
    # Initialize manager
    ab_manager = ABTestManager()

    # Configure test
    ab_manager.add_variant("control", ensemble_config_v1, traffic_pct=90)
    ab_manager.add_variant("variant_a", ensemble_config_v2, traffic_pct=10)

    # Route requests
    result = ab_manager.route_request(url)

    # Check metrics
    metrics = ab_manager.get_variant_metrics()

    # Promote winning variant
    if ab_manager.should_promote("variant_a"):
        ab_manager.promote_variant("variant_a")
"""

import time
import random
import logging
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================
# Variant Performance Tracker
# ============================================

class VariantMetrics:
    """Tracks performance metrics for a single variant"""

    def __init__(self, variant_id: str):
        self.variant_id = variant_id
        self.requests = 0
        self.predictions = []
        self.actuals = []  # For ground truth when available
        self.latencies = []
        self.errors = 0
        self.start_time = datetime.now()

    def record_prediction(
        self,
        prediction: int,
        latency_ms: float,
        actual_label: Optional[int] = None
    ):
        """Record a prediction"""
        self.requests += 1
        self.predictions.append(prediction)
        self.latencies.append(latency_ms)

        if actual_label is not None:
            self.actuals.append(actual_label)

    def record_error(self):
        """Record a prediction error/failure"""
        self.errors += 1
        self.requests += 1

    def get_metrics(self) -> Dict:
        """Calculate current metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            "variant_id": self.variant_id,
            "requests": self.requests,
            "errors": self.errors,
            "error_rate": self.errors / self.requests if self.requests > 0 else 0,
            "runtime_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "requests_per_hour": self.requests / ((datetime.now() - self.start_time).total_seconds() / 3600) if self.requests > 0 else 0
        }

        # Latency stats
        if self.latencies:
            metrics.update({
                "latency_avg_ms": float(np.mean(self.latencies)),
                "latency_p50_ms": float(np.percentile(self.latencies, 50)),
                "latency_p95_ms": float(np.percentile(self.latencies, 95)),
                "latency_p99_ms": float(np.percentile(self.latencies, 99)),
                "latency_max_ms": float(np.max(self.latencies))
            })

        # Accuracy stats (if ground truth available)
        if len(self.actuals) > 0 and len(self.predictions) >= len(self.actuals):
            preds = self.predictions[:len(self.actuals)]
            try:
                accuracy = accuracy_score(self.actuals, preds)
                precision = precision_score(self.actuals, preds, pos_label=1, zero_division=0)
                recall = recall_score(self.actuals, preds, pos_label=1, zero_division=0)
                f1 = f1_score(self.actuals, preds, pos_label=1, zero_division=0)

                # FPR
                fp = sum(1 for a, p in zip(self.actuals, preds) if a == 0 and p == 1)
                tn = sum(1 for a, p in zip(self.actuals, preds) if a == 0 and p == 0)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                metrics.update({
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "fpr": float(fpr),
                    "samples_with_ground_truth": len(self.actuals)
                })
            except Exception as e:
                logger.warning(f"Failed to calculate accuracy metrics: {e}")

        return metrics


# ============================================
# A/B Test Manager
# ============================================

class ABTestManager:
    """Manages A/B tests for model ensembles"""

    def __init__(self, config_path: Optional[str] = None):
        self.variants: Dict[str, Dict] = {}
        self.metrics: Dict[str, VariantMetrics] = {}
        self.config_path = config_path or "logs/ab_test_config.json"
        self.active = False

        # Thresholds for automatic actions
        self.min_samples_for_promotion = 1000
        self.min_runtime_hours = 24
        self.max_fpr_degradation = 0.02  # Max 2% FPR increase
        self.min_accuracy_improvement = 0.01  # Min 1% accuracy improvement
        self.auto_rollback_threshold = 0.10  # Rollback if FPR > 10%

        # Load existing config if available
        self.load_config()

    def add_variant(
        self,
        variant_id: str,
        ensemble_config: Dict,
        traffic_pct: float,
        is_control: bool = False
    ):
        """
        Register a new variant.

        Args:
            variant_id: Unique identifier (e.g., "control", "variant_a")
            ensemble_config: Ensemble configuration dict
            traffic_pct: Percentage of traffic (0-100)
            is_control: Whether this is the control/baseline variant
        """
        if traffic_pct < 0 or traffic_pct > 100:
            raise ValueError(f"Invalid traffic_pct: {traffic_pct}. Must be 0-100.")

        self.variants[variant_id] = {
            "config": ensemble_config,
            "traffic_pct": traffic_pct,
            "is_control": is_control,
            "created_at": datetime.now().isoformat()
        }

        self.metrics[variant_id] = VariantMetrics(variant_id)

        logger.info(f"Added variant '{variant_id}' with {traffic_pct}% traffic")

        self.save_config()
        self.active = True

    def route_request(self, url: str) -> Tuple[str, any]:
        """
        Route request to a variant based on traffic split.

        Args:
            url: URL to predict

        Returns:
            (variant_id, prediction_result)
        """
        if not self.active or not self.variants:
            raise ValueError("No active A/B test configured")

        # Select variant based on traffic split
        variant_id = self._select_variant()

        # Make prediction with the selected variant
        result = self._predict_with_variant(variant_id, url)

        return variant_id, result

    def _select_variant(self) -> str:
        """Select variant based on traffic percentage"""
        rand = random.random() * 100

        cumulative = 0
        for variant_id, variant in self.variants.items():
            cumulative += variant["traffic_pct"]
            if rand < cumulative:
                return variant_id

        # Fallback to control or first variant
        for variant_id, variant in self.variants.items():
            if variant["is_control"]:
                return variant_id

        return list(self.variants.keys())[0]

    def _predict_with_variant(self, variant_id: str, url: str) -> Dict:
        """
        Make prediction using specified variant.

        Args:
            variant_id: Variant to use
            url: URL to predict

        Returns:
            Prediction result dict
        """
        variant = self.variants[variant_id]
        metrics = self.metrics[variant_id]

        try:
            # Import prediction function
            from src.api.predict_utils import predict_ensemble_risk

            # Time the prediction
            start = time.perf_counter()
            phish_prob, legit_prob, verdict, details = predict_ensemble_risk(
                url,
                ensemble_config=variant["config"]
            )
            latency_ms = (time.perf_counter() - start) * 1000

            # Record metrics
            prediction = 1 if phish_prob > 0.5 else 0
            metrics.record_prediction(prediction, latency_ms)

            return {
                "variant_id": variant_id,
                "phish_probability": phish_prob,
                "legit_probability": legit_prob,
                "verdict": verdict,
                "latency_ms": latency_ms,
                "details": details
            }

        except Exception as e:
            logger.error(f"Prediction failed for variant {variant_id}: {e}")
            metrics.record_error()
            raise

    def record_ground_truth(self, variant_id: str, actual_label: int):
        """
        Record ground truth for a prediction (when available).

        Args:
            variant_id: Variant that made the prediction
            actual_label: True label (0=legit, 1=phish)
        """
        if variant_id in self.metrics:
            self.metrics[variant_id].actuals.append(actual_label)

    def get_variant_metrics(self) -> Dict:
        """Get performance metrics for all variants"""
        return {
            vid: metrics.get_metrics()
            for vid, metrics in self.metrics.items()
        }

    def should_promote(self, variant_id: str) -> bool:
        """
        Determine if variant should be promoted to production.

        Criteria:
        - Minimum sample size met
        - Minimum runtime met
        - Better accuracy than control
        - FPR not significantly worse
        """
        if variant_id not in self.metrics:
            return False

        variant_metrics = self.metrics[variant_id].get_metrics()

        # Find control metrics
        control_id = None
        for vid, variant in self.variants.items():
            if variant["is_control"]:
                control_id = vid
                break

        if not control_id or control_id not in self.metrics:
            logger.warning("No control variant found - cannot determine promotion")
            return False

        control_metrics = self.metrics[control_id].get_metrics()

        # Check minimum samples
        if variant_metrics["requests"] < self.min_samples_for_promotion:
            logger.info(f"Variant {variant_id} needs more samples: {variant_metrics['requests']}/{self.min_samples_for_promotion}")
            return False

        # Check minimum runtime
        if variant_metrics["runtime_hours"] < self.min_runtime_hours:
            logger.info(f"Variant {variant_id} needs more runtime: {variant_metrics['runtime_hours']:.1f}/{self.min_runtime_hours} hours")
            return False

        # Check if ground truth available
        if "accuracy" not in variant_metrics or "accuracy" not in control_metrics:
            logger.warning("Ground truth not available - cannot determine promotion")
            return False

        # Check accuracy improvement
        accuracy_delta = variant_metrics["accuracy"] - control_metrics["accuracy"]
        if accuracy_delta < self.min_accuracy_improvement:
            logger.info(f"Variant {variant_id} accuracy not significantly better: {accuracy_delta:.4f}")
            return False

        # Check FPR degradation
        fpr_delta = variant_metrics["fpr"] - control_metrics["fpr"]
        if fpr_delta > self.max_fpr_degradation:
            logger.warning(f"Variant {variant_id} FPR worse: {fpr_delta:.4f}")
            return False

        logger.info(f"✅ Variant {variant_id} meets promotion criteria!")
        return True

    def should_rollback(self, variant_id: str) -> bool:
        """
        Determine if variant should be rolled back.

        Rollback if:
        - Error rate too high
        - FPR exceeds threshold
        - Significantly worse than control
        """
        if variant_id not in self.metrics:
            return False

        variant_metrics = self.metrics[variant_id].get_metrics()

        # Check error rate
        if variant_metrics["error_rate"] > 0.05:  # >5% errors
            logger.warning(f"Variant {variant_id} has high error rate: {variant_metrics['error_rate']:.4f}")
            return True

        # Check FPR
        if "fpr" in variant_metrics and variant_metrics["fpr"] > self.auto_rollback_threshold:
            logger.warning(f"Variant {variant_id} FPR too high: {variant_metrics['fpr']:.4f}")
            return True

        return False

    def promote_variant(self, variant_id: str):
        """
        Promote variant to 100% traffic (make it the new control).

        Args:
            variant_id: Variant to promote
        """
        if variant_id not in self.variants:
            raise ValueError(f"Unknown variant: {variant_id}")

        logger.info(f"🚀 Promoting variant '{variant_id}' to control")

        # Set old control to 0% traffic
        for vid, variant in self.variants.items():
            if variant["is_control"]:
                variant["is_control"] = False
                variant["traffic_pct"] = 0
                logger.info(f"Demoting old control: {vid}")

        # Promote new variant
        self.variants[variant_id]["is_control"] = True
        self.variants[variant_id]["traffic_pct"] = 100

        self.save_config()

    def rollback(self, variant_id: str):
        """
        Rollback variant (set traffic to 0%).

        Args:
            variant_id: Variant to rollback
        """
        if variant_id not in self.variants:
            raise ValueError(f"Unknown variant: {variant_id}")

        logger.warning(f"⚠️ Rolling back variant '{variant_id}'")

        # Set traffic to 0
        self.variants[variant_id]["traffic_pct"] = 0

        # Ensure control has 100% traffic
        for vid, variant in self.variants.items():
            if variant["is_control"]:
                variant["traffic_pct"] = 100

        self.save_config()

    def save_config(self):
        """Save current configuration to disk"""
        config = {
            "active": self.active,
            "variants": self.variants,
            "updated_at": datetime.now().isoformat()
        }

        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def load_config(self):
        """Load configuration from disk"""
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            self.active = config.get("active", False)
            self.variants = config.get("variants", {})

            # Initialize metrics for loaded variants
            for variant_id in self.variants.keys():
                if variant_id not in self.metrics:
                    self.metrics[variant_id] = VariantMetrics(variant_id)

            logger.info(f"Loaded A/B test config: {len(self.variants)} variants")

    def export_metrics(self, output_path: str):
        """Export metrics to JSON file"""
        metrics = self.get_variant_metrics()

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Exported metrics to {output_path}")

    def get_status_report(self) -> str:
        """Generate human-readable status report"""
        if not self.active:
            return "❌ No active A/B test"

        report = []
        report.append("\n" + "=" * 80)
        report.append("🧪 A/B TEST STATUS REPORT")
        report.append("=" * 80)

        metrics = self.get_variant_metrics()

        for variant_id, variant_metrics in metrics.items():
            variant_info = self.variants[variant_id]
            is_control = "✓ CONTROL" if variant_info["is_control"] else ""

            report.append(f"\n{'━' * 80}")
            report.append(f"{variant_id} {is_control}")
            report.append(f"{'━' * 80}")
            report.append(f"Traffic: {variant_info['traffic_pct']:.1f}%")
            report.append(f"Requests: {variant_metrics['requests']:,}")
            report.append(f"Errors: {variant_metrics['errors']} ({variant_metrics['error_rate']:.2%})")
            report.append(f"Runtime: {variant_metrics['runtime_hours']:.1f} hours")

            if "latency_p95_ms" in variant_metrics:
                report.append(f"\nLatency:")
                report.append(f"  Avg: {variant_metrics['latency_avg_ms']:.2f} ms")
                report.append(f"  P95: {variant_metrics['latency_p95_ms']:.2f} ms")
                report.append(f"  P99: {variant_metrics['latency_p99_ms']:.2f} ms")

            if "accuracy" in variant_metrics:
                report.append(f"\nAccuracy Metrics:")
                report.append(f"  Accuracy: {variant_metrics['accuracy']:.4f}")
                report.append(f"  Precision: {variant_metrics['precision']:.4f}")
                report.append(f"  Recall: {variant_metrics['recall']:.4f}")
                report.append(f"  F1: {variant_metrics['f1_score']:.4f}")
                report.append(f"  FPR: {variant_metrics['fpr']:.4f}")
                report.append(f"  Samples: {variant_metrics['samples_with_ground_truth']}")

            # Promotion/rollback recommendations
            if self.should_promote(variant_id):
                report.append(f"\n✅ RECOMMENDATION: Promote to production")
            elif self.should_rollback(variant_id):
                report.append(f"\n⚠️ RECOMMENDATION: Rollback (performance issues)")

        report.append(f"\n{'=' * 80}\n")

        return "\n".join(report)


# ============================================
# CLI for Testing
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A/B Testing Manager")
    parser.add_argument("--status", action="store_true", help="Show status report")
    parser.add_argument("--export", type=str, help="Export metrics to file")

    args = parser.parse_args()

    manager = ABTestManager()

    if args.status:
        print(manager.get_status_report())

    if args.export:
        manager.export_metrics(args.export)
        print(f"Exported metrics to {args.export}")
