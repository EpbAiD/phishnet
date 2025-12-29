#!/usr/bin/env python3
"""
Automated MLOps Pipeline for PhishNet
======================================
End-to-end automated pipeline:
1. Data Collection (VM) → Feature Extraction → Database Update
2. Data Validation → Feature Engineering → Model Training
3. Model Evaluation → A/B Testing → Deployment
4. Monitoring → Auto-Rollback if performance degrades

Architecture:
- Runs continuously in background
- Triggers retraining when data/performance thresholds met
- Automatic model validation and deployment
- Performance monitoring and alerting
"""

import os
import sys
import time
import json
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================
# Configuration
# ============================================
class PipelineConfig:
    # Directories
    VM_DATA_DIR = "data/vm_collected"
    PROCESSED_DIR = "data/processed"
    MODELS_DIR = "models"
    MODELS_BACKUP_DIR = "models_backup"
    LOGS_DIR = "logs"

    # Thresholds
    MIN_NEW_URLS = 100  # Minimum new URLs before triggering retrain
    MAX_TRAINING_INTERVAL_HOURS = 24  # Force retrain after 24 hours (daily updates)
    MIN_ACCURACY_THRESHOLD = 0.85  # Minimum acceptable accuracy
    MAX_FALSE_POSITIVE_RATE = 0.05  # Maximum 5% false positives

    # Model performance tracking
    PERFORMANCE_HISTORY_FILE = f"{LOGS_DIR}/model_performance_history.json"
    PIPELINE_STATE_FILE = f"{LOGS_DIR}/pipeline_state.json"

    # Monitoring
    ALERT_EMAIL = "eb3658@columbia.edu"  # Email for alerts
    SLACK_WEBHOOK = None  # Optional Slack webhook for alerts

os.makedirs(PipelineConfig.LOGS_DIR, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{PipelineConfig.LOGS_DIR}/mlops_pipeline_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================
# Pipeline State Management
# ============================================
class PipelineState:
    """Track pipeline state across runs"""

    def __init__(self):
        self.state = self.load_state()

    def load_state(self) -> dict:
        """Load pipeline state from disk"""
        if os.path.exists(PipelineConfig.PIPELINE_STATE_FILE):
            with open(PipelineConfig.PIPELINE_STATE_FILE, 'r') as f:
                return json.load(f)

        return {
            "last_training_time": None,
            "last_data_sync_time": None,
            "last_training_data_size": 0,
            "current_model_version": "v0.0.0",
            "current_model_accuracy": 0.0,
            "total_runs": 0,
            "successful_deployments": 0,
            "failed_deployments": 0
        }

    def save_state(self):
        """Save pipeline state to disk"""
        with open(PipelineConfig.PIPELINE_STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def update(self, **kwargs):
        """Update state with new values"""
        self.state.update(kwargs)
        self.save_state()


# ============================================
# Stage 1: Data Collection & Sync
# ============================================
class DataCollectionStage:
    """Sync data from VM and check if new data available"""

    @staticmethod
    def run() -> Tuple[bool, int]:
        """
        Sync data from VM and check if new data available.

        Returns:
            (has_new_data, new_data_count)
        """
        logger.info("=" * 80)
        logger.info("STAGE 1: Data Collection & Sync")
        logger.info("=" * 80)

        # Check VM collector status
        logger.info("Checking VM collector status...")
        result = subprocess.run(
            ["./scripts/vm_manager.sh", "status"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if "NOT RUNNING" in result.stdout:
            logger.warning("⚠️ VM collector is not running! Starting it...")
            subprocess.run(["./scripts/vm_manager.sh", "start"], timeout=60)

        # Sync data from VM
        logger.info("Syncing data from VM...")
        result = subprocess.run(
            ["./scripts/vm_manager.sh", "sync"],
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout
        )

        if result.returncode != 0:
            logger.error(f"Data sync failed: {result.stderr}")
            return False, 0

        # Count VM-collected features (not URLs)
        vm_data_files = [
            f"{PipelineConfig.VM_DATA_DIR}/whois_results.csv",
            f"{PipelineConfig.VM_DATA_DIR}/dns_results.csv"
        ]

        total_vm_features = 0
        for filepath in vm_data_files:
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                total_vm_features += len(df)

        # Check existing processed datasets
        processed_datasets = [
            f"{PipelineConfig.PROCESSED_DIR}/url_features_modelready_imputed.csv",
            f"{PipelineConfig.PROCESSED_DIR}/dns_features_modelready_imputed.csv",
            f"{PipelineConfig.PROCESSED_DIR}/whois_features_modelready_imputed.csv"
        ]

        total_processed = 0
        for filepath in processed_datasets:
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                total_processed = max(total_processed, len(df))

        # For daily updates, we don't rely on count comparison
        # Instead, we signal that new data is available for time-based retraining
        # Return the number of VM features collected as an indicator
        new_data_count = total_vm_features

        logger.info(f"📊 VM features collected: {total_vm_features} (WHOIS + DNS)")
        logger.info(f"📊 Processed training data: {total_processed} rows")
        logger.info(f"📊 Data collection status: {'Active' if total_vm_features > 0 else 'Inactive'}")

        # Mark as having new data if VM collector is active
        has_new_data = total_vm_features > 0

        return has_new_data, new_data_count


# ============================================
# Stage 2: Data Processing & Feature Engineering
# ============================================
class DataProcessingStage:
    """Merge VM data with training data and extract features"""

    @staticmethod
    def run() -> bool:
        """
        Process and merge data, extract features.

        Returns:
            success (bool)
        """
        logger.info("=" * 80)
        logger.info("STAGE 2: Data Processing & Feature Engineering")
        logger.info("=" * 80)

        try:
            # Import data preparation modules
            from src.data_prep.dataset_builder import (
                build_url_modelready,
                build_dns_modelready,
                build_whois_modelready
            )

            # Build all model-ready datasets
            logger.info("Building URL model-ready dataset...")
            url_df, url_df_imp = build_url_modelready()

            logger.info("Building DNS model-ready dataset...")
            dns_df, dns_df_imp = build_dns_modelready()

            logger.info("Building WHOIS model-ready dataset...")
            whois_df, whois_df_imp = build_whois_modelready()

            # Verify dataset was created
            url_modelready_path = f"{PipelineConfig.PROCESSED_DIR}/url_features_modelready_imputed.csv"
            if not os.path.exists(url_modelready_path):
                logger.error("Dataset building failed - url_features_modelready_imputed.csv not found")
                return False

            df = pd.read_csv(url_modelready_path)
            logger.info(f"✅ Dataset ready: {len(df)} total URLs")

            return True

        except Exception as e:
            logger.error(f"Data processing failed: {e}", exc_info=True)
            return False


# ============================================
# Stage 3: Model Training
# ============================================
class ModelTrainingStage:
    """Train all models (URL, WHOIS, Ensemble)"""

    @staticmethod
    def run() -> bool:
        """
        Train all models.

        Returns:
            success (bool)
        """
        logger.info("=" * 80)
        logger.info("STAGE 3: Model Training")
        logger.info("=" * 80)

        try:
            # Backup current models
            if os.path.exists(PipelineConfig.MODELS_DIR):
                backup_path = f"{PipelineConfig.MODELS_BACKUP_DIR}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                logger.info(f"Backing up current models to {backup_path}")
                os.makedirs(backup_path, exist_ok=True)

                import shutil
                for model_file in Path(PipelineConfig.MODELS_DIR).glob("*.pkl"):
                    shutil.copy2(model_file, backup_path)

            # Train top 4 models for each feature type
            best_models = ["catboost", "lgbm", "xgb", "rf"]

            # Train URL models
            logger.info(f"Training URL models: {best_models}...")
            from src.training.url_train import train_all_url_models
            train_all_url_models(subset=best_models)

            # Train DNS models
            logger.info(f"Training DNS models: {best_models}...")
            from src.training.dns_train import train_all_dns_models
            train_all_dns_models(subset=best_models)

            # Train WHOIS models
            logger.info(f"Training WHOIS models: {best_models}...")
            from src.training.whois_train import train_all_whois_models
            train_all_whois_models()

            logger.info("✅ Model training complete! (4 models × 3 feature types = 12 models)")
            return True

        except Exception as e:
            logger.error(f"Model training failed: {e}", exc_info=True)
            return False


# ============================================
# Stage 4: Model Evaluation
# ============================================
class ModelEvaluationStage:
    """Evaluate trained models and compare with previous version"""

    @staticmethod
    def run(state: PipelineState) -> Tuple[bool, dict]:
        """
        Evaluate models and compare with previous version.

        Returns:
            (should_deploy, metrics)
        """
        logger.info("=" * 80)
        logger.info("STAGE 4: Model Evaluation")
        logger.info("=" * 80)

        try:
            # Load test data from model-ready dataset
            test_df = pd.read_csv(f"{PipelineConfig.PROCESSED_DIR}/url_features_modelready_imputed.csv")

            # Filter to valid labels only
            test_df = test_df[test_df['label'].isin([0, 1])]

            # Take random 20% as test set (not just last 20%)
            test_size = int(len(test_df) * 0.2)
            test_df = test_df.sample(n=test_size, random_state=42)

            logger.info(f"Test set size: {len(test_df)} URLs")

            # Evaluate using predict_utils
            from src.api.predict_utils import predict_ensemble_risk

            predictions = []
            actuals = []

            for _, row in test_df.iterrows():
                url = row['url']

                # Skip rows with invalid labels
                try:
                    actual_label = int(row['label'])  # 0=legitimate, 1=phishing
                    if actual_label not in [0, 1]:
                        continue
                except (ValueError, TypeError):
                    logger.warning(f"Skipping row with invalid label: {row['label']}")
                    continue

                try:
                    phish_prob, legit_prob, verdict, _ = predict_ensemble_risk(url)
                    prediction = 1 if phish_prob > 0.5 else 0

                    predictions.append(prediction)
                    actuals.append(actual_label)
                except Exception as e:
                    logger.warning(f"Prediction failed for {url}: {e}")

            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            accuracy = accuracy_score(actuals, predictions)
            precision = precision_score(actuals, predictions, pos_label=1, zero_division=0)
            recall = recall_score(actuals, predictions, pos_label=1, zero_division=0)
            f1 = f1_score(actuals, predictions, pos_label=1, zero_division=0)

            # Calculate false positive rate
            fp = sum(1 for a, p in zip(actuals, predictions) if a == 0 and p == 1)
            tn = sum(1 for a, p in zip(actuals, predictions) if a == 0 and p == 0)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "false_positive_rate": fpr,
                "test_size": len(actuals),
                "timestamp": datetime.now().isoformat()
            }

            logger.info("📊 Model Performance:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1 Score: {f1:.4f}")
            logger.info(f"  False Positive Rate: {fpr:.4f}")

            # Save metrics history
            ModelEvaluationStage.save_metrics_history(metrics)

            # Decide if should deploy
            should_deploy = (
                accuracy >= PipelineConfig.MIN_ACCURACY_THRESHOLD and
                fpr <= PipelineConfig.MAX_FALSE_POSITIVE_RATE and
                accuracy >= state.state["current_model_accuracy"]  # Better than current
            )

            if should_deploy:
                logger.info("✅ New model meets deployment criteria!")
            else:
                logger.warning(f"⚠️ New model does not meet deployment criteria:")
                logger.warning(f"  - Accuracy: {accuracy:.4f} (required: {PipelineConfig.MIN_ACCURACY_THRESHOLD})")
                logger.warning(f"  - FPR: {fpr:.4f} (max: {PipelineConfig.MAX_FALSE_POSITIVE_RATE})")
                logger.warning(f"  - Current model accuracy: {state.state['current_model_accuracy']:.4f}")

            return should_deploy, metrics

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}", exc_info=True)
            return False, {}

    @staticmethod
    def save_metrics_history(metrics: dict):
        """Append metrics to history file"""
        history = []
        if os.path.exists(PipelineConfig.PERFORMANCE_HISTORY_FILE):
            with open(PipelineConfig.PERFORMANCE_HISTORY_FILE, 'r') as f:
                history = json.load(f)

        history.append(metrics)

        with open(PipelineConfig.PERFORMANCE_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)


# ============================================
# Stage 5: Model Deployment
# ============================================
class ModelDeploymentStage:
    """Deploy new models to production"""

    @staticmethod
    def run(state: PipelineState, metrics: dict) -> bool:
        """
        Deploy new models to production.

        Returns:
            success (bool)
        """
        logger.info("=" * 80)
        logger.info("STAGE 5: Model Deployment")
        logger.info("=" * 80)

        try:
            # Models are already in models/ directory from training
            # Just need to update state

            # Increment version
            current_version = state.state["current_model_version"]
            major, minor, patch = map(int, current_version.lstrip('v').split('.'))
            new_version = f"v{major}.{minor}.{patch + 1}"

            state.update(
                current_model_version=new_version,
                current_model_accuracy=metrics["accuracy"],
                last_training_time=datetime.now().isoformat(),
                successful_deployments=state.state["successful_deployments"] + 1
            )

            logger.info(f"✅ Models deployed! Version: {new_version}")
            logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")

            return True

        except Exception as e:
            logger.error(f"Model deployment failed: {e}", exc_info=True)
            state.update(
                failed_deployments=state.state["failed_deployments"] + 1
            )
            return False


# ============================================
# Stage 6: Model Rollback (if needed)
# ============================================
class ModelRollbackStage:
    """Rollback to previous model version"""

    @staticmethod
    def run(state: PipelineState) -> bool:
        """
        Rollback to most recent backup.

        Returns:
            success (bool)
        """
        logger.info("=" * 80)
        logger.info("STAGE 6: Model Rollback")
        logger.info("=" * 80)

        try:
            # Find most recent backup
            backups = sorted(Path(PipelineConfig.MODELS_BACKUP_DIR).glob("*"))
            if not backups:
                logger.error("No model backups found!")
                return False

            latest_backup = backups[-1]
            logger.info(f"Rolling back to backup: {latest_backup}")

            # Restore models
            import shutil
            for model_file in latest_backup.glob("*.pkl"):
                shutil.copy2(model_file, PipelineConfig.MODELS_DIR)

            logger.info("✅ Models rolled back successfully")
            return True

        except Exception as e:
            logger.error(f"Model rollback failed: {e}", exc_info=True)
            return False


# ============================================
# Main Pipeline Orchestrator
# ============================================
class MLOpsPipeline:
    """Main pipeline orchestrator"""

    def __init__(self):
        self.state = PipelineState()

    def should_trigger_training(self, new_data_count: int) -> bool:
        """
        Decide if training should be triggered.

        Triggers if:
        - Enough new data (>= MIN_NEW_URLS)
        - OR it's been > MAX_TRAINING_INTERVAL_HOURS since last training
        """
        # Check data threshold
        if new_data_count >= PipelineConfig.MIN_NEW_URLS:
            logger.info(f"✅ Training triggered: {new_data_count} new URLs (>= {PipelineConfig.MIN_NEW_URLS})")
            return True

        # Check time threshold
        if self.state.state["last_training_time"]:
            last_training = datetime.fromisoformat(self.state.state["last_training_time"])
            hours_since_training = (datetime.now() - last_training).total_seconds() / 3600

            if hours_since_training >= PipelineConfig.MAX_TRAINING_INTERVAL_HOURS:
                logger.info(f"✅ Training triggered: {hours_since_training:.1f} hours since last training")
                return True
        else:
            # Never trained before
            logger.info("✅ Training triggered: First time training")
            return True

        logger.info(f"⏭️  Training skipped: Only {new_data_count} new URLs, {hours_since_training:.1f} hours since last")
        return False

    def run_pipeline(self) -> bool:
        """
        Run full MLOps pipeline.

        Returns:
            success (bool)
        """
        logger.info("\n\n")
        logger.info("=" * 80)
        logger.info("🚀 MLOPS PIPELINE STARTED")
        logger.info(f"Run #{self.state.state['total_runs'] + 1}")
        logger.info(f"Current model: {self.state.state['current_model_version']} (Accuracy: {self.state.state['current_model_accuracy']:.4f})")
        logger.info("=" * 80)

        self.state.update(total_runs=self.state.state["total_runs"] + 1)

        try:
            # Stage 1: Data Collection & Sync
            has_new_data, new_data_count = DataCollectionStage.run()

            if not self.should_trigger_training(new_data_count):
                logger.info("Pipeline execution skipped - no training needed")
                return True

            # Stage 2: Data Processing
            if not DataProcessingStage.run():
                logger.error("❌ Data processing failed - aborting pipeline")
                return False

            # Stage 3: Model Training
            if not ModelTrainingStage.run():
                logger.error("❌ Model training failed - aborting pipeline")
                return False

            # Stage 4: Model Evaluation
            should_deploy, metrics = ModelEvaluationStage.run(self.state)

            if not should_deploy:
                logger.warning("❌ New model does not meet deployment criteria")
                logger.info("Rolling back to previous model...")
                ModelRollbackStage.run(self.state)
                return False

            # Stage 5: Model Deployment
            if not ModelDeploymentStage.run(self.state, metrics):
                logger.error("❌ Model deployment failed")
                logger.info("Rolling back to previous model...")
                ModelRollbackStage.run(self.state)
                return False

            logger.info("=" * 80)
            logger.info("✅ MLOPS PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return False


# ============================================
# Continuous Execution Loop
# ============================================
def run_continuous_pipeline():
    """Run pipeline continuously with configurable intervals"""

    pipeline = MLOpsPipeline()
    CHECK_INTERVAL_HOURS = 6  # Check every 6 hours

    logger.info("🚀 Starting continuous MLOps pipeline...")
    logger.info(f"   Check interval: {CHECK_INTERVAL_HOURS} hours")
    logger.info(f"   Min new URLs for training: {PipelineConfig.MIN_NEW_URLS}")
    logger.info(f"   Max training interval: {PipelineConfig.MAX_TRAINING_INTERVAL_HOURS} hours")

    while True:
        try:
            pipeline.run_pipeline()

            # Wait before next check
            logger.info(f"\n⏰ Next pipeline check in {CHECK_INTERVAL_HOURS} hours...")
            time.sleep(CHECK_INTERVAL_HOURS * 3600)

        except KeyboardInterrupt:
            logger.info("\n\n⚠️ Pipeline stopped by user")
            break
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            logger.info("Retrying in 1 hour...")
            time.sleep(3600)


# ============================================
# Entry Point
# ============================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PhishNet MLOps Pipeline")
    parser.add_argument("--once", action="store_true", help="Run pipeline once and exit")
    parser.add_argument("--continuous", action="store_true", help="Run pipeline continuously")

    args = parser.parse_args()

    if args.continuous:
        run_continuous_pipeline()
    else:
        # Run once (default)
        pipeline = MLOpsPipeline()
        success = pipeline.run_pipeline()
        sys.exit(0 if success else 1)
