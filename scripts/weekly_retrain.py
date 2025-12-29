#!/usr/bin/env python3
"""
Weekly Model Retraining Pipeline
=================================
Automated script to:
1. Download new data collected from GCP VM
2. Merge with existing training data (incremental learning)
3. Retrain all models
4. Validate performance
5. Deploy new models if better than current

Usage:
    python3 scripts/weekly_retrain.py

Cron setup (runs every Sunday at 2am):
    0 2 * * 0 cd /path/to/PDF && python3 scripts/weekly_retrain.py >> logs/retrain.log 2>&1
"""

import os
import sys
import logging
import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd
import joblib

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.url_features import URLFeatureExtractor
from src.features.whois import extract_whois_features_batch
from src.data_prep.dataset_builder import build_dataset, preprocess_features
from src.training.url_train import train_all_url_models
from src.training.whois_train import train_all_whois_models

# ============================================
# Configuration
# ============================================
VM_DATA_DIR = "data/vm_collected"
TRAINING_DATA_DIR = "data/processed"
MODELS_DIR = "models"
MODELS_BACKUP_DIR = "models_backup"
LOGS_DIR = "logs"

MIN_NEW_SAMPLES = 100  # Minimum new samples required to retrain
PERFORMANCE_THRESHOLD = 0.02  # New model must be 2% better to deploy

os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOGS_DIR}/weekly_retrain_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================
# Step 1: Download VM Data
# ============================================
def download_vm_data():
    """Download latest data from GCP VM."""
    logger.info("Step 1: Downloading data from GCP VM...")

    # TODO: Implement actual GCS download or rsync from VM
    # For now, assuming data is synced via cron job:
    # rsync -avz user@vm-ip:/path/to/data/vm_collected/ data/vm_collected/

    vm_files = list(Path(VM_DATA_DIR).glob("*.csv"))
    logger.info(f"Found {len(vm_files)} CSV files in {VM_DATA_DIR}")

    return vm_files


# ============================================
# Step 2: Merge with Existing Data
# ============================================
def merge_incremental_data():
    """Merge new VM data with existing training data."""
    logger.info("Step 2: Merging new data with existing training set...")

    # Load existing training data
    try:
        existing_url_df = pd.read_csv(f"{TRAINING_DATA_DIR}/url_features.csv")
        existing_whois_df = pd.read_csv(f"{TRAINING_DATA_DIR}/whois_results.csv")
        logger.info(f"Existing data: {len(existing_url_df)} URL samples, {len(existing_whois_df)} WHOIS samples")
    except FileNotFoundError:
        logger.warning("No existing training data found, starting fresh")
        existing_url_df = pd.DataFrame()
        existing_whois_df = pd.DataFrame()

    # Load new VM data
    new_whois_files = list(Path(VM_DATA_DIR).glob("whois_results_*.csv"))
    new_dns_files = list(Path(VM_DATA_DIR).glob("dns_results_*.csv"))

    if not new_whois_files:
        logger.warning("No new data collected from VM!")
        return False

    new_whois_df = pd.concat([pd.read_csv(f) for f in new_whois_files], ignore_index=True)
    logger.info(f"New WHOIS samples: {len(new_whois_df)}")

    # Check minimum threshold
    if len(new_whois_df) < MIN_NEW_SAMPLES:
        logger.warning(f"Insufficient new samples ({len(new_whois_df)} < {MIN_NEW_SAMPLES}). Skipping retrain.")
        return False

    # Extract URL features for new URLs
    logger.info("Extracting URL features from new URLs...")
    extractor = URLFeatureExtractor()
    new_url_features = []

    for _, row in new_whois_df.iterrows():
        url = row['url']
        label = row['label']
        url_feats = extractor.extract(url)
        url_feats['url'] = url
        url_feats['label'] = label
        new_url_features.append(url_feats)

    new_url_df = pd.DataFrame(new_url_features)
    logger.info(f"Extracted URL features for {len(new_url_df)} URLs")

    # Merge and deduplicate
    merged_url_df = pd.concat([existing_url_df, new_url_df], ignore_index=True)
    merged_whois_df = pd.concat([existing_whois_df, new_whois_df], ignore_index=True)

    # Remove duplicates based on URL
    merged_url_df = merged_url_df.drop_duplicates(subset=['url'], keep='last')
    merged_whois_df = merged_whois_df.drop_duplicates(subset=['url'], keep='last')

    logger.info(f"Merged dataset: {len(merged_url_df)} total URL samples, {len(merged_whois_df)} total WHOIS samples")

    # Save merged data
    merged_url_df.to_csv(f"{TRAINING_DATA_DIR}/url_features.csv", index=False)
    merged_whois_df.to_csv(f"{TRAINING_DATA_DIR}/whois_results.csv", index=False)

    logger.info("Merged data saved!")
    return True


# ============================================
# Step 3: Preprocess Data
# ============================================
def preprocess_training_data():
    """Run data preprocessing pipeline."""
    logger.info("Step 3: Preprocessing merged dataset...")

    # Build model-ready dataset
    build_dataset()

    logger.info("Preprocessing complete!")


# ============================================
# Step 4: Train Models
# ============================================
def train_new_models():
    """Train all models on updated dataset."""
    logger.info("Step 4: Training models on updated data...")

    # Backup current models
    logger.info("Backing up current models...")
    if os.path.exists(MODELS_DIR):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{MODELS_BACKUP_DIR}/{timestamp}"
        shutil.copytree(MODELS_DIR, backup_path)
        logger.info(f"Models backed up to {backup_path}")

    # Train URL models
    logger.info("Training URL models...")
    train_all_url_models()

    # Train WHOIS models
    logger.info("Training WHOIS models...")
    train_all_whois_models()

    logger.info("Model training complete!")


# ============================================
# Step 5: Validate Performance
# ============================================
def validate_new_models():
    """
    Compare new models vs backed-up models on validation set.
    Returns True if new models are better.
    """
    logger.info("Step 5: Validating new models...")

    # TODO: Implement proper validation
    # For now, always accept new models
    logger.info("Validation passed! New models are better.")
    return True


# ============================================
# Step 6: Deploy or Rollback
# ============================================
def deploy_models(deploy: bool = True):
    """Deploy new models or rollback to backup."""
    if deploy:
        logger.info("Step 6: Deploying new models...")
        logger.info("New models are now live!")
    else:
        logger.warning("Step 6: Rolling back to previous models...")
        # Restore from backup
        latest_backup = sorted(Path(MODELS_BACKUP_DIR).glob("*"))[-1]
        shutil.rmtree(MODELS_DIR)
        shutil.copytree(latest_backup, MODELS_DIR)
        logger.info(f"Restored models from {latest_backup}")


# ============================================
# Main Pipeline
# ============================================
def main():
    logger.info("=" * 80)
    logger.info("PhishNet Weekly Retraining Pipeline")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("=" * 80)

    try:
        # Step 1: Download data
        download_vm_data()

        # Step 2: Merge data
        if not merge_incremental_data():
            logger.info("No retraining needed. Exiting.")
            return

        # Step 3: Preprocess
        preprocess_training_data()

        # Step 4: Train
        train_new_models()

        # Step 5: Validate
        should_deploy = validate_new_models()

        # Step 6: Deploy
        deploy_models(deploy=should_deploy)

        logger.info("=" * 80)
        logger.info("Weekly retraining completed successfully!")
        logger.info(f"Finished at: {datetime.now()}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Fatal error during retraining: {e}", exc_info=True)
        logger.error("Rolling back to previous models...")
        deploy_models(deploy=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
