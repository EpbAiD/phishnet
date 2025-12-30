#!/usr/bin/env python3
"""
Prepare Model-Ready Datasets
============================
Converts main datasets into model-ready format with:
  - url, label, bucket columns
  - Feature columns only (no metadata)
  - Stratification bucket for cross-validation

Creates 3 model-ready datasets:
  - data/processed/url_features_modelready.csv
  - data/processed/dns_features_modelready.csv
  - data/processed/whois_features_modelready.csv

Also creates imputed versions for models that require NaN-free data.

Usage:
  python3 scripts/prepare_modelready.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.impute import SimpleImputer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Main datasets
MAIN_DNS = os.getenv("MAIN_DNS", "data/phishing_features_complete_dns.csv")
MAIN_WHOIS = os.getenv("MAIN_WHOIS", "data/phishing_features_complete_whois.csv")
MAIN_URL = os.getenv("MAIN_URL", "data/phishing_features_complete_url.csv")

# Model-ready outputs
MODELREADY_DNS = os.getenv("MODELREADY_DNS", "data/processed/dns_features_modelready.csv")
MODELREADY_WHOIS = os.getenv("MODELREADY_WHOIS", "data/processed/whois_features_modelready.csv")
MODELREADY_URL = os.getenv("MODELREADY_URL", "data/processed/url_features_modelready.csv")

# Imputed versions
MODELREADY_DNS_IMP = os.getenv("MODELREADY_DNS_IMP", "data/processed/dns_features_modelready_imputed.csv")
MODELREADY_WHOIS_IMP = os.getenv("MODELREADY_WHOIS_IMP", "data/processed/whois_features_modelready_imputed.csv")
MODELREADY_URL_IMP = os.getenv("MODELREADY_URL_IMP", "data/processed/url_features_modelready_imputed.csv")


def log(message):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def assign_bucket(df):
    """
    Assign stratification bucket to each URL for cross-validation.

    Bucket is determined by:
      - source column if present
      - label (phishing vs legitimate)

    Returns DataFrame with 'bucket' column.
    """
    df = df.copy()

    # Normalize label
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    df['label'] = df['label'].replace({
        'legit': 'legitimate',
        'benign': 'legitimate',
        '0': 'legitimate',
        '1': 'phishing',
    })

    # Convert to binary: phishing=1, legitimate=0
    df['label'] = df['label'].apply(lambda x: 1 if x == 'phishing' else 0)

    # Assign bucket
    if 'source' in df.columns:
        df['bucket'] = df['source'].fillna('unknown')
    else:
        # Fallback: use label as bucket
        df['bucket'] = df['label'].apply(lambda x: 'phishing' if x == 1 else 'legitimate')

    return df


def prepare_dns_modelready():
    """Prepare DNS model-ready dataset."""
    log("=" * 60)
    log("Preparing DNS Model-Ready Dataset")
    log("=" * 60)

    if not os.path.exists(MAIN_DNS):
        log(f"❌ Main DNS dataset not found: {MAIN_DNS}")
        return None

    # Load
    df = pd.read_csv(MAIN_DNS)
    log(f"Loaded {len(df)} rows from {MAIN_DNS}")

    # Assign bucket and normalize label
    df = assign_bucket(df)

    # Select columns: url, label, bucket + DNS features
    meta_cols = ['url', 'label', 'bucket']
    feature_cols = [c for c in df.columns if c not in meta_cols and c not in ['source', 'error']]

    df_modelready = df[meta_cols + feature_cols]
    log(f"Model-ready dataset: {len(df_modelready)} rows, {len(feature_cols)} features")

    # Check for NaN
    nan_ratio = df_modelready[feature_cols].isna().mean().mean()
    log(f"NaN ratio: {nan_ratio:.2%}")

    # Save native version
    os.makedirs(os.path.dirname(MODELREADY_DNS), exist_ok=True)
    df_modelready.to_csv(MODELREADY_DNS, index=False)
    log(f"✅ Saved: {MODELREADY_DNS}")

    # Create imputed version (for models that don't handle NaN)
    df_imputed = df_modelready.copy()

    # Impute numeric columns with median
    numeric_cols = df_imputed[feature_cols].select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Remove columns that are all NaN (cannot impute)
        valid_numeric_cols = [c for c in numeric_cols if not df_imputed[c].isna().all()]
        if len(valid_numeric_cols) > 0:
            imputer = SimpleImputer(strategy='median')
            df_imputed[valid_numeric_cols] = imputer.fit_transform(df_imputed[valid_numeric_cols])

    # Impute categorical columns with most frequent
    cat_cols = df_imputed[feature_cols].select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        # Remove columns that are all NaN (cannot impute)
        valid_cat_cols = [c for c in cat_cols if not df_imputed[c].isna().all()]
        if len(valid_cat_cols) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df_imputed[valid_cat_cols] = imputer_cat.fit_transform(df_imputed[valid_cat_cols])

    df_imputed.to_csv(MODELREADY_DNS_IMP, index=False)
    log(f"✅ Saved imputed: {MODELREADY_DNS_IMP}")
    log("")

    return df_modelready


def prepare_whois_modelready():
    """Prepare WHOIS model-ready dataset."""
    log("=" * 60)
    log("Preparing WHOIS Model-Ready Dataset")
    log("=" * 60)

    if not os.path.exists(MAIN_WHOIS):
        log(f"❌ Main WHOIS dataset not found: {MAIN_WHOIS}")
        return None

    # Load
    df = pd.read_csv(MAIN_WHOIS)
    log(f"Loaded {len(df)} rows from {MAIN_WHOIS}")

    # Assign bucket and normalize label
    df = assign_bucket(df)

    # Select columns: url, label, bucket + WHOIS features
    meta_cols = ['url', 'label', 'bucket']
    feature_cols = [c for c in df.columns if c not in meta_cols and c not in ['source', 'error']]

    df_modelready = df[meta_cols + feature_cols]
    log(f"Model-ready dataset: {len(df_modelready)} rows, {len(feature_cols)} features")

    # Check for NaN
    nan_ratio = df_modelready[feature_cols].isna().mean().mean()
    log(f"NaN ratio: {nan_ratio:.2%}")

    # Save native version
    os.makedirs(os.path.dirname(MODELREADY_WHOIS), exist_ok=True)
    df_modelready.to_csv(MODELREADY_WHOIS, index=False)
    log(f"✅ Saved: {MODELREADY_WHOIS}")

    # Create imputed version
    df_imputed = df_modelready.copy()

    # Impute numeric columns with median
    numeric_cols = df_imputed[feature_cols].select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Remove columns that are all NaN (cannot impute)
        valid_numeric_cols = [c for c in numeric_cols if not df_imputed[c].isna().all()]
        if len(valid_numeric_cols) > 0:
            imputer = SimpleImputer(strategy='median')
            df_imputed[valid_numeric_cols] = imputer.fit_transform(df_imputed[valid_numeric_cols])

    # Impute categorical columns with most frequent
    cat_cols = df_imputed[feature_cols].select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        # Remove columns that are all NaN (cannot impute)
        valid_cat_cols = [c for c in cat_cols if not df_imputed[c].isna().all()]
        if len(valid_cat_cols) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df_imputed[valid_cat_cols] = imputer_cat.fit_transform(df_imputed[valid_cat_cols])

    df_imputed.to_csv(MODELREADY_WHOIS_IMP, index=False)
    log(f"✅ Saved imputed: {MODELREADY_WHOIS_IMP}")
    log("")

    return df_modelready


def prepare_url_modelready():
    """Prepare URL model-ready dataset."""
    log("=" * 60)
    log("Preparing URL Model-Ready Dataset")
    log("=" * 60)

    if not os.path.exists(MAIN_URL):
        log(f"❌ Main URL dataset not found: {MAIN_URL}")
        return None

    # Load
    df = pd.read_csv(MAIN_URL)
    log(f"Loaded {len(df)} rows from {MAIN_URL}")

    # Assign bucket and normalize label
    df = assign_bucket(df)

    # Select columns: url, label, bucket + URL features
    meta_cols = ['url', 'label', 'bucket']
    feature_cols = [c for c in df.columns if c not in meta_cols and c not in ['source', 'error']]

    df_modelready = df[meta_cols + feature_cols]
    log(f"Model-ready dataset: {len(df_modelready)} rows, {len(feature_cols)} features")

    # Check for NaN
    nan_ratio = df_modelready[feature_cols].isna().mean().mean()
    log(f"NaN ratio: {nan_ratio:.2%}")

    # Save native version
    os.makedirs(os.path.dirname(MODELREADY_URL), exist_ok=True)
    df_modelready.to_csv(MODELREADY_URL, index=False)
    log(f"✅ Saved: {MODELREADY_URL}")

    # Create imputed version
    df_imputed = df_modelready.copy()

    # Impute numeric columns with median
    numeric_cols = df_imputed[feature_cols].select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Remove columns that are all NaN (cannot impute)
        valid_numeric_cols = [c for c in numeric_cols if not df_imputed[c].isna().all()]
        if len(valid_numeric_cols) > 0:
            imputer = SimpleImputer(strategy='median')
            df_imputed[valid_numeric_cols] = imputer.fit_transform(df_imputed[valid_numeric_cols])

    # Impute categorical columns with most frequent
    cat_cols = df_imputed[feature_cols].select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        # Remove columns that are all NaN (cannot impute)
        valid_cat_cols = [c for c in cat_cols if not df_imputed[c].isna().all()]
        if len(valid_cat_cols) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df_imputed[valid_cat_cols] = imputer_cat.fit_transform(df_imputed[valid_cat_cols])

    df_imputed.to_csv(MODELREADY_URL_IMP, index=False)
    log(f"✅ Saved imputed: {MODELREADY_URL_IMP}")
    log("")

    return df_modelready


def check_row_counts():
    """Verify all model-ready datasets have matching row counts."""
    log("=" * 60)
    log("Verification: Checking row counts")
    log("=" * 60)

    counts = {}

    if os.path.exists(MODELREADY_DNS):
        counts['DNS'] = len(pd.read_csv(MODELREADY_DNS))

    if os.path.exists(MODELREADY_WHOIS):
        counts['WHOIS'] = len(pd.read_csv(MODELREADY_WHOIS))

    if os.path.exists(MODELREADY_URL):
        counts['URL'] = len(pd.read_csv(MODELREADY_URL))

    log(f"Row counts:")
    for name, count in counts.items():
        log(f"  {name}: {count}")

    if len(set(counts.values())) > 1:
        log("⚠️  WARNING: Row counts don't match across datasets!")
    else:
        log("✅ All datasets have matching row counts")

    log("")


def check_data_growth_and_prepare():
    """
    Intelligent data growth detection + model-ready preparation.

    Returns True if retraining needed, False otherwise.
    """
    METADATA_FILE = "models/production_metadata.json"

    log("=" * 60)
    log("Model-Ready Dataset Preparation with Intelligent Detection")
    log("=" * 60)
    log("")

    # Check if main datasets exist
    if not all([os.path.exists(MAIN_DNS), os.path.exists(MAIN_WHOIS), os.path.exists(MAIN_URL)]):
        log("⚠️  Main datasets not found - first time setup")
        return False

    # Get current row counts from main datasets
    current_dns_rows = len(pd.read_csv(MAIN_DNS))
    current_whois_rows = len(pd.read_csv(MAIN_WHOIS))
    current_url_rows = len(pd.read_csv(MAIN_URL))

    log(f"Current main dataset rows:")
    log(f"  DNS: {current_dns_rows}")
    log(f"  WHOIS: {current_whois_rows}")
    log(f"  URL: {current_url_rows}")

    # Load previous training metadata
    if os.path.exists(METADATA_FILE):
        import json
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)

        prev_rows = metadata.get('dataset_rows', {})
        prev_dns = prev_rows.get('dns', 0)
        prev_whois = prev_rows.get('whois', 0)
        prev_url = prev_rows.get('url', 0)

        log(f"\nPrevious training dataset rows:")
        log(f"  DNS: {prev_dns}")
        log(f"  WHOIS: {prev_whois}")
        log(f"  URL: {prev_url}")

        # Check for growth
        dns_growth = current_dns_rows - prev_dns
        whois_growth = current_whois_rows - prev_whois
        url_growth = current_url_rows - prev_url

        log(f"\nData growth since last training:")
        log(f"  DNS: +{dns_growth}")
        log(f"  WHOIS: +{whois_growth}")
        log(f"  URL: +{url_growth}")

        if all([dns_growth == 0, whois_growth == 0, url_growth == 0]):
            log("")
            log("=" * 60)
            log("ℹ️  NO DATA GROWTH - Skipping retraining")
            log("=" * 60)
            return False
    else:
        log("\nNo previous training metadata - first time training")

    log("")
    log("=" * 60)
    log("✅ DATA GROWTH DETECTED - Proceeding with model-ready preparation")
    log("=" * 60)
    log("")

    # Prepare each dataset
    prepare_url_modelready()
    prepare_dns_modelready()
    prepare_whois_modelready()

    # Verify row counts
    check_row_counts()

    # Save metadata for next run
    import json
    metadata = {
        'last_training_date': datetime.now().isoformat(),
        'dataset_rows': {
            'dns': current_dns_rows,
            'whois': current_whois_rows,
            'url': current_url_rows
        }
    }
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    log(f"Saved training metadata: {METADATA_FILE}")

    log("")
    log("=" * 60)
    log("✅ DONE - Ready for model training")
    log("=" * 60)

    return True


def main():
    """Main entry point."""
    needs_retraining = check_data_growth_and_prepare()
    sys.exit(0 if needs_retraining else 1)


if __name__ == "__main__":
    main()
