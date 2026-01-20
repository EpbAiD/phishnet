#!/usr/bin/env python3
"""
Extract DNS and WHOIS features on VM and accumulate to master dataset
VM is responsible for all feature merging and dataset accumulation
"""

import sys
import os
import pandas as pd
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.dns_ipwhois import extract_single_domain_features
from src.features.whois import extract_single_whois_features


def extract_and_accumulate(batch_date: str):
    """
    Extract DNS/WHOIS features and accumulate to master dataset in GCS

    Steps:
    1. Download URL batch and URL features from GCS
    2. Extract DNS features
    3. Extract WHOIS features
    4. Merge URL + DNS + WHOIS features
    5. Download existing master dataset from GCS
    6. Accumulate (append + deduplicate)
    7. Upload updated master back to GCS
    """

    # Setup directories
    os.makedirs("vm_data/url_queue", exist_ok=True)
    os.makedirs("vm_data/incremental", exist_ok=True)
    os.makedirs("vm_data/master", exist_ok=True)

    batch_name = f"batch_{batch_date}.csv"
    url_features_name = f"url_features_{batch_date}.csv"

    # Step 1: Download batch files from GCS
    print("=" * 60)
    print("STEP 1: Download batch data from GCS")
    print("=" * 60)

    subprocess.run([
        "gcloud", "storage", "cp",
        f"gs://phishnet-pipeline-data/queue/{batch_name}",
        f"vm_data/url_queue/{batch_name}"
    ], check=True)

    subprocess.run([
        "gcloud", "storage", "cp",
        f"gs://phishnet-pipeline-data/queue/{url_features_name}",
        f"vm_data/url_queue/{url_features_name}"
    ], check=True)

    # Load batch
    batch_file = f"vm_data/url_queue/{batch_name}"
    url_features_file = f"vm_data/url_queue/{url_features_name}"

    df_batch = pd.read_csv(batch_file)
    df_url_features = pd.read_csv(url_features_file)

    print(f"✅ Downloaded {len(df_batch)} URLs")
    print(f"✅ Downloaded URL features for {len(df_url_features)} URLs")

    # Step 2: Extract DNS features
    print("\n" + "=" * 60)
    print("STEP 2: Extract DNS features")
    print("=" * 60)

    dns_features = []
    for idx, row in df_batch.iterrows():
        print(f"[{idx+1}/{len(df_batch)}] {row['url']}")
        features = extract_single_domain_features(row['url'])
        dns_features.append(features)

    df_dns = pd.DataFrame(dns_features)
    print(f"✅ Extracted DNS features for {len(df_dns)} URLs")

    # Step 3: Extract WHOIS features
    print("\n" + "=" * 60)
    print("STEP 3: Extract WHOIS features")
    print("=" * 60)

    whois_features = []
    for idx, row in df_batch.iterrows():
        print(f"[{idx+1}/{len(df_batch)}] {row['url']}")
        features = extract_single_whois_features(row['url'])
        whois_features.append(features)

    df_whois = pd.DataFrame(whois_features)
    print(f"✅ Extracted WHOIS features for {len(df_whois)} URLs")

    # Step 4: Merge all features
    print("\n" + "=" * 60)
    print("STEP 4: Merge URL + DNS + WHOIS features")
    print("=" * 60)

    merged = df_url_features.merge(df_dns, on='url', how='left')
    merged = merged.merge(df_whois, on='url', how='left')
    print(f"✅ Merged features: {len(merged)} rows × {len(merged.columns)} columns")

    # Step 5: Download existing master dataset from GCS
    print("\n" + "=" * 60)
    print("STEP 5: Download existing master dataset from GCS")
    print("=" * 60)

    # Use new filename for validation testing (phishing_features_master_v2.csv)
    master_file = "vm_data/master/phishing_features_master_v2.csv"
    result = subprocess.run([
        "gcloud", "storage", "cp",
        "gs://phishnet-pipeline-data/master/phishing_features_master_v2.csv",
        master_file
    ], capture_output=True, text=True)

    # Step 6: Accumulate (append + deduplicate)
    print("\n" + "=" * 60)
    print("STEP 6: Accumulate dataset")
    print("=" * 60)

    if result.returncode == 0:
        df_existing = pd.read_csv(master_file)
        print(f"📊 Existing master dataset: {len(df_existing)} rows")
        print(f"📊 New batch: {len(merged)} rows")

        # Combine and deduplicate
        df_combined = pd.concat([df_existing, merged], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['url'], keep='last')

        added = len(df_combined) - len(df_existing)
        duplicates = len(merged) - added

        print(f"✅ Combined dataset: {len(df_combined)} rows")
        print(f"   └─ Added: {added} new URLs")
        print(f"   └─ Skipped: {duplicates} duplicates")
    else:
        print("ℹ️  No existing master dataset - this is the first run")
        df_combined = merged
        print(f"✅ Initial dataset: {len(df_combined)} rows")

    # Save updated master
    df_combined.to_csv(master_file, index=False)

    # Step 7: Upload updated master back to GCS
    print("\n" + "=" * 60)
    print("STEP 7: Upload updated master dataset to GCS")
    print("=" * 60)

    subprocess.run([
        "gcloud", "storage", "cp",
        master_file,
        "gs://phishnet-pipeline-data/master/"
    ], check=True)

    print(f"✅ Uploaded master dataset to GCS ({len(df_combined)} rows)")

    print("\n" + "=" * 60)
    print("✅ COMPLETE: Feature extraction and accumulation finished!")
    print("=" * 60)
    print(f"Final dataset size: {len(df_combined)} rows × {len(df_combined.columns)} columns")


if __name__ == "__main__":
    # Support both old and new calling conventions
    # Old: python extract_vm_features.py vm_data/url_queue/batch_X.csv YYYYMMDD
    # New: python extract_vm_features.py YYYYMMDD
    if len(sys.argv) == 2:
        # New format: just batch_date
        batch_date = sys.argv[1]
    elif len(sys.argv) == 3:
        # Old format: batch_file batch_date (ignore batch_file, get from GCS)
        batch_date = sys.argv[2]
    else:
        print("Usage: python extract_vm_features.py <batch_date>")
        print("Example: python extract_vm_features.py 20260120")
        sys.exit(1)

    extract_and_accumulate(batch_date)
