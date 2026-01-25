#!/usr/bin/env python3
"""
Extract DNS and WHOIS features on AWS EC2 and accumulate to master dataset in S3
"""

import sys
import os
import pandas as pd
import boto3
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.dns_ipwhois import extract_single_domain_features
from src.features.whois import extract_single_whois_features

S3_BUCKET = "phishnet-data"
AWS_REGION = "us-east-1"


def extract_and_accumulate(batch_date: str):
    """
    Extract DNS/WHOIS features and accumulate to master dataset in S3

    Steps:
    1. Download URL batch and URL features from S3
    2. Extract DNS features
    3. Extract WHOIS features
    4. Merge URL + DNS + WHOIS features
    5. Download existing master dataset from S3
    6. Accumulate (append + deduplicate)
    7. Upload updated master back to S3
    """

    s3 = boto3.client('s3', region_name=AWS_REGION)

    # Setup directories
    os.makedirs("vm_data/url_queue", exist_ok=True)
    os.makedirs("vm_data/incremental", exist_ok=True)
    os.makedirs("vm_data/master", exist_ok=True)

    batch_name = f"batch_{batch_date}.csv"
    url_features_name = f"url_features_{batch_date}.csv"

    # Step 1: Download batch files from S3
    print("=" * 60)
    print("STEP 1: Download batch data from S3")
    print("=" * 60)

    s3.download_file(S3_BUCKET, f"queue/{batch_name}", f"vm_data/url_queue/{batch_name}")
    s3.download_file(S3_BUCKET, f"queue/{url_features_name}", f"vm_data/url_queue/{url_features_name}")

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
        print(f"[{idx+1}/{len(df_batch)}] {row['url'][:50]}...")
        features = extract_single_domain_features(row['url'])
        features['url'] = row['url']
        dns_features.append(features)

    df_dns = pd.DataFrame(dns_features)
    print(f"✅ Extracted DNS features for {len(df_dns)} URLs")

    # Step 3: Extract WHOIS features
    print("\n" + "=" * 60)
    print("STEP 3: Extract WHOIS features")
    print("=" * 60)

    whois_features = []
    for idx, row in df_batch.iterrows():
        print(f"[{idx+1}/{len(df_batch)}] {row['url'][:50]}...")
        features = extract_single_whois_features(row['url'])
        features['url'] = row['url']
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

    # Step 5: Download existing master dataset from S3
    print("\n" + "=" * 60)
    print("STEP 5: Download existing master dataset from S3")
    print("=" * 60)

    master_file = "vm_data/master/phishing_features_master.csv"
    try:
        s3.download_file(S3_BUCKET, "master/phishing_features_master.csv", master_file)
        df_existing = pd.read_csv(master_file)
        print(f"📊 Existing master dataset: {len(df_existing)} rows")
        has_existing = True
    except Exception as e:
        print(f"ℹ️  No existing master dataset - this is the first run")
        has_existing = False

    # Step 6: Accumulate (append + deduplicate)
    print("\n" + "=" * 60)
    print("STEP 6: Accumulate dataset")
    print("=" * 60)

    if has_existing:
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
        df_combined = merged
        print(f"✅ Initial dataset: {len(df_combined)} rows")

    # Save updated master
    df_combined.to_csv(master_file, index=False)

    # Step 7: Upload updated master back to S3
    print("\n" + "=" * 60)
    print("STEP 7: Upload updated master dataset to S3")
    print("=" * 60)

    s3.upload_file(master_file, S3_BUCKET, "master/phishing_features_master.csv")

    print(f"✅ Uploaded master dataset to S3 ({len(df_combined)} rows)")

    print("\n" + "=" * 60)
    print("✅ COMPLETE: Feature extraction and accumulation finished!")
    print("=" * 60)
    print(f"Final dataset size: {len(df_combined)} rows × {len(df_combined.columns)} columns")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        batch_date = sys.argv[1]
    elif len(sys.argv) == 3:
        batch_date = sys.argv[2]
    else:
        print("Usage: python extract_vm_features_aws.py <batch_date>")
        print("Example: python extract_vm_features_aws.py 20260125")
        sys.exit(1)

    extract_and_accumulate(batch_date)
