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

    # Step 4: Prepare THREE SEPARATE master files (not combined!)
    # This ensures each model type uses ONLY its dedicated features
    print("\n" + "=" * 60)
    print("STEP 4: Prepare SEPARATE master files (URL, DNS, WHOIS)")
    print("=" * 60)

    # Add label column to DNS and WHOIS from batch
    df_dns = df_dns.merge(df_batch[['url', 'label']], on='url', how='left')
    df_whois = df_whois.merge(df_batch[['url', 'label']], on='url', how='left')

    print(f"✅ URL features: {len(df_url_features)} rows × {len(df_url_features.columns)} columns")
    print(f"✅ DNS features: {len(df_dns)} rows × {len(df_dns.columns)} columns")
    print(f"✅ WHOIS features: {len(df_whois)} rows × {len(df_whois.columns)} columns")

    # Step 5: Download existing master datasets from S3 (3 separate files)
    print("\n" + "=" * 60)
    print("STEP 5: Download existing SEPARATE master datasets from S3")
    print("=" * 60)

    masters = {
        'url': {'new': df_url_features, 'file': 'vm_data/master/url_features_master.csv', 's3_key': 'master/url_features_master.csv'},
        'dns': {'new': df_dns, 'file': 'vm_data/master/dns_features_master.csv', 's3_key': 'master/dns_features_master.csv'},
        'whois': {'new': df_whois, 'file': 'vm_data/master/whois_features_master.csv', 's3_key': 'master/whois_features_master.csv'}
    }

    for feature_type, config in masters.items():
        try:
            s3.download_file(S3_BUCKET, config['s3_key'], config['file'])
            config['existing'] = pd.read_csv(config['file'])
            print(f"📊 Existing {feature_type.upper()} master: {len(config['existing'])} rows")
        except Exception as e:
            print(f"ℹ️  No existing {feature_type.upper()} master - creating new")
            config['existing'] = None

    # Step 6: Accumulate each master separately
    print("\n" + "=" * 60)
    print("STEP 6: Accumulate EACH master dataset separately")
    print("=" * 60)

    for feature_type, config in masters.items():
        print(f"\n--- {feature_type.upper()} ---")
        print(f"📊 New batch: {len(config['new'])} rows")

        if config['existing'] is not None:
            # Combine and deduplicate
            df_combined = pd.concat([config['existing'], config['new']], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['url'], keep='last')

            added = len(df_combined) - len(config['existing'])
            duplicates = len(config['new']) - added

            print(f"✅ Combined: {len(df_combined)} rows (+{added} new, {duplicates} duplicates)")
        else:
            df_combined = config['new']
            print(f"✅ Initial: {len(df_combined)} rows")

        config['combined'] = df_combined
        df_combined.to_csv(config['file'], index=False)

    # Step 7: Upload all three masters back to S3
    print("\n" + "=" * 60)
    print("STEP 7: Upload THREE SEPARATE master datasets to S3")
    print("=" * 60)

    for feature_type, config in masters.items():
        s3.upload_file(config['file'], S3_BUCKET, config['s3_key'])
        print(f"✅ Uploaded {feature_type.upper()} master: {len(config['combined'])} rows")

    # Also upload combined for backwards compatibility (optional)
    merged = df_url_features.merge(df_dns.drop(columns=['label'], errors='ignore'), on='url', how='left')
    merged = merged.merge(df_whois.drop(columns=['label'], errors='ignore'), on='url', how='left')
    combined_file = "vm_data/master/phishing_features_master.csv"

    # Download existing combined and accumulate
    try:
        s3.download_file(S3_BUCKET, "master/phishing_features_master.csv", combined_file)
        df_existing_combined = pd.read_csv(combined_file)
        df_combined_all = pd.concat([df_existing_combined, merged], ignore_index=True)
        df_combined_all = df_combined_all.drop_duplicates(subset=['url'], keep='last')
    except:
        df_combined_all = merged

    df_combined_all.to_csv(combined_file, index=False)
    s3.upload_file(combined_file, S3_BUCKET, "master/phishing_features_master.csv")
    print(f"✅ Uploaded combined master (backwards compat): {len(df_combined_all)} rows")

    print("\n" + "=" * 60)
    print("✅ COMPLETE: THREE separate feature masters created!")
    print("=" * 60)
    print(f"URL master:   {len(masters['url']['combined'])} rows")
    print(f"DNS master:   {len(masters['dns']['combined'])} rows")
    print(f"WHOIS master: {len(masters['whois']['combined'])} rows")


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
