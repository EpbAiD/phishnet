#!/usr/bin/env python3
"""
Extract DNS and WHOIS features on VM (synchronous)
Used by daily pipeline workflow
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.dns_ipwhois import extract_single_domain_features
from src.features.whois import extract_single_whois_features

def extract_features(batch_file: str, batch_date: str):
    """Extract DNS and WHOIS features for a batch"""

    print(f"Loading URLs from {batch_file}...")
    df = pd.read_csv(batch_file)
    print(f"Loaded {len(df)} URLs")

    # Extract DNS features
    print("\nExtracting DNS features...")
    dns_features = []
    for idx, row in df.iterrows():
        print(f"[{idx+1}/{len(df)}] {row['url']}")
        features = extract_single_domain_features(row['url'])
        dns_features.append(features)

    dns_df = pd.DataFrame(dns_features)
    dns_output = f"vm_data/incremental/dns_{batch_date}.csv"
    os.makedirs("vm_data/incremental", exist_ok=True)
    dns_df.to_csv(dns_output, index=False)
    print(f"✅ DNS features saved: {dns_output} ({len(dns_df)} rows)")

    # Extract WHOIS features
    print("\nExtracting WHOIS features...")
    whois_features = []
    for idx, row in df.iterrows():
        print(f"[{idx+1}/{len(df)}] {row['url']}")
        features = extract_single_whois_features(row['url'])
        whois_features.append(features)

    whois_df = pd.DataFrame(whois_features)
    whois_output = f"vm_data/incremental/whois_{batch_date}.csv"
    whois_df.to_csv(whois_output, index=False)
    print(f"✅ WHOIS features saved: {whois_output} ({len(whois_df)} rows)")

    print("\n✅ Feature extraction complete!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_vm_features.py <batch_file> <batch_date>")
        print("Example: python extract_vm_features.py vm_data/url_queue/batch_20260115.csv 20260115")
        sys.exit(1)

    batch_file = sys.argv[1]
    batch_date = sys.argv[2]

    extract_features(batch_file, batch_date)
