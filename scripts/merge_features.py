#!/usr/bin/env python3
"""
Merge URL, DNS, and WHOIS features into single dataset
"""

import sys
import pandas as pd

def merge_features(url_file: str, dns_file: str, whois_file: str, output_file: str):
    """Merge URL, DNS, and WHOIS features"""

    print(f"Loading URL features from {url_file}...")
    url_df = pd.read_csv(url_file)
    print(f"  {len(url_df)} rows")

    print(f"Loading DNS features from {dns_file}...")
    dns_df = pd.read_csv(dns_file)
    print(f"  {len(dns_df)} rows")

    print(f"Loading WHOIS features from {whois_file}...")
    whois_df = pd.read_csv(whois_file)
    print(f"  {len(whois_df)} rows")

    # Merge on URL column
    print("\nMerging features...")
    merged = url_df.copy()

    # Add DNS features (skip 'url' column if present)
    dns_cols = [col for col in dns_df.columns if col != 'url']
    for col in dns_cols:
        merged[col] = dns_df[col]

    # Add WHOIS features (skip 'url' column if present)
    whois_cols = [col for col in whois_df.columns if col != 'url']
    for col in whois_cols:
        merged[col] = whois_df[col]

    print(f"\nMerged dataset: {len(merged)} rows, {len(merged.columns)} columns")

    # Save
    merged.to_csv(output_file, index=False)
    print(f"✅ Saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python merge_features.py <url_file> <dns_file> <whois_file> <output_file>")
        sys.exit(1)

    merge_features(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
