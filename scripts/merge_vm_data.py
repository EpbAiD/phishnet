#!/usr/bin/env python3
"""
Merge VM-Processed Features with Main Dataset
=============================================
Merges DNS and WHOIS features from VM into main dataset.
Called by daily_model_retrain.sh
"""

import os
import sys
import pandas as pd
import glob
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def merge_vm_data(vm_processed_dir: str = "data/vm_processed",
                  main_dataset: str = "data/processed/url_features_modelready.csv"):
    """
    Merge VM-processed DNS/WHOIS features with main dataset.

    Args:
        vm_processed_dir: Directory with VM-processed CSV files
        main_dataset: Path to main dataset

    Returns:
        Number of URLs in merged dataset
    """
    print(f"Merging VM-processed data...")
    print(f"  VM directory: {vm_processed_dir}")
    print(f"  Main dataset: {main_dataset}")
    print()

    # Load main dataset
    if os.path.exists(main_dataset):
        df_main = pd.read_csv(main_dataset)
        print(f"  Loaded main dataset: {len(df_main)} URLs")
    else:
        df_main = pd.DataFrame()
        print(f"  No existing main dataset - will create new one")

    # Load all VM-processed files
    dns_files = sorted(glob.glob(f"{vm_processed_dir}/dns_*.csv"))
    whois_files = sorted(glob.glob(f"{vm_processed_dir}/whois_*.csv"))

    print(f"  Found {len(dns_files)} DNS files and {len(whois_files)} WHOIS files")

    if len(dns_files) == 0:
        print(f"  No files to merge")
        return len(df_main)

    # Merge DNS and WHOIS features for each batch
    for dns_file in dns_files:
        batch_name = os.path.basename(dns_file).replace('dns_', '').replace('.csv', '')
        whois_file = f"{vm_processed_dir}/whois_{batch_name}.csv"

        if not os.path.exists(whois_file):
            print(f"  ⚠️  Missing WHOIS file for {batch_name} - skipping")
            continue

        # Load features
        df_dns = pd.read_csv(dns_file)
        df_whois = pd.read_csv(whois_file)

        print(f"  Merging batch {batch_name}: {len(df_dns)} URLs")

        # Merge on URL
        df_batch = df_dns.merge(df_whois, on='url', how='inner', suffixes=('', '_whois'))

        # Remove duplicate label columns
        if 'label_whois' in df_batch.columns:
            df_batch = df_batch.drop(columns=['label_whois'])

        # Append to main dataset
        df_main = pd.concat([df_main, df_batch], ignore_index=True)

    # Remove duplicates by URL (keep latest)
    original_count = len(df_main)
    df_main = df_main.drop_duplicates(subset=['url'], keep='last')
    duplicates_removed = original_count - len(df_main)

    print()
    print(f"  ✅ Merged dataset: {len(df_main)} URLs ({duplicates_removed} duplicates removed)")

    # Save updated main dataset
    os.makedirs(os.path.dirname(main_dataset), exist_ok=True)
    df_main.to_csv(main_dataset, index=False)
    print(f"  ✅ Saved to: {main_dataset}")

    # Archive processed VM files
    archive_dir = f"{vm_processed_dir}/archived"
    os.makedirs(archive_dir, exist_ok=True)

    archived_count = 0
    for f in dns_files + whois_files:
        archive_path = f.replace("vm_processed/", "vm_processed/archived/")
        os.rename(f, archive_path)
        archived_count += 1

    print(f"  ✅ Archived {archived_count} files to {archive_dir}")

    return len(df_main)


if __name__ == "__main__":
    vm_dir = sys.argv[1] if len(sys.argv) > 1 else "data/vm_processed"
    main_ds = sys.argv[2] if len(sys.argv) > 2 else "data/processed/url_features_modelready.csv"

    count = merge_vm_data(vm_dir, main_ds)
    sys.exit(0 if count > 0 else 1)
