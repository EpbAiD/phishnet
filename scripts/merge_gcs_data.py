#!/usr/bin/env python3
"""
Merge GCS Incremental Data into Main Datasets
==============================================
Downloads DNS/WHOIS data from GCS incremental folder and merges into
separate main datasets:
  - data/phishing_features_complete_dns.csv
  - data/phishing_features_complete_whois.csv

Also handles URL features from local batch files.

Usage:
  python3 scripts/merge_gcs_data.py <days_to_merge>
"""

import os
import sys
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from google.cloud import storage

# Configuration
BUCKET_NAME = "phishnet-pipeline-data"
INCREMENTAL_FOLDER = "incremental"
LOCAL_INCREMENTAL = "vm_data/merged"
MAIN_DNS = "data/phishing_features_complete_dns.csv"
MAIN_WHOIS = "data/phishing_features_complete_whois.csv"
MAIN_URL = "data/phishing_features_complete_url.csv"


def log(message):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def download_incremental_data(days: int = 7):
    """
    Download incremental DNS/WHOIS/URL data from GCS.

    Args:
        days: Number of days to look back

    Returns:
        (dns_files, whois_files, url_files): Lists of local file paths
    """
    log(f"Downloading last {days} days of incremental data from GCS...")

    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    # Get all blobs in incremental folder
    blobs = list(bucket.list_blobs(prefix=f"{INCREMENTAL_FOLDER}/"))

    # Filter by modification time (last N days)
    cutoff_date = datetime.now() - timedelta(days=days)

    dns_blobs = []
    whois_blobs = []
    url_blobs = []

    for blob in blobs:
        if blob.name.endswith('.csv'):
            if blob.updated >= cutoff_date.replace(tzinfo=blob.updated.tzinfo):
                if 'dns_' in blob.name:
                    dns_blobs.append(blob)
                elif 'whois_' in blob.name:
                    whois_blobs.append(blob)
                elif 'url_features_' in blob.name:
                    url_blobs.append(blob)

    log(f"Found {len(dns_blobs)} DNS files, {len(whois_blobs)} WHOIS files, and {len(url_blobs)} URL files")

    if len(dns_blobs) == 0 and len(whois_blobs) == 0 and len(url_blobs) == 0:
        log("No new data to download")
        return [], [], []

    # Download files
    os.makedirs(LOCAL_INCREMENTAL, exist_ok=True)

    dns_files = []
    whois_files = []
    url_files = []

    for blob in dns_blobs:
        filename = os.path.basename(blob.name)
        local_path = os.path.join(LOCAL_INCREMENTAL, filename)
        log(f"  Downloading {filename}...")
        blob.download_to_filename(local_path)
        dns_files.append(local_path)

    for blob in whois_blobs:
        filename = os.path.basename(blob.name)
        local_path = os.path.join(LOCAL_INCREMENTAL, filename)
        log(f"  Downloading {filename}...")
        blob.download_to_filename(local_path)
        whois_files.append(local_path)

    for blob in url_blobs:
        filename = os.path.basename(blob.name)
        local_path = os.path.join(LOCAL_INCREMENTAL, filename)
        log(f"  Downloading {filename}...")
        blob.download_to_filename(local_path)
        url_files.append(local_path)

    log(f"Downloaded {len(dns_files)} DNS files, {len(whois_files)} WHOIS files, and {len(url_files)} URL files")

    return dns_files, whois_files, url_files


def merge_into_main_datasets(dns_files, whois_files, url_feature_files):
    """
    Merge incremental data into main datasets.

    Args:
        dns_files: List of DNS CSV files
        whois_files: List of WHOIS CSV files
        url_batches: List of URL batch CSV files (with URL features)
    """
    log("=" * 60)
    log("MERGING INTO MAIN DATASETS")
    log("=" * 60)

    # Load or create main datasets
    if os.path.exists(MAIN_DNS):
        df_dns_main = pd.read_csv(MAIN_DNS)
        log(f"Main DNS dataset: {len(df_dns_main)} rows")
    else:
        df_dns_main = pd.DataFrame()
        log("No existing DNS dataset - will create new one")

    if os.path.exists(MAIN_WHOIS):
        df_whois_main = pd.read_csv(MAIN_WHOIS)
        log(f"Main WHOIS dataset: {len(df_whois_main)} rows")
    else:
        df_whois_main = pd.DataFrame()
        log("No existing WHOIS dataset - will create new one")

    if os.path.exists(MAIN_URL):
        df_url_main = pd.read_csv(MAIN_URL)
        log(f"Main URL dataset: {len(df_url_main)} rows")
    else:
        df_url_main = pd.DataFrame()
        log("No existing URL dataset - will create new one")

    # Merge DNS files
    if dns_files:
        dns_dfs = []
        for f in dns_files:
            log(f"  Reading {os.path.basename(f)}...")
            df = pd.read_csv(f)
            dns_dfs.append(df)

        df_dns_new = pd.concat(dns_dfs, ignore_index=True)
        log(f"New DNS data: {len(df_dns_new)} rows")

        # Combine with main
        df_dns_combined = pd.concat([df_dns_main, df_dns_new], ignore_index=True)
        df_dns_combined = df_dns_combined.drop_duplicates(subset=['url'], keep='last')
        log(f"DNS dataset after merge: {len(df_dns_combined)} rows")

        # Save
        os.makedirs(os.path.dirname(MAIN_DNS), exist_ok=True)
        df_dns_combined.to_csv(MAIN_DNS, index=False)
        log(f"✅ Saved DNS: {MAIN_DNS}")

    # Merge WHOIS files
    if whois_files:
        whois_dfs = []
        for f in whois_files:
            log(f"  Reading {os.path.basename(f)}...")
            df = pd.read_csv(f)
            whois_dfs.append(df)

        df_whois_new = pd.concat(whois_dfs, ignore_index=True)
        log(f"New WHOIS data: {len(df_whois_new)} rows")

        # Combine with main
        df_whois_combined = pd.concat([df_whois_main, df_whois_new], ignore_index=True)
        df_whois_combined = df_whois_combined.drop_duplicates(subset=['url'], keep='last')
        log(f"WHOIS dataset after merge: {len(df_whois_combined)} rows")

        # Save
        os.makedirs(os.path.dirname(MAIN_WHOIS), exist_ok=True)
        df_whois_combined.to_csv(MAIN_WHOIS, index=False)
        log(f"✅ Saved WHOIS: {MAIN_WHOIS}")

    # Merge URL features (from local extraction, not from batches)
    url_feature_files = sorted(glob.glob("data/url_queue/url_features_*.csv"))
    if url_feature_files:
        url_dfs = []
        for f in url_feature_files:
            log(f"  Reading {os.path.basename(f)}...")
            df = pd.read_csv(f)
            url_dfs.append(df)

        df_url_new = pd.concat(url_dfs, ignore_index=True)
        log(f"New URL data: {len(df_url_new)} rows")

        # Combine with main
        df_url_combined = pd.concat([df_url_main, df_url_new], ignore_index=True)
        df_url_combined = df_url_combined.drop_duplicates(subset=['url'], keep='last')
        log(f"URL dataset after merge: {len(df_url_combined)} rows")

        # Save
        os.makedirs(os.path.dirname(MAIN_URL), exist_ok=True)
        df_url_combined.to_csv(MAIN_URL, index=False)
        log(f"✅ Saved URL: {MAIN_URL}")

    log("=" * 60)
    log("MERGE COMPLETE")
    log("=" * 60)


def main():
    """Main entry point."""
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7

    log("=" * 60)
    log("GCS Data Merger")
    log("=" * 60)
    log(f"Bucket: gs://{BUCKET_NAME}")
    log(f"Looking back: {days} days")
    log("")

    # Download from GCS (includes DNS, WHOIS, and URL features)
    dns_files, whois_files, url_files = download_incremental_data(days)

    # Merge into main datasets
    if dns_files or whois_files or url_files:
        merge_into_main_datasets(dns_files, whois_files, url_files)
    else:
        log("No new data to merge")

    log("")
    log("✅ Done!")


if __name__ == "__main__":
    main()
