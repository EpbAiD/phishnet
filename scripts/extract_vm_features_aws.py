#!/usr/bin/env python3
"""
Extract DNS and WHOIS features on AWS EC2 and accumulate to master dataset in S3.

Includes:
- Checkpointing every 50 URLs (survives SSH drops / crashes)
- Progress heartbeat every 30s to keep SSH alive
- Per-URL timeout to prevent hanging on unresponsive domains
"""

import sys
import os
import time
import json
import signal
import threading
import pandas as pd
import boto3
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.dns_ipwhois import extract_single_domain_features
from src.features.whois import extract_single_whois_features

S3_BUCKET = "phishnet-data"
AWS_REGION = "us-east-1"
CHECKPOINT_EVERY = 50  # Save progress every N URLs
HEARTBEAT_INTERVAL = 30  # Print heartbeat every N seconds to keep SSH alive
PER_URL_TIMEOUT = 60  # Max seconds per URL for DNS+WHOIS combined


class Heartbeat:
    """Background thread that prints periodic heartbeat to keep SSH alive."""
    def __init__(self, interval=HEARTBEAT_INTERVAL):
        self.interval = interval
        self.status = "initializing"
        self.count = 0
        self.total = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def update(self, status, count=0, total=0):
        self.status = status
        self.count = count
        self.total = total

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(self.interval):
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - self._start_time))
            print(f"💓 [{elapsed}] {self.status} ({self.count}/{self.total})", flush=True)

    def __enter__(self):
        self._start_time = time.time()
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def _extract_with_timeout(func, url, timeout=PER_URL_TIMEOUT):
    """Run extraction function with a timeout to prevent hanging."""
    result = [None]
    error = [None]

    def target():
        try:
            result[0] = func(url)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=target)
    t.start()
    t.join(timeout)

    if t.is_alive():
        # Thread is stuck - return empty features
        print(f"  ⏰ TIMEOUT after {timeout}s for {url[:50]}", flush=True)
        return {}

    if error[0]:
        print(f"  ❌ Error for {url[:50]}: {error[0]}", flush=True)
        return {}

    return result[0] or {}


def extract_and_accumulate(batch_date: str):
    """
    Extract DNS/WHOIS features and accumulate to master dataset in S3.
    Includes checkpointing to survive SSH disconnections.
    """

    s3 = boto3.client('s3', region_name=AWS_REGION)

    # Setup directories
    os.makedirs("vm_data/url_queue", exist_ok=True)
    os.makedirs("vm_data/incremental", exist_ok=True)
    os.makedirs("vm_data/master", exist_ok=True)
    os.makedirs("vm_data/checkpoints", exist_ok=True)

    batch_name = f"batch_{batch_date}.csv"
    url_features_name = f"url_features_{batch_date}.csv"
    checkpoint_file = f"vm_data/checkpoints/checkpoint_{batch_date}.json"
    dns_checkpoint = f"vm_data/checkpoints/dns_{batch_date}.csv"
    whois_checkpoint = f"vm_data/checkpoints/whois_{batch_date}.csv"

    # Step 1: Download batch files from S3
    print("=" * 60, flush=True)
    print("STEP 1: Download batch data from S3", flush=True)
    print("=" * 60, flush=True)

    s3.download_file(S3_BUCKET, f"queue/{batch_name}", f"vm_data/url_queue/{batch_name}")
    s3.download_file(S3_BUCKET, f"queue/{url_features_name}", f"vm_data/url_queue/{url_features_name}")

    batch_file = f"vm_data/url_queue/{batch_name}"
    url_features_file = f"vm_data/url_queue/{url_features_name}"

    df_batch = pd.read_csv(batch_file)
    df_url_features = pd.read_csv(url_features_file)

    total_urls = len(df_batch)
    print(f"✅ Downloaded {total_urls} URLs", flush=True)
    print(f"✅ Downloaded URL features for {len(df_url_features)} URLs", flush=True)

    # Load checkpoint if exists (resume from crash)
    dns_start_idx = 0
    whois_start_idx = 0
    dns_features = []
    whois_features = []

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            cp = json.load(f)
        dns_start_idx = cp.get("dns_completed", 0)
        whois_start_idx = cp.get("whois_completed", 0)
        print(f"📋 Resuming from checkpoint: DNS={dns_start_idx}, WHOIS={whois_start_idx}", flush=True)

        if os.path.exists(dns_checkpoint) and dns_start_idx > 0:
            dns_features = pd.read_csv(dns_checkpoint).to_dict('records')
            print(f"  Loaded {len(dns_features)} cached DNS results", flush=True)
        if os.path.exists(whois_checkpoint) and whois_start_idx > 0:
            whois_features = pd.read_csv(whois_checkpoint).to_dict('records')
            print(f"  Loaded {len(whois_features)} cached WHOIS results", flush=True)

    with Heartbeat() as hb:
        # Step 2: Extract DNS features
        if dns_start_idx < total_urls:
            print(f"\n{'=' * 60}", flush=True)
            print(f"STEP 2: Extract DNS features (starting at {dns_start_idx})", flush=True)
            print("=" * 60, flush=True)

            for idx in range(dns_start_idx, total_urls):
                row = df_batch.iloc[idx]
                hb.update("DNS extraction", idx + 1, total_urls)
                print(f"[{idx+1}/{total_urls}] {row['url'][:50]}...", flush=True)

                features = _extract_with_timeout(extract_single_domain_features, row['url'])
                features['url'] = row['url']
                dns_features.append(features)

                # Checkpoint every N URLs
                if (idx + 1) % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(dns_features).to_csv(dns_checkpoint, index=False)
                    with open(checkpoint_file, 'w') as f:
                        json.dump({"dns_completed": idx + 1, "whois_completed": whois_start_idx}, f)
                    print(f"  💾 DNS checkpoint saved at {idx + 1}/{total_urls}", flush=True)

            # Final DNS checkpoint
            pd.DataFrame(dns_features).to_csv(dns_checkpoint, index=False)
            with open(checkpoint_file, 'w') as f:
                json.dump({"dns_completed": total_urls, "whois_completed": whois_start_idx}, f)

        df_dns = pd.DataFrame(dns_features)
        print(f"✅ Extracted DNS features for {len(df_dns)} URLs", flush=True)

        # Step 3: Extract WHOIS features
        if whois_start_idx < total_urls:
            print(f"\n{'=' * 60}", flush=True)
            print(f"STEP 3: Extract WHOIS features (starting at {whois_start_idx})", flush=True)
            print("=" * 60, flush=True)

            for idx in range(whois_start_idx, total_urls):
                row = df_batch.iloc[idx]
                hb.update("WHOIS extraction", idx + 1, total_urls)
                print(f"[{idx+1}/{total_urls}] {row['url'][:50]}...", flush=True)

                features = _extract_with_timeout(extract_single_whois_features, row['url'])
                features['url'] = row['url']
                whois_features.append(features)

                # Checkpoint every N URLs
                if (idx + 1) % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(whois_features).to_csv(whois_checkpoint, index=False)
                    with open(checkpoint_file, 'w') as f:
                        json.dump({"dns_completed": total_urls, "whois_completed": idx + 1}, f)
                    print(f"  💾 WHOIS checkpoint saved at {idx + 1}/{total_urls}", flush=True)

            # Final WHOIS checkpoint
            pd.DataFrame(whois_features).to_csv(whois_checkpoint, index=False)

        df_whois = pd.DataFrame(whois_features)
        print(f"✅ Extracted WHOIS features for {len(df_whois)} URLs", flush=True)
        hb.update("Accumulating masters", 0, 0)

    # Step 4: Prepare THREE SEPARATE master files
    print(f"\n{'=' * 60}", flush=True)
    print("STEP 4: Prepare SEPARATE master files (URL, DNS, WHOIS)", flush=True)
    print("=" * 60, flush=True)

    # Add label column to DNS and WHOIS from batch
    df_dns = df_dns.merge(df_batch[['url', 'label']], on='url', how='left')
    df_whois = df_whois.merge(df_batch[['url', 'label']], on='url', how='left')

    print(f"✅ URL features: {len(df_url_features)} rows × {len(df_url_features.columns)} columns", flush=True)
    print(f"✅ DNS features: {len(df_dns)} rows × {len(df_dns.columns)} columns", flush=True)
    print(f"✅ WHOIS features: {len(df_whois)} rows × {len(df_whois.columns)} columns", flush=True)

    # Step 5: Download existing master datasets from S3
    print(f"\n{'=' * 60}", flush=True)
    print("STEP 5: Download existing SEPARATE master datasets from S3", flush=True)
    print("=" * 60, flush=True)

    masters = {
        'url': {'new': df_url_features, 'file': 'vm_data/master/url_features_master.csv', 's3_key': 'master/url_features_master.csv'},
        'dns': {'new': df_dns, 'file': 'vm_data/master/dns_features_master.csv', 's3_key': 'master/dns_features_master.csv'},
        'whois': {'new': df_whois, 'file': 'vm_data/master/whois_features_master.csv', 's3_key': 'master/whois_features_master.csv'}
    }

    for feature_type, config in masters.items():
        try:
            s3.download_file(S3_BUCKET, config['s3_key'], config['file'])
            config['existing'] = pd.read_csv(config['file'])
            print(f"📊 Existing {feature_type.upper()} master: {len(config['existing'])} rows", flush=True)
        except Exception as e:
            print(f"ℹ️  No existing {feature_type.upper()} master - creating new", flush=True)
            config['existing'] = None

    # Step 6: Accumulate each master separately
    print(f"\n{'=' * 60}", flush=True)
    print("STEP 6: Accumulate EACH master dataset separately", flush=True)
    print("=" * 60, flush=True)

    for feature_type, config in masters.items():
        print(f"\n--- {feature_type.upper()} ---", flush=True)
        print(f"📊 New batch: {len(config['new'])} rows", flush=True)

        if config['existing'] is not None:
            df_combined = pd.concat([config['existing'], config['new']], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['url'], keep='last')

            added = len(df_combined) - len(config['existing'])
            duplicates = len(config['new']) - added

            print(f"✅ Combined: {len(df_combined)} rows (+{added} new, {duplicates} duplicates)", flush=True)
        else:
            df_combined = config['new']
            print(f"✅ Initial: {len(df_combined)} rows", flush=True)

        config['combined'] = df_combined
        df_combined.to_csv(config['file'], index=False)

    # Step 7: Upload all three masters back to S3
    print(f"\n{'=' * 60}", flush=True)
    print("STEP 7: Upload THREE SEPARATE master datasets to S3", flush=True)
    print("=" * 60, flush=True)

    for feature_type, config in masters.items():
        s3.upload_file(config['file'], S3_BUCKET, config['s3_key'])
        print(f"✅ Uploaded {feature_type.upper()} master: {len(config['combined'])} rows", flush=True)

    # Also upload combined for backwards compatibility
    merged = df_url_features.merge(df_dns.drop(columns=['label'], errors='ignore'), on='url', how='left')
    merged = merged.merge(df_whois.drop(columns=['label'], errors='ignore'), on='url', how='left')
    combined_file = "vm_data/master/phishing_features_master.csv"

    try:
        s3.download_file(S3_BUCKET, "master/phishing_features_master.csv", combined_file)
        df_existing_combined = pd.read_csv(combined_file)
        df_combined_all = pd.concat([df_existing_combined, merged], ignore_index=True)
        df_combined_all = df_combined_all.drop_duplicates(subset=['url'], keep='last')
    except Exception:
        df_combined_all = merged

    df_combined_all.to_csv(combined_file, index=False)
    s3.upload_file(combined_file, S3_BUCKET, "master/phishing_features_master.csv")
    print(f"✅ Uploaded combined master (backwards compat): {len(df_combined_all)} rows", flush=True)

    # Cleanup checkpoint files on success
    for f in [checkpoint_file, dns_checkpoint, whois_checkpoint]:
        if os.path.exists(f):
            os.remove(f)

    print(f"\n{'=' * 60}", flush=True)
    print("✅ COMPLETE: THREE separate feature masters created!", flush=True)
    print("=" * 60, flush=True)
    print(f"URL master:   {len(masters['url']['combined'])} rows", flush=True)
    print(f"DNS master:   {len(masters['dns']['combined'])} rows", flush=True)
    print(f"WHOIS master: {len(masters['whois']['combined'])} rows", flush=True)


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
