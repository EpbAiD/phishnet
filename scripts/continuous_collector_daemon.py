#!/usr/bin/env python3
"""
PhishNet Continuous URL Collector Daemon
=========================================
Runs 24/7 on EC2, collecting URLs from 13 threat intelligence feeds,
extracting all features (URL, DNS, WHOIS), and accumulating to S3.

This daemon:
- Fetches URLs every 15 minutes from 13+ sources
- Extracts URL features locally
- Extracts DNS features (requires network)
- Extracts WHOIS features (requires network)
- Deduplicates against existing S3 master datasets
- Appends new unique data to S3 masters
- Logs metrics for monitoring

Run as systemd service:
    sudo systemctl start phishnet-collector
    sudo systemctl enable phishnet-collector

Or run directly:
    python3 scripts/continuous_collector_daemon.py
"""

import os
import sys
import time
import signal
import logging
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import traceback

# Add project root to path - use absolute path for systemd service compatibility
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Also set PYTHONPATH for subprocess imports
os.environ['PYTHONPATH'] = str(PROJECT_ROOT)

# Verify src module is accessible
try:
    import src.features.url_features
    print(f"[OK] Project root: {PROJECT_ROOT}")
except ImportError as e:
    print(f"[ERROR] Cannot import src.features.url_features from {PROJECT_ROOT}")
    print(f"[ERROR] sys.path = {sys.path[:3]}")
    print(f"[ERROR] {e}")

# Configuration
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
S3_BUCKET = os.environ.get('S3_BUCKET', 'phishnet-data')
COLLECTION_INTERVAL_SECONDS = int(os.environ.get('COLLECTION_INTERVAL', 900))  # 15 minutes
MAX_URLS_PER_CYCLE = int(os.environ.get('MAX_URLS_PER_CYCLE', 1000))
LOG_FILE = os.environ.get('LOG_FILE', '/var/log/phishnet/collector.log')

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Graceful shutdown flag
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


class ContinuousCollector:
    """24/7 URL collector daemon."""

    def __init__(self):
        self.s3 = boto3.client('s3', region_name=AWS_REGION)
        self.stats = {
            'cycles_completed': 0,
            'urls_collected': 0,
            'urls_added_to_master': 0,
            'errors': 0,
            'start_time': datetime.now()
        }

    def fetch_urls_from_all_sources(self) -> pd.DataFrame:
        """Fetch URLs from all 13 threat intelligence sources."""
        from scripts.fetch_urls import (
            fetch_phishtank, fetch_openphish, fetch_urlhaus,
            fetch_phishstats, fetch_phishing_army, fetch_urlabuse,
            fetch_threatview, fetch_digitalside, fetch_mitchellkrogza,
            fetch_malwaredomainlist, fetch_vxvault, fetch_cybercrime_tracker,
            fetch_alienvault_otx, generate_legitimate_urls
        )

        all_urls = []

        fetchers = [
            ("PhishTank", fetch_phishtank),
            ("OpenPhish", fetch_openphish),
            ("URLhaus", fetch_urlhaus),
            ("PhishStats", fetch_phishstats),
            ("Phishing Army", fetch_phishing_army),
            ("URLAbuse", fetch_urlabuse),
            ("ThreatView", fetch_threatview),
            ("DigitalSide", fetch_digitalside),
            ("Mitchell Krogza", fetch_mitchellkrogza),
            ("Malware Domain List", fetch_malwaredomainlist),
            ("VXVault", fetch_vxvault),
            ("Cybercrime Tracker", fetch_cybercrime_tracker),
            ("AlienVault OTX", fetch_alienvault_otx),
        ]

        for name, fetcher in fetchers:
            try:
                urls = fetcher()
                all_urls.extend(urls)
                logger.info(f"  {name}: {len(urls)} URLs")
            except Exception as e:
                logger.warning(f"  {name}: FAILED - {e}")
            time.sleep(0.5)  # Rate limiting

        # Create DataFrame
        df = pd.DataFrame(all_urls)
        if len(df) == 0:
            return df

        # Deduplicate within batch
        df = df.drop_duplicates(subset=['url'])

        # Add legitimate URLs for balance (100% of phishing count for 50:50 split)
        # This is CRITICAL for model performance - class imbalance causes false positives
        # Use unique_suffix=True to ensure URLs aren't deduplicated across batches
        phishing_count = len(df[df['label'] == 'phishing'])
        legit_count = max(phishing_count, 100)  # Match phishing count for balance
        legit_urls = generate_legitimate_urls(legit_count, unique_suffix=True)
        df_legit = pd.DataFrame(legit_urls).drop_duplicates(subset=['url'])

        df = pd.concat([df, df_legit], ignore_index=True)

        return df

    def download_master_from_s3(self, feature_type: str) -> pd.DataFrame:
        """Download existing master dataset from S3."""
        master_key = f"master/{feature_type}_features_master.csv"
        local_path = f"/tmp/{feature_type}_master.csv"

        try:
            self.s3.download_file(S3_BUCKET, master_key, local_path)
            df = pd.read_csv(local_path)
            logger.info(f"Downloaded {feature_type} master: {len(df)} rows")
            return df
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.info(f"No existing {feature_type} master, starting fresh")
                return pd.DataFrame()
            raise

    def upload_master_to_s3(self, df: pd.DataFrame, feature_type: str):
        """Upload updated master dataset to S3."""
        master_key = f"master/{feature_type}_features_master.csv"
        local_path = f"/tmp/{feature_type}_master_upload.csv"

        df.to_csv(local_path, index=False)
        self.s3.upload_file(local_path, S3_BUCKET, master_key)
        logger.info(f"Uploaded {feature_type} master: {len(df)} rows")

    def extract_url_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract URL-based features."""
        from src.features.url_features import extract_single_url_features

        features_list = []
        for _, row in df.iterrows():
            try:
                features = extract_single_url_features(row['url'])
                features['url'] = row['url']
                features['label'] = row['label']
                features['source'] = row.get('source', 'unknown')
                features['collected_at'] = datetime.now().isoformat()
                features_list.append(features)
            except Exception as e:
                logger.debug(f"URL feature extraction failed for {row['url']}: {e}")

        return pd.DataFrame(features_list)

    def extract_dns_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract DNS-based features (requires network)."""
        from src.features.dns_ipwhois import extract_single_domain_features

        features_list = []
        for _, row in df.iterrows():
            try:
                features = extract_single_domain_features(row['url'])
                features['url'] = row['url']
                features['label'] = row['label']
                features['collected_at'] = datetime.now().isoformat()
                features_list.append(features)
            except Exception as e:
                logger.debug(f"DNS feature extraction failed for {row['url']}: {e}")

        return pd.DataFrame(features_list)

    def extract_whois_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract WHOIS-based features (requires network)."""
        from src.features.whois import extract_single_whois_features

        features_list = []
        for _, row in df.iterrows():
            try:
                features = extract_single_whois_features(row['url'])
                features['url'] = row['url']
                features['label'] = row['label']
                features['collected_at'] = datetime.now().isoformat()
                features_list.append(features)
            except Exception as e:
                logger.debug(f"WHOIS feature extraction failed for {row['url']}: {e}")

        return pd.DataFrame(features_list)

    def merge_with_master(self, new_df: pd.DataFrame, master_df: pd.DataFrame) -> tuple:
        """Merge new data with master, deduplicating by URL."""
        if len(new_df) == 0:
            return master_df, 0

        if len(master_df) == 0:
            return new_df, len(new_df)

        # Get existing URLs
        existing_urls = set(master_df['url'].tolist())

        # Filter to only truly new URLs
        new_unique = new_df[~new_df['url'].isin(existing_urls)]

        if len(new_unique) == 0:
            return master_df, 0

        # Append new data
        merged = pd.concat([master_df, new_unique], ignore_index=True)

        return merged, len(new_unique)

    def run_collection_cycle(self):
        """Run a single collection cycle."""
        cycle_start = time.time()
        logger.info("=" * 60)
        logger.info(f"COLLECTION CYCLE {self.stats['cycles_completed'] + 1}")
        logger.info("=" * 60)

        try:
            # Step 1: Fetch URLs from all sources
            logger.info("Step 1: Fetching URLs from 13 threat feeds...")
            df_urls = self.fetch_urls_from_all_sources()

            if len(df_urls) == 0:
                logger.warning("No URLs fetched, skipping cycle")
                return

            # Limit to max per cycle
            if len(df_urls) > MAX_URLS_PER_CYCLE:
                df_urls = df_urls.sample(n=MAX_URLS_PER_CYCLE, random_state=42)

            self.stats['urls_collected'] += len(df_urls)
            logger.info(f"Fetched {len(df_urls)} unique URLs")

            # Step 2: Extract URL features
            logger.info("Step 2: Extracting URL features...")
            df_url_features = self.extract_url_features(df_urls)
            logger.info(f"Extracted URL features for {len(df_url_features)} URLs")

            # Step 3: Extract DNS features
            logger.info("Step 3: Extracting DNS features...")
            df_dns_features = self.extract_dns_features(df_urls)
            logger.info(f"Extracted DNS features for {len(df_dns_features)} URLs")

            # Step 4: Extract WHOIS features
            logger.info("Step 4: Extracting WHOIS features...")
            df_whois_features = self.extract_whois_features(df_urls)
            logger.info(f"Extracted WHOIS features for {len(df_whois_features)} URLs")

            # Step 5: Merge with S3 masters
            logger.info("Step 5: Merging with S3 master datasets...")

            total_added = 0

            for feature_type, df_new in [
                ('url', df_url_features),
                ('dns', df_dns_features),
                ('whois', df_whois_features)
            ]:
                if len(df_new) == 0:
                    continue

                master_df = self.download_master_from_s3(feature_type)
                merged_df, added = self.merge_with_master(df_new, master_df)

                if added > 0:
                    self.upload_master_to_s3(merged_df, feature_type)
                    total_added += added
                    logger.info(f"  {feature_type.upper()}: +{added} new rows (total: {len(merged_df)})")
                else:
                    logger.info(f"  {feature_type.upper()}: No new unique URLs")

            self.stats['urls_added_to_master'] += total_added
            self.stats['cycles_completed'] += 1

            cycle_duration = time.time() - cycle_start
            logger.info("-" * 60)
            logger.info(f"Cycle completed in {cycle_duration:.1f}s")
            logger.info(f"URLs added to master: {total_added}")
            logger.info(f"Total stats: {self.stats}")

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Collection cycle failed: {e}")
            logger.error(traceback.format_exc())

    def run(self):
        """Main daemon loop."""
        logger.info("=" * 60)
        logger.info("PhishNet Continuous Collector Started")
        logger.info(f"Collection interval: {COLLECTION_INTERVAL_SECONDS}s")
        logger.info(f"Max URLs per cycle: {MAX_URLS_PER_CYCLE}")
        logger.info(f"S3 bucket: {S3_BUCKET}")
        logger.info("=" * 60)

        while not shutdown_requested:
            try:
                self.run_collection_cycle()
            except Exception as e:
                logger.error(f"Unexpected error in daemon loop: {e}")
                logger.error(traceback.format_exc())

            # Wait for next cycle
            if not shutdown_requested:
                logger.info(f"Sleeping {COLLECTION_INTERVAL_SECONDS}s until next cycle...")

                # Sleep in small intervals to check shutdown flag
                sleep_remaining = COLLECTION_INTERVAL_SECONDS
                while sleep_remaining > 0 and not shutdown_requested:
                    time.sleep(min(10, sleep_remaining))
                    sleep_remaining -= 10

        logger.info("=" * 60)
        logger.info("Daemon shutting down gracefully")
        logger.info(f"Final stats: {self.stats}")
        logger.info("=" * 60)


def main():
    """Entry point."""
    collector = ContinuousCollector()
    collector.run()


if __name__ == "__main__":
    main()
