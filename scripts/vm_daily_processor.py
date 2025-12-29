#!/usr/bin/env python3
"""
VM Daily Processor - Extract DNS and WHOIS features for queued URLs
============================================================================
Runs on GCP VM (triggered by daily_url_collector.sh)
Processes all URLs in vm_data/url_queue/
Extracts DNS (38 features) + WHOIS (12 features)
Saves results to vm_data/incremental/
Signals completion with .ready file
============================================================================
"""

import os
import sys
import time
import glob
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.dns_ipwhois import extract_single_domain_features
from src.features.whois import extract_single_whois_features

# ============================================================================
# Configuration
# ============================================================================
QUEUE_DIR = "vm_data/url_queue"
OUTPUT_DIR = "vm_data/incremental"
READY_DIR = "vm_data/ready"
LOG_DIR = "logs"

# Create directories
os.makedirs(QUEUE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(READY_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Logging
DATE = datetime.now().strftime('%Y%m%d')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_DIR}/vm_processor_{DATE}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Feature Extraction
# ============================================================================
def extract_features_for_batch(df_urls: pd.DataFrame, batch_date: str):
    """
    Extract DNS and WHOIS features for batch of URLs.

    Args:
        df_urls: DataFrame with 'url' and 'label' columns
        batch_date: Date string (YYYYMMDD) for output files

    Returns:
        (dns_file, whois_file) paths
    """
    logger.info(f"Processing batch: {batch_date}")
    logger.info(f"  Total URLs: {len(df_urls)}")

    dns_results = []
    whois_results = []

    start_time = time.time()

    for idx, row in df_urls.iterrows():
        url = row['url']
        label = row.get('label', 'unknown')

        # DNS features
        try:
            dns_feats = extract_single_domain_features(url)
            dns_feats['url'] = url
            dns_feats['label'] = label
            dns_results.append(dns_feats)
        except Exception as e:
            logger.warning(f"  DNS extraction failed for {url}: {e}")
            dns_results.append({'url': url, 'label': label, 'error': str(e)})

        # WHOIS features
        try:
            whois_feats = extract_single_whois_features(url, live_lookup=True)
            whois_feats['url'] = url
            whois_feats['label'] = label
            whois_results.append(whois_feats)
        except Exception as e:
            logger.warning(f"  WHOIS extraction failed for {url}: {e}")
            whois_results.append({'url': url, 'label': label, 'error': str(e)})

        # Progress logging
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            remaining = (len(df_urls) - idx - 1) / rate if rate > 0 else 0
            logger.info(f"  Progress: {idx + 1}/{len(df_urls)} URLs ({rate:.1f} URLs/sec, {remaining/60:.1f} min remaining)")

    # Save results
    dns_df = pd.DataFrame(dns_results)
    whois_df = pd.DataFrame(whois_results)

    dns_output = f"{OUTPUT_DIR}/dns_{batch_date}.csv"
    whois_output = f"{OUTPUT_DIR}/whois_{batch_date}.csv"

    dns_df.to_csv(dns_output, index=False)
    whois_df.to_csv(whois_output, index=False)

    elapsed = time.time() - start_time
    logger.info(f"  ✅ Batch complete in {elapsed/60:.1f} minutes")
    logger.info(f"     DNS output: {dns_output} ({len(dns_df)} records)")
    logger.info(f"     WHOIS output: {whois_output} ({len(whois_df)} records)")

    return dns_output, whois_output


def create_ready_signal(batch_date: str, dns_file: str, whois_file: str, url_count: int):
    """
    Create .ready file to signal MacBook that batch is complete.

    Args:
        batch_date: Date string (YYYYMMDD)
        dns_file: Path to DNS output file
        whois_file: Path to WHOIS output file
        url_count: Number of URLs processed
    """
    ready_file = f"{READY_DIR}/batch_{batch_date}.ready"

    with open(ready_file, 'w') as f:
        f.write(f"batch_date: {batch_date}\n")
        f.write(f"completed_at: {datetime.now().isoformat()}\n")
        f.write(f"dns_file: {dns_file}\n")
        f.write(f"whois_file: {whois_file}\n")
        f.write(f"url_count: {url_count}\n")

    logger.info(f"  ✅ Ready signal created: {ready_file}")


# ============================================================================
# Main Processing Loop
# ============================================================================
def process_queue():
    """Process all batches in queue (one iteration)"""
    # Find queued batches
    queue_files = sorted(glob.glob(f"{QUEUE_DIR}/batch_*.csv"))

    if not queue_files:
        return 0  # No batches to process

    logger.info(f"📦 Found {len(queue_files)} batches to process")
    processed_count = 0

    for queue_file in queue_files:
        batch_filename = os.path.basename(queue_file)
        batch_date = batch_filename.replace('batch_', '').replace('.csv', '')

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Processing: {batch_filename}")
        logger.info("=" * 80)

        # Check if already processed
        ready_file = f"{READY_DIR}/batch_{batch_date}.ready"
        if os.path.exists(ready_file):
            logger.info(f"⏭️  Already processed - skipping")
            continue

        # Load URLs
        try:
            df = pd.read_csv(queue_file)
            logger.info(f"  Loaded {len(df)} URLs from {queue_file}")
        except Exception as e:
            logger.error(f"  ❌ Failed to load {queue_file}: {e}")
            continue

        # Extract features
        try:
            dns_file, whois_file = extract_features_for_batch(df, batch_date)
        except Exception as e:
            logger.error(f"  ❌ Feature extraction failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Create ready signal
        create_ready_signal(batch_date, dns_file, whois_file, len(df))

        # Archive processed queue file
        archive_path = queue_file.replace("/url_queue/", "/url_queue/processed_")
        os.makedirs(os.path.dirname(archive_path), exist_ok=True)
        os.rename(queue_file, archive_path)
        logger.info(f"  ✅ Archived queue file: {archive_path}")

        processed_count += 1

    return processed_count


def main():
    """Continuous processing loop - runs 24/7"""
    logger.info("=" * 80)
    logger.info("VM Daily Processor Started (Continuous Mode)")
    logger.info("=" * 80)
    logger.info(f"Queue directory: {QUEUE_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Ready signals: {READY_DIR}")
    logger.info("")
    logger.info("Watching for new batches... (Ctrl+C to stop)")
    logger.info("")

    check_interval = 60  # Check every 60 seconds
    iteration = 0

    while True:
        iteration += 1

        try:
            processed = process_queue()

            if processed > 0:
                logger.info("")
                logger.info("=" * 80)
                logger.info(f"✅ Processed {processed} batches successfully")
                logger.info("=" * 80)
                logger.info("")
                logger.info("Waiting for new batches...")
            else:
                # Only log every 10 minutes when idle
                if iteration % 10 == 0:
                    logger.info(f"⏳ Still watching... (checked {iteration} times)")

        except Exception as e:
            logger.error(f"❌ Error processing queue: {e}")
            import traceback
            traceback.print_exc()

        # Sleep before next check
        time.sleep(check_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user - shutting down gracefully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
